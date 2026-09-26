"""Embed every readable payload segment exactly once, into a cache Phase 2 reuses.

Sources (config `payload.encoder_source`):
    lora    phi* from this run's Phase 1 (E1, E5)                 tag lora_<enc>_r<r>_R<R1>_a<alpha>_s<seed>
    frozen  the same pre-trained encoder, no adaptation (A3)      tag frozen_<enc>
    hash    HashingVectorizer char 3-5-grams, 768-d, no fit (A2)   tag hash768
Ablations that reuse E1 embeddings look the `lora` tag up and fail loudly if
it is missing instead of re-running Phase 1 (plan: never rerun Phase 1 unless
the experiment changes the payload representation).

Cache layout: <work>/<ds>/cache/embeddings/<tag>/{emb.npy (fp16), index.parquet, meta.json}.
Only flows with m_i = 1 are encoded; everything else skips the encoder.
"""

import json
import logging
import os
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from fhf.data.store import Store
from fhf.phase2.build_heterograph import EmbeddingLookup

logger = logging.getLogger(__name__)


def encoder_short(cfg) -> str:
    return cfg.encoder.hf_id.split('/')[-1].lower().replace('_', '-')


def embedding_tag(cfg, source: str) -> str:
    smoke = ('_smoke' if cfg.get('smoke') else '') + (f"_probe{cfg.subset_flows}" if cfg.get('subset_flows') else '')
    if source == 'lora':
        # Phase 1 is trained on the clients of one partition draw, so the draw is part of the tag.
        # partition.seed 0 (the main partition) keeps the original tag, so existing caches stay valid.
        pseed = int(cfg.partition.get('seed', 0) or 0)
        return (f"lora_{encoder_short(cfg)}_r{cfg.encoder.lora.r}_R{cfg.phase1.rounds}"
                f"_a{cfg.partition.alpha}_s{cfg.seed}{f'_p{pseed}' if pseed else ''}{smoke}")
    if source == 'frozen':
        return f"frozen_{encoder_short(cfg)}{smoke}"
    if source == 'hash':
        return f"hash{int(cfg.payload.get('hash_features', 768))}{smoke}"
    raise ValueError(f"Unknown payload.encoder_source '{source}' (lora | frozen | hash | none)")


def segments_to_embed(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.loc[frame['has_payload'] == 1, ['flow_uid', 'payload_texts']].explode('payload_texts')
    rows = rows.dropna(subset=['payload_texts'])
    rows['seg_pos'] = rows.groupby('flow_uid').cumcount()
    return rows.rename(columns={'payload_texts': 'text'}).reset_index(drop=True)


@torch.no_grad()
def encode_texts(model, texts, batch_size: int) -> np.ndarray:
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), int(model.identity['hidden_size'])), dtype=np.float16)
    model.eval()
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        out[idx] = model.embed([texts[j] for j in idx]).float().cpu().numpy().astype(np.float16)
    return out


def hash_texts(texts, n_features: int = 768) -> np.ndarray:
    """A2: fixed char 3-5-gram hashing, no vocabulary, no fit step (plan §2.3)."""
    from sklearn.feature_extraction.text import HashingVectorizer
    vec = HashingVectorizer(analyzer='char', ngram_range=(3, 5), n_features=n_features, lowercase=False,
                            alternate_sign=False, norm='l2')
    return vec.transform(texts).toarray().astype(np.float16)


def cache_exists(cfg, tag: str) -> bool:
    d = Store(cfg).embeddings_dir(tag)
    return all(os.path.exists(os.path.join(d, f)) for f in ('emb.npy', 'index.parquet', 'meta.json'))


def write_cache(cfg, tag: str, seg: pd.DataFrame, emb: np.ndarray, meta: Dict):
    d = Store(cfg).embeddings_dir(tag)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, 'emb.npy'), emb)
    seg[['flow_uid', 'seg_pos']].to_parquet(os.path.join(d, 'index.parquet'), index=False)
    with open(os.path.join(d, 'meta.json'), 'w') as f:
        json.dump({'tag': tag, 'segments': int(len(seg)), 'dim': int(emb.shape[1]), **meta}, f, indent=2, default=str)


def load_cache(cfg, tag: str, flow_uids: Optional[np.ndarray] = None) -> EmbeddingLookup:
    d = Store(cfg).embeddings_dir(tag)
    if not cache_exists(cfg, tag):
        raise FileNotFoundError(
            f"Embedding cache '{tag}' not found in {d}. Paired ablations reuse E1's Phase-1 embeddings: run E1 "
            f"(same dataset, alpha, seed) first.")
    emb = np.load(os.path.join(d, 'emb.npy'), mmap_mode='r')
    index = pd.read_parquet(os.path.join(d, 'index.parquet'))
    if flow_uids is not None:
        keep = index['flow_uid'].isin(set(flow_uids)).to_numpy()
        emb, index = np.asarray(emb[keep]), index[keep]
    return EmbeddingLookup(np.asarray(emb, dtype=np.float32), index['flow_uid'].to_numpy(), index['seg_pos'].to_numpy())


def cache_meta(cfg, tag: str) -> Dict:
    with open(os.path.join(Store(cfg).embeddings_dir(tag), 'meta.json')) as f:
        return json.load(f)


def build_cache(cfg, frame: pd.DataFrame, source: str, tag: str, model=None) -> Dict:
    """Encodes every readable segment of `frame` (all clients, all roles). Each
    client only ever encodes its own flows; running them together here is the
    same computation."""
    seg = segments_to_embed(frame)
    t0 = time.perf_counter()
    if source == 'hash':
        emb = hash_texts(seg['text'].tolist(), int(cfg.payload.get('hash_features', 768)))
        meta = {'source': 'hash', 'vectorizer': "HashingVectorizer(analyzer='char', ngram_range=(3,5), "
                f"n_features={emb.shape[1]}, lowercase=False, alternate_sign=False, norm='l2')"}
    else:
        if model is None:
            raise ValueError("build_cache needs the encoder for lora / frozen sources")
        emb = encode_texts(model, seg['text'].tolist(), int(cfg.encoder.embed_batch_size))
        meta = {'source': source, 'encoder': model.identity, 'lora_targets': model.lora_targets}
    seconds = time.perf_counter() - t0
    meta.update({'embed_seconds': seconds, 'flows_encoded': int(seg['flow_uid'].nunique()),
                 'flows_skipped_m0': int((frame['has_payload'] == 0).sum())})
    write_cache(cfg, tag, seg, emb, meta)
    logger.info(f"Embedded {len(seg):,} segments ({meta['flows_encoded']:,} flows) as '{tag}' in {seconds:.1f}s")
    return meta
