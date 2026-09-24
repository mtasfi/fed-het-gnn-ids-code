"""Payload encoder: pre-trained transformer + LoRA adapters + temporary linear head.

* Encoder: AutoModel (ModernBERT-base by default), attention-mask mean pooling.
* LoRA targets are resolved against named_modules() of the LOADED model with
  `encoder.lora.target_regex`. ModernBERT fuses Q, K and V into `attn.Wqkv`;
  there are no `query` / `value` modules, so name-based defaults would adapt
  nothing. The resolved names are logged and written to the run directory, and
  a run with zero trainable LoRA parameters fails loudly (plan pitfall 7).
* Trainable weights stay fp32; on CUDA the forward pass runs under fp16
  autocast (T4: no bf16, no FlashAttention-2). `encoder.precision: fp32` turns
  autocast off; the reason is logged.
"""

import logging
import re
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def load_backbone(enc_cfg, device):
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    kwargs = {'revision': enc_cfg.get('revision') or 'main'}
    tokenizer = AutoTokenizer.from_pretrained(enc_cfg.hf_id, **kwargs)
    config = AutoConfig.from_pretrained(enc_cfg.hf_id, **kwargs)
    if hasattr(config, 'reference_compile'):
        config.reference_compile = False   # older ModernBERT code may torch.compile on first use
    model = AutoModel.from_pretrained(enc_cfg.hf_id, config=config, torch_dtype=torch.float32,
                                      attn_implementation=enc_cfg.get('attn_implementation', 'sdpa'), **kwargs)
    identity = {
        'hf_id': enc_cfg.hf_id,
        'requested_revision': kwargs['revision'],
        'resolved_commit': getattr(model.config, '_commit_hash', None),
        'tokenizer_class': type(tokenizer).__name__,
        'tokenizer_commit': getattr(tokenizer, 'init_kwargs', {}).get('_commit_hash'),
        'hidden_size': int(model.config.hidden_size),
    }
    return tokenizer, model.to(device), identity


def resolve_lora_targets(model: nn.Module, pattern: str) -> List[str]:
    regex = re.compile(pattern)
    names = [n for n, m in model.named_modules() if regex.fullmatch(n) and isinstance(m, nn.Linear)]
    if not names:
        linear = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)][:40]
        raise RuntimeError(
            f"LoRA target regex '{pattern}' matched no Linear module. First Linear modules of the loaded "
            f"model: {linear}. Fix encoder.lora.target_regex (ModernBERT: '.*\\.attn\\.Wqkv$').")
    return names


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1.0)


class PayloadEncoder(nn.Module):
    def __init__(self, cfg, num_classes: int, device, use_lora: bool = True):
        super().__init__()
        enc = cfg.encoder
        self.tokenizer, backbone, self.identity = load_backbone(enc, device)
        self.max_tokens = int(enc.max_tokens)
        self.device = device
        self.use_lora = use_lora
        self.lora_targets: List[str] = []

        for p in backbone.parameters():
            p.requires_grad = False
        if use_lora:
            from peft import LoraConfig, get_peft_model
            self.lora_targets = resolve_lora_targets(backbone, enc.lora.target_regex)
            lcfg = LoraConfig(r=int(enc.lora.r), lora_alpha=int(enc.lora.alpha), lora_dropout=float(enc.lora.dropout),
                              target_modules=self.lora_targets, bias='none')
            backbone = get_peft_model(backbone, lcfg)
        self.backbone = backbone
        d = int(self.identity['hidden_size'])
        self.head = nn.Linear(d, num_classes).to(device)   # W_c: temporary, discarded after Phase 1

        lora_params = sum(p.numel() for n, p in self.backbone.named_parameters() if p.requires_grad)
        if use_lora and lora_params == 0:
            raise RuntimeError(f"LoRA resolved {len(self.lora_targets)} target modules but produced 0 trainable "
                               f"parameters; refusing to run a Phase 1 that would train nothing.")
        self.param_counts = {
            'encoder_total': sum(p.numel() for p in self.backbone.parameters()),
            'lora_trainable': lora_params,
            'head': sum(p.numel() for p in self.head.parameters()),
        }
        self.amp = device.type == 'cuda' and enc.get('precision', 'fp16') == 'fp16'
        if device.type == 'cuda' and not self.amp:
            logger.warning(f"Phase-1 autocast disabled: encoder.precision={enc.get('precision')} "
                           f"({enc.get('precision_reason', 'no reason given in config')})")
        logger.info(f"Encoder {enc.hf_id} ({self.identity['resolved_commit']}), LoRA targets "
                    f"{len(self.lora_targets)} modules, trainable LoRA params {lora_params:,}, "
                    f"head {self.param_counts['head']:,}, total encoder {self.param_counts['encoder_total']:,}")

    def tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer(list(texts), truncation=True, max_length=self.max_tokens, padding='longest',
                               return_tensors='pt')
        return {k: v.to(self.device) for k, v in batch.items() if k in ('input_ids', 'attention_mask')}

    def forward(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = self.tokenize(texts)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
            hidden = self.backbone(**batch).last_hidden_state
            pooled = mean_pool(hidden.float(), batch['attention_mask'])
        logits = self.head(pooled)
        return pooled, logits

    @torch.no_grad()
    def embed(self, texts: List[str]) -> torch.Tensor:
        self.eval()
        pooled, _ = self.forward(texts)
        return pooled

    # ---------------------------------------------------------------- federated state
    def trainable_state(self) -> Dict[str, torch.Tensor]:
        """What a client uploads: LoRA adapters Delta and the head W_c."""
        state = {f'backbone.{n}': p.detach().cpu().clone()
                 for n, p in self.backbone.named_parameters() if p.requires_grad}
        state.update({f'head.{n}': p.detach().cpu().clone() for n, p in self.head.named_parameters()})
        return state

    def load_trainable_state(self, state: Dict[str, torch.Tensor]):
        params = dict(self.named_parameters())
        missing = [k for k in state if k not in params]
        if missing:
            raise KeyError(f"State keys not in model: {missing[:5]}")
        with torch.no_grad():
            for key, value in state.items():
                params[key].copy_(value.to(params[key].device))

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
