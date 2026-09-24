"""A randomly initialised, tiny ModernBERT + byte-level BPE tokenizer saved to disk,
so the pipeline can be exercised offline. TESTS ONLY: never used for any result.

    python tests/make_tiny_encoder.py /tmp/tiny-modernbert
"""

import sys

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast

CORPUS = ["GET /index.html HTTP/1.1", "POST /login user=admin&pass=123456", "<ScRiPt>alert(1)</ScRiPt>",
          "' OR 1=1 --", "UNION SELECT user FROM users", "HTTP/1.1 200 OK", "<NP>"] * 50


def build(path: str):
    tok = Tokenizer(models.BPE(unk_token='[UNK]'))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=['[PAD]', '[CLS]', '[SEP]', '[UNK]', '[MASK]'],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(CORPUS, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token='[PAD]', cls_token='[CLS]', sep_token='[SEP]',
                                   unk_token='[UNK]', mask_token='[MASK]')
    fast.save_pretrained(path)
    cfg = ModernBertConfig(vocab_size=fast.vocab_size, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                           intermediate_size=64, max_position_embeddings=512, pad_token_id=0, cls_token_id=1,
                           sep_token_id=2, bos_token_id=1, eos_token_id=2, global_attn_every_n_layers=1)
    ModernBertModel(cfg).save_pretrained(path)
    return path


if __name__ == '__main__':
    build(sys.argv[1])
    print(f"tiny encoder written to {sys.argv[1]}")
