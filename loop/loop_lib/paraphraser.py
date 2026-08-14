"""Style-injection 재서술기 G — ko-BART + 학습가능 스타일 임베딩 + fusion layer.

MASH §3 구조 (notes/30 § Method 2):
  x_ai → encoder → content representation h
  h ⊕ s_style (learnable, ai/human 2종) → fusion W_p·[h; s] + b_p → decoder
  L_recon = CE(디코더 출력 | x_ai, s_ai   ; 정답 x_ai)
  L_trans = CE(디코더 출력 | x_ai, s_human; 정답 x_human)

구현 결정:
- fusion 은 항등 초기화 (W=[I; 0], b=0) — 학습 시작 시점의 모델이 사전학습 ko-BART 와
  정확히 같아서 SFT 초반이 안정된다. 스타일 임베딩은 std 0.02 정규분포.
- 인터페이스는 (input_ids, attention_mask, style_ids, labels). 손실은 호출부가
  per-sample 로 계산한다 (λ 가중 결합·DPO logp 에 공통으로 필요).
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import io_utils

STYLE_AI, STYLE_HUMAN = 0, 1


def _ensure_special_ids(model, tok):
    """kobart config 에 빠져 있을 수 있는 생성용 특수 토큰을 확정한다 (BART 관례: start=eos)."""
    c = model.config
    if c.pad_token_id is None:
        c.pad_token_id = tok.pad_token_id
    if c.eos_token_id is None:
        c.eos_token_id = tok.eos_token_id
    if c.decoder_start_token_id is None:
        c.decoder_start_token_id = c.eos_token_id
    g = model.generation_config
    g.pad_token_id = c.pad_token_id
    g.eos_token_id = c.eos_token_id
    g.decoder_start_token_id = c.decoder_start_token_id
    # forced BOS 는 끈다. kobart 는 bos==eos(=1) 라, 첫 토큰을 bos 로 강제하면 그게 곧 eos 라
    # min_new_tokens 를 무시하고 즉시 생성이 끝난다(빈 문자열). BART/kobart 는 mBART 와 달리
    # forced BOS 가 필요 없다 — decoder_start_token_id 로 시작해 자유 생성한다.
    c.forced_bos_token_id = None
    g.forced_bos_token_id = None


def shift_right(labels, pad_id, start_id):
    """labels → decoder_input_ids. -100 은 pad 로 치환."""
    x = labels.new_full(labels.shape, pad_id)
    x[:, 1:] = labels[:, :-1]
    x[:, 0] = start_id
    x.masked_fill_(x == -100, pad_id)
    return x


class StyleParaphraser(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        d = base.config.d_model
        self.style_emb = nn.Embedding(2, d)
        self.fusion = nn.Linear(2 * d, d)
        self._init_identity()

    def _init_identity(self):
        d = self.base.config.d_model
        with torch.no_grad():
            nn.init.normal_(self.style_emb.weight, std=0.02)
            self.fusion.weight.zero_()
            self.fusion.weight[:, :d].copy_(torch.eye(d))
            self.fusion.bias.zero_()

    def fuse(self, input_ids, attention_mask, style_ids):
        h = self.base.get_encoder()(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        s = self.style_emb(style_ids).unsqueeze(1).expand(-1, h.size(1), -1)
        return self.fusion(torch.cat([h, s], dim=-1))

    def logits_for(self, input_ids, attention_mask, style_ids, labels):
        """teacher forcing 로짓 (B, T, V)."""
        from transformers.modeling_outputs import BaseModelOutput

        fused = self.fuse(input_ids, attention_mask, style_ids)
        dec_in = shift_right(labels, self.base.config.pad_token_id, self.base.config.decoder_start_token_id)
        out = self.base(
            encoder_outputs=BaseModelOutput(last_hidden_state=fused),
            attention_mask=attention_mask,
            decoder_input_ids=dec_in,
        )
        return out.logits

    def label_scores(self, input_ids, attention_mask, style_ids, labels, reduce="mean"):
        """per-sample 손실/로그확률.

        reduce="mean" → 토큰 평균 CE (SFT 용, 길이에 강건)
        reduce="sum"  → Σ log p(y|x,s)  (DPO 용)
        """
        logits = self.logits_for(input_ids, attention_mask, style_ids, labels)
        V = logits.size(-1)
        ce = F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-100,
                             reduction="none").view(labels.size())
        valid = (labels != -100).float()
        per_tok_sum = (ce * valid).sum(dim=1)
        if reduce == "mean":
            return per_tok_sum / valid.sum(dim=1).clamp(min=1)
        if reduce == "sum":
            return -per_tok_sum  # sum log-prob
        raise ValueError(reduce)

    def _decode_logprob(self, fused, attention_mask, labels):
        """인코딩 결과를 재사용해 Σ log p(labels) 만 계산."""
        from transformers.modeling_outputs import BaseModelOutput

        dec_in = shift_right(labels, self.base.config.pad_token_id, self.base.config.decoder_start_token_id)
        logits = self.base(
            encoder_outputs=BaseModelOutput(last_hidden_state=fused),
            attention_mask=attention_mask,
            decoder_input_ids=dec_in,
        ).logits
        V = logits.size(-1)
        ce = F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-100,
                             reduction="none").view(labels.size())
        return -(ce * (labels != -100).float()).sum(dim=1)

    def pair_logprobs(self, input_ids, attention_mask, style_ids, labels_w, labels_l):
        """DPO 용 — 인코더 1회, 디코더 2회로 (Σlogp_w, Σlogp_l)."""
        fused = self.fuse(input_ids, attention_mask, style_ids)
        return (self._decode_logprob(fused, attention_mask, labels_w),
                self._decode_logprob(fused, attention_mask, labels_l))

    @torch.no_grad()
    def generate_from(self, input_ids, attention_mask, style_ids, **gen_kwargs):
        from transformers.modeling_outputs import BaseModelOutput

        fused = self.fuse(input_ids, attention_mask, style_ids)
        return self.base.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=fused),
            attention_mask=attention_mask,
            **gen_kwargs,
        )


# ── 생성/저장/로드 ──────────────────────────────────────────────
def create(model_name, tiny=False):
    from transformers import AutoTokenizer, BartConfig, BartForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(model_name)
    if tiny:
        c = BartConfig(
            vocab_size=len(tok), d_model=64, encoder_layers=2, decoder_layers=2,
            encoder_attention_heads=2, decoder_attention_heads=2,
            encoder_ffn_dim=128, decoder_ffn_dim=128, max_position_embeddings=512,
            pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id,
            eos_token_id=tok.eos_token_id,
        )
        base = BartForConditionalGeneration(c)
    else:
        base = BartForConditionalGeneration.from_pretrained(model_name)
    _ensure_special_ids(base, tok)
    return StyleParaphraser(base), tok


def save(model: StyleParaphraser, tok, out_dir, meta):
    out_dir = Path(out_dir)
    base_dir = out_dir / "base"
    model.base.save_pretrained(str(base_dir))
    tok.save_pretrained(str(base_dir))
    torch.save({"style_emb": model.style_emb.state_dict(), "fusion": model.fusion.state_dict()},
               str(out_dir / "style_head.pt"))
    io_utils.write_json(out_dir / "meta.json", meta)


def load(dir_path, device):
    from transformers import AutoTokenizer, BartForConditionalGeneration

    dir_path = Path(dir_path)
    base = BartForConditionalGeneration.from_pretrained(str(dir_path / "base"))
    tok = AutoTokenizer.from_pretrained(str(dir_path / "base"))
    _ensure_special_ids(base, tok)
    model = StyleParaphraser(base)
    head = torch.load(str(dir_path / "style_head.pt"), map_location="cpu")
    model.style_emb.load_state_dict(head["style_emb"])
    model.fusion.load_state_dict(head["fusion"])
    model.to(device)
    model.eval()
    return model, tok


# ── 일괄 재서술 ────────────────────────────────────────────────
@torch.no_grad()
def paraphrase_texts(model, tok, texts, device, para_cfg, style=STYLE_HUMAN,
                     num_return=1, greedy=True, temperature=1.0, top_p=0.95,
                     batch_size=4, seed=None, min_new_tokens=0):
    """texts → 재서술 list[list[str]] (원문당 num_return 개). 입력 순서 보존.

    min_new_tokens: EOS 조기 방출로 빈 문자열이 나오는 것을 막는 하한. 초록(수백 토큰)에는
    항상 만족돼 실제 모델엔 영향 없고, 퇴화한 빈 재서술만 걸러진다."""
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)
    max_src, max_tgt = para_cfg.max_src_len, para_cfg.max_tgt_len
    gen = dict(max_length=max_tgt, num_return_sequences=num_return)
    if min_new_tokens:
        gen["min_new_tokens"] = min_new_tokens
    if greedy:
        gen.update(do_sample=False, num_beams=1)
    else:
        gen.update(do_sample=True, temperature=temperature, top_p=top_p)

    out = []
    for s in range(0, len(texts), batch_size):
        chunk = texts[s : s + batch_size]
        enc = tok(chunk, truncation=True, max_length=max_src, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        style_ids = torch.full((len(chunk),), style, dtype=torch.long, device=device)
        ids = model.generate_from(enc["input_ids"], enc["attention_mask"], style_ids, **gen)
        texts_out = tok.batch_decode(ids, skip_special_tokens=True)
        for i in range(len(chunk)):
            out.append([t.strip() for t in texts_out[i * num_return : (i + 1) * num_return]])
    return out
