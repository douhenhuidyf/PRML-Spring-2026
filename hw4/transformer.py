import os
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import einops


@dataclass
class Config:
    num_tokens: int = 10000
    num_blocks: int = 6
    d_model: int = 512
    d_key: int = 64
    num_heads: int = 8
    d_ff: int = 4 * d_model

    max_src_len: int = 256
    max_tgt_len: int = 128
    dropout: float = 0.1
    pad_token_id: int = 0

    use_attn_value: bool = True
    use_residual: bool = True

    use_moe: bool = False
    num_experts: int = 4

    use_pre_norm: bool = True


class Attention(nn.Module):

    def __init__(self, *args: Any, config: Config, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.scale = config.d_key**-0.5

    def forward(self,
                q: Tensor,
                k: Tensor,
                v: Tensor,
                mask: Tensor | None = None) -> Tensor:
        score = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            score = score.masked_fill(mask == 0, float('-inf'))
        score = F.softmax(score, dim=-1)
        x = torch.matmul(score, v)
        return x


class MHA(nn.Module):

    def __init__(self, *args: Any, config: Config, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        d_key = config.d_key
        d_model = config.d_model
        self.num_heads = config.num_heads
        self.use_attn_value = config.use_attn_value

        self.attn = Attention(config=config)

        self.wq = nn.Linear(in_features=d_model, out_features=d_model)
        self.wk = nn.Linear(in_features=d_model, out_features=d_model)
        self.wv = nn.Linear(in_features=d_model, out_features=d_model)
        self.wo = nn.Linear(in_features=d_model, out_features=d_model)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x) if self.use_attn_value else k
        q = einops.rearrange(q, "b seq (h d) -> b h seq d", h=self.num_heads)
        k = einops.rearrange(k, "b seq (h d) -> b h seq d", h=self.num_heads)
        v = einops.rearrange(v, "b seq (h d) -> b h seq d", h=self.num_heads)
        if mask is not None:
            mask = einops.rearrange(mask, "b seq1 seq2 -> b 1 seq1 seq2")
        x = self.attn(q, k, v, mask)  # b, h, seq, d_key
        x = einops.rearrange(x, "b h seq d -> b seq (h d)")  # b, seq, d_model
        x = self.wo(x)  # b, seq, d_model
        return x


class CrossMHA(nn.Module):

    def __init__(self, *args: Any, config: Config, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        d_model = config.d_model
        self.num_heads = config.num_heads
        self.use_attn_value = config.use_attn_value

        self.attn = Attention(config=config)

        self.wq = nn.Linear(in_features=d_model, out_features=d_model)
        self.wk = nn.Linear(in_features=d_model, out_features=d_model)
        self.wv = nn.Linear(in_features=d_model, out_features=d_model)
        self.wo = nn.Linear(in_features=d_model, out_features=d_model)

    def forward(self,
                q_input: Tensor,
                kv_input: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
        q = self.wq(q_input)
        k = self.wk(kv_input)
        v = self.wv(kv_input) if self.use_attn_value else k
        q = einops.rearrange(q, "b seq (h d) -> b h seq d", h=self.num_heads)
        k = einops.rearrange(k, "b seq (h d) -> b h seq d", h=self.num_heads)
        v = einops.rearrange(v, "b seq (h d) -> b h seq d", h=self.num_heads)
        if mask is not None:
            mask = einops.rearrange(mask, "b seq1 seq2 -> b 1 seq1 seq2")
        x = self.attn(q, k, v, mask)
        x = einops.rearrange(x, "b h seq d -> b seq (h d)")
        x = self.wo(x)
        return x


class FFN(nn.Module):

    def __init__(self, *args: Any, config: Config, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        d_model = config.d_model
        d_ff = config.d_ff
        self.use_moe = config.use_moe

        if self.use_moe:
            num_experts = config.num_experts
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(in_features=d_model, out_features=d_ff),
                    nn.ReLU(),
                    nn.Linear(in_features=d_ff, out_features=d_model),
                ) for _ in range(num_experts)
            ])
            self.gate = nn.Linear(in_features=d_model,
                                  out_features=num_experts)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(in_features=d_model, out_features=d_ff),
                nn.ReLU(),
                nn.Linear(in_features=d_ff, out_features=d_model),
            )

    def forward(self, x: Tensor) -> Tensor:
        if self.use_moe:
            gate_score = F.softmax(self.gate(x), dim=-1)  # b, seq, num_experts
            expert_outputs = torch.stack(
                [expert(x) for expert in self.experts],
                dim=-1)  # b, seq, d_model, num_experts
            x = torch.einsum("b s e, b s d e -> b s d", gate_score,
                             expert_outputs)  # b, seq, d_model
        else:
            x = self.ffn(x)  # b, seq, d_model
        return x


class TransformerBlock(nn.Module):

    def __init__(self, *args: Any, config: Config, **kwargs: Any) -> None:

        super().__init__(*args, **kwargs)

        d_model = config.d_model
        self.use_pre_norm = config.use_pre_norm
        self.use_residual = config.use_residual

        self.mha = MHA(config=config)
        self.ffn = FFN(config=config)

        self.norm1 = nn.LayerNorm(normalized_shape=d_model)
        self.norm2 = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if self.use_pre_norm:
            if self.use_residual:
                x = x + self.mha(self.norm1(x), mask)
                x = x + self.ffn(self.norm2(x))
            else:
                x = self.mha(self.norm1(x), mask)
                x = self.ffn(self.norm2(x))
        else:
            if self.use_residual:
                x = self.norm1(x + self.mha(x, mask))
                x = self.norm2(x + self.ffn(x))
            else:
                x = self.norm1(self.mha(x, mask))
                x = self.norm2(self.ffn(x))
        return x


class Encoder(nn.Module):
    """
    Transformer encoder for translation.
    """

    def __init__(self, *args: Any, config: Config, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.embedding = nn.Embedding(num_embeddings=config.num_tokens,
                                      embedding_dim=config.d_model)
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, config.max_src_len, config.d_model))
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.Sequential()
        for _ in range(config.num_blocks):
            self.blocks.append(TransformerBlock(config=config))

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        pos = self.positional_embedding[:, :x.size(1), :]
        x = self.dropout(self.embedding(x) + pos)
        for block in self.blocks:
            x = block(x, mask)
        return x


class Decoder(nn.Module):
    """
    Transformer decoder for translation.
    """

    def __init__(self, *args: Any, config: Config, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.embedding = nn.Embedding(num_embeddings=config.num_tokens,
                                      embedding_dim=config.d_model)
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, config.max_tgt_len, config.d_model))
        self.dropout = nn.Dropout(config.dropout)

        self.self_blocks = nn.ModuleList([
            TransformerBlock(config=config) for _ in range(config.num_blocks)
        ])
        self.cross_blocks = nn.ModuleList(
            [CrossMHA(config=config) for _ in range(config.num_blocks)])
        self.norms = nn.ModuleList(
            [nn.LayerNorm(config.d_model) for _ in range(config.num_blocks)])
        self.use_pre_norm = config.use_pre_norm
        self.use_residual = config.use_residual

    def forward(self,
                x: Tensor,
                enc_out: Tensor,
                self_mask: Optional[Tensor] = None,
                cross_mask: Optional[Tensor] = None) -> Tensor:
        pos = self.positional_embedding[:, :x.size(1), :]
        x = self.dropout(self.embedding(x) + pos)
        for idx, block in enumerate(self.self_blocks):
            x = block(x, self_mask)
            if self.use_pre_norm:
                normed = self.norms[idx](x)
                if self.use_residual:
                    x = x + self.cross_blocks[idx](normed, enc_out, cross_mask)
                else:
                    x = self.cross_blocks[idx](normed, enc_out, cross_mask)
            else:
                if self.use_residual:
                    x = self.norms[idx](x + self.cross_blocks[idx]
                                        (x, enc_out, cross_mask))
                else:
                    x = self.norms[idx](self.cross_blocks[idx](x, enc_out,
                                                               cross_mask))
        return x


class Transformer(nn.Module):

    def __init__(self, *args: Any, config: Config, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if config.d_model != config.num_heads * config.d_key:
            raise ValueError("d_model must equal num_heads * d_key")
        self.config = config
        self.encoder = Encoder(config=config)
        self.decoder = Decoder(config=config)
        self.lm_head = nn.Linear(config.d_model, config.num_tokens)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def _make_src_mask(self, attention_mask: Tensor) -> Tensor:
        return attention_mask.unsqueeze(1).expand(-1, attention_mask.size(1),
                                                  -1)

    def _make_cross_mask(self, src_attention_mask: Tensor,
                         tgt_len: int) -> Tensor:
        return src_attention_mask.unsqueeze(1).expand(-1, tgt_len, -1)

    def _make_tgt_mask(self, attention_mask: Tensor) -> Tensor:
        seq_len = attention_mask.size(1)
        causal = torch.tril(
            torch.ones((seq_len, seq_len),
                       device=attention_mask.device,
                       dtype=attention_mask.dtype))
        mask = attention_mask.unsqueeze(1).expand(-1, seq_len, -1)
        return mask * causal

    def forward(self,
                src_input_ids: Tensor,
                src_attention_mask: Tensor,
                tgt_input_ids: Tensor,
                tgt_attention_mask: Tensor,
                labels: Optional[Tensor] = None) -> dict:
        src_mask = self._make_src_mask(src_attention_mask)
        tgt_mask = self._make_tgt_mask(tgt_attention_mask)
        enc_out = self.encoder(src_input_ids, src_mask)
        dec_out = self.decoder(tgt_input_ids,
                               enc_out,
                               self_mask=tgt_mask,
                               cross_mask=self._make_cross_mask(
                                   src_attention_mask, tgt_input_ids.size(1)))
        logits = self.lm_head(dec_out)
        output = {"logits": logits}
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.size(-1)),
                                 labels.view(-1))
            output["loss"] = loss
        return output

    @torch.no_grad()
    def generate(self,
                 src_input_ids: Tensor,
                 src_attention_mask: Tensor,
                 max_new_tokens: int,
                 bos_token_id: int,
                 eos_token_id: Optional[int] = None) -> Tensor:
        self.eval()
        batch_size = src_input_ids.size(0)
        device = src_input_ids.device
        tgt_input_ids = torch.full((batch_size, 1),
                                   bos_token_id,
                                   dtype=src_input_ids.dtype,
                                   device=device)
        tgt_attention_mask = torch.ones_like(tgt_input_ids)
        enc_out = self.encoder(src_input_ids,
                               self._make_src_mask(src_attention_mask))
        for _ in range(max_new_tokens):
            tgt_mask = self._make_tgt_mask(tgt_attention_mask)
            dec_out = self.decoder(tgt_input_ids,
                                   enc_out,
                                   self_mask=tgt_mask,
                                   cross_mask=self._make_cross_mask(
                                       src_attention_mask,
                                       tgt_input_ids.size(1)))
            logits = self.lm_head(dec_out[:, -1:, :])
            next_token = torch.argmax(logits, dim=-1)
            tgt_input_ids = torch.cat([tgt_input_ids, next_token], dim=1)
            next_mask = torch.ones_like(next_token)
            tgt_attention_mask = torch.cat([tgt_attention_mask, next_mask],
                                           dim=1)
            if eos_token_id is not None:
                if torch.all(next_token.squeeze(1) == eos_token_id):
                    break
        return tgt_input_ids
