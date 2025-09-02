# ==============================================================================
# 파일: models/parallel_decoder.py
# 역할: Whisper 디코더 레이어를 병렬(Self/Cross 병렬) + γ-이행 지원
#       - γ=1.0이면 원본(순차)과 동일한 Cross-Attn 입력, γ→0.0로 내려가면 병렬 입력
#       - 어텐션 시그니처 버전차 대응(inspection)
# ==============================================================================

from typing import Optional, Tuple
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.whisper.modeling_whisper import (
    WhisperDecoderLayer,
    WhisperConfig,
)

class ParallelWhisperDecoderLayer(nn.Module):
    def __init__(self, ref_layer: WhisperDecoderLayer, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # ====== 가중치/모듈을 "그대로" 재사용 (프리트레인 보존) ======
        self.embed_dim = ref_layer.embed_dim
        self.self_attn = ref_layer.self_attn
        self.cross_attn = ref_layer.encoder_attn
        self.fc1 = ref_layer.fc1
        self.fc2 = ref_layer.fc2
        self.activation_fn = ref_layer.activation_fn

        self.self_attn_layer_norm   = ref_layer.self_attn_layer_norm
        self.encoder_attn_layer_norm= ref_layer.encoder_attn_layer_norm
        self.final_layer_norm       = ref_layer.final_layer_norm

        # ====== dropout 확률은 ref_layer의 속성에서 읽기 ======
        # (WhisperDecoderLayer는 self.dropout / self.activation_dropout을 가짐)
        self.dropout_p = float(getattr(ref_layer, "dropout", 0.0))
        self.activation_dropout_p = float(getattr(ref_layer, "activation_dropout", 0.0))

        # 감마 스위치
        self.register_buffer("gamma", torch.tensor(1.0), persistent=False)

    # --- 어텐션 호출(버전 차이 안전 처리) ---
    def _call_self_attn(self, hidden_states, attention_mask, layer_head_mask,
                        past_key_value, output_attentions, causal_mask):
        sig = inspect.signature(self.self_attn.forward).parameters
        kwargs = dict(hidden_states=hidden_states,
                      past_key_value=past_key_value,
                      output_attentions=output_attentions)
        if "layer_head_mask" in sig: kwargs["layer_head_mask"] = layer_head_mask
        if "attention_mask" in sig: kwargs["attention_mask"] = attention_mask
        if "causal_mask" in sig and causal_mask is not None:
            kwargs["causal_mask"] = causal_mask
        if "is_causal" in sig and "causal_mask" not in sig:
            kwargs["is_causal"] = True
        return self.self_attn(**kwargs)

    def _call_cross_attn(self, hidden_states, key_value_states, encoder_attention_mask,
                         layer_head_mask, past_key_value, output_attentions):
        sig = inspect.signature(self.cross_attn.forward).parameters
        kwargs = dict(hidden_states=hidden_states,
                      key_value_states=key_value_states,
                      past_key_value=past_key_value,
                      output_attentions=output_attentions)
        if "layer_head_mask" in sig: kwargs["layer_head_mask"] = layer_head_mask
        if "attention_mask" in sig: kwargs["attention_mask"] = encoder_attention_mask
        if "is_causal" in sig: kwargs["is_causal"] = False
        return self.cross_attn(**kwargs)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value: Optional[Tuple] = None,  # (past_self, past_cross) or None
        output_attentions: bool = False,
        use_cache: bool = True,                  # 상위 요구; 어텐션엔 넘기지 않음
        causal_mask=None,                        # 포지셔널로 전달될 수 있음
        **kwargs,
    ):
        # past 분리
        past_self = past_cross = None
        if past_key_value is not None:
            if isinstance(past_key_value, (tuple, list)) and len(past_key_value) == 2:
                past_self, past_cross = past_key_value
            else:
                past_self = past_key_value

        # ===== Self-Attn =====
        residual = hidden_states
        self_in = self.self_attn_layer_norm(hidden_states)
        self_outputs = self._call_self_attn(
            hidden_states=self_in,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            past_key_value=past_self,
            output_attentions=output_attentions,
            causal_mask=causal_mask,
        )
        self_out = self_outputs[0]
        self_weights = self_outputs[1] if output_attentions and len(self_outputs) >= 2 else None
        self_present = (
            self_outputs[2] if output_attentions and len(self_outputs) >= 3
            else (self_outputs[1] if (not output_attentions and len(self_outputs) >= 2) else None)
        )

        # ===== Cross-Attn 입력(γ-혼합) =====
        # 원본: LN2(residual + dropout(self_out))  ← γ=1.0
        # 완전 병렬: LN2(residual)                 ← γ=0.0
        self_out_d = F.dropout(self_out, p=self.dropout_p, training=self.training)

        # ★ gamma를 hidden_states dtype으로 맞춰서 업캐스트 방지
        gamma = self.gamma
        if gamma.dtype != hidden_states.dtype:
            gamma = gamma.to(hidden_states.dtype)
                    
        cross_in = self.encoder_attn_layer_norm(residual + gamma * self_out_d)

        # ===== Cross-Attn =====
        cross_outputs = self._call_cross_attn(
            hidden_states=cross_in,
            key_value_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            layer_head_mask=cross_attn_layer_head_mask,
            past_key_value=past_cross,
            output_attentions=output_attentions,
        )
        cross_out = cross_outputs[0]
        cross_weights = cross_outputs[1] if output_attentions and len(cross_outputs) >= 2 else None
        cross_present = (
            cross_outputs[2] if output_attentions and len(cross_outputs) >= 3
            else (cross_outputs[1] if (not output_attentions and len(cross_outputs) >= 2) else None)
        )

        # ===== 병합 + FFN =====
        # self_out = F.dropout(self_out, p=self.dropout_p, training=self.training)
        cross_out = F.dropout(cross_out, p=self.dropout_p, training=self.training)
        merged = residual + self_out_d + cross_out

        ffn_residual = merged
        ffn_in = self.final_layer_norm(merged)
        ffn_hidden = self.fc1(ffn_in)
        ffn_hidden = self.activation_fn(ffn_hidden)
        ffn_hidden = F.dropout(ffn_hidden, p=self.activation_dropout_p, training=self.training)
        ffn_hidden = self.fc2(ffn_hidden)
        ffn_hidden = F.dropout(ffn_hidden, p=self.dropout_p, training=self.training)
        hidden_states = ffn_residual + ffn_hidden

        present = (self_present, cross_present) if use_cache else None

        # 리턴 포맷 (Whisper 기대 순서)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_weights,)
        if use_cache:
            outputs += (present,)
        if output_attentions:
            outputs += (cross_weights,)
        return outputs
