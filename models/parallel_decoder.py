# ==============================================================================
# 파일: models/parallel_decoder.py
# 역할: Whisper 디코더 레이어를 병렬(Self-Attn / Cross-Attn) 구조로 재구성
#       - layer_idx 전달(캐시 호환)
#       - 어텐션에 use_cache 인자 미전달 (WhisperSdpaAttention 미지원)
#       - is_causal / causal_mask / attention_mask 는 시그니처를 점검해 선택적으로 전달
#       - dropout/activation_dropout은 확률값으로 보관하고 F.dropout로 적용
#       - HF Whisper와 forward 시그니처/리턴 포맷 호환
# ==============================================================================

from typing import Optional, Tuple
import inspect
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.whisper.modeling_whisper import (
    WhisperDecoderLayer,
    WhisperConfig,
)

class ParallelWhisperDecoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig, layer_idx: int):
        super().__init__()
        ref = WhisperDecoderLayer(config, layer_idx=layer_idx)

        self.embed_dim = config.d_model
        self.layer_idx = layer_idx

        # --- 모듈: 원본 그대로 재사용 ---
        self.self_attn = ref.self_attn
        self.cross_attn = ref.encoder_attn
        self.fc1 = ref.fc1
        self.fc2 = ref.fc2
        self.activation_fn = ref.activation_fn

        self.self_attn_layer_norm = ref.self_attn_layer_norm
        self.encoder_attn_layer_norm = ref.encoder_attn_layer_norm
        self.final_layer_norm = ref.final_layer_norm

        # --- 드롭아웃 확률값 (float) ---
        self.dropout_p = getattr(config, "dropout", 0.0)
        self.activation_dropout_p = getattr(config, "activation_dropout", 0.0)

    def _call_self_attn(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        past_key_value,
        output_attentions,
        causal_mask,
    ):
        sig = inspect.signature(self.self_attn.forward).parameters
        kwargs = dict(hidden_states=hidden_states, past_key_value=past_key_value, output_attentions=output_attentions)

        if "layer_head_mask" in sig:
            kwargs["layer_head_mask"] = layer_head_mask
        if "attention_mask" in sig:
            kwargs["attention_mask"] = attention_mask
        # 일부 버전은 causal_mask를 지원
        if "causal_mask" in sig and causal_mask is not None:
            kwargs["causal_mask"] = causal_mask
        # 일부 구버전은 is_causal을 받음
        if "is_causal" in sig and "causal_mask" not in sig:
            kwargs["is_causal"] = True

        return self.self_attn(**kwargs)

    def _call_cross_attn(
        self,
        hidden_states,
        key_value_states,
        encoder_attention_mask,
        layer_head_mask,
        past_key_value,
        output_attentions,
    ):
        sig = inspect.signature(self.cross_attn.forward).parameters
        kwargs = dict(hidden_states=hidden_states, key_value_states=key_value_states,
                      past_key_value=past_key_value, output_attentions=output_attentions)

        if "layer_head_mask" in sig:
            kwargs["layer_head_mask"] = layer_head_mask
        if "attention_mask" in sig:
            kwargs["attention_mask"] = encoder_attention_mask
        # cross-attn은 causal 아님. 구버전 is_causal을 받는다면 False로.
        if "is_causal" in sig:
            kwargs["is_causal"] = False

        return self.cross_attn(**kwargs)

    def forward(
        self,
        hidden_states,
        attention_mask=None,                        # decoder causal mask(기본)
        encoder_hidden_states=None,
        encoder_attention_mask=None,                # cross-attn mask
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value: Optional[Tuple] = None,     # (past_self, past_cross) 또는 None
        output_attentions: bool = False,
        use_cache: bool = True,                     # 상위에서 요구하지만 어텐션엔 넘기지 않음
        causal_mask=None,                           # HF 디코더가 마지막 포지셔널로 주입 가능
        **kwargs,
    ):
        """
        Returns (HF Whisper 호환):
          - hidden_states
          - (self_attn_weights?)        if output_attentions
          - (present_key_value?)        if use_cache
          - (cross_attn_weights?)       if output_attentions
        """
        # past 분리
        past_self = past_cross = None
        if past_key_value is not None:
            if isinstance(past_key_value, (tuple, list)) and len(past_key_value) == 2:
                past_self, past_cross = past_key_value
            else:
                past_self = past_key_value  # 호환성

        # ===== Self-Attn branch =====
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

        # 반환 파싱(버전별 길이 차이 흡수)
        self_out = self_outputs[0]
        self_weights = None
        self_present = None
        if output_attentions:
            if len(self_outputs) >= 2:
                self_weights = self_outputs[1]
            if len(self_outputs) >= 3:
                self_present = self_outputs[2]
        else:
            if isinstance(self_outputs, (tuple, list)) and len(self_outputs) >= 2:
                self_present = self_outputs[1]

        # ===== Cross-Attn branch =====
        cross_in = self.encoder_attn_layer_norm(hidden_states)
        cross_outputs = self._call_cross_attn(
            hidden_states=cross_in,
            key_value_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            layer_head_mask=cross_attn_layer_head_mask,
            past_key_value=past_cross,
            output_attentions=output_attentions,
        )
        cross_out = cross_outputs[0]
        cross_weights = None
        cross_present = None
        if output_attentions:
            if len(cross_outputs) >= 2:
                cross_weights = cross_outputs[1]
            if len(cross_outputs) >= 3:
                cross_present = cross_outputs[2]
        else:
            if isinstance(cross_outputs, (tuple, list)) and len(cross_outputs) >= 2:
                cross_present = cross_outputs[1]

        # ===== 병합 + FFN =====
        self_out = F.dropout(self_out, p=self.dropout_p, training=self.training)
        cross_out = F.dropout(cross_out, p=self.dropout_p, training=self.training)
        merged = residual + self_out + cross_out

        ffn_residual = merged
        ffn_in = self.final_layer_norm(merged)
        ffn_hidden = self.fc1(ffn_in)
        ffn_hidden = self.activation_fn(ffn_hidden)
        ffn_hidden = F.dropout(ffn_hidden, p=self.activation_dropout_p, training=self.training)
        ffn_hidden = self.fc2(ffn_hidden)
        ffn_hidden = F.dropout(ffn_hidden, p=self.dropout_p, training=self.training)
        hidden_states = ffn_residual + ffn_hidden

        # ===== present_key_value =====
        present = (self_present, cross_present) if use_cache else None

        # ===== 리턴 포맷 (Whisper 기대 순서) =====
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_weights,)
        if use_cache:
            outputs += (present,)
        if output_attentions:
            outputs += (cross_weights,)

        return outputs

