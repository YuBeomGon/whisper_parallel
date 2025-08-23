# ==============================================================================
# 파일: models/whisper_decoder_lm.py
# 역할: Whisper 디코더 + 출력 proj 만 떼어낸 Causal LM 래퍼 (Cross-Attn 미사용)
#       - 출력층 이름이 버전에 따라 proj_out / lm_head 로 달라지는 문제를 유연하게 처리
#       - 둘 다 없으면 Linear를 생성 후 embed_tokens와 weight tying
#       - 디코더 forward 시그니처 차이( past_key_value vs past_key_values 등 )도 안전 처리
# ==============================================================================

from typing import Optional, Tuple, Any, Dict
import inspect
import torch
import torch.nn as nn
from transformers.models.whisper.modeling_whisper import WhisperForConditionalGeneration

class WhisperDecoderLM(nn.Module):
    def __init__(self, base: WhisperForConditionalGeneration):
        super().__init__()
        self.config = base.config
        self.decoder = base.model.decoder

        # --- 출력 projection 잡기: proj_out -> lm_head -> (없으면 새로 생성 + tie)
        out = None
        if hasattr(base, "proj_out"):
            out = getattr(base, "proj_out")
        elif hasattr(base, "lm_head"):
            out = getattr(base, "lm_head")

        if out is None:
            print("Warning!!!!!!, out is None")
            # 둘 다 없으면 직접 만들기 (bias=False가 보통의 Whisper 설정)
            out = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
            # 임베딩과 tie
            try:
                out.weight = self.decoder.embed_tokens.weight
            except Exception:
                pass

        self.output_proj = out  # 이후엔 이 이름만 사용

        # (선택) generation_config를 남겨두면 일부 툴 호환이 낫다
        self.generation_config = getattr(base, "generation_config", None)

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, new_emb):
        self.decoder.embed_tokens = new_emb
        # weight tying 유지
        try:
            self.output_proj.weight = new_emb.weight
        except Exception:
            pass

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[Tuple] = None,
        **kwargs,
    ):
        # ===== 디코더 시그니처 점검 후 존재하는 인자만 전달 =====
        sig = inspect.signature(self.decoder.forward).parameters
        dec_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask if "attention_mask" in sig else None,
            "use_cache": use_cache if "use_cache" in sig else False,
            "output_attentions": False if "output_attentions" in sig else None,
            "output_hidden_states": False if "output_hidden_states" in sig else None,
        }
        # past 이름 호환
        if "past_key_values" in sig:
            dec_kwargs["past_key_values"] = past_key_values
        elif "past_key_value" in sig:
            dec_kwargs["past_key_value"] = past_key_values
        # cross 비활성 (있을 때만 None으로 전달)
        if "encoder_hidden_states" in sig:
            dec_kwargs["encoder_hidden_states"] = None
        if "encoder_attention_mask" in sig:
            dec_kwargs["encoder_attention_mask"] = None

        # None 값은 키에서 제거 (일부 버전이 None 인자를 싫어함)
        dec_kwargs = {k: v for k, v in dec_kwargs.items() if v is not None}

        dec_out = self.decoder(**dec_kwargs)
        hidden = dec_out[0]
        logits = self.output_proj(hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        return {"loss": loss, "logits": logits}
