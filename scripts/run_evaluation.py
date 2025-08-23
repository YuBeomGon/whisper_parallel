# scripts/run_evaluation.py
import os, glob, argparse
import torch
import jiwer
from tqdm import tqdm
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from safetensors.torch import load_file, safe_open

# normalizers
from whisper.normalizers.basic import BasicTextNormalizer
from whisper.normalizers.english import EnglishTextNormalizer


# 커스텀 병렬 디코더
from models.parallel_decoder import ParallelWhisperDecoderLayer

def load_checkpoint_state_dict(model_path: str):
    index_json = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_json):
        from collections import OrderedDict
        sd = OrderedDict()
        for shard in sorted(glob.glob(os.path.join(model_path, "model-*.safetensors"))):
            with safe_open(shard, framework="pt", device="cpu") as f:
                for k in f.keys():
                    sd[k] = f.get_tensor(k)
        return sd
    st_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(st_path):
        return load_file(st_path, device="cpu")
    bin_path = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(bin_path):
        return torch.load(bin_path, map_location="cpu")
    raise FileNotFoundError(f"No model weights found in: {model_path}")

def build_model(base_model: str, arch: str, torch_dtype):
    model = WhisperForConditionalGeneration.from_pretrained(base_model, torch_dtype=torch_dtype)
    if arch == "parallel":
        # 병렬 디코더로 교체
        for i in range(model.config.decoder_layers):
            model.model.decoder.layers[i] = ParallelWhisperDecoderLayer(model.config, layer_idx=i)
    return model

def set_decoder_gamma(model, gamma: float):
    """병렬 디코더일 때만 gamma 주입(원본은 무시)."""
    for layer in model.model.decoder.layers:
        if hasattr(layer, "gamma"):
            layer.gamma.fill_(float(gamma))

def setup_arg_parser():
    p = argparse.ArgumentParser(description="Whisper 평가 (원본/병렬 + 데이터셋 선택)")

    # 아키텍처/모델
    p.add_argument("--arch", choices=["original", "parallel"], default="parallel",
                   help="모델 아키텍처: original(원본) | parallel(병렬 디코더)")
    p.add_argument("--base_model", type=str, default="openai/whisper-large-v3-turbo",
                   help="베이스 모델 ID (학습에 사용한 것과 동일 권장)")
    p.add_argument("--model_path", type=str, default="",
                   help="체크포인트 경로(미지정 시 base_model 가중치 사용)")

    # 데이터셋
    p.add_argument("--dataset", choices=["zeroth_ko", "librispeech_en"], default="zeroth_ko",
                   help="평가 데이터셋 선택")
    # LibriSpeech일 때만 사용(일반적으로 config: clean|other|all, split: test.clean 등)
    p.add_argument("--ls_config", type=str, default="clean",
                   help="LibriSpeech config (예: clean / other / all)")
    p.add_argument("--ls_eval_split", type=str, default="test",
                   help="LibriSpeech eval split (예: test for zeroth / validation.clean for librispeech)")

    # 생성 옵션
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=225)

    # 병렬 디코더 전환 계수
    p.add_argument("--gamma", type=float, default=None,
                   help="디코더 병렬 전환 계수. 1.0=원본(순차), 0.0=완전 병렬. 미지정 시 체크포인트값 사용")

    return p.parse_args()

            
def load_eval_dataset(args, processor):
    """데이터셋/정규화기/언어 프롬프트를 한 번에 세팅."""
    if args.dataset == "zeroth_ko":
        language = "ko"
        normalizer = BasicTextNormalizer()
        print("Loading 'Bingsu/zeroth-korean' test split...")
        ds = load_dataset("Bingsu/zeroth-korean", split="test")
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    else:
        language = "en"
        normalizer = EnglishTextNormalizer()
        print(f"Loading 'librispeech_asr' config='{args.ls_config}' split='{args.ls_eval_split}' ...")
        # 예: config='clean', split='test.clean'
        ds = load_dataset("librispeech_asr", args.ls_config, split=args.ls_eval_split)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))  # LibriSpeech도 16kHz

    forced = processor.get_decoder_prompt_ids(language=language, task="transcribe")
    return ds, normalizer, forced            

def evaluate_model(arch: str, model_path: str, base_model: str, batch_size: int,
                   num_beams: int, max_new_tokens: int, gamma: float,
                   dataset: str, ls_config: str, ls_eval_split: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # GPU Ampere+면 bf16, 아니면 fp16로 추론(대부분 안전)
    torch_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

    print(f"[EVAL] arch={arch} | base_model={base_model} | ckpt='{model_path or 'N/A'}' | device={device} | dtype={torch_dtype}")

    # 1) 프로세서
    processor = WhisperProcessor.from_pretrained(base_model)

    # 2) 모델 구성
    model = build_model(base_model, arch, torch_dtype)
    if model_path:
        sd = load_checkpoint_state_dict(model_path)
        incomp = model.load_state_dict(sd, strict=False)  # 구조 혼용 대비
        print("Missing keys:", len(incomp.missing_keys))
        print("Unexpected keys:", len(incomp.unexpected_keys))

    # generation 설정
    model.generation_config.num_beams = num_beams
    model.generation_config.max_new_tokens = max_new_tokens
    model.generation_config.pad_token_id = processor.tokenizer.eos_token_id

    # gamma 강제 주입(병렬일 때만 의미)
    if gamma is not None:
        if arch == "parallel":
            set_decoder_gamma(model, gamma)
            print(f"[EVAL] Set decoder gamma = {gamma:.3f}")
        else:
            print("[EVAL] --gamma는 parallel 아키텍처에만 적용됩니다. (무시)")

    # 디바이스/dtype 정렬
    model.to(device=device, dtype=torch_dtype)
    model.eval()

    # 3) 데이터셋/정규화기/언어 프롬프트
    ds, normalizer, forced = load_eval_dataset(
        argparse.Namespace(dataset=dataset, ls_config=ls_config, ls_eval_split=ls_eval_split),
        processor
    )

    # 4) 추론 루프
    preds, refs = [], []
    print(f"Inference with batch_size={batch_size}, beams={num_beams} ...")
    for i in tqdm(range(0, len(ds), batch_size)):
        sub = ds[i:i+batch_size]
        audio_inputs = [x["array"] for x in sub["audio"]]
        ref_texts = sub["text"]

        inputs = processor(audio_inputs, return_tensors="pt", sampling_rate=16000).input_features
        # 입력도 모델 dtype으로 확실히 정렬
        inputs = inputs.to(device=device, dtype=next(model.parameters()).dtype)

        with torch.no_grad():
            ids = model.generate(inputs, forced_decoder_ids=forced)
        hyp = processor.batch_decode(ids, skip_special_tokens=True)

        preds.extend([normalizer(t) for t in hyp])
        refs.extend([normalizer(t) for t in ref_texts])

    # 5) CER
    cer = jiwer.cer(refs, preds)
    print("="*50)
    print(f"완료! 아키텍처: {arch} | 모델: {model_path or base_model}")
    print(f"데이터셋: {dataset}" + (f" (ls_config={ls_config}, split={ls_eval_split})" if dataset == "librispeech_en" else ""))
    print(f"CER: {cer*100:.2f}%")
    print("="*50)

    # 샘플 확인
    print("— sample predictions —")
    for k in range(min(2, len(preds))):
        print(f"[pred] {preds[k]}")
        print(f"[ref ] {refs[k]}")
        print("---")


if __name__ == "__main__":
    args = setup_arg_parser()
    evaluate_model(
        arch=args.arch,
        model_path=args.model_path,
        base_model=args.base_model,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        gamma=args.gamma,
        dataset=args.dataset,
        ls_config=args.ls_config,
        ls_eval_split=args.ls_eval_split,
    )
