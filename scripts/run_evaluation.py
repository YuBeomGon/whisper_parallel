# scripts/run_evaluation.py
import os, glob, argparse
import torch
import jiwer
from tqdm import tqdm
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from whisper.normalizers.basic import BasicTextNormalizer
from safetensors.torch import load_file, safe_open

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

def setup_arg_parser():
    p = argparse.ArgumentParser(description="Whisper 평가 (원본/병렬 디코더 모두 지원)")
    p.add_argument("--arch", choices=["original", "parallel"], default="parallel",
                   help="모델 아키텍처 선택: original(원본) | parallel(병렬 디코더)")
    p.add_argument("--model_path", type=str, default="",
                   help="체크포인트 경로(미지정 시 base_model 가중치 사용)")
    p.add_argument("--base_model", type=str, default="openai/whisper-large-v3-turbo",
                   help="베이스 모델 ID (학습에 사용한 것과 동일 권장)")
    p.add_argument("--batch_size", type=int, default=1)
    return p.parse_args()

def evaluate_model(arch: str, model_path: str, base_model: str, batch_size: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[EVAL] arch={arch} | base_model={base_model} | ckpt='{model_path or 'N/A'}' | device={device}")

    # 1) 프로세서
    processor = WhisperProcessor.from_pretrained(base_model)

    # 2) 모델 구성
    model = build_model(base_model, arch, torch_dtype)
    if model_path:
        sd = load_checkpoint_state_dict(model_path)
        # 원본 체크포인트면 strict=True도 가능. 혼용 대비 기본은 False로.
        incomp = model.load_state_dict(sd, strict=False)
        print("Missing keys:", len(incomp.missing_keys))
        print("Unexpected keys:", len(incomp.unexpected_keys))
    # dtype/디바이스 완전 통일
    model.to(device=device, dtype=torch_dtype)
    model.eval()
    # deprecation/마스크 경고 완화
    model.generation_config.pad_token_id = processor.tokenizer.eos_token_id

    # 3) 데이터셋
    print("Loading 'Bingsu/zeroth-korean' test split...")
    ds = load_dataset("Bingsu/zeroth-korean", split="test")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    # 4) 추론
    normalizer = BasicTextNormalizer()
    preds, refs = [], []
    # 추천: task/language 플래그 사용
    forced = processor.get_decoder_prompt_ids(language="ko", task="transcribe")

    print(f"Inference with batch_size={batch_size} ...")
    for i in tqdm(range(0, len(ds), batch_size)):
        sub = ds[i:i+batch_size]
        audio_inputs = [x["array"] for x in sub["audio"]]
        ref_texts = sub["text"]
        inputs = processor(audio_inputs, return_tensors="pt", sampling_rate=16000).input_features
        inputs = inputs.to(device, dtype=torch_dtype)
        with torch.no_grad():
            ids = model.generate(inputs, forced_decoder_ids=forced)
        hyp = processor.batch_decode(ids, skip_special_tokens=True)
        preds.extend([normalizer(t) for t in hyp])
        refs.extend([normalizer(t) for t in ref_texts])

    # 5) CER
    cer = jiwer.cer(refs, preds)
    print("="*50)
    print(f"완료! 아키텍처: {arch} | 모델: {model_path or base_model}")
    print(f"CER: {cer*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    args = setup_arg_parser()
    evaluate_model(args.arch, args.model_path, args.base_model, args.batch_size)
