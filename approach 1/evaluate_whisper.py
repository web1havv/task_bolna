import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
import soundfile as sf
from datasets import load_dataset
from jiwer import wer
from peft import PeftModel
from rich import print
from transformers import WhisperForConditionalGeneration, WhisperProcessor


ROOT = Path(__file__).resolve().parent
WHISPER_DATASET_DIR = ROOT / "data" / "whisper_dataset"


def load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    # Use soundfile to read WAV robustly (avoids torchcodec issues)
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)  # Convert to mono
    waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # (1, T)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform


def load_eval_dataset(max_samples: Optional[int] = 40) -> List[Dict]:
    # Use train.jsonl since eval.jsonl is empty in this run
    data_files = {"eval": str(WHISPER_DATASET_DIR / "train.jsonl")}
    ds = load_dataset("json", data_files=data_files)["eval"]
    if max_samples is not None and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    return list(ds)


def transcribe_batch(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    batch: List[Dict],
) -> List[str]:
    model.eval()
    device = next(model.parameters()).device
    texts = []
    with torch.no_grad():
        # Process each audio individually (they have different lengths)
        for ex in batch:
            waveform = load_audio(ex["audio"])
            audio_array = waveform.squeeze(0).numpy()
            
            # Extract features for single audio
            feats = processor.feature_extractor(
                audio_array, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device)
            
            generated = model.generate(input_features=feats)
            text = processor.batch_decode(generated, skip_special_tokens=True)[0]
            texts.append(text)
    return texts


def evaluate_models(
    base_model_name: str,
    lora_dir: str,
    max_samples: Optional[int] = 40,
) -> None:
    print(f"[bold]Loading base Whisper model: {base_model_name}[/bold]")
    processor = WhisperProcessor.from_pretrained(base_model_name)
    base_model = WhisperForConditionalGeneration.from_pretrained(base_model_name)
    base_model.config.forced_decoder_ids = None  # leave generation_config default

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model.to(device)

    print(f"[bold]Loading LoRA fine-tuned model from: {lora_dir}[/bold]")
    ft_model = WhisperForConditionalGeneration.from_pretrained(base_model_name)
    ft_model = PeftModel.from_pretrained(ft_model, lora_dir)
    ft_model.config.forced_decoder_ids = None  # leave generation_config default
    ft_model.to(device)

    eval_examples = load_eval_dataset(max_samples=max_samples)
    print(f"[bold]Evaluating on {len(eval_examples)} eval examples.[/bold]")

    gold_texts: List[str] = []
    base_preds: List[str] = []
    ft_preds: List[str] = []

    gold_addr: List[str] = []
    base_addr_preds: List[str] = []
    ft_addr_preds: List[str] = []

    batch_size = 8
    for i in range(0, len(eval_examples), batch_size):
        batch = eval_examples[i : i + batch_size]
        gold_batch = [ex["text"] for ex in batch]
        base_batch = transcribe_batch(base_model, processor, batch)
        ft_batch = transcribe_batch(ft_model, processor, batch)

        gold_texts.extend(gold_batch)
        base_preds.extend(base_batch)
        ft_preds.extend(ft_batch)

        for ex, b_pred, f_pred in zip(batch, base_batch, ft_batch):
            spoken_addr = (ex.get("spoken_address") or "").strip()
            if not spoken_addr:
                continue
            gold_addr.append(spoken_addr.lower())
            # Use full prediction for address WER (simple heuristic)
            base_addr_preds.append(b_pred.lower())
            ft_addr_preds.append(f_pred.lower())

    # Overall WER
    overall_wer_base = wer(gold_texts, base_preds)
    overall_wer_ft = wer(gold_texts, ft_preds)

    print("[bold cyan]=== Overall WER ===[/bold cyan]")
    print(f"Base Whisper:      {overall_wer_base:.4f}")
    print(f"Fine-tuned Whisper:{overall_wer_ft:.4f}")

    # Address-only WER
    if gold_addr:
        addr_wer_base = wer(gold_addr, base_addr_preds)
        addr_wer_ft = wer(gold_addr, ft_addr_preds)

        print("[bold magenta]=== Address-span WER (Bangalore addresses only) ===[/bold magenta]")
        print(f"Base Whisper:      {addr_wer_base:.4f}")
        print(f"Fine-tuned Whisper:{addr_wer_ft:.4f}")
    else:
        print("[yellow]No address examples found in eval set for address-specific WER.[/yellow]")


def main():
    parser = argparse.ArgumentParser(
        description="Compare base Whisper vs fine-tuned Whisper+LoRA on address-heavy eval set."
    )
    parser.add_argument(
        "--base-model-name",
        type=str,
        default="openai/whisper-small",
        help="Base Whisper model name.",
    )
    parser.add_argument(
        "--lora-dir",
        type=str,
        default=str(ROOT / "models" / "whisper_bangalore_lora"),
        help="Directory containing LoRA adapter and config.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=40,
        help="Max number of eval samples to use.",
    )
    args = parser.parse_args()

    evaluate_models(
        base_model_name=args.base_model_name,
        lora_dir=args.lora_dir,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()

