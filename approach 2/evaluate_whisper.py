import argparse
import json
import re
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


def load_eval_dataset(max_samples: Optional[int] = None) -> List[Dict]:
    # Load eval dataset
    data_files = {"eval": str(WHISPER_DATASET_DIR / "eval.jsonl")}
    if not Path(WHISPER_DATASET_DIR / "eval.jsonl").exists():
        # Fallback to train if eval doesn't exist
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


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, remove punctuation, normalize whitespace"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text


def extract_address_tokens(text: str, processor: WhisperProcessor) -> List[int]:
    """Extract token IDs from text, excluding special tokens"""
    tokens = processor.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    special_tokens = {
        processor.tokenizer.bos_token_id,
        processor.tokenizer.eos_token_id,
        processor.tokenizer.pad_token_id,
    }
    return [t.item() for t in tokens if t.item() not in special_tokens]


def compute_address_token_f1(
    gold_addresses: List[str],
    pred_texts: List[str],
    processor: WhisperProcessor,
) -> float:
    """Compute token-level F1 score for address tokens"""
    all_precision = []
    all_recall = []
    
    for gold_addr, pred_text in zip(gold_addresses, pred_texts):
        gold_tokens = set(extract_address_tokens(gold_addr, processor))
        pred_tokens = set(extract_address_tokens(pred_text, processor))
        
        if len(gold_tokens) == 0:
            continue
            
        intersection = gold_tokens & pred_tokens
        precision = len(intersection) / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
        recall = len(intersection) / len(gold_tokens) if len(gold_tokens) > 0 else 0.0
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            all_precision.append(precision)
            all_recall.append(recall)
    
    if len(all_precision) == 0:
        return 0.0
    
    avg_precision = sum(all_precision) / len(all_precision)
    avg_recall = sum(all_recall) / len(all_recall)
    
    if avg_precision + avg_recall == 0:
        return 0.0
    
    return 2 * avg_precision * avg_recall / (avg_precision + avg_recall)


def compute_exact_canonical_accuracy(
    canonical_addresses: List[str],
    pred_texts: List[str],
) -> float:
    """Compute exact match accuracy for canonical addresses"""
    correct = 0
    total = 0
    
    for canonical, pred_text in zip(canonical_addresses, pred_texts):
        if not canonical.strip():
            continue
        
        canonical_norm = normalize_text(canonical)
        pred_norm = normalize_text(pred_text)
        
        # Check if canonical address appears in prediction (substring match)
        if canonical_norm in pred_norm:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0.0


def evaluate_models(
    base_model_name: str,
    lora_dir: str,
    max_samples: Optional[int] = None,
) -> None:
    print(f"[bold]Loading base Whisper model: {base_model_name}[/bold]")
    processor = WhisperProcessor.from_pretrained(base_model_name)
    base_model = WhisperForConditionalGeneration.from_pretrained(base_model_name)
    base_model.config.forced_decoder_ids = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model.to(device)

    print(f"[bold]Loading LoRA fine-tuned model from: {lora_dir}[/bold]")
    ft_model = WhisperForConditionalGeneration.from_pretrained(base_model_name)
    ft_model = PeftModel.from_pretrained(ft_model, lora_dir)
    ft_model.config.forced_decoder_ids = None
    ft_model.to(device)

    eval_examples = load_eval_dataset(max_samples=max_samples)
    print(f"[bold]Evaluating on {len(eval_examples)} eval examples.[/bold]")

    # Separate address and non-address examples
    address_examples = [ex for ex in eval_examples if ex.get("canonical_address", "").strip()]
    all_examples = eval_examples

    print(f"[bold]Found {len(address_examples)} address examples out of {len(all_examples)} total.[/bold]")

    # Transcribe all examples
    gold_texts: List[str] = []
    base_preds: List[str] = []
    ft_preds: List[str] = []

    batch_size = 8
    for i in range(0, len(all_examples), batch_size):
        batch = all_examples[i : i + batch_size]
        gold_batch = [ex["text"] for ex in batch]
        base_batch = transcribe_batch(base_model, processor, batch)
        ft_batch = transcribe_batch(ft_model, processor, batch)

        gold_texts.extend(gold_batch)
        base_preds.extend(base_batch)
        ft_preds.extend(ft_batch)

    # 1. Overall WER (only to check regression)
    overall_wer_base = wer(gold_texts, base_preds)
    overall_wer_ft = wer(gold_texts, ft_preds)

    print("\n[bold cyan]=== Overall WER (regression check only) ===[/bold cyan]")
    print(f"Base Whisper:      {overall_wer_base:.4f}")
    print(f"Fine-tuned Whisper:{overall_wer_ft:.4f}")
    if overall_wer_ft > overall_wer_base * 1.1:
        print("[yellow]⚠️  Warning: Overall WER increased significantly (>10%). This may indicate regression.[/yellow]")
    else:
        print("[green]✓ Overall WER is acceptable (no significant regression)[/green]")

    # Address-specific metrics (only on address examples)
    if address_examples:
        address_indices = [i for i, ex in enumerate(all_examples) if ex.get("canonical_address", "").strip()]
        address_gold_texts = [gold_texts[i] for i in address_indices]
        address_base_preds = [base_preds[i] for i in address_indices]
        address_ft_preds = [ft_preds[i] for i in address_indices]
        
        canonical_addresses = [ex["canonical_address"] for ex in address_examples]
        spoken_addresses = [ex.get("spoken_address", "") for ex in address_examples]

        # 2. Exact canonical address accuracy
        exact_acc_base = compute_exact_canonical_accuracy(canonical_addresses, address_base_preds)
        exact_acc_ft = compute_exact_canonical_accuracy(canonical_addresses, address_ft_preds)

        print("\n[bold magenta]=== Exact Canonical Address Accuracy ===[/bold magenta]")
        print(f"Base Whisper:      {exact_acc_base:.4f} ({exact_acc_base*100:.2f}%)")
        print(f"Fine-tuned Whisper:{exact_acc_ft:.4f} ({exact_acc_ft*100:.2f}%)")
        improvement = exact_acc_ft - exact_acc_base
        if improvement > 0:
            print(f"[green]✓ Improvement: +{improvement:.4f} ({improvement*100:.2f}%)[/green]")
        else:
            print(f"[red]✗ Degradation: {improvement:.4f} ({improvement*100:.2f}%)[/red]")

        # 3. Address-token F1
        addr_f1_base = compute_address_token_f1(spoken_addresses, address_base_preds, processor)
        addr_f1_ft = compute_address_token_f1(spoken_addresses, address_ft_preds, processor)

        print("\n[bold magenta]=== Address-Token F1 Score ===[/bold magenta]")
        print(f"Base Whisper:      {addr_f1_base:.4f}")
        print(f"Fine-tuned Whisper:{addr_f1_ft:.4f}")
        improvement_f1 = addr_f1_ft - addr_f1_base
        if improvement_f1 > 0:
            print(f"[green]✓ Improvement: +{improvement_f1:.4f}[/green]")
        else:
            print(f"[red]✗ Degradation: {improvement_f1:.4f}[/red]")

        # Summary
        print("\n[bold green]=== Summary ===[/bold green]")
        print(f"Address accuracy improved: {exact_acc_ft > exact_acc_base}")
        print(f"Address F1 improved: {addr_f1_ft > addr_f1_base}")
        print(f"Overall WER acceptable: {overall_wer_ft <= overall_wer_base * 1.1}")
        
        if exact_acc_ft > exact_acc_base and addr_f1_ft > addr_f1_base and overall_wer_ft <= overall_wer_base * 1.1:
            print("[bold green]✓ SUCCESS: Address accuracy improved without significant overall WER regression[/bold green]")
        elif exact_acc_ft > exact_acc_base or addr_f1_ft > addr_f1_base:
            print("[bold yellow]⚠️  PARTIAL SUCCESS: Some address metrics improved[/bold yellow]")
        else:
            print("[bold red]✗ FAILURE: Address metrics did not improve[/bold red]")
    else:
        print("[yellow]No address examples found in eval set for address-specific metrics.[/yellow]")


def main():
    parser = argparse.ArgumentParser(
        description="Compare base Whisper vs fine-tuned Whisper+LoRA on address-heavy eval set."
    )
    parser.add_argument(
        "--base-model-name",
        type=str,
        default="openai/whisper-tiny",
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
        default=None,
        help="Max number of eval samples to use (None = use all).",
    )
    args = parser.parse_args()

    evaluate_models(
        base_model_name=args.base_model_name,
        lora_dir=args.lora_dir,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()

