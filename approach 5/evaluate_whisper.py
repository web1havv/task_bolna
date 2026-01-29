import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, DefaultDict

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


def load_eval_dataset(
    max_samples: Optional[int] = None,
    split: str = "eval",
) -> List[Dict]:
    """Load eval/test dataset. Approach 4: use split='eval' or 'test'."""
    target_file = WHISPER_DATASET_DIR / f"{split}.jsonl"
    if not target_file.exists() or target_file.stat().st_size == 0:
        target_file = ROOT / "audio" / split / f"{split}_metadata.jsonl"
    if not target_file.exists() or target_file.stat().st_size == 0:
        print(f"[yellow]No {split} dataset found. Trying eval/train fallback...[/yellow]")
        if split != "eval":
            target_file = WHISPER_DATASET_DIR / "eval.jsonl"
        if not target_file.exists() or target_file.stat().st_size == 0:
            target_file = ROOT / "audio" / "eval" / "eval_metadata.jsonl"
        if not target_file.exists() or target_file.stat().st_size == 0:
            target_file = WHISPER_DATASET_DIR / "train.jsonl"
        if not target_file.exists() or target_file.stat().st_size == 0:
            target_file = ROOT / "audio" / "train" / "train_metadata.jsonl"
        if not target_file.exists() or target_file.stat().st_size == 0:
            print("[yellow]No eval/train dataset available. Returning empty list.[/yellow]")
            return []

    if target_file.stat().st_size == 0:
        print("[yellow]Dataset file is empty. Returning empty list.[/yellow]")
        return []

    try:
        data_files = {"eval": str(target_file)}
        ds = load_dataset("json", data_files=data_files)["eval"]
        if max_samples is not None and len(ds) > max_samples:
            ds = ds.select(range(max_samples))
        return list(ds)
    except Exception as e:
        print(f"[yellow]Error loading dataset: {e}. Returning empty list.[/yellow]")
        return []


def transcribe_batch(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    batch: List[Dict],
    show_progress: bool = True,
) -> List[str]:
    model.eval()
    device = next(model.parameters()).device
    texts = []
    total = len(batch)
    
    with torch.no_grad():
        # Process each audio individually (they have different lengths)
        for idx, ex in enumerate(batch):
            if show_progress and (idx + 1) % 10 == 0:
                print(f"[yellow]Transcribing: {idx + 1}/{total} ({100 * (idx + 1) / total:.1f}%)[/yellow]")
            
            try:
                waveform = load_audio(ex["audio"])
                audio_array = waveform.squeeze(0).numpy()
                
                # Extract features for single audio
                feats = processor.feature_extractor(
                    audio_array, sampling_rate=16000, return_tensors="pt"
                ).input_features.to(device)
                
                # Generate with language='en' to avoid language detection overhead
                generated = model.generate(
                    input_features=feats,
                    language="en",
                    task="transcribe",
                )
                text = processor.batch_decode(generated, skip_special_tokens=True)[0]
                texts.append(text)
            except Exception as e:
                print(f"[red]Error transcribing example {idx}: {e}[/red]")
                texts.append("")  # Append empty string on error
    return texts


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, remove punctuation, normalize whitespace"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text


def pseudo_phonemize(text: str) -> List[str]:
    """
    Lightweight "phoneme-ish" tokenization to approximate phoneme-aware WER without extra deps.
    Not linguistically perfect; intended for relative comparisons.
    """
    t = normalize_text(text)
    # common digraphs / clusters
    replacements = [
        ("tion", "SHUN"),
        ("ph", "F"),
        ("ch", "CH"),
        ("sh", "SH"),
        ("th", "TH"),
        ("ng", "NG"),
        ("aa", "AA"),
        ("ee", "EE"),
        ("oo", "OO"),
    ]
    for a, b in replacements:
        t = t.replace(a, f" {b} ")
    # split to tokens (keep single letters as proxy)
    toks: List[str] = []
    for w in t.split():
        if w.isalpha() and len(w) > 3:
            toks.extend(list(w))
        else:
            toks.append(w)
    return [x for x in toks if x]


def edit_alignment(ref: List[str], hyp: List[str]) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """
    Levenshtein alignment producing ops:
    ('ok', r, h), ('sub', r, h), ('ins', None, h), ('del', r, None)
    """
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = ("del", i - 1, 0)
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = ("ins", 0, j - 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cands = [
                (dp[i - 1][j] + 1, ("del", i - 1, j)),
                (dp[i][j - 1] + 1, ("ins", i, j - 1)),
                (dp[i - 1][j - 1] + cost, ("sub" if cost else "ok", i - 1, j - 1)),
            ]
            dp[i][j], bt[i][j] = min(cands, key=lambda x: x[0])
    ops: List[Tuple[str, Optional[str], Optional[str]]] = []
    i, j = n, m
    while i > 0 or j > 0:
        kind, ii, jj = bt[i][j]
        if kind == "del":
            ops.append(("del", ref[ii], None))
            i -= 1
        elif kind == "ins":
            ops.append(("ins", None, hyp[jj]))
            j -= 1
        else:
            ops.append((kind, ref[ii], hyp[jj]))
            i -= 1
            j -= 1
    ops.reverse()
    return ops


def span_recall_f1(spoken_address: str, pred_text: str) -> Tuple[float, float]:
    """
    Approximate address span recall/F1 using token overlap between gold spoken_address
    and the best-matching contiguous window in pred.
    """
    gold = normalize_text(spoken_address).split()
    pred = normalize_text(pred_text).split()
    if not gold:
        return 0.0, 0.0
    if not pred:
        return 0.0, 0.0
    gset = set(gold)
    # best window length near gold length
    L = len(gold)
    best_overlap = 0
    best_len = 0
    for start in range(0, max(1, len(pred) - 1)):
        for win_len in [max(1, L - 2), L, L + 2]:
            end = min(len(pred), start + win_len)
            window = pred[start:end]
            ov = len(set(window) & gset)
            if ov > best_overlap:
                best_overlap = ov
                best_len = len(window)
    recall = best_overlap / len(gset) if gset else 0.0
    precision = best_overlap / best_len if best_len else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return recall, f1


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
    compare_with_model: Optional[str] = None,
    eval_split: str = "eval",
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load processor from fine-tuned model's base (tiny)
    print(f"[bold]Loading processor from: {base_model_name}[/bold]")
    processor = WhisperProcessor.from_pretrained(base_model_name)
    
    # Load fine-tuned model (tiny)
    print(f"[bold]Loading LoRA fine-tuned model from: {lora_dir}[/bold]")
    ft_model = WhisperForConditionalGeneration.from_pretrained(base_model_name)
    ft_model = PeftModel.from_pretrained(ft_model, lora_dir)
    ft_model.config.forced_decoder_ids = None
    ft_model.to(device)
    
    # Load comparison model (if specified, e.g., whisper-small)
    if compare_with_model:
        print(f"[bold]Loading comparison model: {compare_with_model}[/bold]")
        base_model = WhisperForConditionalGeneration.from_pretrained(compare_with_model)
        base_model.config.forced_decoder_ids = None
        base_model.to(device)
        comparison_name = compare_with_model.split("/")[-1]
    else:
        # Use same base model as fine-tuned
        print(f"[bold]Loading base Whisper model: {base_model_name}[/bold]")
        base_model = WhisperForConditionalGeneration.from_pretrained(base_model_name)
        base_model.config.forced_decoder_ids = None
        base_model.to(device)
        comparison_name = base_model_name.split("/")[-1]

    eval_examples = load_eval_dataset(max_samples=max_samples, split=eval_split)
    if not eval_examples:
        print("[yellow]No evaluation examples available. Skipping evaluation.[/yellow]")
        return
    print(f"[bold]Evaluating on {len(eval_examples)} {eval_split} examples.[/bold]")

    # Separate address and non-address examples
    address_examples = [ex for ex in eval_examples if ex.get("canonical_address", "").strip()]
    all_examples = eval_examples

    print(f"[bold]Found {len(address_examples)} address examples out of {len(all_examples)} total.[/bold]")

    # Transcribe all examples
    gold_texts: List[str] = []
    base_preds: List[str] = []
    ft_preds: List[str] = []

    batch_size = 8
    total_batches = (len(all_examples) + batch_size - 1) // batch_size
    
    print(f"[bold]Transcribing {len(all_examples)} examples in {total_batches} batches...[/bold]")
    
    for batch_idx in range(0, len(all_examples), batch_size):
        batch_num = (batch_idx // batch_size) + 1
        batch = all_examples[batch_idx : batch_idx + batch_size]
        gold_batch = [ex["text"] for ex in batch]
        
        print(f"[cyan]Batch {batch_num}/{total_batches}: Transcribing with base model...[/cyan]")
        base_batch = transcribe_batch(base_model, processor, batch, show_progress=False)
        
        print(f"[cyan]Batch {batch_num}/{total_batches}: Transcribing with fine-tuned model...[/cyan]")
        ft_batch = transcribe_batch(ft_model, processor, batch, show_progress=False)

        gold_texts.extend(gold_batch)
        base_preds.extend(base_batch)
        ft_preds.extend(ft_batch)
        
        print(f"[green]Completed batch {batch_num}/{total_batches} ({100 * batch_num / total_batches:.1f}%)[/green]")

    # 1. Overall WER (only to check regression)
    overall_wer_base = wer(gold_texts, base_preds)
    overall_wer_ft = wer(gold_texts, ft_preds)

    print("\n[bold cyan]=== Overall WER (regression check only) ===[/bold cyan]")
    print(f"{comparison_name}:      {overall_wer_base:.4f}")
    print(f"Fine-tuned {base_model_name.split('/')[-1]}:{overall_wer_ft:.4f}")
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
        print(f"{comparison_name}:      {addr_f1_base:.4f}")
        print(f"Fine-tuned {base_model_name.split('/')[-1]}:{addr_f1_ft:.4f}")
        improvement_f1 = addr_f1_ft - addr_f1_base
        if improvement_f1 > 0:
            print(f"[green]✓ Improvement: +{improvement_f1:.4f}[/green]")
        else:
            print(f"[red]✗ Degradation: {improvement_f1:.4f}[/red]")

        # 4. Span-level recall/F1 (relaxed)
        span_rec_base = []
        span_f1_base = []
        span_rec_ft = []
        span_f1_ft = []
        for gold_spk, pb, pf in zip(spoken_addresses, address_base_preds, address_ft_preds):
            rb, fb = span_recall_f1(gold_spk, pb)
            rf, ff = span_recall_f1(gold_spk, pf)
            span_rec_base.append(rb); span_f1_base.append(fb)
            span_rec_ft.append(rf); span_f1_ft.append(ff)
        avg_span_rec_base = sum(span_rec_base) / len(span_rec_base) if span_rec_base else 0.0
        avg_span_f1_base = sum(span_f1_base) / len(span_f1_base) if span_f1_base else 0.0
        avg_span_rec_ft = sum(span_rec_ft) / len(span_rec_ft) if span_rec_ft else 0.0
        avg_span_f1_ft = sum(span_f1_ft) / len(span_f1_ft) if span_f1_ft else 0.0

        print("\n[bold magenta]=== Address Span (relaxed) ===[/bold magenta]")
        print(f"{comparison_name}:      recall={avg_span_rec_base:.4f}  f1={avg_span_f1_base:.4f}")
        print(f"Fine-tuned {base_model_name.split('/')[-1]}: recall={avg_span_rec_ft:.4f}  f1={avg_span_f1_ft:.4f}")

        # 5. Phoneme-aware WER (pseudo-phonemes) + confusion top pairs
        base_ops: DefaultDict[Tuple[str, str], int] = {}
        ft_ops: DefaultDict[Tuple[str, str], int] = {}
        base_subs = {}
        ft_subs = {}
        base_ph_gold = []
        base_ph_pred = []
        ft_ph_pred = []
        for g, pb, pf in zip(address_gold_texts, address_base_preds, address_ft_preds):
            gph = pseudo_phonemize(g)
            bph = pseudo_phonemize(pb)
            fph = pseudo_phonemize(pf)
            base_ph_gold.append(" ".join(gph))
            base_ph_pred.append(" ".join(bph))
            ft_ph_pred.append(" ".join(fph))
            for kind, r, h in edit_alignment(gph, bph):
                if kind == "sub" and r and h:
                    base_subs[(r, h)] = base_subs.get((r, h), 0) + 1
            for kind, r, h in edit_alignment(gph, fph):
                if kind == "sub" and r and h:
                    ft_subs[(r, h)] = ft_subs.get((r, h), 0) + 1
        ph_wer_base = wer(base_ph_gold, base_ph_pred)
        ph_wer_ft = wer(base_ph_gold, ft_ph_pred)
        print("\n[bold magenta]=== Phoneme-aware WER (pseudo) ===[/bold magenta]")
        print(f"{comparison_name}:      {ph_wer_base:.4f}")
        print(f"Fine-tuned {base_model_name.split('/')[-1]}:{ph_wer_ft:.4f}")
        top_base = sorted(base_subs.items(), key=lambda x: -x[1])[:15]
        top_ft = sorted(ft_subs.items(), key=lambda x: -x[1])[:15]
        print("\n[bold magenta]=== Top pseudo-phoneme confusions (base) ===[/bold magenta]")
        for (r, h), c in top_base:
            print(f"{r} -> {h}: {c}")
        print("\n[bold magenta]=== Top pseudo-phoneme confusions (fine-tuned) ===[/bold magenta]")
        for (r, h), c in top_ft:
            print(f"{r} -> {h}: {c}")

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
        default="openai/whisper-small",
        help="Base Whisper model (must match fine-tuned base).",
    )
    parser.add_argument(
        "--lora-dir",
        type=str,
        default=str(ROOT / "models" / "whisper_small_bangalore_lora"),
        help="Directory containing LoRA adapter and config.",
    )
    parser.add_argument(
        "--compare-with-model",
        type=str,
        default=None,
        help="Model to compare against (e.g., 'openai/whisper-small'). If None, compares with base model.",
    )
    def parse_max_samples(value):
        """Parse max_samples: handle 'None' string or integer"""
        if value is None or (isinstance(value, str) and value.lower() == "none"):
            return None
        return int(value)
    
    parser.add_argument(
        "--max-samples",
        type=parse_max_samples,
        default=None,
        help="Max number of eval samples to use (omit or 'None' = use all).",
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        default="eval",
        choices=("eval", "test"),
        help="Split to evaluate on (eval or test). Approach 4: both available.",
    )
    args = parser.parse_args()

    evaluate_models(
        base_model_name=args.base_model_name,
        lora_dir=args.lora_dir,
        max_samples=args.max_samples,
        compare_with_model=args.compare_with_model,
        eval_split=args.eval_split,
    )


if __name__ == "__main__":
    main()

