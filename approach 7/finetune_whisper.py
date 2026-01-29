"""
Approach 7: Whisper-Base Fine-tuning for Bangalore Address Transcription

Key changes from Approach 6:
1. Use whisper-base (74M params) instead of whisper-small (244M params)
   - Smaller model may respond better to fine-tuning (like whisper-tiny in Approach 2)
   - Faster training, less memory
   - Higher LoRA rank to compensate for smaller model capacity

Inherited from Approach 6:
- Focal loss on address tokens
- Sequence-level address bonus
- Label smoothing
- Wider LoRA targets (q, k, v)
- No curriculum (direct canonical training)

Goal: Test if smaller base model achieves better accuracy like Approach 2 did with tiny.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import math

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from rich import print
from torch import nn
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
import torchaudio
import soundfile as sf


ROOT = Path(__file__).resolve().parent
WHISPER_DATASET_DIR = ROOT / "data" / "whisper_dataset"
DEFAULT_MODEL_NAME = "openai/whisper-base"  # Approach 7: Use base instead of small


def load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    # Use soundfile to read WAV robustly (avoids torchcodec issues on Colab)
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)  # Convert to mono
    waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # (1, T)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform


def prepare_datasets(
    processor: WhisperProcessor,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
):
    # Check which files exist
    train_file = WHISPER_DATASET_DIR / "train.jsonl"
    eval_file = WHISPER_DATASET_DIR / "eval.jsonl"
    
    data_files = {"train": str(train_file)}
    if eval_file.exists() and eval_file.stat().st_size > 0:
        data_files["eval"] = str(eval_file)
    
    raw_datasets = load_dataset("json", data_files=data_files)
    
    # If eval doesn't exist, create empty split
    if "eval" not in raw_datasets:
        from datasets import Dataset
        raw_datasets["eval"] = Dataset.from_dict({})

    if max_train_samples is not None:
        # Safety: don't process more than available
        available_train = len(raw_datasets["train"])
        actual_train = min(max_train_samples, available_train)
        raw_datasets["train"] = raw_datasets["train"].select(range(actual_train))
        print(f"[yellow]Processing {actual_train} train samples (requested {max_train_samples}, available {available_train})[/yellow]")
    if max_eval_samples is not None:
        available_eval = len(raw_datasets["eval"])
        actual_eval = min(max_eval_samples, available_eval)
        raw_datasets["eval"] = raw_datasets["eval"].select(range(actual_eval))
        print(f"[yellow]Processing {actual_eval} eval samples (requested {max_eval_samples}, available {available_eval})[/yellow]")

    def preprocess_text_only(example: Dict[str, Any]) -> Dict[str, Any]:
        """Only preprocess text, NOT audio - audio will be loaded on-the-fly"""
        audio_path = example.get("audio", "")
        text = example.get("text", "")
        spoken_address = example.get("spoken_address", "")
        
        if not audio_path or not text:
            raise ValueError(f"Missing required fields: audio={bool(audio_path)}, text={bool(text)}")

        # Tokenize full transcript for labels
        label_ids = processor.tokenizer(
            text,
            return_tensors="pt",
        ).input_ids[0]

        # Build address token mask by subsequence search (if spoken_address present)
        if spoken_address:
            addr_ids = processor.tokenizer(
                spoken_address,
                return_tensors="pt",
            ).input_ids[0]

            # Remove special tokens for subsequence matching (heuristic)
            def strip_special(ids: torch.Tensor) -> torch.Tensor:
                specials = {
                    processor.tokenizer.bos_token_id,
                    processor.tokenizer.eos_token_id,
                    processor.tokenizer.pad_token_id,
                }
                return torch.tensor([i for i in ids.tolist() if i not in specials])

            label_core = strip_special(label_ids)
            addr_core = strip_special(addr_ids)

            addr_mask = [0] * len(label_ids)
            if len(addr_core) > 0 and len(label_core) >= len(addr_core):
                # naive subsequence search on core ids
                core = label_core.tolist()
                sub = addr_core.tolist()
                for start in range(len(core) - len(sub) + 1):
                    if core[start : start + len(sub)] == sub:
                        # Map core indices back to label_ids indices
                        core_indices = [
                            idx
                            for idx, i in enumerate(label_ids.tolist())
                            if i not in {
                                processor.tokenizer.bos_token_id,
                                processor.tokenizer.eos_token_id,
                                processor.tokenizer.pad_token_id,
                            }
                        ]
                        for pos in range(start, start + len(sub)):
                            addr_mask[core_indices[pos]] = 1
                        break
        else:
            addr_mask = [0] * len(label_ids)

        # Keep audio_path for on-the-fly loading
        example_out = {
            "audio_path": audio_path,  # Keep path, don't load audio yet
            "labels": label_ids,
            "address_token_mask": addr_mask,
            "is_address_example": int(bool(spoken_address)),
        }
        return example_out

    # Only preprocess text (fast, no memory issues)
    # IMPORTANT: map **per split** so we don't try to remove columns on empty eval datasets
    from datasets import Dataset, DatasetDict

    processed_splits = {}
    cols_to_remove = [
        "audio",
        "text",
        "spoken_address",
        "canonical_address",
        "address_id",
        "locality",
        "pincode",
    ]

    # Train split (must exist)
    train_ds = raw_datasets["train"]
    remove_cols_train = [c for c in train_ds.column_names if c in cols_to_remove]
    # IMPORTANT: Don't remove audio_path - we need it in data collator!
    remove_cols_train = [c for c in remove_cols_train if c != "audio"]
    processed_splits["train"] = train_ds.map(
        preprocess_text_only,
        remove_columns=remove_cols_train,
        num_proc=1,
        desc="Preprocessing text only (train)",
    )

    # Eval split (may be empty or missing)
    if "eval" in raw_datasets and len(raw_datasets["eval"]) > 0:
        eval_ds = raw_datasets["eval"]
        remove_cols_eval = [c for c in eval_ds.column_names if c in cols_to_remove]
        # IMPORTANT: Don't remove audio_path - we need it in data collator!
        remove_cols_eval = [c for c in remove_cols_eval if c != "audio"]
        processed_splits["eval"] = eval_ds.map(
            preprocess_text_only,
            remove_columns=remove_cols_eval,
            num_proc=1,
            desc="Preprocessing text only (eval)",
        )
    else:
        processed_splits["eval"] = Dataset.from_dict({})

    # Clear cache after preprocessing
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return DatasetDict(processed_splits)


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, ignore_index: int = -100) -> torch.Tensor:
    """
    Focal loss: down-weight easy examples, focus on hard ones.
    FL(p) = -alpha * (1-p)^gamma * log(p)
    """
    ce_loss = F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction='none')
    pt = torch.exp(-ce_loss)  # p_t
    focal_weight = (1 - pt) ** gamma
    return focal_weight * ce_loss


class AddressWeightedTrainer(Seq2SeqTrainer):
    """
    Enhanced trainer with:
    - Focal loss on address tokens
    - Sequence-level address bonus
    - R-Drop consistency regularization
    - Label smoothing
    """
    def __init__(
        self, 
        alpha_start: float = 1.0, 
        alpha_end: float = 2.5,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        rdrop_alpha: float = 0.0,  # 0 = disabled, try 0.1-0.5
        sequence_bonus: float = 0.0,  # bonus for full address match
        **kwargs
    ):
        super().__init__(**kwargs)
        self.alpha_start = float(alpha_start)
        self.alpha_end = float(alpha_end)
        self._alpha_curr = float(alpha_start)
        self.focal_gamma = float(focal_gamma)
        self.label_smoothing = float(label_smoothing)
        self.rdrop_alpha = float(rdrop_alpha)
        self.sequence_bonus = float(sequence_bonus)

    def _update_alpha(self) -> None:
        max_steps = getattr(self.state, "max_steps", None) or self.args.max_steps
        if not max_steps or max_steps <= 0:
            self._alpha_curr = self.alpha_end
            return
        p = min(1.0, max(0.0, float(self.state.global_step) / float(max_steps)))
        self._alpha_curr = self.alpha_start + p * (self.alpha_end - self.alpha_start)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override to remove input_ids before Seq2SeqTrainer processes it"""
        inputs.pop("input_ids", None)
        inputs.pop("attention_mask", None)
        return super().training_step(model, inputs, num_items_in_batch)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self._update_alpha()
        labels = inputs.pop("labels")
        address_mask = inputs.pop("address_token_mask", None)

        # Extract Whisper-compatible inputs
        whisper_inputs = {}
        if "input_features" in inputs:
            whisper_inputs["input_features"] = inputs.pop("input_features")
        if "decoder_input_ids" in inputs:
            whisper_inputs["decoder_input_ids"] = inputs.pop("decoder_input_ids")
        inputs.clear()
        
        if not whisper_inputs:
            raise ValueError("Missing required inputs: input_features and/or decoder_input_ids")
        
        # Forward pass 1
        outputs = model(**whisper_inputs)
        logits = outputs.logits

        # R-Drop: second forward pass for consistency
        if self.rdrop_alpha > 0 and self.training:
            outputs2 = model(**whisper_inputs)
            logits2 = outputs2.logits
        else:
            logits2 = None

        # Shift for teacher forcing
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        batch_size, seq_len, vocab_size = shift_logits.shape

        ignore_idx = self.label_smoother.ignore_index if self.label_smoother else -100
        active_positions = shift_labels.view(-1) != ignore_idx

        # === Loss computation ===
        
        # 1. Base loss with optional focal weighting
        if self.focal_gamma > 0:
            base_loss = focal_loss(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                gamma=self.focal_gamma,
                ignore_index=ignore_idx,
            )
        else:
            base_loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=ignore_idx,
                reduction="none",
                label_smoothing=self.label_smoothing,
            )

        # 2. Address token weighting
        weight = torch.ones_like(base_loss)
        if address_mask is not None:
            addr = address_mask[:, 1:].contiguous().view(-1)
            addr = addr.to(base_loss.device).float()
            weight = weight + self._alpha_curr * addr

        weighted_loss = base_loss * weight
        main_loss = weighted_loss[active_positions].mean()

        # 3. R-Drop consistency loss (KL divergence between two passes)
        rdrop_loss = torch.tensor(0.0, device=main_loss.device)
        if logits2 is not None and self.rdrop_alpha > 0:
            shift_logits2 = logits2[:, :-1, :].contiguous()
            p = F.log_softmax(shift_logits.view(-1, vocab_size), dim=-1)
            q = F.log_softmax(shift_logits2.view(-1, vocab_size), dim=-1)
            # Symmetric KL
            kl_pq = F.kl_div(p, q.exp(), reduction='none').sum(-1)
            kl_qp = F.kl_div(q, p.exp(), reduction='none').sum(-1)
            rdrop_loss = 0.5 * (kl_pq + kl_qp)[active_positions].mean()

        # 4. Sequence-level address bonus (negative loss for correct address spans)
        seq_bonus_loss = torch.tensor(0.0, device=main_loss.device)
        if self.sequence_bonus > 0 and address_mask is not None:
            # Check if predicted tokens match labels for address positions
            with torch.no_grad():
                pred_tokens = shift_logits.argmax(dim=-1)  # (batch, seq)
                addr_shifted = address_mask[:, 1:].to(pred_tokens.device)
                
                # Per-sample: what fraction of address tokens are correct?
                for b in range(batch_size):
                    addr_pos = addr_shifted[b].bool()
                    if addr_pos.any():
                        correct = (pred_tokens[b][addr_pos] == shift_labels[b][addr_pos]).float().mean()
                        # Bonus (negative loss) proportional to correctness
                        seq_bonus_loss = seq_bonus_loss - self.sequence_bonus * correct / batch_size

        # Total loss
        total_loss = main_loss + self.rdrop_alpha * rdrop_loss + seq_bonus_loss

        return (total_loss, outputs) if return_outputs else total_loss


def create_lora_config(rank: int = 16, include_k_proj: bool = True) -> LoraConfig:
    """
    Decoder LoRA config.
    
    Approach 6 change: Include k_proj for more capacity (helps with address structure).
    Keep rank moderate (16-24) to avoid overfitting.
    """
    target_modules = ["q_proj", "v_proj"]
    if include_k_proj:
        target_modules.append("k_proj")
    
    return LoraConfig(
        r=rank,
        lora_alpha=rank * 2,  # Standard ratio
        target_modules=target_modules,
        lora_dropout=0.10,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )

class EncoderLastBlockLoRA(nn.Module):
    """
    Minimal LoRA wrapper for a single nn.Linear (q_proj or v_proj) in the last encoder block.
    Keeps encoder frozen except these injected params.
    """
    def __init__(self, base: nn.Linear, r: int = 8, dropout: float = 0.1, alpha: float = 16.0):
        super().__init__()
        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.r)
        self.dropout = nn.Dropout(p=float(dropout))
        self.lora_A = nn.Linear(base.in_features, self.r, bias=False)
        self.lora_B = nn.Linear(self.r, base.out_features, bias=False)
        # init
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)
        # Freeze base
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def inject_encoder_last_block_lora(model: WhisperForConditionalGeneration, *, r: int = 8, dropout: float = 0.1) -> None:
    """
    Inject LoRA into ONLY last encoder block self-attn q_proj/v_proj.
    """
    enc = model.model.encoder
    last = enc.layers[-1]
    # Replace q_proj/v_proj with wrapped modules
    last.self_attn.q_proj = EncoderLastBlockLoRA(last.self_attn.q_proj, r=r, dropout=dropout, alpha=2 * r)
    last.self_attn.v_proj = EncoderLastBlockLoRA(last.self_attn.v_proj, r=r, dropout=dropout, alpha=2 * r)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper with LoRA for Bangalore address transcription."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Base Whisper model name (Hugging Face).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "models" / "whisper_base_bangalore_lora"),
        help="Directory to save LoRA adapter.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=10,
        help="Number of training epochs (up to 10, with early stopping).",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=8,  # Approach 7: base model is smaller, can use larger batch
        help="Per-device batch size (default 8 for whisper-base).",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (effective batch = batch_size * this).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.5,
        help="(Deprecated) kept for compatibility. Use --alpha-start/--alpha-end.",
    )
    parser.add_argument(
        "--alpha-start",
        type=float,
        default=1.0,
        help="Initial address token weighting multiplier (ramped up).",
    )
    parser.add_argument(
        "--alpha-end",
        type=float,
        default=2.5,
        help="Final address token weighting multiplier (cap <= 2.5x).",
    )
    parser.add_argument(
        "--encoder-lora-r",
        type=int,
        default=8,
        help="Rank for partial encoder LoRA (last encoder block q/v only). Use 4–8.",
    )
    parser.add_argument(
        "--specaugment",
        action="store_true",
        help="Enable mild SpecAugment on log-mel features during training.",
    )
    parser.add_argument(
        "--phoneme-epochs",
        type=int,
        default=0,  # Approach 6: skip phoneme stage by default (was confusing model)
        help="Curriculum stage 1 epochs (train on text_phoneme). Default 0 = skip.",
    )
    parser.add_argument(
        "--canonical-epochs",
        type=int,
        default=5,  # Approach 6: moderate epochs for balanced training
        help="Main training epochs (train on text_canonical).",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional max number of train samples.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Optional max number of eval samples.",
    )
    # Optimizations
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.06,
        help="Fraction of steps for linear LR warmup (default 0.06).",
    )
    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        default="cosine",
        choices=("cosine", "linear", "constant", "constant_with_warmup"),
        help="LR scheduler (cosine often works best for fine-tuning).",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm (default 1.0).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW (default 0.01).",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bf16 instead of fp16 on Ampere+ GPUs (faster, more stable).",
    )
    parser.add_argument(
        "--max-label-length",
        type=int,
        default=None,
        help="Cap label length (truncate longer); reduces OOM and speeds training.",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=0,
        help="DataLoader workers (use 2–4 on Colab if no pickling errors).",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=2,
        help="Stop if eval loss does not improve for this many evals (0 = disabled).",
    )
    # Approach 6 new options
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma (0=disabled, 2.0=standard). Focus on hard examples.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor (0=none, 0.1=mild). Reduces overconfidence.",
    )
    parser.add_argument(
        "--rdrop-alpha",
        type=float,
        default=0.0,
        help="R-Drop consistency loss weight (0=disabled, 0.1-0.5=mild). Regularization.",
    )
    parser.add_argument(
        "--sequence-bonus",
        type=float,
        default=0.5,
        help="Bonus for correct address sequence (0=disabled, 0.5=mild). Encourages full match.",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=24,  # Approach 7: Higher rank for smaller base model
        help="LoRA rank for decoder (24-32 recommended for base model).",
    )
    parser.add_argument(
        "--no-k-proj",
        action="store_true",
        help="Exclude k_proj from LoRA targets (default includes it for more capacity).",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing (saves memory, slightly slower).",
    )
    args = parser.parse_args()

    print(f"[bold]Loading Whisper processor and model: {args.model_name}[/bold]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[bold]Using device: {device}[/bold]")
    
    processor = WhisperProcessor.from_pretrained(args.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    model = model.to(device)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # Optional gradient checkpointing (saves memory)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("[bold yellow]Gradient checkpointing enabled[/bold yellow]")

    # Freeze encoder completely (decoder-only LoRA approach)
    for param in model.model.encoder.parameters():
        param.requires_grad = False
    print("[bold yellow]Encoder frozen completely (decoder-only fine-tuning)[/bold yellow]")

    # Prepare model for decoder-only LoRA fine-tuning
    include_k = not args.no_k_proj
    lora_config = create_lora_config(rank=args.lora_rank, include_k_proj=include_k)
    print(f"[bold]LoRA config: rank={args.lora_rank}, targets={lora_config.target_modules}[/bold]")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Inject partial encoder LoRA on last encoder block q/v only (low rank).
    inject_encoder_last_block_lora(model.get_base_model() if hasattr(model, "get_base_model") else model, r=args.encoder_lora_r, dropout=0.10)
    print(f"[bold yellow]Injected encoder last-block LoRA (q/v only), rank={args.encoder_lora_r}[/bold yellow]")
    
    # Wrap the actual Whisper model's forward to filter out input_ids
    # PEFT internally calls model.base_model.model.forward (the actual Whisper model)
    # We need to wrap that specific forward method
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        whisper_model = model.base_model.model
    elif hasattr(model, 'get_base_model'):
        whisper_model = model.get_base_model()
    else:
        whisper_model = model.model if hasattr(model, 'model') else model
    
    original_whisper_forward = whisper_model.forward
    def filtered_whisper_forward(*args, **kwargs):
        # Whisper ONLY accepts: input_features, decoder_input_ids
        # Filter out everything else
        filtered_kwargs = {}
        if "input_features" in kwargs:
            filtered_kwargs["input_features"] = kwargs["input_features"]
        if "decoder_input_ids" in kwargs:
            filtered_kwargs["decoder_input_ids"] = kwargs["decoder_input_ids"]
        # Pass only the filtered kwargs
        return original_whisper_forward(*args, **filtered_kwargs)
    whisper_model.forward = filtered_whisper_forward

    print("[bold]Preparing datasets...[/bold]")
    # Clear memory before preprocessing
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Curriculum: build two dataset views (phoneme-faithful first, then canonical).
    max_label_length = getattr(args, "max_label_length", None)

    def prepare_datasets_with_label_field(label_field: str):
        # locally shadow prepare_datasets preprocess to use a specific text field
        train_file = WHISPER_DATASET_DIR / "train.jsonl"
        eval_file = WHISPER_DATASET_DIR / "eval.jsonl"
        data_files = {"train": str(train_file)}
        if eval_file.exists() and eval_file.stat().st_size > 0:
            data_files["eval"] = str(eval_file)
        raw_datasets = load_dataset("json", data_files=data_files)
        if max_train_samples is not None:
            available_train = len(raw_datasets["train"])
            actual_train = min(max_train_samples, available_train)
            raw_datasets["train"] = raw_datasets["train"].select(range(actual_train))
        if max_eval_samples is not None and "eval" in raw_datasets:
            available_eval = len(raw_datasets["eval"])
            actual_eval = min(max_eval_samples, available_eval)
            raw_datasets["eval"] = raw_datasets["eval"].select(range(actual_eval))

        def preprocess_text_only(example: Dict[str, Any]) -> Dict[str, Any]:
            audio_path = example.get("audio", "")
            text = example.get(label_field) or example.get("text") or ""
            spoken_address = example.get("spoken_address", "")
            if not audio_path or not text:
                raise ValueError("Missing required fields")
            label_ids = processor.tokenizer(text, return_tensors="pt").input_ids[0]

            if spoken_address:
                addr_ids = processor.tokenizer(spoken_address, return_tensors="pt").input_ids[0]
                def strip_special(ids: torch.Tensor) -> torch.Tensor:
                    specials = {
                        processor.tokenizer.bos_token_id,
                        processor.tokenizer.eos_token_id,
                        processor.tokenizer.pad_token_id,
                    }
                    return torch.tensor([i for i in ids.tolist() if i not in specials])
                label_core = strip_special(label_ids)
                addr_core = strip_special(addr_ids)
                addr_mask = [0] * len(label_ids)
                if len(addr_core) > 0 and len(label_core) >= len(addr_core):
                    core = label_core.tolist()
                    sub = addr_core.tolist()
                    for start in range(len(core) - len(sub) + 1):
                        if core[start:start+len(sub)] == sub:
                            core_indices = [
                                idx for idx, i in enumerate(label_ids.tolist())
                                if i not in {
                                    processor.tokenizer.bos_token_id,
                                    processor.tokenizer.eos_token_id,
                                    processor.tokenizer.pad_token_id,
                                }
                            ]
                            for pos in range(start, start + len(sub)):
                                addr_mask[core_indices[pos]] = 1
                            break
            else:
                addr_mask = [0] * len(label_ids)

            # Convert tensor to list for HuggingFace datasets compatibility
            label_ids_list = label_ids.tolist()
            
            if max_label_length is not None and len(label_ids_list) > max_label_length:
                label_ids_list = label_ids_list[:max_label_length]
                addr_mask = addr_mask[:max_label_length]

            return {
                "audio_path": audio_path,
                "labels": label_ids_list,
                "address_token_mask": addr_mask,
                "is_address_example": int(bool(spoken_address)),
            }

        from datasets import Dataset, DatasetDict
        processed_splits = {}
        cols_to_remove = [
            "audio",
            "text",
            "text_phoneme",
            "text_canonical",
            "spoken_address",
            "canonical_address",
            "address_id",
            "locality",
            "pincode",
            "address_position",
            "variant_type",
            "speaker",
            "speed",
        ]
        train_ds = raw_datasets["train"]
        remove_cols_train = [c for c in train_ds.column_names if c in cols_to_remove and c != "audio"]
        processed_splits["train"] = train_ds.map(preprocess_text_only, remove_columns=remove_cols_train, num_proc=1)
        if "eval" in raw_datasets and len(raw_datasets["eval"]) > 0:
            eval_ds = raw_datasets["eval"]
            remove_cols_eval = [c for c in eval_ds.column_names if c in cols_to_remove and c != "audio"]
            processed_splits["eval"] = eval_ds.map(preprocess_text_only, remove_columns=remove_cols_eval, num_proc=1)
        else:
            processed_splits["eval"] = Dataset.from_dict({})
        return DatasetDict(processed_splits)

    max_train_samples = args.max_train_samples
    max_eval_samples = args.max_eval_samples

    processed_phoneme = prepare_datasets_with_label_field("text_phoneme")
    processed_canonical = prepare_datasets_with_label_field("text_canonical")
    
    # Clear again after preprocessing
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Check if eval dataset exists (use phoneme split; eval is same for both)
    has_eval = "eval" in processed_phoneme and len(processed_phoneme["eval"]) > 0
    
    # Total epochs are curriculum sum.
    total_epochs = int(args.phoneme_epochs) + int(args.canonical_epochs)
    use_bf16 = args.bf16 and torch.cuda.is_available()
    use_fp16 = not use_bf16 and torch.cuda.is_available()
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=2,
        learning_rate=args.learning_rate,
        num_train_epochs=total_epochs,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        predict_with_generate=False,
        include_inputs_for_metrics=False,
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        report_to=[],
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        remove_unused_columns=False,  # Keep audio_path for data collator
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False,
        eval_strategy="epoch" if has_eval else "no",
        save_strategy="epoch",
    )

    # Make processor available to collator
    def data_collator(features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Load audio on-the-fly during training to avoid memory issues"""
        # Keep tensors on CPU - DataLoader will move them to GPU automatically
        
        # Load audio and extract features on-the-fly (one batch at a time)
        input_features_list = []
        labels_list = []
        addr_mask_list = []
        
        for sample in features:
            try:
                # Load audio and extract features
                audio_path = sample.get("audio_path")
                if not audio_path:
                    print("[yellow]Warning: Missing audio_path in feature, skipping[/yellow]")
                    continue
                    
                waveform = load_audio(audio_path, target_sr=16000)
                input_feat = processor.feature_extractor(
                    waveform.squeeze(0), sampling_rate=16000
                ).input_features[0]
                feat_t = torch.tensor(input_feat, dtype=torch.float32)
                # Mild SpecAugment on CPU
                if args.specaugment:
                    # feat_t: (80, T)
                    freq_bins = feat_t.shape[0]
                    time_steps = feat_t.shape[1]
                    # frequency mask
                    if freq_bins > 16 and torch.rand(1).item() < 0.5:
                        freq_width = int(torch.randint(low=0, high=8, size=(1,)).item())
                        freq_start = int(torch.randint(low=0, high=max(1, freq_bins - freq_width), size=(1,)).item())
                        feat_t[freq_start:freq_start+freq_width, :] = 0.0
                    # time mask
                    if time_steps > 40 and torch.rand(1).item() < 0.5:
                        time_width = int(torch.randint(low=0, high=30, size=(1,)).item())
                        time_start = int(torch.randint(low=0, high=max(1, time_steps - time_width), size=(1,)).item())
                        feat_t[:, time_start:time_start+time_width] = 0.0
                input_features_list.append(feat_t)
                
                # Labels and mask are already preprocessed
                if "labels" not in sample:
                    print("[yellow]Warning: Missing labels in feature, skipping[/yellow]")
                    continue
                if "address_token_mask" not in sample:
                    print("[yellow]Warning: Missing address_token_mask in feature, skipping[/yellow]")
                    continue
                    
                labels_list.append(torch.tensor(sample["labels"], dtype=torch.long))
                addr_mask_list.append(torch.tensor(sample["address_token_mask"], dtype=torch.long))
                
                # Clear waveform immediately
                del waveform
            except KeyError as e:
                print(f"[red]Error: Missing key in feature: {e}, skipping batch item[/red]")
                continue
            except Exception as e:
                audio_path_str = sample.get("audio_path", "unknown") if isinstance(sample, dict) else "unknown"
                print(f"[red]Error loading {audio_path_str}: {e}, skipping batch item[/red]")
                continue
        
        if not input_features_list:
            raise RuntimeError("No valid audio files in batch")
        
        # Keep on CPU - DataLoader handles GPU transfer
        input_features = torch.stack(input_features_list)
        labels = nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
        addr_mask = nn.utils.rnn.pad_sequence(addr_mask_list, batch_first=True, padding_value=0)
        
        # Whisper needs decoder_input_ids: labels shifted right (teacher forcing)
        # decoder_input_ids = [BOS] + labels[:-1]
        bos_token_id = processor.tokenizer.bos_token_id
        if bos_token_id is None:
            bos_token_id = processor.tokenizer.pad_token_id
        
        decoder_input_ids = labels.clone()
        # Shift right: prepend BOS, remove last token
        decoder_input_ids = torch.cat([
            torch.full((decoder_input_ids.shape[0], 1), bos_token_id, dtype=torch.long),
            decoder_input_ids[:, :-1]
        ], dim=1)
        
        # Ensure padding tokens are correct (no-op if already correct, but ensures consistency)
        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            # Find actual padding positions (where labels are -100 or pad_token_id)
            padding_mask = (labels == pad_token_id) | (labels == -100)
            decoder_input_ids[padding_mask] = pad_token_id
        
        return {
            "input_features": input_features,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
            "address_token_mask": addr_mask,
        }

    # Create data collator with processor reference
    def make_collator():
        return lambda features: data_collator(features)
    
    # Handle empty eval dataset
    eval_dataset = None
    if has_eval:
        eval_dataset = processed_phoneme["eval"]
    else:
        print("[yellow]No eval dataset available, training without evaluation.[/yellow]")
        # Disable evaluation in training args
        training_args.eval_strategy = "no"
        training_args.load_best_model_at_end = False
    
    callbacks = []
    if has_eval and getattr(args, "early_stopping_patience", 0) > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=0.0,
            )
        )

    trainer = AddressWeightedTrainer(
        alpha_start=args.alpha_start,
        alpha_end=args.alpha_end,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        rdrop_alpha=args.rdrop_alpha,
        sequence_bonus=args.sequence_bonus,
        model=model,
        args=training_args,
        train_dataset=processed_phoneme["train"],
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    print(f"[bold]Training config:[/bold]")
    print(f"  - Focal gamma: {args.focal_gamma}")
    print(f"  - Label smoothing: {args.label_smoothing}")
    print(f"  - R-Drop alpha: {args.rdrop_alpha}")
    print(f"  - Sequence bonus: {args.sequence_bonus}")
    print(f"  - Address weight: {args.alpha_start} → {args.alpha_end}")

    print("[bold]Starting training...[/bold]")
    
    # Approach 6: Simplified training (skip phoneme curriculum by default)
    # Based on learnings: phoneme stage may confuse the model for canonical accuracy
    if args.phoneme_epochs > 0:
        print(f"[bold cyan]Phase 1: Phoneme-faithful training ({args.phoneme_epochs} epochs)[/bold cyan]")
        trainer.args.num_train_epochs = int(args.phoneme_epochs)
        trainer.train_dataset = processed_phoneme["train"]
        trainer.train()
    
    if args.canonical_epochs > 0:
        phase_num = 2 if args.phoneme_epochs > 0 else 1
        print(f"[bold cyan]Phase {phase_num}: Canonical training ({args.canonical_epochs} epochs)[/bold cyan]")
        trainer.args.num_train_epochs = int(args.canonical_epochs)
        trainer.train_dataset = processed_canonical["train"]
        trainer.train()

    print("[bold green]Saving LoRA adapter and config...[/bold green]")
    ROOT.joinpath("models").mkdir(exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

