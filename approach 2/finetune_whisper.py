import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from rich import print
from torch import nn
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import torchaudio
import soundfile as sf


ROOT = Path(__file__).resolve().parent
WHISPER_DATASET_DIR = ROOT / "data" / "whisper_dataset"
DEFAULT_MODEL_NAME = "openai/whisper-tiny"


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
    data_files = {
        "train": str(WHISPER_DATASET_DIR / "train.jsonl"),
        "eval": str(WHISPER_DATASET_DIR / "eval.jsonl"),
    }

    raw_datasets = load_dataset("json", data_files=data_files)

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
        audio_path = example["audio"]
        text = example["text"]
        spoken_address = example.get("spoken_address", "")

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
    processed = raw_datasets.map(
        preprocess_text_only,
        remove_columns=["audio", "text", "spoken_address", "canonical_address", "address_id", "locality", "pincode"],
        num_proc=1,
        desc="Preprocessing text only",
    )
    
    # Clear cache after preprocessing
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return processed


class AddressWeightedTrainer(Seq2SeqTrainer):
    def __init__(self, alpha: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        address_mask = inputs.pop("address_token_mask", None)

        outputs = model(**inputs)
        logits = outputs.logits

        # Shift for teacher forcing
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        vocab_size = shift_logits.size(-1)
        loss_fct = nn.CrossEntropyLoss(
            ignore_index=self.label_smoother.ignore_index if self.label_smoother else -100,
            reduction="none",
        )

        loss = loss_fct(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
        )  # (batch * seq)

        active_positions = shift_labels.view(-1) != (
            self.label_smoother.ignore_index if self.label_smoother else -100
        )

        weight = torch.ones_like(loss)
        if address_mask is not None:
            # shift address mask to align with shift_labels
            addr = address_mask[:, 1:].contiguous().view(-1)
            addr = addr.to(loss.device).float()
            weight = weight + self.alpha * addr

        loss = (loss * weight)
        loss = loss[active_positions].mean()

        return (loss, outputs) if return_outputs else loss


def create_lora_config() -> LoraConfig:
    # Decoder-only LoRA: Only target decoder attention projection layers
    # Target modules: decoder.layers.*.self_attn.{q_proj, k_proj, v_proj, o_proj}
    # PEFT uses substring matching - patterns must be contained in the full module name
    # Since encoder is frozen, only decoder modules will be affected
    # Using patterns that match decoder attention projections specifically
    return LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "decoder.layers.self_attn.q_proj",
            "decoder.layers.self_attn.k_proj",
            "decoder.layers.self_attn.v_proj",
            "decoder.layers.self_attn.o_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )


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
        default=str(ROOT / "models" / "whisper_bangalore_lora"),
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
        default=8,
        help="Per-device batch size.",
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
        default=2.0,
        help="Address token loss weight multiplier.",
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
    args = parser.parse_args()

    print(f"[bold]Loading Whisper processor and model: {args.model_name}[/bold]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[bold]Using device: {device}[/bold]")
    
    processor = WhisperProcessor.from_pretrained(args.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    model = model.to(device)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # Freeze encoder completely (decoder-only LoRA approach)
    for param in model.model.encoder.parameters():
        param.requires_grad = False
    print("[bold yellow]Encoder frozen completely (decoder-only fine-tuning)[/bold yellow]")

    # Prepare model for decoder-only LoRA fine-tuning
    lora_config = create_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("[bold]Preparing datasets...[/bold]")
    # Clear memory before preprocessing
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    processed = prepare_datasets(
        processor,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    
    # Clear again after preprocessing
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=4,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        predict_with_generate=False,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=2,  # Accumulate gradients to simulate larger batch
        report_to=[],
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        evaluation_strategy="epoch" if len(processed.get("eval", [])) > 0 else "no",
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
        
        for f in features:
            try:
                # Load audio and extract features
                audio_path = f["audio_path"]
                waveform = load_audio(audio_path, target_sr=16000)
                input_feat = processor.feature_extractor(
                    waveform.squeeze(0), sampling_rate=16000
                ).input_features[0]
                input_features_list.append(torch.tensor(input_feat, dtype=torch.float32))
                
                # Labels and mask are already preprocessed
                labels_list.append(torch.tensor(f["labels"], dtype=torch.long))
                addr_mask_list.append(torch.tensor(f["address_token_mask"], dtype=torch.long))
                
                # Clear waveform immediately
                del waveform
            except Exception as e:
                print(f"[red]Error loading {audio_path}: {e}, skipping batch item[/red]")
                continue
        
        if not input_features_list:
            raise RuntimeError("No valid audio files in batch")
        
        # Keep on CPU - DataLoader handles GPU transfer
        input_features = torch.stack(input_features_list)
        labels = nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
        addr_mask = nn.utils.rnn.pad_sequence(addr_mask_list, batch_first=True, padding_value=0)
        
        # Whisper needs decoder_input_ids: labels shifted right (teacher forcing)
        # decoder_input_ids = [BOS] + labels[:-1]
        bos_token_id = processor.tokenizer.bos_token_id or processor.tokenizer.pad_token_id
        decoder_input_ids = labels.clone()
        # Shift right: prepend BOS, remove last token
        decoder_input_ids = torch.cat([
            torch.full((decoder_input_ids.shape[0], 1), bos_token_id, dtype=torch.long),
            decoder_input_ids[:, :-1]
        ], dim=1)
        # Replace padding with pad_token_id
        decoder_input_ids[decoder_input_ids == processor.tokenizer.pad_token_id] = processor.tokenizer.pad_token_id
        
        return {
            "input_features": input_features,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
            "address_token_mask": addr_mask,
        }

    # Create data collator with processor reference
    def make_collator():
        return lambda features: data_collator(features)
    
    trainer = AddressWeightedTrainer(
        alpha=args.alpha,
        model=model,
        args=training_args,
        train_dataset=processed["train"],
        eval_dataset=processed["eval"] if len(processed["eval"]) > 0 else None,
        data_collator=data_collator,
    )

    print("[bold]Starting training...[/bold]")
    trainer.train()

    print("[bold green]Saving LoRA adapter and config...[/bold green]")
    ROOT.joinpath("models").mkdir(exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

