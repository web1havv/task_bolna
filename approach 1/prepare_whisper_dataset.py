import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from rich import print


ROOT = Path(__file__).resolve().parent
AUDIO_ROOT = ROOT / "audio"
WHISPER_DATASET_DIR = ROOT / "data" / "whisper_dataset"


def load_metadata(split: str) -> List[Dict]:
    meta_path = AUDIO_ROOT / split / f"{split}_metadata.jsonl"
    if not meta_path.exists():
        print(
            f"[yellow]{meta_path} not found for split={split}. "
            "Continuing with an empty split.[/yellow]"
        )
        return []
    records: List[Dict] = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(rec)
    return records


def sanitize_record(rec: Dict) -> Optional[Dict]:
    text = (rec.get("transcript") or "").strip()
    if not text:
        return None

    audio_rel = rec.get("audio_path")
    if not audio_rel:
        return None

    audio_path = (ROOT / audio_rel).resolve()
    if not audio_path.exists():
        # Skip if audio missing
        return None

    spoken_address = (rec.get("spoken_address") or "").strip()

    return {
        "audio": str(audio_path),
        "text": text,
        "spoken_address": spoken_address,
        "canonical_address": (rec.get("canonical_address") or "").strip(),
        "address_id": (rec.get("address_id") or "").strip(),
        "locality": (rec.get("locality") or "").strip(),
        "pincode": (rec.get("pincode") or "").strip(),
        "is_address_example": bool(spoken_address),
    }


def write_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare JSONL dataset for Whisper fine-tuning from synthetic audio."
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Optional max number of train examples.",
    )
    parser.add_argument(
        "--max-eval",
        type=int,
        default=None,
        help="Optional max number of eval examples.",
    )
    args = parser.parse_args(argv)

    train_meta = load_metadata("train")
    eval_meta = load_metadata("eval")

    train_records: List[Dict] = []
    eval_records: List[Dict] = []

    for rec in train_meta:
        s = sanitize_record(rec)
        if s is not None:
            train_records.append(s)
    for rec in eval_meta:
        s = sanitize_record(rec)
        if s is not None:
            eval_records.append(s)

    if args.max_train is not None:
        train_records = train_records[: args.max_train]
    if args.max_eval is not None:
        eval_records = eval_records[: args.max_eval]

    train_out = WHISPER_DATASET_DIR / "train.jsonl"
    eval_out = WHISPER_DATASET_DIR / "eval.jsonl"

    write_jsonl(train_out, train_records)
    write_jsonl(eval_out, eval_records)

    print(
        f"[green]Prepared Whisper dataset at {WHISPER_DATASET_DIR} with "
        f"{len(train_records)} train and {len(eval_records)} eval examples.[/green]"
    )


if __name__ == "__main__":
    main()

