"""
Approach 4: Rebuild metadata files from existing audio and sentence JSONL.
Use this if audio generation was stopped before metadata files were created.
"""
import json
from pathlib import Path
from typing import Dict, List

from rich import print


ROOT = Path(__file__).resolve().parent
DATA_SENTENCES = ROOT / "data" / "sentences"
AUDIO_ROOT = ROOT / "audio"


def load_sentences(split: str) -> Dict[str, Dict]:
    """Load sentences and index by sample_id."""
    jsonl_path = DATA_SENTENCES / f"{split}_sentences.jsonl"
    if not jsonl_path.exists():
        return {}

    out: Dict[str, Dict] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["sample_id"]] = rec
    return out


def rebuild_metadata(split: str) -> None:
    """Rebuild metadata file from existing audio files."""
    audio_dir = AUDIO_ROOT / split
    if not audio_dir.exists():
        print(f"[yellow]Audio directory {audio_dir} does not exist.[/yellow]")
        return

    sentences = load_sentences(split)
    if not sentences:
        print(f"[yellow]No sentences found for split={split}.[/yellow]")
        return

    wav_files = list(audio_dir.glob("*.wav"))
    if not wav_files:
        print(f"[yellow]No WAV files found in {audio_dir}.[/yellow]")
        return

    print(f"[bold]Found {len(wav_files)} WAV files in {audio_dir}[/bold]")

    metadata: List[Dict] = []
    matched = 0
    missing = 0

    for wav_file in sorted(wav_files):
        sample_id = wav_file.stem

        if sample_id not in sentences:
            print(f"[yellow]Warning: No sentence found for {sample_id}[/yellow]")
            missing += 1
            continue

        rec = sentences[sample_id]
        rec_meta = {
            "audio_path": str(wav_file.relative_to(ROOT)),
            "transcript": rec.get("text", ""),
            "conversation_id": sample_id,
            "canonical_address": rec.get("canonical_address", ""),
            "spoken_address": rec.get("spoken_address", ""),
            "address_id": rec.get("address_id", ""),
            "locality": rec.get("locality", ""),
            "pincode": rec.get("pincode", ""),
        }
        metadata.append(rec_meta)
        matched += 1

    meta_path = audio_dir / f"{split}_metadata.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for m in metadata:
            json.dump(m, f, ensure_ascii=False)
            f.write("\n")

    print(
        f"[green]Rebuilt {split}_metadata.jsonl with {matched} entries "
        f"({missing} audio files without matching sentences)[/green]"
    )


def main() -> None:
    print("[bold]Rebuilding audio metadata files (Approach 4: train / test / eval)...[/bold]")
    rebuild_metadata("train")
    rebuild_metadata("test")
    rebuild_metadata("eval")
    print("[bold green]Done![/bold green]")


if __name__ == "__main__":
    main()
