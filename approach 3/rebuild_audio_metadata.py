"""
Rebuild metadata files from existing audio files and conversation JSONL.
Use this if audio generation was stopped before metadata files were created.
"""
import json
from pathlib import Path
from typing import Dict, List

from rich import print


ROOT = Path(__file__).resolve().parent
DATA_CONV = ROOT / "data" / "conversations"
AUDIO_ROOT = ROOT / "audio"


def load_conversations(split: str) -> Dict[str, Dict]:
    """Load conversations and index by conversation_id"""
    jsonl_path = DATA_CONV / f"{split}_conversations.jsonl"
    if not jsonl_path.exists():
        return {}
    
    conversations = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            conv = json.loads(line)
            conversations[conv["conversation_id"]] = conv
    return conversations


def rebuild_metadata(split: str) -> None:
    """Rebuild metadata file from existing audio files"""
    audio_dir = AUDIO_ROOT / split
    if not audio_dir.exists():
        print(f"[yellow]Audio directory {audio_dir} does not exist.[/yellow]")
        return
    
    conversations = load_conversations(split)
    if not conversations:
        print(f"[yellow]No conversations found for split={split}.[/yellow]")
        return
    
    # Find all WAV files
    wav_files = list(audio_dir.glob("*.wav"))
    if not wav_files:
        print(f"[yellow]No WAV files found in {audio_dir}.[/yellow]")
        return
    
    print(f"[bold]Found {len(wav_files)} WAV files in {audio_dir}[/bold]")
    
    metadata = []
    matched = 0
    missing = 0
    
    for wav_file in sorted(wav_files):
        # Extract conversation_id from filename (e.g., train_addr_000054.wav -> train_addr_000054)
        conv_id = wav_file.stem
        
        if conv_id not in conversations:
            print(f"[yellow]Warning: No conversation found for {conv_id}[/yellow]")
            missing += 1
            continue
        
        conv = conversations[conv_id]
        
        # Build metadata record (same format as generate_audio.py)
        rec_meta = {
            "audio_path": str(wav_file.relative_to(ROOT)),
            "transcript": conv.get("transcript", ""),
            "conversation_id": conv_id,
            "canonical_address": conv.get("canonical_address", ""),
            "spoken_address": conv.get("spoken_address", ""),
            "address_id": conv.get("address_id", ""),
            "locality": conv.get("locality", ""),
            "pincode": conv.get("pincode", ""),
        }
        metadata.append(rec_meta)
        matched += 1
    
    # Write metadata file
    meta_path = audio_dir / f"{split}_metadata.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for m in metadata:
            json.dump(m, f, ensure_ascii=False)
            f.write("\n")
    
    print(
        f"[green]Rebuilt {split}_metadata.jsonl with {matched} entries "
        f"({missing} audio files without matching conversations)[/green]"
    )


def main():
    print("[bold]Rebuilding audio metadata files...[/bold]")
    rebuild_metadata("train")
    rebuild_metadata("eval")
    print("[bold green]Done![/bold green]")


if __name__ == "__main__":
    main()
