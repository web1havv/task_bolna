import json
from pathlib import Path
from typing import Dict, List

from rich import print


ROOT = Path(__file__).resolve().parent
DATA_CONV = ROOT / "data" / "conversations"
AUDIO_ROOT = ROOT / "audio"


def rebuild_split(split: str) -> None:
    conv_path = DATA_CONV / f"{split}_conversations.jsonl"
    audio_dir = AUDIO_ROOT / split
    meta_path = audio_dir / f"{split}_metadata.jsonl"

    if not conv_path.exists():
        print(f"[yellow]{conv_path} not found; skipping {split}.[/yellow]")
        return
    if not audio_dir.exists():
        print(f"[yellow]{audio_dir} not found; skipping {split}.[/yellow]")
        return

    # Load conversations into dict by conversation_id
    conv_by_id: Dict[str, Dict] = {}
    with conv_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec.get("conversation_id")
            if cid:
                conv_by_id[cid] = rec

    records: List[Dict] = []
    for wav in sorted(audio_dir.glob("*.wav")):
        conv_id = wav.stem
        rec = conv_by_id.get(conv_id)
        if not rec:
            print(f"[yellow]No conversation found for audio {wav.name}; skipping.[/yellow]")
            continue

        records.append(
            {
                "audio_path": str(wav.relative_to(ROOT)),
                "transcript": rec.get("transcript", ""),
                "conversation_id": conv_id,
                "canonical_address": rec.get("canonical_address", ""),
                "spoken_address": rec.get("spoken_address", ""),
                "address_id": rec.get("address_id", ""),
                "locality": rec.get("locality", ""),
                "pincode": rec.get("pincode", ""),
            }
        )

    if not records:
        print(f"[yellow]No records built for split={split}.[/yellow]")
        return

    with meta_path.open("w", encoding="utf-8") as f:
        for r in records:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")

    print(
        f"[green]Rebuilt {len(records)} metadata entries for split={split} "
        f"at {meta_path}.[/green]"
    )


def main() -> None:
    rebuild_split("train")
    rebuild_split("eval")


if __name__ == "__main__":
    main()

