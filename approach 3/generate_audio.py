import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydub import AudioSegment, effects
from scipy import signal
from rich import print
from gtts import gTTS  # Always use gTTS for TTS (simpler & consistent)


ROOT = Path(__file__).resolve().parent
DATA_CONV = ROOT / "data" / "conversations"
AUDIO_ROOT = ROOT / "audio"


def ensure_dirs() -> None:
    (AUDIO_ROOT / "train").mkdir(parents=True, exist_ok=True)
    (AUDIO_ROOT / "eval").mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def synthesize_gtts(
    text: str,
    lang: str = "en",
    tld: str = "co.in",
    max_retries: int = 3,
) -> Optional[AudioSegment]:
    """
    Use gTTS to synthesize an utterance and return a pydub AudioSegment.
    Fallback method when GPU TTS is not available.
    """
    tmp_path = ROOT / "_tmp_gtts.mp3"
    last_err: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            tts = gTTS(text=text, lang=lang, tld=tld)
            tts.save(str(tmp_path))

            # Basic sanity check: file should exist and be non-empty
            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                raise RuntimeError("gTTS produced empty file")

            audio = AudioSegment.from_file(tmp_path)
            tmp_path.unlink(missing_ok=True)
            return audio
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(
                f"[red]gTTS/ffmpeg failed on attempt {attempt}/{max_retries}: "
                f"{e}[/red]"
            )
            tmp_path.unlink(missing_ok=True)

    print(
        f"[yellow]Giving up on this utterance after {max_retries} failed attempts.[/yellow]"
    )
    if last_err:
        print(f"[yellow]Last error: {last_err}[/yellow]")
    return None


def apply_telephone_corruption(audio: AudioSegment, seed: Optional[int] = None) -> AudioSegment:
    """
    Return clean TTS audio, only resampled to 16kHz mono.
    NO added noise, filters, speed changes, or clipping.
    """
    # Just ensure 16kHz mono – nothing else
    return audio.set_frame_rate(16000).set_channels(1)


def build_call_audio(transcript: str) -> Optional[AudioSegment]:
    """
    TTS the full conversation transcript as a single utterance.
    Uses gTTS (CPU, free).
    """
    return synthesize_gtts(transcript)


def process_split(
    split: str,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> None:
    random.seed(seed)
    jsonl_path = DATA_CONV / f"{split}_conversations.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"{jsonl_path} not found. Run generate_conversations.py first."
        )

    out_dir = AUDIO_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(jsonl_path)
    if max_samples is not None:
        records = records[:max_samples]

    meta: List[Dict] = []

    for idx, rec in enumerate(records):
        conv_id = rec["conversation_id"]
        transcript = rec["transcript"]

        # Synthesize base audio with gTTS
        base_audio = build_call_audio(transcript)
        if base_audio is None:
            print(f"[yellow]Skipping conversation {conv_id} due to TTS failures.[/yellow]")
            continue
        
        # Format audio to 16kHz mono (no artificial noise/corruption)
        audio = apply_telephone_corruption(base_audio, seed=seed + idx)

        # Save as WAV (16kHz mono)
        file_name = f"{conv_id}.wav"
        out_path = out_dir / file_name
        audio.set_channels(1).set_frame_rate(16000).export(out_path, format="wav")

        rec_meta = {
            "audio_path": str(out_path.relative_to(ROOT)),
            "transcript": transcript,
            "conversation_id": conv_id,
            "canonical_address": rec.get("canonical_address", ""),
            "spoken_address": rec.get("spoken_address", ""),
            "address_id": rec.get("address_id", ""),
            "locality": rec.get("locality", ""),
            "pincode": rec.get("pincode", ""),
        }
        meta.append(rec_meta)

        if (idx + 1) % 100 == 0:
            print(f"[blue]{split}: processed {idx + 1} conversations...[/blue]")

    meta_path = out_dir / f"{split}_metadata.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for m in meta:
            json.dump(m, f, ensure_ascii=False)
            f.write("\n")

    print(
        f"[green]Saved {len(meta)} {split} audio files and metadata to {out_dir}.[/green]"
    )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic telephone-style audio for conversations using gTTS (processes all conversations by default)."
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Optional max number of train conversations to synthesize.",
    )
    parser.add_argument(
        "--max-eval",
        type=int,
        default=None,
        help="Optional max number of eval conversations to synthesize.",
    )
    args = parser.parse_args(argv)

    ensure_dirs()
    process_split("train", max_samples=args.max_train)
    process_split("eval", max_samples=args.max_eval)


if __name__ == "__main__":
    main()

