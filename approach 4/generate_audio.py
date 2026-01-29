"""
Approach 4: Generate synthetic audio from single-sentence samples using Sarvam TTS API.
Reads from data/sentences/ (train, test, eval), outputs WAV to audio/{split}.
"""
import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
from rich import print

ROOT = Path(__file__).resolve().parent
DATA_SENTENCES = ROOT / "data" / "sentences"
AUDIO_ROOT = ROOT / "audio"
TARGET_SR = 16000

# Sarvam API key: set SARVAM_API_KEY environment variable
SARVAM_API_KEY_ENV = "SARVAM_API_KEY"


def ensure_dirs() -> None:
    (AUDIO_ROOT / "train").mkdir(parents=True, exist_ok=True)
    (AUDIO_ROOT / "test").mkdir(parents=True, exist_ok=True)
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


def synthesize_sarvam(
    client,
    text: str,
    *,
    lang: str = "en-IN",
    sample_rate: int = 16000,
    speaker: str = "anushka",
    model: str = "bulbul:v2",
    max_retries: int = 3,
) -> Optional[np.ndarray]:
    """
    Use Sarvam TTS to synthesize text. Returns mono float32 array at 16kHz, or None on failure.
    """
    from sarvamai.play import save as sarvam_save

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.text_to_speech.convert(
                text=text,
                target_language_code=lang,
                speech_sample_rate=sample_rate,
                speaker=speaker,
                model=model,
                enable_preprocessing=True,
            )
            if not resp:
                raise RuntimeError("Sarvam API returned no response")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                sarvam_save(resp, tmp_path)
                data, sr = sf.read(tmp_path, dtype="float32")
            finally:
                Path(tmp_path).unlink(missing_ok=True)

            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != sample_rate:
                from scipy import signal as scipy_signal

                num = int(len(data) * sample_rate / sr)
                data = scipy_signal.resample(data, num).astype(np.float32)
            return data
        except Exception as e:
            print(f"[red]Sarvam TTS attempt {attempt}/{max_retries}: {e}[/red]")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    return None


def process_split(
    client,
    split: str,
    max_samples: Optional[int] = None,
) -> None:
    jsonl_path = DATA_SENTENCES / f"{split}_sentences.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"{jsonl_path} not found. Run generate_sentences.py first."
        )

    out_dir = AUDIO_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(jsonl_path)
    if max_samples is not None:
        records = records[:max_samples]

    meta: List[Dict] = []

    for idx, rec in enumerate(records):
        sample_id = rec["sample_id"]
        text = rec["text"]

        audio_arr = synthesize_sarvam(client, text)
        if audio_arr is None:
            print(f"[yellow]Skipping {sample_id} due to TTS failures.[/yellow]")
            continue

        file_name = f"{sample_id}.wav"
        out_path = out_dir / file_name
        sf.write(out_path, audio_arr, 16000, subtype="PCM_16")

        rec_meta = {
            "audio_path": str(out_path.relative_to(ROOT)),
            "transcript": text,
            "conversation_id": sample_id,
            "canonical_address": rec.get("canonical_address", ""),
            "spoken_address": rec.get("spoken_address", ""),
            "address_id": rec.get("address_id", ""),
            "locality": rec.get("locality", ""),
            "pincode": rec.get("pincode", ""),
        }
        meta.append(rec_meta)

        if (idx + 1) % 50 == 0:
            print(f"[blue]{split}: processed {idx + 1} sentences...[/blue]")

        # Rate limiting: avoid hitting Sarvam API too fast
        time.sleep(0.15)

    meta_path = out_dir / f"{split}_metadata.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for m in meta:
            json.dump(m, f, ensure_ascii=False)
            f.write("\n")

    print(f"[green]Saved {len(meta)} {split} audio files and metadata to {out_dir}.[/green]")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic audio from sentences using Sarvam TTS API."
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Optional max number of train sentences to synthesize.",
    )
    parser.add_argument(
        "--max-test",
        type=int,
        default=None,
        help="Optional max number of test sentences to synthesize.",
    )
    parser.add_argument(
        "--max-eval",
        type=int,
        default=None,
        help="Optional max number of eval sentences to synthesize.",
    )
    args = parser.parse_args(argv)

    api_key = os.environ.get(SARVAM_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"Set {SARVAM_API_KEY_ENV} environment variable with your Sarvam API key. "
            "Get it from https://dashboard.sarvam.ai"
        )

    from sarvamai import SarvamAI
    client = SarvamAI(api_subscription_key=api_key)

    ensure_dirs()
    process_split(client, "train", max_samples=args.max_train)
    process_split(client, "test", max_samples=args.max_test)
    process_split(client, "eval", max_samples=args.max_eval)


if __name__ == "__main__":
    main()
