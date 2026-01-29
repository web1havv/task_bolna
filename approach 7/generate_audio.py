"""
Approach 5: Generate synthetic audio from single-turn samples using Sarvam TTS API.
Reads from data/sentences/ (train, test, eval), outputs WAV to audio/{split}.

Features:
- Incremental: only synthesize missing WAVs (doesn't redo existing ones).
- Multi-speaker (configurable list) and mild speed variation (0.97–1.03x).
- Metadata includes curriculum fields: text_phoneme / text_canonical.
"""
import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from rich import print

ROOT = Path(__file__).resolve().parent
DATA_SENTENCES = ROOT / "data" / "sentences"
AUDIO_ROOT = ROOT / "audio"
TARGET_SR = 16000

# Sarvam API key: set SARVAM_API_KEY environment variable
SARVAM_API_KEY_ENV = "SARVAM_API_KEY"
SARVAM_SPEAKERS_ENV = "SARVAM_SPEAKERS"  # optional comma-separated list


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


def apply_speed_variation(audio: np.ndarray, speed: float, sample_rate: int = 16000) -> np.ndarray:
    """
    Apply mild speed variation by resampling (changes tempo & pitch).
    Keeps output sample_rate constant by resampling back to sample_rate.
    """
    if speed <= 0:
        return audio
    if abs(speed - 1.0) < 1e-6:
        return audio
    from scipy import signal as scipy_signal

    # First: change duration
    target_len = max(1, int(round(len(audio) / speed)))
    audio2 = scipy_signal.resample(audio, target_len).astype(np.float32)
    # Second: ensure exactly sample_rate (already), just return
    return audio2


def choose_speaker_and_speed(speakers: List[str]) -> Tuple[str, float]:
    spk = random_choice(speakers)
    speed = float(np.random.uniform(0.97, 1.03))
    return spk, speed


def random_choice(xs: List[str]) -> str:
    return xs[int(np.random.randint(0, len(xs)))]


def existing_wavs(out_dir: Path) -> set:
    return {p.stem for p in out_dir.glob("*.wav")}


def process_split(
    client,
    split: str,
    max_samples: Optional[int] = None,
    *,
    speakers: Optional[List[str]] = None,
    model: str = "bulbul:v2",
    skip_existing: bool = True,
    sleep_s: float = 0.12,
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

    speakers = speakers or ["anushka"]
    present = existing_wavs(out_dir) if skip_existing else set()
    meta: List[Dict] = []

    for idx, rec in enumerate(records):
        sample_id = rec["sample_id"]
        # Curriculum fields (fallback to legacy)
        text_phoneme = rec.get("text_phoneme") or rec.get("text") or ""
        text_canonical = rec.get("text_canonical") or rec.get("text") or ""
        text = text_phoneme
        if skip_existing and sample_id in present:
            continue
        if not text.strip():
            print(f"[yellow]Skipping {sample_id}: empty text[/yellow]")
            continue

        speaker, speed = choose_speaker_and_speed(speakers)
        audio_arr = synthesize_sarvam(client, text, speaker=speaker, model=model)
        if audio_arr is None:
            print(f"[yellow]Skipping {sample_id} due to TTS failures.[/yellow]")
            continue
        audio_arr = apply_speed_variation(audio_arr, speed, sample_rate=TARGET_SR)

        file_name = f"{sample_id}.wav"
        out_path = out_dir / file_name
        sf.write(out_path, audio_arr, TARGET_SR, subtype="PCM_16")

        rec_meta = {
            "audio_path": str(out_path.relative_to(ROOT)),
            "transcript": text,  # phoneme-faithful label for training stage 1
            "text_phoneme": text_phoneme,
            "text_canonical": text_canonical,
            "conversation_id": sample_id,
            "canonical_address": rec.get("canonical_address", ""),
            "spoken_address": rec.get("spoken_address", ""),
            "address_id": rec.get("address_id", ""),
            "locality": rec.get("locality", ""),
            "pincode": rec.get("pincode", ""),
            "speaker": speaker,
            "speed": speed,
            "address_position": rec.get("address_position", ""),
            "variant_type": rec.get("variant_type", ""),
        }
        meta.append(rec_meta)

        if (idx + 1) % 50 == 0:
            print(f"[blue]{split}: processed {idx + 1} sentences...[/blue]")

        # Rate limiting: avoid hitting Sarvam API too fast
        time.sleep(sleep_s)

    meta_path = out_dir / f"{split}_metadata.jsonl"
    # Append metadata so incremental runs don't wipe older entries
    with meta_path.open("a", encoding="utf-8") as f:
        for m in meta:
            json.dump(m, f, ensure_ascii=False)
            f.write("\n")

    print(f"[green]Saved {len(meta)} {split} audio files and metadata to {out_dir}.[/green]")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic audio from sentences using Sarvam TTS API (incremental)."
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
    parser.add_argument(
        "--speakers",
        type=str,
        default=None,
        help="Comma-separated Sarvam speaker names. If omitted, uses $SARVAM_SPEAKERS or a safe default.",
    )
    parser.add_argument(
        "--sarvam-model",
        type=str,
        default="bulbul:v2",
        help="Sarvam TTS model name (default: bulbul:v2).",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="If set, re-synthesize even if WAV already exists.",
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

    speakers_str = args.speakers or os.environ.get(SARVAM_SPEAKERS_ENV)
    # Default: keep conservative; user can override to known valid Sarvam speakers.
    speakers = [s.strip() for s in (speakers_str.split(",") if speakers_str else ["anushka"]) if s.strip()]

    ensure_dirs()
    process_split(
        client,
        "train",
        max_samples=args.max_train,
        speakers=speakers,
        model=args.sarvam_model,
        skip_existing=not args.no_skip_existing,
    )
    process_split(
        client,
        "test",
        max_samples=args.max_test,
        speakers=speakers,
        model=args.sarvam_model,
        skip_existing=not args.no_skip_existing,
    )
    process_split(
        client,
        "eval",
        max_samples=args.max_eval,
        speakers=speakers,
        model=args.sarvam_model,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    main()
