## Approach 1 — Bangalore Address-Aware Whisper Fine-Tuning

This folder implements **Approach 1**: an end-to-end, fully local and free pipeline to improve address transcription inside long telephone conversations by fine-tuning an open-source Whisper model for **Bangalore / Bengaluru** addresses only.

The pipeline has three stages:

1. **Bangalore address dataset creation**
2. **Synthetic telephone conversation + audio generation**
3. **Whisper fine-tuning (LoRA) + evaluation vs. base Whisper**

All scripts are written for **Python 3.10+** and use only free/open-source libraries.

---

## Project Structure

- `data/`
  - `raw/`
    - `osm.csv` — raw OpenStreetMap-derived localities/roads (Bangalore only)
    - `pincode.csv` — India Post 560xxx PIN-based localities
    - `bbmp.csv` — (optional) scraped BBMP ward/locality names
  - `final/`
    - `bangalore_addresses.csv` — canonical + spoken variants (primary address dataset)
  - `conversations/`
    - `train_conversations.jsonl` — training split (text + metadata)
    - `eval_conversations.jsonl` — evaluation split (text + metadata)
- `audio/`
  - `train/` — synthetic call audio for training
  - `eval/` — synthetic call audio for evaluation
- `generate_addresses.py` — builds the Bangalore address dataset from OSM + PINs (+ optional BBMP)
- `generate_conversations.py` — generates synthetic call-style conversations with embedded spoken addresses
- `generate_audio.py` — converts conversations to telephone-style audio using gTTS + simple DSP
- `prepare_whisper_dataset.py` — builds HF-style dataset (JSONL/Parquet) for Whisper fine-tuning
- `finetune_whisper.py` — LoRA fine-tuning script for Whisper (encoder+decoder attention only)
- `evaluate_whisper.py` — compares **base vs. fine-tuned** Whisper on an address-heavy eval set
- `requirements.txt` — Python dependencies for this pipeline

---

## 1. Why Synthetic Data Works Here

- **We lack real call recordings**, especially for noisy, Indian-English Bangalore addresses.
- Open-source Whisper models already model **general conversation** well but fail on:
  - Localities (`Koramangala`, `BTM`, `Indiranagar`)
  - Structured address fragments (`4th Block`, `2nd Stage`, `Sector 6`, `100 Feet Road`)
- Synthetic data lets us:
  - Precisely control **where** in the conversation the address appears (mid-call, once).
  - Generate **multiple spoken variants** of the same canonical address.
  - Systematically cover **dozens of thousands** of canonical Bangalore localities.

We use:

- **OSM** (Overpass API) for roads, suburbs, neighbourhoods, residential localities.
- **India Post** (PINs starting with 560) for locality names tied to Bangalore Urban.
- Optional **BBMP ward/locality lists** for extra coverage.

---

## 2. Why Long Conversations Matter

Real customer calls:

- Contain **casual chit-chat, hesitations, confirmations**, and unrelated context.
- Mention the address **once**, often in the middle, sometimes with noise or overlap.
- Have **variable length** (30 seconds to a few minutes).

Short “address-only” clips do **not** reflect this distribution.

This pipeline:

- Generates **30s–3min** synthetic calls.
- Embeds the address **once per conversation** in a realistic confirmation snippet.
- Includes **fillers and corrections** (“uh”, “near the signal”, “opposite the park”).

This better matches **Bolna-style contact center calls**, so the fine-tuned Whisper adapts to **in-situ address mentions**, not just clean dictation.

---

## 3. Why Address-Weighted Fine-Tuning Is Needed

Whisper’s general ASR is strong; we **do not want to break it**.

The goal is to:

- **Improve accuracy** for:
  - Proper nouns (localities, landmarks)
  - Ordinals and numbers in address context
  - Bangalore-specific tokens (e.g. `HSR`, `BTM`, `Indiranagar`)
- While **preserving** generic conversational WER.

To do this:

- We use **parameter-efficient fine-tuning (PEFT)** with **LoRA** on:
  - Later **encoder** attention layers
  - Later **decoder** attention layers
- We **mix data** per batch:
  - **70–80%** general conversation (no address)
  - **20–30%** address-containing conversations
- We add an **address-weighted loss** term:
  - Extra weight for tokens that belong to:
    - Locality names
    - Address numbers/ordinals
    - Bangalore-specific address tokens

This nudges Whisper to “care more” about getting addresses right when they appear, without overfitting to addresses or hallucinating them elsewhere.

---

## 4. How to Run the Pipeline

Assuming you are inside `approach 1/`:

### 4.1. Install Dependencies

Create a virtual environment (recommended) and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4.2. Generate Bangalore Address Dataset

This step calls **Overpass API** + **India Post PIN API** (and optional BBMP scraping).

```bash
python generate_addresses.py
```

Outputs:

- `data/raw/osm.csv`
- `data/raw/pincode.csv`
- `data/raw/bbmp.csv` (if enabled)
- `data/final/bangalore_addresses.csv` (canonical + spoken variants)

If the final dataset has fewer than ~15k canonical addresses or 100k spoken variants, the script will **fail loudly**.

### 4.3. Generate Synthetic Conversations

Uses the final address dataset to create call-style conversations.

```bash
python generate_conversations.py
```

Outputs:

- `data/conversations/train_conversations.jsonl`
- `data/conversations/eval_conversations.jsonl`

Each line contains:

- Conversation ID
- Turn-by-turn transcript
- Embedded canonical address ID
- Spoken variant used

### 4.4. Generate Synthetic Audio

Uses **gTTS** (with Indian English accent) and post-processes audio to simulate phone calls.

```bash
python generate_audio.py
```

Outputs:

- `audio/train/*.wav`
- `audio/eval/*.wav`

Effects per call:

- Speed variation (0.9–1.1×)
- Random silences
- Band-pass filtering for telephone effect
- Optional background noise

Transcripts are stored alongside audio, and are **perfectly correct**, even if the audio is noisy.

### 4.5. Prepare Whisper Fine-Tuning Dataset

Converts conversations + audio into a Hugging Face dataset format for Whisper.

```bash
python prepare_whisper_dataset.py
```

Outputs:

- A small HF dataset directory under `data/whisper_dataset/` (train + eval splits).

### 4.6. Fine-Tune Whisper with LoRA

Runs LoRA-based fine-tuning on top of an open-source Whisper checkpoint (e.g. `openai/whisper-small`).

```bash
python finetune_whisper.py
```

Outputs:

- LoRA adapter + config under `models/whisper_bangalore_lora/`.

The script:

- Uses mixed batches (70–80% non-address, 20–30% address-containing).
- Applies an address-weighted loss term.
- Logs training curves and address-specific metrics.

---

## 5. Evaluation vs. Original Whisper

To confirm we **actually improved address transcription**, we compare:

- **Base Whisper model**
- **Fine-tuned Whisper+LoRA model**

on the same held-out, address-heavy eval split.

Run:

```bash
python evaluate_whisper.py
```

You will get:

- **Global WER/CER** for both models.
- **Address-span WER/CER**, i.e. error rates computed only on the address tokens.
- A few **qualitative examples** where:
  - Base Whisper mis-transcribes the address.
  - Fine-tuned Whisper gets it right (or at least closer).

This is the metric a **Bolna AI founder** actually cares about: _“Did the fine-tuned model stop butchering Bangalore addresses inside real-ish conversations?”_

---

## 6. Limitations

- **Synthetic data only**:
  - Real-world accents, co-channel speech, and background noise distributions may differ.
  - Performance on real calls must be validated with an actual contact-center dataset.
- **Bangalore-only**:
  - The logic is deliberately constrained to **Bangalore/Bengaluru**.
  - Extending pan-India requires **separate city-specific pipelines**.
- **No paid APIs**:
  - We use only free sources (OSM, India Post, gTTS).
  - TTS quality may be lower than commercial options.
- **Hardware requirements**:
  - Whisper fine-tuning requires a GPU with sufficient VRAM; LoRA helps, but CPU-only training is not realistic for large models.

---

## 7. Extending City-by-City Later

To extend this approach to another city:

1. **Clone the pipeline config**:
   - New Overpass bounding box.
   - New PIN code prefixes.
   - New city/district filters.
2. **Regenerate addresses**:
   - Build a new `city_addresses.csv` with canonical + spoken variants.
3. **Regenerate conversations + audio**:
   - Keep templates, just swap in the new address dataset.
4. **Fine-tune per city**:
   - Either:
     - Train a city-specific LoRA adapter.
     - Or train a multi-city adapter with city tokens in the prompt.

This keeps the approach **scalable but local**, instead of an under-specified “pan-India” model.

