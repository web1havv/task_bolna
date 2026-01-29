"""
Approach 5: Expand existing dataset to ~10k samples (APPEND mode supported).

Key properties:
- Keep existing samples (default: append remaining to reach target split sizes).
- ≥2k unique Bangalore addresses overall.
- 5–10 spoken variants per address (including phoneme-confusable forms).
- No fixed full-sentence templates; utterances are composed from varied fragments.
- Address position randomized: start / middle / end.
- Curriculum fields:
  - `text_phoneme`: phoneme-faithful text label (matches what we asked TTS to speak).
  - `text_canonical`: canonicalized label for later-stage training.
"""
import argparse
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rich import print


ROOT = Path(__file__).resolve().parent
DATA_FINAL = ROOT / "data" / "final"
DATA_SENTENCES = ROOT / "data" / "sentences"

ADDRESSES_CSV = DATA_FINAL / "bangalore_addresses.csv"

TOTAL_SAMPLES = 10_000
TRAIN_RATIO = 0.70   # 7000
TEST_RATIO = 0.15    # 1500
EVAL_RATIO = 0.15    # 1500

# For Approach 5 we bias heavily toward address-containing utterances.
# Keep a small portion of non-address utterances to avoid regressions.
ADDRESS_RATIO = 0.90   # 90% with address, 10% without


@dataclass
class SentenceSample:
    sample_id: str
    split: str  # "train", "test", or "eval"
    # Back-compat: `text` is what downstream scripts used previously.
    # We set it to phoneme-faithful label (matches the audio we synthesize).
    text: str
    # Curriculum targets
    text_phoneme: str
    text_canonical: str
    canonical_address: str
    spoken_address: str
    address_id: str
    locality: str
    pincode: str
    address_position: str  # "start" | "middle" | "end"
    variant_type: str      # "base" | "confusable"


def _norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def canonicalize_for_label(s: str) -> str:
    """
    Canonical-ish label string for curriculum stage 2.
    Keep it simple/rule-based to avoid pulling extra deps.
    """
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = _norm_spaces(s)
    # normalize a few common spoken forms
    s = s.replace("cross road", "cross")
    s = s.replace("main road", "main")
    s = s.replace("rd", "road")
    s = s.replace("st", "street")
    return s


def make_confusable_variant(spoken: str) -> str:
    """
    Create a phoneme-confusable variant. This is intentionally lightweight and stochastic.
    """
    s = spoken.lower()
    # multi-char first
    rules = [
        ("ph", "f"),
        ("th", random.choice(["t", "d"])),
        ("sh", "s"),
        ("ch", "tsh"),
        ("tion", "shun"),
        ("v", random.choice(["w", "v"])),
        ("z", "j"),
        ("x", "ks"),
    ]
    for a, b in rules:
        if a in s and random.random() < 0.35:
            s = s.replace(a, b, 1)
    # sometimes insert a tiny filler that doesn't change semantics but changes acoustics
    if random.random() < 0.25:
        s = random.choice(["near ", "around ", "in ", "at "]) + s
    return _norm_spaces(s)


def build_natural_utterance(addr: Optional[str], *, position: str) -> str:
    """
    Compose a single-turn natural utterance with varied fragments.
    No fixed full-sentence templates.
    """
    # Intent fragments
    intents = [
        "i need to update my delivery address",
        "please deliver my order",
        "can you send the package",
        "i want to set the drop location",
        "the delivery should come",
        "i'm sharing the address for delivery",
        "deliver it",
    ]
    # Softening / politeness / discourse markers
    openers = ["hey", "hi", "listen", "okay", "so", "actually", "umm", "sorry"]
    hedges = ["please", "if possible", "kindly", "just"]
    suffixes = [
        "that's my place",
        "that's where i stay",
        "you can mark it for delivery",
        "can you confirm once updated",
        "thanks",
        "that's it",
    ]
    # Extra context fragments
    context_bits = [
        "for today",
        "for this order",
        "for my subscription delivery",
        "for the next delivery",
        "for the parcel",
        "",
        "",
    ]
    disfluencies = ["uh", "um", "like", "you know", ""]

    parts: List[str] = []
    if random.random() < 0.6:
        parts.append(random.choice(openers))
    if random.random() < 0.35:
        parts.append(random.choice(disfluencies))

    parts.append(random.choice(intents))
    cb = random.choice(context_bits)
    if cb:
        parts.append(cb)
    if random.random() < 0.55:
        parts.append(random.choice(hedges))

    addr = _norm_spaces(addr or "")
    if addr:
        if position == "start":
            core = [addr, *parts]
        elif position == "end":
            core = [*parts, addr]
        else:
            # middle: split parts around address
            cut = max(1, min(len(parts) - 1, int(len(parts) * random.uniform(0.35, 0.65))))
            core = [*parts[:cut], addr, *parts[cut:]]
    else:
        core = parts

    if random.random() < 0.7:
        core.append(random.choice(suffixes))
    return _norm_spaces(" ".join([p for p in core if p]))


def build_non_address_utterance() -> str:
    intents = [
        "when will my order arrive",
        "can i cancel my subscription",
        "i want to change the delivery slot",
        "i need help with a refund",
        "my last order was delayed",
        "how do i track my package",
        "i was charged twice",
        "i need to talk to support",
    ]
    # Make it less template-y by adding optional fragments
    prefixes = ["hi", "hey", "sorry", "actually", "so", ""]
    suffixes = ["please", "thanks", "can you help", "right now", ""]
    return _norm_spaces(" ".join([random.choice(prefixes), random.choice(intents), random.choice(suffixes)]))


def ensure_dirs() -> None:
    DATA_SENTENCES.mkdir(parents=True, exist_ok=True)


def load_addresses(limit: Optional[int] = None) -> pd.DataFrame:
    if not ADDRESSES_CSV.exists():
        raise FileNotFoundError(
            f"Address file {ADDRESSES_CSV} not found. Run generate_addresses.py first."
        )
    df = pd.read_csv(ADDRESSES_CSV)
    if limit is not None:
        df = df.sample(n=limit, random_state=42)
    return df


def build_address_sentence(
    sample_id: str, addr_row: pd.Series, split: str, *, position: str, variant_type: str
) -> SentenceSample:
    canonical = str(addr_row["canonical_address"])
    spoken = str(addr_row["spoken_variant"])
    address_id = str(addr_row["address_id"])
    locality = str(addr_row.get("locality", ""))
    pincode = str(addr_row.get("pincode", ""))

    # Create utterance with address at requested position
    text_phoneme = build_natural_utterance(spoken, position=position)
    # Curriculum stage-2 label uses canonicalized address (not necessarily spoken)
    # Keep same utterance "shape" but swap in canonical address where possible.
    text_canonical = build_natural_utterance(canonicalize_for_label(canonical), position=position)

    return SentenceSample(
        sample_id=sample_id,
        split=split,
        text=text_phoneme,
        text_phoneme=text_phoneme,
        text_canonical=text_canonical,
        canonical_address=canonical,
        spoken_address=spoken,
        address_id=address_id,
        locality=locality,
        pincode=pincode,
        address_position=position,
        variant_type=variant_type,
    )


def build_non_address_sentence(sample_id: str, split: str) -> SentenceSample:
    text = build_non_address_utterance()
    return SentenceSample(
        sample_id=sample_id,
        split=split,
        text=text,
        text_phoneme=text,
        text_canonical=text,
        canonical_address="",
        spoken_address="",
        address_id="",
        locality="",
        pincode="",
        address_position=random.choice(["start", "middle", "end"]),
        variant_type="base",
    )

def load_existing_samples(split: str) -> List[Dict]:
    path = DATA_SENTENCES / f"{split}_sentences.jsonl"
    if not path.exists() or path.stat().st_size == 0:
        return []
    out: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _next_index(existing: List[Dict], prefix: str) -> int:
    """
    Find next numeric index for IDs like '{prefix}{i:06d}'.
    """
    best = -1
    for r in existing:
        sid = str(r.get("sample_id", ""))
        if sid.startswith(prefix):
            m = re.search(r"(\d+)$", sid)
            if m:
                best = max(best, int(m.group(1)))
    return best + 1


def _existing_unique_address_ids(existing_all: List[Dict]) -> set:
    return {str(r.get("address_id", "")).strip() for r in existing_all if str(r.get("address_id", "")).strip()}


def _choose_positions(n: int) -> List[str]:
    # Enforce near-balanced positions to avoid positional shortcut learning.
    base = (["start", "middle", "end"] * ((n // 3) + 1))[:n]
    random.shuffle(base)
    return base


def sample_sentences(
    addresses: pd.DataFrame,
    total: int = TOTAL_SAMPLES,
    train_ratio: float = TRAIN_RATIO,
    test_ratio: float = TEST_RATIO,
    eval_ratio: float = EVAL_RATIO,
    address_ratio: float = ADDRESS_RATIO,
    seed: int = 42,
    append: bool = True,
    min_unique_addresses: int = 2000,
) -> Tuple[List[SentenceSample], List[SentenceSample], List[SentenceSample]]:
    random.seed(seed)

    target_train = int(total * train_ratio)
    target_test = int(total * test_ratio)
    target_eval = total - target_train - target_test  # remainder

    existing_train = load_existing_samples("train") if append else []
    existing_test = load_existing_samples("test") if append else []
    existing_eval = load_existing_samples("eval") if append else []

    n_train_existing = len(existing_train)
    n_test_existing = len(existing_test)
    n_eval_existing = len(existing_eval)

    need_train = max(0, target_train - n_train_existing)
    need_test = max(0, target_test - n_test_existing)
    need_eval = max(0, target_eval - n_eval_existing)

    n_addr_train = int(need_train * address_ratio)
    n_addr_test = int(need_test * address_ratio)
    n_addr_eval = int(need_eval * address_ratio)

    existing_all = [*existing_train, *existing_test, *existing_eval]
    existing_unique = _existing_unique_address_ids(existing_all)

    addr_ids_all = addresses["address_id"].unique().tolist()
    random.shuffle(addr_ids_all)
    available_new = [a for a in addr_ids_all if a not in existing_unique]
    if not available_new:
        # Fall back to reusing if we already consumed everything (unlikely)
        available_new = addr_ids_all[:]

    # Ensure we hit min_unique_addresses overall by forcing enough NEW unique address_ids
    needed_unique_new = max(0, min_unique_addresses - len(existing_unique))
    # We'll generate multiple samples per address; estimate avg variants/address ~7
    est_avg = 7
    forced_unique_for_new_samples = min(len(available_new), max(needed_unique_new, int((n_addr_train + n_addr_test + n_addr_eval) / est_avg)))
    seed_unique = available_new[:forced_unique_for_new_samples]

    train_samples: List[SentenceSample] = []
    test_samples: List[SentenceSample] = []
    eval_samples: List[SentenceSample] = []

    # ID counters (continue numbering)
    train_addr_i = _next_index(existing_train, "train_addr_")
    train_no_i = _next_index(existing_train, "train_noaddr_")
    test_addr_i = _next_index(existing_test, "test_addr_")
    test_no_i = _next_index(existing_test, "test_noaddr_")
    eval_addr_i = _next_index(existing_eval, "eval_addr_")
    eval_no_i = _next_index(existing_eval, "eval_noaddr_")

    # Helper: sample a row for an address_id, optionally overriding spoken variant
    def pick_row(addr_id: str, *, spoken_override: Optional[str] = None, rs: int = 0) -> pd.Series:
        row = addresses[addresses["address_id"] == addr_id].sample(1, random_state=rs).iloc[0].copy()
        if spoken_override is not None:
            row["spoken_variant"] = spoken_override
        return row

    # Build a pool of (address_id, spoken_variant) candidates
    # Start by taking multiple spoken_variants per address_id for the forced unique seed set.
    addr_variant_pool: List[Tuple[str, str, str]] = []  # (addr_id, spoken_variant, variant_type)
    for j, aid in enumerate(seed_unique):
        # 5–10 variants/address from existing spoken_variant rows + confusables
        variants_rows = addresses[addresses["address_id"] == aid]
        variants = variants_rows["spoken_variant"].dropna().astype(str).tolist()
        random.shuffle(variants)
        k = random.randint(5, 10)
        picked = (variants * ((k // max(1, len(variants))) + 1))[:k]
        # sprinkle confusables
        for v in picked:
            if random.random() < 0.35:
                addr_variant_pool.append((aid, make_confusable_variant(v), "confusable"))
            addr_variant_pool.append((aid, v, "base"))

    # If still short, add more variants by sampling from remaining addresses (may reuse some address_ids).
    total_addr_needed = n_addr_train + n_addr_test + n_addr_eval
    while len(addr_variant_pool) < total_addr_needed:
        aid = random.choice(addr_ids_all)
        row = addresses[addresses["address_id"] == aid].sample(1, random_state=seed + len(addr_variant_pool)).iloc[0]
        v = str(row["spoken_variant"])
        addr_variant_pool.append((aid, v, "base"))
        if random.random() < 0.30:
            addr_variant_pool.append((aid, make_confusable_variant(v), "confusable"))
    # Aggressively shuffle to prevent shortcut learning by ordering
    random.shuffle(addr_variant_pool)

    def consume_addr_samples(n: int, split: str) -> List[SentenceSample]:
        nonlocal train_addr_i, test_addr_i, eval_addr_i
        positions = _choose_positions(n)
        out: List[SentenceSample] = []
        for idx in range(n):
            aid, spoken_v, vtype = addr_variant_pool.pop()
            if split == "train":
                sid = f"train_addr_{train_addr_i:06d}"
                train_addr_i += 1
                rs = seed + 10_000 + train_addr_i
            elif split == "test":
                sid = f"test_addr_{test_addr_i:06d}"
                test_addr_i += 1
                rs = seed + 20_000 + test_addr_i
            else:
                sid = f"eval_addr_{eval_addr_i:06d}"
                eval_addr_i += 1
                rs = seed + 30_000 + eval_addr_i
            row = pick_row(aid, spoken_override=spoken_v, rs=rs)
            out.append(build_address_sentence(sid, row, split=split, position=positions[idx], variant_type=vtype))
        return out

    def consume_noaddr_samples(n: int, split: str) -> List[SentenceSample]:
        nonlocal train_no_i, test_no_i, eval_no_i
        out: List[SentenceSample] = []
        for _ in range(n):
            if split == "train":
                sid = f"train_noaddr_{train_no_i:06d}"
                train_no_i += 1
            elif split == "test":
                sid = f"test_noaddr_{test_no_i:06d}"
                test_no_i += 1
            else:
                sid = f"eval_noaddr_{eval_no_i:06d}"
                eval_no_i += 1
            out.append(build_non_address_sentence(sid, split=split))
        return out

    train_samples.extend(consume_addr_samples(n_addr_train, "train"))
    train_samples.extend(consume_noaddr_samples(need_train - n_addr_train, "train"))
    test_samples.extend(consume_addr_samples(n_addr_test, "test"))
    test_samples.extend(consume_noaddr_samples(need_test - n_addr_test, "test"))
    eval_samples.extend(consume_addr_samples(n_addr_eval, "eval"))
    eval_samples.extend(consume_noaddr_samples(need_eval - n_addr_eval, "eval"))

    random.shuffle(train_samples)
    random.shuffle(test_samples)
    random.shuffle(eval_samples)

    return train_samples, test_samples, eval_samples


def save_jsonl(path: Path, samples: List[SentenceSample], *, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for s in samples:
            obj = asdict(s)
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate/append single-turn utterances (Approach 5)."
    )
    parser.add_argument(
        "--total",
        type=int,
        default=TOTAL_SAMPLES,
        help=f"Target total number of samples across splits (default {TOTAL_SAMPLES}).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=TRAIN_RATIO,
        help="Fraction for train split.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=TEST_RATIO,
        help="Fraction for test split.",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=EVAL_RATIO,
        help="Fraction for evaluation split.",
    )
    parser.add_argument(
        "--address-ratio",
        type=float,
        default=ADDRESS_RATIO,
        help="Fraction of address-containing samples among NEWLY GENERATED samples per split.",
    )
    parser.add_argument(
        "--address-limit",
        type=int,
        default=None,
        help="Optional limit on unique addresses to sample from.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append missing samples to existing JSONL files (default behavior).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite sentence JSONLs from scratch (dangerous; will discard existing samples).",
    )
    parser.add_argument(
        "--min-unique-addresses",
        type=int,
        default=2000,
        help="Ensure at least this many unique address_ids across (existing + newly generated) samples.",
    )
    args = parser.parse_args(argv)
    if args.overwrite:
        args.append = False
    elif not args.append:
        # default behavior
        args.append = True

    ensure_dirs()
    addresses = load_addresses(limit=args.address_limit)

    print(
        f"[bold]Loaded {len(addresses)} address spoken variants "
        f"from {ADDRESSES_CSV}.[/bold]"
    )

    train, test, eval_ = sample_sentences(
        addresses=addresses,
        total=args.total,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        eval_ratio=args.eval_ratio,
        address_ratio=args.address_ratio,
        append=args.append,
        min_unique_addresses=args.min_unique_addresses,
    )

    # Append (default) or overwrite (if requested).
    save_jsonl(DATA_SENTENCES / "train_sentences.jsonl", train, append=args.append)
    save_jsonl(DATA_SENTENCES / "test_sentences.jsonl", test, append=args.append)
    save_jsonl(DATA_SENTENCES / "eval_sentences.jsonl", eval_, append=args.append)

    print(
        f"[green]Appended {len(train)} train, {len(test)} test, {len(eval_)} eval "
        f"new sentences to {DATA_SENTENCES}.[/green]"
    )
    print(f"[bold]Newly generated: {len(train) + len(test) + len(eval_)} samples.[/bold]")


if __name__ == "__main__":
    main()
