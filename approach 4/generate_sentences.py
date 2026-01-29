"""
Approach 4: Generate ~10k samples of longer, dialogue-style single sentences.
Split: train / test / evaluation. No user/agent turns; one utterance per sample.
"""
import argparse
import json
import random
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

ADDRESS_RATIO = 0.40   # 40% with address, 60% without


@dataclass
class SentenceSample:
    sample_id: str
    split: str  # "train", "test", or "eval"
    text: str
    canonical_address: str
    spoken_address: str
    address_id: str
    locality: str
    pincode: str


# Single-sentence templates with {ADDR} placeholder (no user/agent turns)
ADDRESS_TEMPLATES = [
    "can you please deliver my order to {ADDR} i live there",
    "my address is {ADDR} please send it to this place",
    "i need to update my delivery address to {ADDR}",
    "please deliver to {ADDR} thats where i live",
    "could you deliver my package to {ADDR} its my address",
    "i live at {ADDR} please deliver there",
    "send it to {ADDR} thats my address",
    "my new address is {ADDR} please update your records",
    "please deliver my address to this place i live at {ADDR}",
    "i want to change my pickup location to {ADDR}",
]

# Non-address sentences (generic, no address)
NON_ADDRESS_TEMPLATES = [
    "when will my order arrive",
    "i was double charged for my last order",
    "can i cancel my subscription",
    "what are your delivery hours",
    "i just wanted to check on my order status",
    "how do i track my package",
    "i need to speak to customer support",
    "can you help me with a refund",
    "what is your return policy",
    "i have a question about my account",
]


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
    sample_id: str, addr_row: pd.Series, split: str
) -> SentenceSample:
    canonical = str(addr_row["canonical_address"])
    spoken = str(addr_row["spoken_variant"])
    address_id = str(addr_row["address_id"])
    locality = str(addr_row.get("locality", ""))
    pincode = str(addr_row.get("pincode", ""))

    template = random.choice(ADDRESS_TEMPLATES)
    text = template.replace("{ADDR}", spoken)

    return SentenceSample(
        sample_id=sample_id,
        split=split,
        text=text,
        canonical_address=canonical,
        spoken_address=spoken,
        address_id=address_id,
        locality=locality,
        pincode=pincode,
    )


def build_non_address_sentence(sample_id: str, split: str) -> SentenceSample:
    text = random.choice(NON_ADDRESS_TEMPLATES)
    return SentenceSample(
        sample_id=sample_id,
        split=split,
        text=text,
        canonical_address="",
        spoken_address="",
        address_id="",
        locality="",
        pincode="",
    )


def sample_sentences(
    addresses: pd.DataFrame,
    total: int = TOTAL_SAMPLES,
    train_ratio: float = TRAIN_RATIO,
    test_ratio: float = TEST_RATIO,
    eval_ratio: float = EVAL_RATIO,
    address_ratio: float = ADDRESS_RATIO,
    seed: int = 42,
) -> Tuple[List[SentenceSample], List[SentenceSample], List[SentenceSample]]:
    random.seed(seed)

    n_train = int(total * train_ratio)
    n_test = int(total * test_ratio)
    n_eval = total - n_train - n_test  # remainder

    n_addr_train = int(n_train * address_ratio)
    n_addr_test = int(n_test * address_ratio)
    n_addr_eval = int(n_eval * address_ratio)

    unique_addr_ids = addresses["address_id"].unique().tolist()
    random.shuffle(unique_addr_ids)

    needed = n_addr_train + n_addr_test + n_addr_eval
    if len(unique_addr_ids) < needed:
        raise RuntimeError(
            f"Not enough unique address IDs ({len(unique_addr_ids)}) "
            f"for requested {needed} address-containing samples."
        )

    train_ids = set(unique_addr_ids[:n_addr_train])
    test_ids = set(unique_addr_ids[n_addr_train : n_addr_train + n_addr_test])
    eval_ids = set(
        unique_addr_ids[
            n_addr_train + n_addr_test : n_addr_train + n_addr_test + n_addr_eval
        ]
    )

    train_samples: List[SentenceSample] = []
    test_samples: List[SentenceSample] = []
    eval_samples: List[SentenceSample] = []

    # TRAIN: address sentences
    for i, addr_id in enumerate(train_ids):
        row = addresses[addresses["address_id"] == addr_id].sample(1, random_state=seed + i).iloc[0]
        sid = f"train_addr_{i:06d}"
        train_samples.append(build_address_sentence(sid, row, split="train"))

    # TRAIN: non-address sentences
    for i in range(n_train - n_addr_train):
        sid = f"train_noaddr_{i:06d}"
        train_samples.append(build_non_address_sentence(sid, split="train"))

    # TEST: address sentences
    for i, addr_id in enumerate(test_ids):
        row = addresses[addresses["address_id"] == addr_id].sample(
            1, random_state=seed + 2000 + i
        ).iloc[0]
        sid = f"test_addr_{i:06d}"
        test_samples.append(build_address_sentence(sid, row, split="test"))

    # TEST: non-address sentences
    for i in range(n_test - n_addr_test):
        sid = f"test_noaddr_{i:06d}"
        test_samples.append(build_non_address_sentence(sid, split="test"))

    # EVAL: address sentences
    for i, addr_id in enumerate(eval_ids):
        row = addresses[addresses["address_id"] == addr_id].sample(
            1, random_state=seed + 3000 + i
        ).iloc[0]
        sid = f"eval_addr_{i:06d}"
        eval_samples.append(build_address_sentence(sid, row, split="eval"))

    # EVAL: non-address sentences
    for i in range(n_eval - n_addr_eval):
        sid = f"eval_noaddr_{i:06d}"
        eval_samples.append(build_non_address_sentence(sid, split="eval"))

    random.shuffle(train_samples)
    random.shuffle(test_samples)
    random.shuffle(eval_samples)

    return train_samples, test_samples, eval_samples


def save_jsonl(path: Path, samples: List[SentenceSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            obj = asdict(s)
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate 2500 single-sentence samples (train/test/eval) for Approach 4."
    )
    parser.add_argument(
        "--total",
        type=int,
        default=TOTAL_SAMPLES,
        help=f"Total number of samples (default {TOTAL_SAMPLES}).",
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
        help="Fraction of address-containing samples per split.",
    )
    parser.add_argument(
        "--address-limit",
        type=int,
        default=None,
        help="Optional limit on unique addresses to sample from.",
    )
    args = parser.parse_args(argv)

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
    )

    save_jsonl(DATA_SENTENCES / "train_sentences.jsonl", train)
    save_jsonl(DATA_SENTENCES / "test_sentences.jsonl", test)
    save_jsonl(DATA_SENTENCES / "eval_sentences.jsonl", eval_)

    print(
        f"[green]Saved {len(train)} train, {len(test)} test, {len(eval_)} eval "
        f"sentences to {DATA_SENTENCES}.[/green]"
    )
    print(f"[bold]Total: {len(train) + len(test) + len(eval_)} samples.[/bold]")


if __name__ == "__main__":
    main()
