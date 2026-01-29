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
DATA_CONV = ROOT / "data" / "conversations"

ADDRESSES_CSV = DATA_FINAL / "bangalore_addresses.csv"


@dataclass
class ConversationSample:
    conversation_id: str
    split: str  # "train" or "eval"
    turns: List[Dict[str, str]]  # [{speaker, text}]
    canonical_address: str
    spoken_address: str
    address_id: str
    locality: str
    pincode: str


AGENT_NAMES = ["Agent", "Executive", "Support"]
USER_NAMES = ["User", "Customer", "Caller"]


ADDRESS_TEMPLATES = [
    # Address appears in the middle of the call (exactly one address mention)
    [
        ("agent", "hi thanks for calling bolna how can i help you today"),
        ("user", "yeah uh i just needed to update my delivery address"),
        ("agent", "sure i can do that can you confirm the full address for me"),
        ("user", "yeah uh so my address is {ADDR} near the signal"),
        ("agent", "okay got it one second while i update that"),
    ],
    [
        ("agent", "hello this is bolna calling about your recent order is this a good time"),
        ("user", "yeah thats fine whats up"),
        ("agent", "we just need to confirm your address before dispatch"),
        ("user", "yeah uh so my address is {ADDR} opposite the park"),
        ("agent", "alright noted thank you"),
    ],
    [
        ("agent", "hey just calling to reconfirm your address for todays pickup"),
        ("user", "okay sure"),
        ("agent", "could you tell me your address once"),
        ("user", "yeah so my address is {ADDR} backside of the petrol bunk"),
        ("agent", "okay perfect thanks"),
    ],
    [
        ("agent", "good evening am i speaking to the owner of the number"),
        ("user", "yeah speaking"),
        ("agent", "we are scheduling a technician visit can you share your address"),
        ("user", "yeah uh so my address is {ADDR} its close to the main road"),
        ("agent", "alright ill pass that to the technician"),
    ],
    [
        ("agent", "hi this is bolna support how can i assist you"),
        ("user", "i need to change my pickup location"),
        ("agent", "no problem can you give me the new address"),
        ("user", "yeah so my address is {ADDR} near the metro station"),
        ("agent", "got it thanks for confirming"),
    ],
]


NON_ADDRESS_TEMPLATES = [
    [
        ("agent", "hi this is bolna support how can i help"),
        ("user", "i think i was double charged for my order"),
        ("agent", "okay let me quickly check that for you"),
        ("user", "yeah i see two sms messages for the same amount"),
        ("agent", "dont worry if its a duplicate it will auto reverse in twenty four hours"),
    ],
    [
        ("agent", "hello thanks for choosing bolna"),
        ("user", "i just wanted to know when my order will arrive"),
        ("agent", "it shows out for delivery you should get it before eight pm"),
        ("user", "okay fine ill wait"),
    ],
    [
        ("agent", "good morning calling from bolna about your feedback"),
        ("user", "oh yeah the call quality was a bit low last time"),
        ("agent", "thanks for letting us know we are working on it"),
    ],
]


def ensure_dirs() -> None:
    DATA_CONV.mkdir(parents=True, exist_ok=True)


def load_addresses(limit: Optional[int] = None) -> pd.DataFrame:
    if not ADDRESSES_CSV.exists():
        raise FileNotFoundError(
            f"Address file {ADDRESSES_CSV} not found. Run generate_addresses.py first."
        )
    df = pd.read_csv(ADDRESSES_CSV)
    if limit is not None:
        df = df.sample(n=limit, random_state=42)
    return df


def build_address_conversation(
    conv_id: str, addr_row: pd.Series, split: str
) -> ConversationSample:
    canonical = str(addr_row["canonical_address"])
    spoken = str(addr_row["spoken_variant"])
    address_id = str(addr_row["address_id"])
    locality = str(addr_row.get("locality", ""))
    pincode = str(addr_row.get("pincode", ""))

    tmpl = random.choice(ADDRESS_TEMPLATES)
    turns: List[Dict[str, str]] = []
    for role, text in tmpl:
        speaker = random.choice(AGENT_NAMES) if role == "agent" else random.choice(USER_NAMES)
        # Insert spoken variant in place of {ADDR}
        text_filled = text.replace("{ADDR}", spoken)
        turns.append({"speaker": speaker, "text": text_filled})

    return ConversationSample(
        conversation_id=conv_id,
        split=split,
        turns=turns,
        canonical_address=canonical,
        spoken_address=spoken,
        address_id=address_id,
        locality=locality,
        pincode=pincode,
    )


def build_non_address_conversation(conv_id: str, split: str) -> ConversationSample:
    tmpl = random.choice(NON_ADDRESS_TEMPLATES)
    turns: List[Dict[str, str]] = []
    for role, text in tmpl:
        speaker = random.choice(AGENT_NAMES) if role == "agent" else random.choice(USER_NAMES)
        turns.append({"speaker": speaker, "text": text})

    # Dummy values for address fields
    return ConversationSample(
        conversation_id=conv_id,
        split=split,
        turns=turns,
        canonical_address="",
        spoken_address="",
        address_id="",
        locality="",
        pincode="",
    )


def sample_conversations(
    addresses: pd.DataFrame,
    num_train_with_address: int,
    num_eval_with_address: int,
    ratio_non_address: float,
    seed: int = 42,
) -> Tuple[List[ConversationSample], List[ConversationSample]]:
    random.seed(seed)

    # Sample distinct address rows for train/eval (by address_id)
    unique_addr_ids = addresses["address_id"].unique().tolist()
    random.shuffle(unique_addr_ids)

    needed = num_train_with_address + num_eval_with_address
    if len(unique_addr_ids) < needed:
        raise RuntimeError(
            f"Not enough unique address IDs ({len(unique_addr_ids)}) "
            f"for requested {needed} address-containing conversations."
        )

    train_ids = set(unique_addr_ids[:num_train_with_address])
    eval_ids = set(unique_addr_ids[num_train_with_address:needed])

    train_samples: List[ConversationSample] = []
    eval_samples: List[ConversationSample] = []

    # TRAIN: with address
    for i, addr_id in enumerate(train_ids):
        row = addresses[addresses["address_id"] == addr_id].sample(1, random_state=seed + i).iloc[0]
        conv_id = f"train_addr_{i:06d}"
        train_samples.append(build_address_conversation(conv_id, row, split="train"))

    # EVAL: with address
    for i, addr_id in enumerate(eval_ids):
        row = addresses[addresses["address_id"] == addr_id].sample(1, random_state=seed + 1000 + i).iloc[0]
        conv_id = f"eval_addr_{i:06d}"
        eval_samples.append(build_address_conversation(conv_id, row, split="eval"))

    # Add non-address conversations according to ratio
    num_train_non = int(len(train_samples) * ratio_non_address)
    num_eval_non = int(len(eval_samples) * ratio_non_address)

    for i in range(num_train_non):
        conv_id = f"train_noaddr_{i:06d}"
        train_samples.append(build_non_address_conversation(conv_id, split="train"))

    for i in range(num_eval_non):
        conv_id = f"eval_noaddr_{i:06d}"
        eval_samples.append(build_non_address_conversation(conv_id, split="eval"))

    random.shuffle(train_samples)
    random.shuffle(eval_samples)

    return train_samples, eval_samples


def save_jsonl(path: Path, samples: List[ConversationSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            obj = asdict(s)
            # Flatten turns to a single transcript string for convenience
            obj["transcript"] = " ".join(t["text"] for t in s.turns)
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic telephone-style conversations with Bangalore addresses."
    )
    parser.add_argument(
        "--train-address-convs",
        type=int,
        default=800,  # 40% of 2000 train conversations (for 2500 total)
        help="Number of TRAIN conversations containing an address (40% of total).",
    )
    parser.add_argument(
        "--eval-address-convs",
        type=int,
        default=200,  # 40% of 500 eval conversations (for 2500 total)
        help="Number of EVAL conversations containing an address (40% of total).",
    )
    parser.add_argument(
        "--non-address-ratio",
        type=float,
        default=1.5,  # 60% / 40% = 1.5 ratio
        help="Ratio of non-address to address conversations per split (60% generic / 40% address-heavy).",
    )
    parser.add_argument(
        "--address-limit",
        type=int,
        default=None,
        help="Optional limit on number of unique addresses to sample from.",
    )
    args = parser.parse_args(argv)

    ensure_dirs()
    addresses = load_addresses(limit=args.address_limit)

    print(
        f"[bold]Loaded {len(addresses)} address spoken variants "
        f"from {ADDRESSES_CSV}.[/bold]"
    )

    train_samples, eval_samples = sample_conversations(
        addresses=addresses,
        num_train_with_address=args.train_address_convs,
        num_eval_with_address=args.eval_address_convs,
        ratio_non_address=args.non_address_ratio,
    )

    train_path = DATA_CONV / "train_conversations.jsonl"
    eval_path = DATA_CONV / "eval_conversations.jsonl"
    save_jsonl(train_path, train_samples)
    save_jsonl(eval_path, eval_samples)

    print(
        f"[green]Saved {len(train_samples)} train and {len(eval_samples)} eval "
        f"conversations to {DATA_CONV}.[/green]"
    )


if __name__ == "__main__":
    main()

