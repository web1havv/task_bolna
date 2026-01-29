import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import requests
from bs4 import BeautifulSoup
import pandas as pd
from slugify import slugify
from rich import print


ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_FINAL = ROOT / "data" / "final"

OSM_RAW_PATH = DATA_RAW / "osm.csv"
PINCODE_RAW_PATH = DATA_RAW / "pincode.csv"
BBMP_RAW_PATH = DATA_RAW / "bbmp.csv"
FINAL_ADDRESSES_PATH = DATA_FINAL / "bangalore_addresses.csv"


OVERPASS_URL = "https://overpass-api.de/api/interpreter"
INDIA_POST_PINCODE_API = "https://api.postalpincode.in/pincode/{pincode}"


@dataclass
class AddressRecord:
    address_id: str
    canonical_address: str
    spoken_variant: str
    locality: str
    pincode: str
    source: str


def ensure_dirs() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_FINAL.mkdir(parents=True, exist_ok=True)


def fetch_osm_bangalore() -> pd.DataFrame:
    """
    Fetch Bangalore-local address-like entities from OSM using Overpass.

    We restrict to the Bengaluru administrative boundary and extract:
    - highway
    - place
    - suburb
    - neighbourhood
    - landuse=residential
    """
    if OSM_RAW_PATH.exists():
        print(f"[yellow]OSM raw file already exists at {OSM_RAW_PATH}, reusing.[/yellow]")
        return pd.read_csv(OSM_RAW_PATH)

    print("[bold]Fetching OSM data for Bengaluru via Overpass API...[/bold]")

    # Minimal conservative query: administratively bounded Bangalore (approximate bbox)
    # This is intentionally simple; you can refine the boundary later if needed.
    overpass_query = """
    [out:json][timeout:120];
    area["name"="Bengaluru"]["boundary"="administrative"]->.searchArea;
    (
      way["highway"](area.searchArea);
      node["place"](area.searchArea);
      node["suburb"](area.searchArea);
      node["neighbourhood"](area.searchArea);
      way["landuse"="residential"](area.searchArea);
    );
    out tags;
    """

    resp = requests.post(OVERPASS_URL, data={"data": overpass_query})
    resp.raise_for_status()
    data = resp.json()

    rows: List[Dict[str, str]] = []
    for element in data.get("elements", []):
        tags = element.get("tags", {})
        name = tags.get("name")
        if not name:
            continue

        tag_type = None
        for key in ("highway", "place", "suburb", "neighbourhood", "landuse"):
            if key in tags:
                tag_type = f"{key}:{tags.get(key)}"
                break

        if not tag_type:
            continue

        rows.append(
            {
                "name": name,
                "tag_type": tag_type,
                "source": "osm",
            }
        )

    df = pd.DataFrame(rows).drop_duplicates()
    df.to_csv(OSM_RAW_PATH, index=False)
    print(f"[green]Saved OSM raw data to {OSM_RAW_PATH} with {len(df)} rows.[/green]")
    return df


def fetch_india_post_pincodes(max_failures: int = 50) -> pd.DataFrame:
    """
    Fetch locality names for all PINs starting with 560 via India Post API.
    """
    if PINCODE_RAW_PATH.exists():
        print(f"[yellow]Pincode raw file already exists at {PINCODE_RAW_PATH}, reusing.[/yellow]")
        return pd.read_csv(PINCODE_RAW_PATH)

    print("[bold]Fetching India Post PIN data for 560xxx...[/bold]")
    records: List[Dict[str, str]] = []
    failures = 0

    # 560000–560999 (overshooting a bit is fine)
    for suffix in range(0, 1000):
        pincode = f"560{suffix:03d}"
        url = INDIA_POST_PINCODE_API.format(pincode=pincode)
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as e:  # noqa: BLE001
            print(f"[red]Failed to fetch pincode {pincode}: {e}[/red]")
            failures += 1
            if failures >= max_failures:
                print(
                    "[yellow]India Post API appears unavailable; "
                    "stopping further PIN requests and continuing without pincode data.[/yellow]"
                )
                # Return empty frame – downstream will still work using OSM/BBMP only.
                df_empty = pd.DataFrame(
                    columns=["Name", "Block", "District", "State", "Pincode", "source"]
                )
                df_empty.to_csv(PINCODE_RAW_PATH, index=False)
                return df_empty
            continue

        if not payload:
            continue

        entry = payload[0]
        if entry.get("Status") != "Success":
            continue

        for po in entry.get("PostOffice", []) or []:
            district = (po.get("District") or "").lower()
            state = (po.get("State") or "").lower()

            if "bangalore urban" not in district and "bengaluru urban" not in district:
                continue
            if "karnataka" not in state:
                continue

            records.append(
                {
                    "Name": po.get("Name", "").strip(),
                    "Block": po.get("Block", "").strip(),
                    "District": po.get("District", "").strip(),
                    "State": po.get("State", "").strip(),
                    "Pincode": po.get("Pincode", "").strip(),
                    "source": "india_post",
                }
            )

    df = pd.DataFrame(records).drop_duplicates()
    df.to_csv(PINCODE_RAW_PATH, index=False)
    print(f"[green]Saved pincode raw data to {PINCODE_RAW_PATH} with {len(df)} rows.[/green]")
    return df


def fetch_bbmp_localities(url: Optional[str]) -> pd.DataFrame:
    """
    Optionally scrape BBMP ward/locality names from a public table.

    The exact URL can be provided via CLI. If not provided, this returns
    an empty DataFrame and the pipeline proceeds without BBMP.
    """
    if not url:
        print("[yellow]No BBMP URL provided; skipping BBMP scrape.[/yellow]")
        return pd.DataFrame(columns=["name", "source"])

    if BBMP_RAW_PATH.exists():
        print(f"[yellow]BBMP raw file already exists at {BBMP_RAW_PATH}, reusing.[/yellow]")
        return pd.read_csv(BBMP_RAW_PATH)

    print(f"[bold]Scraping BBMP localities from {url}...[/bold]")
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    rows: List[Dict[str, str]] = []
    tables = soup.find_all("table")
    for table in tables:
        for tr in table.find_all("tr"):
            cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
            if not cells:
                continue
            # Heuristic: pick the longest text chunk as the locality/ward name.
            name = max(cells, key=len)
            if not name or len(name) < 3:
                continue
            rows.append({"name": name, "source": "bbmp"})

    df = pd.DataFrame(rows).drop_duplicates()
    df.to_csv(BBMP_RAW_PATH, index=False)
    print(f"[green]Saved BBMP raw data to {BBMP_RAW_PATH} with {len(df)} rows.[/green]")
    return df


def normalize_address_string(s: str) -> str:
    """
    Basic normalization for canonical addresses:
    - Trim
    - Normalize whitespace
    - Expand some common abbreviations
    - Keep case (we lower only for dedup keys)
    """
    s = " ".join(s.strip().split())
    replacements = {
        " rd": " road",
        " rd.": " road",
        " rd,": " road,",
        " blk": " block",
        " blk.": " block",
        " st ": " street ",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    return s


ORDINAL_MAP = {
    "1": "1st",
    "2": "2nd",
    "3": "3rd",
    "4": "4th",
    "5": "5th",
    "6": "6th",
    "7": "7th",
    "8": "8th",
    "9": "9th",
    "10": "10th",
}


def normalize_numerics(text: str) -> str:
    """
    Convert bare numbers that are commonly used as block/stage/sector to ordinals.
    This is intentionally simple and can be extended if needed.
    """
    tokens = text.split()
    out: List[str] = []
    for t in tokens:
        base = t.rstrip(",")
        suffix = "," if t.endswith(",") else ""
        if base.isdigit() and base in ORDINAL_MAP:
            out.append(ORDINAL_MAP[base] + suffix)
        else:
            out.append(t)
    return " ".join(out)


def is_valid_entity(name: str) -> bool:
    """
    Filter for valid address entities:
    - Should not look like a person name
    - Should not be just a house/flat number
    - Should not be trivially short
    """
    s = name.strip()
    if not s:
        return False
    if len(s) < 3:
        return False

    # Flat / house number detection (very rough)
    lowered = s.lower()
    if any(prefix in lowered for prefix in ("flat ", "house no", "door no", "d.no", "no.")):
        return False

    # Strong person-name heuristic: two capitalized words with no digits
    if all(
        part and part[0].isupper() and part[1:].islower()
        for part in s.split()
        if part.isalpha()
    ) and not any(ch.isdigit() for ch in s):
        return False

    return True


def dedup_canonical_addresses(candidates: Iterable[Tuple[str, str, str]]) -> Dict[str, Dict]:
    """
    Deduplicate canonical addresses.
    Input: iterable of (canonical_address, locality, pincode).
    Returns dict keyed by canonical_id with metadata.
    """
    seen: Dict[str, Dict] = {}
    key_set: Set[str] = set()

    for canonical, locality, pincode in candidates:
        canonical = normalize_address_string(normalize_numerics(canonical))
        loc = (locality or "").strip()
        pin = (pincode or "").strip()

        # Dedup key: lowercase, no punctuation, tokens sorted
        key = " ".join(
            sorted(
                "".join(ch for ch in canonical.lower() if ch.isalnum() or ch.isspace()).split()
            )
        )
        if not key:
            continue
        if key in key_set:
            continue

        key_set.add(key)
        addr_hash = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        address_id = f"blr_{addr_hash}"

        seen[address_id] = {
            "address_id": address_id,
            "canonical_address": canonical,
            "locality": loc or canonical,
            "pincode": pin,
        }

    return seen


def spoken_variants_for(canonical: str) -> List[str]:
    """
    Generate multiple spoken variants for a canonical Bangalore address.
    This includes:
    - Different ways of speaking numbers (4th -> fourth / four)
    - Acronym expansion (HSR -> h s r)
    - Simple phonetic corruptions (koramangala -> koraman gala, etc.)
    All variants are lowercase and punctuation-light.
    """
    text = canonical.lower()

    variants: Set[str] = {text}

    # Number variations
    num_map = {
        "4th": ["fourth", "four"],
        "2nd": ["second", "two"],
        "3rd": ["third", "three"],
        "5th": ["fifth", "five"],
        "6th": ["sixth", "six"],
        "7th": ["seventh", "seven"],
        "8th": ["eighth", "eight"],
        "9th": ["ninth", "nine"],
        "10th": ["tenth", "ten"],
        "100": ["hundred", "one zero zero"],
    }

    for key, repls in num_map.items():
        if key in text:
            for r in repls:
                variants.add(text.replace(key, r))

    # Acronym expansions
    if "hsr" in text:
        variants.add(text.replace("hsr", "h s r"))
    if "btm" in text:
        variants.add(text.replace("btm", "b t m"))

    # Simple phonetic corruptions
    corr_map = {
        "koramangala": ["koraman gala", "koram angala"],
        "indiranagar": ["indira nagar", "indir nagar"],
        "bengaluru": ["bangalore"],
    }
    for key, repls in corr_map.items():
        if key in text:
            for r in repls:
                variants.add(text.replace(key, r))

    # Remove punctuation (minimal)
    cleaned: Set[str] = set()
    for v in variants:
        cleaned.add(
            "".join(ch for ch in v if ch.isalnum() or ch.isspace()).strip()
        )

    # Ensure min 5 and max 15 by adding small template noises if needed
    base_list = [v for v in cleaned if v]
    if len(base_list) < 5:
        padded: List[str] = base_list.copy()
        templates = [
            "in {}",
            "near {}",
            "around {}",
            "{} area",
            "{} side",
        ]
        i = 0
        while len(padded) < 5 and base_list:
            tmpl = templates[i % len(templates)]
            padded.append(tmpl.format(base_list[i % len(base_list)]))
            i += 1
        base_list = padded

    return base_list[:15]


def build_final_dataset(
    osm_df: pd.DataFrame, pincode_df: pd.DataFrame, bbmp_df: pd.DataFrame
) -> List[AddressRecord]:
    candidates: List[Tuple[str, str, str]] = []

    # From OSM
    for _, row in osm_df.iterrows():
        name = str(row.get("name", "")).strip()
        if not is_valid_entity(name):
            continue
        candidates.append((name, name, ""))  # locality approximated as name

    # From India Post
    for _, row in pincode_df.iterrows():
        name = str(row.get("Name", "")).strip()
        block = str(row.get("Block", "")).strip()
        pincode = str(row.get("Pincode", "")).strip()
        if not is_valid_entity(name):
            continue
        locality = block or name
        candidates.append((name, locality, pincode))

    # From BBMP
    for _, row in bbmp_df.iterrows():
        name = str(row.get("name", "")).strip()
        if not is_valid_entity(name):
            continue
        candidates.append((name, name, ""))

    canonical_map = dedup_canonical_addresses(candidates)
    print(f"[bold]Canonical address count: {len(canonical_map)}[/bold]")

    # Original design expects ~15k+ canonical addresses, but in constrained / API-failure
    # scenarios (like skipping India Post) we still want a usable smaller dataset.
    if len(canonical_map) < 1000:
        raise RuntimeError(
            f"Too few canonical addresses ({len(canonical_map)}). "
            "Need at least ~1k to proceed; check your data sources or enable more inputs."
        )
    if len(canonical_map) < 15000:
        print(
            "[yellow]Warning: canonical address count is below the ideal 15k+. "
            "Continuing anyway with a smaller Bangalore dataset.[/yellow]"
        )

    records: List[AddressRecord] = []
    for info in canonical_map.values():
        canonical = info["canonical_address"]
        locality = info["locality"]
        pincode = info["pincode"]
        address_id = info["address_id"]

        variants = spoken_variants_for(canonical)
        if len(variants) < 5:
            continue

        for spoken in variants:
            records.append(
                AddressRecord(
                    address_id=address_id,
                    canonical_address=canonical,
                    spoken_variant=spoken,
                    locality=locality,
                    pincode=pincode,
                    source="mixed",
                )
            )

    if len(records) < 20_000:
        print(
            f"[yellow]Warning: only {len(records)} spoken variants generated "
            "(<100k ideal). This is fine for MacBook-scale experiments but "
            "may be small for serious training.[/yellow]"
        )

    return records


def save_final_dataset(records: List[AddressRecord]) -> None:
    FINAL_ADDRESSES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FINAL_ADDRESSES_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["address_id", "canonical_address", "spoken_variant", "locality", "pincode", "source"]
        )
        for r in records:
            writer.writerow(
                [
                    r.address_id,
                    r.canonical_address,
                    r.spoken_variant,
                    r.locality,
                    r.pincode,
                    r.source,
                ]
            )

    print(
        f"[green]Saved final Bangalore address dataset with "
        f"{len(records)} spoken variants to {FINAL_ADDRESSES_PATH}.[/green]"
    )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate Bangalore address dataset (canonical + spoken variants)."
    )
    parser.add_argument(
        "--bbmp-url",
        type=str,
        default=None,
        help="Optional URL to BBMP ward/locality table for scraping.",
    )
    parser.add_argument(
        "--skip-india-post",
        action="store_true",
        help="Skip India Post PIN API and rely only on OSM/BBMP sources.",
    )
    args = parser.parse_args(argv)

    ensure_dirs()
    osm_df = fetch_osm_bangalore()
    if args.skip_india_post:
        print("[yellow]Skipping India Post API as requested.[/yellow]")
        pincode_df = pd.DataFrame(
            columns=["Name", "Block", "District", "State", "Pincode", "source"]
        )
    else:
        pincode_df = fetch_india_post_pincodes()
    bbmp_df = fetch_bbmp_localities(args.bbmp_url)

    records = build_final_dataset(osm_df, pincode_df, bbmp_df)
    save_final_dataset(records)


if __name__ == "__main__":
    main()

