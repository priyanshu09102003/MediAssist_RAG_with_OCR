"""
download_kb.py
--------------
Auto-downloads medical knowledge base documents into data/medical_kb/.

Sources:
    - WHO Fact Sheets (scraped as clean .txt)
    - MedlinePlus Health Topics XML (bulk download)

Run from project root:
    python download_kb.py
"""

import re
import time
import urllib.request
from pathlib import Path

import requests
from bs4 import BeautifulSoup

import config

KB_DIR = config.MEDICAL_KB_PATH
KB_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ── WHO Fact Sheet URLs ───────────────────────────────────────────────────────

WHO_FACT_SHEETS = {
    "who_diabetes":          "https://www.who.int/news-room/fact-sheets/detail/diabetes",
    "who_hypertension":      "https://www.who.int/news-room/fact-sheets/detail/hypertension",
    "who_dengue":            "https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue",
    "who_tuberculosis":      "https://www.who.int/news-room/fact-sheets/detail/tuberculosis",
    "who_malaria":           "https://www.who.int/news-room/fact-sheets/detail/malaria",
    "who_asthma":            "https://www.who.int/news-room/fact-sheets/detail/asthma",
    "who_cancer":            "https://www.who.int/news-room/fact-sheets/detail/cancer",
    "who_depression":        "https://www.who.int/news-room/fact-sheets/detail/depression",
    "who_anaemia":           "https://www.who.int/news-room/fact-sheets/detail/anaemia",
    "who_cholera":           "https://www.who.int/news-room/fact-sheets/detail/cholera",
    "who_typhoid":           "https://www.who.int/news-room/fact-sheets/detail/typhoid",
    "who_hepatitis_b":       "https://www.who.int/news-room/fact-sheets/detail/hepatitis-b",
    "who_hepatitis_c":       "https://www.who.int/news-room/fact-sheets/detail/hepatitis-c",
    "who_hiv_aids":          "https://www.who.int/news-room/fact-sheets/detail/hiv-aids",
    "who_pneumonia":         "https://www.who.int/news-room/fact-sheets/detail/pneumonia",
    "who_diarrhoea":         "https://www.who.int/news-room/fact-sheets/detail/diarrhoeal-disease",
    "who_obesity":           "https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight",
    "who_cardiovascular":    "https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)",
    "who_stroke":            "https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death",
    "who_mental_health":     "https://www.who.int/news-room/fact-sheets/detail/mental-health-strengthening-our-response",
    "who_chronic_kidney":    "https://www.who.int/news-room/fact-sheets/detail/chronic-kidney-disease",
    "who_epilepsy":          "https://www.who.int/news-room/fact-sheets/detail/epilepsy",
    "who_influenza":         "https://www.who.int/news-room/fact-sheets/detail/influenza-(seasonal)",
    "who_blindness":         "https://www.who.int/news-room/fact-sheets/detail/blindness-and-visual-impairment",
    "who_deafness":          "https://www.who.int/news-room/fact-sheets/detail/deafness-and-hearing-loss",
}

# ── MedlinePlus XML ───────────────────────────────────────────────────────────

MEDLINEPLUS_XML_URL = (
    "https://medlineplus.gov/xml/mplus_topics_2026-05-22.xml"
)


# ── Scraper ───────────────────────────────────────────────────────────────────

def scrape_who_fact_sheet(url: str) -> str:
    """
    Scrape clean text from a WHO fact sheet page.
    Targets the main article body and removes nav/footer noise.
    """
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # WHO fact sheets have content in <div class="sf-detail-body-wrapper">
    # or inside <article> or <main> depending on page version
    content_div = (
        soup.find("div", class_="sf-detail-body-wrapper")
        or soup.find("article")
        or soup.find("main")
        or soup.find("div", class_="content")
    )

    if not content_div:
        content_div = soup.body

    # Remove script, style, nav, footer, header, aside
    for tag in content_div.find_all(["script","style","nav","footer",
                                      "header","aside","figure","button"]):
        tag.decompose()

    # Extract text
    raw_text = content_div.get_text(separator="\n")

    # Clean up excessive whitespace
    lines = [line.strip() for line in raw_text.splitlines()]
    lines = [l for l in lines if l and len(l) > 2]
    clean = "\n".join(lines)

    # Remove lines that look like nav/cookie/share noise
    noise_patterns = [
        r"^Share$", r"^Print$", r"^Listen$", r"^Español$",
        r"^中文$", r"^Français$", r"^العربية$", r"^Русский$",
        r"cookie", r"©\s*WHO", r"All rights reserved",
        r"^Credits$", r"^\+$", r"^-$",
    ]
    filtered_lines = []
    for line in clean.splitlines():
        if not any(re.search(p, line, re.IGNORECASE) for p in noise_patterns):
            filtered_lines.append(line)

    return "\n".join(filtered_lines).strip()


def save_txt(filename: str, content: str) -> Path:
    path = KB_DIR / f"{filename}.txt"
    path.write_text(content, encoding="utf-8")
    return path


# ── Downloaders ───────────────────────────────────────────────────────────────

def download_who_sheets():
    print("\n📥 Downloading WHO Fact Sheets...")
    print(f"   Target: {KB_DIR}\n")

    success = 0
    failed  = []

    for name, url in WHO_FACT_SHEETS.items():
        out_path = KB_DIR / f"{name}.txt"

        if out_path.exists() and out_path.stat().st_size > 500:
            print(f"   ⏭️  Already exists: {name}.txt")
            success += 1
            continue

        try:
            print(f"   ⬇️  {name}...", end=" ", flush=True)
            text = scrape_who_fact_sheet(url)

            if len(text) < 200:
                print("⚠️  Too short, skipping")
                failed.append(name)
                continue

            save_txt(name, text)
            print(f"✅  ({len(text):,} chars)")
            success += 1
            time.sleep(1.5)     # polite delay between requests

        except Exception as e:
            print(f"❌  Error: {e}")
            failed.append(name)
            time.sleep(2)

    print(f"\n   WHO sheets: {success} downloaded, {len(failed)} failed")
    if failed:
        print(f"   Failed: {', '.join(failed)}")
    return success


def download_medlineplus_xml():
    print("\n📥 Downloading MedlinePlus Health Topics XML...")
    out_path = KB_DIR / "medlineplus_health_topics.xml"

    if out_path.exists() and out_path.stat().st_size > 100_000:
        print(f"   ⏭️  Already exists ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return True

    try:
        print("   ⬇️  Downloading (~15 MB)...", end=" ", flush=True)

        def progress(count, block_size, total):
            if total > 0:
                pct = min(count * block_size / total * 100, 100)
                print(f"\r   ⬇️  Downloading: {pct:.0f}%", end="", flush=True)

        urllib.request.urlretrieve(MEDLINEPLUS_XML_URL, str(out_path), progress)
        print(f"\n   ✅  Saved: {out_path.stat().st_size / 1024 / 1024:.1f} MB")
        return True

    except Exception as e:
        print(f"\n   ❌  Failed: {e}")
        print("   Try manually: https://medlineplus.gov/xml.html")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print(" MediAssist — Knowledge Base Downloader")
    print("=" * 55)

    who_count = download_who_sheets()
    ml_ok     = download_medlineplus_xml()

    print("\n" + "=" * 55)
    print(f" Download complete!")
    print(f" WHO fact sheets  : {who_count} files")
    print(f" MedlinePlus XML  : {'✅ downloaded' if ml_ok else '❌ failed'}")
    print(f" Location         : {KB_DIR}")
    print("=" * 55)
    print("\n✅ Now run the ingestion pipeline:")
    print("   python -m core.ingest\n")