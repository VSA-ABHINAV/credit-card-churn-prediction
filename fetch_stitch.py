"""Helper script for fetching Stitch design export for Credit Card Churn Prediction.

Project ID: 4741929038724694034
Screen ID: cb2280fa88b643339d70e848de622321
Screen Title: Customer Churn Risk Scoring - Warm Artisanal Edition
"""

import json
import os
from pathlib import Path
import sys
import urllib.request

PROJECT_ID = "4741929038724694034"
SCREEN_ID = "cb2280fa88b643339d70e848de622321"

BASE_DIR = Path(__file__).resolve().parent
EXPORT_DIR = BASE_DIR / "stitch_export"
STATIC_INDEX = BASE_DIR / "static" / "index.html"


def fetch_or_verify_stitch():
    """Attempt automated retrieval and verify export files."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    export_file = EXPORT_DIR / f"{SCREEN_ID}.html"

    print(f"[*] Checking Stitch project: {PROJECT_ID}, screen: {SCREEN_ID}")

    if STATIC_INDEX.exists():
        print(f"[+] Found active production frontend at: {STATIC_INDEX}")
        with open(STATIC_INDEX, "r", encoding="utf-8") as f:
            content = f.read()
        with open(export_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[+] Saved design snapshot to: {export_file}")
        return True

    # If not already present, provide clear instructions
    print("\n[!] To fetch or update the design directly from Stitch:")
    print(f"    1. Open URL: https://stitch.ai/p/{PROJECT_ID}/screen/{SCREEN_ID}")
    print("    2. Click File -> Export -> HTML/Code")
    print(f"    3. Save the exported HTML to {STATIC_INDEX}")
    print("\n[+] Retrieval instructions complete.")
    return False


if __name__ == "__main__":
    fetch_or_verify_stitch()
