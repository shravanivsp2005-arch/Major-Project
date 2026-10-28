"""
CivicPulse Supabase Database Inspector & Exporter
=================================================
Connects directly to your Supabase PostgreSQL instance and displays
all live documents stored across collections (complaints, users, notifications, etc.).
"""

import os
import json
import dotenv
import requests
import pandas as pd
from collections import Counter

# Load environment
dotenv.load_dotenv("Major-Project/backend/.env")
if not os.environ.get("SUPABASE_URL"):
    dotenv.load_dotenv(".env")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("[!] Error: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env")
    exit(1)

headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}"
}


def fetch_all_documents():
    endpoint = f"{SUPABASE_URL}/rest/v1/civic_documents?select=collection,id,data,created_at"
    resp = requests.get(endpoint, headers=headers)
    if resp.status_code != 200:
        print(f"[!] Error fetching from Supabase: {resp.status_code} - {resp.text}")
        return []
    return resp.json()


def main():
    print("=" * 75)
    print(f"       CONNECTING TO SUPABASE: {SUPABASE_URL}")
    print("=" * 75)

    docs = fetch_all_documents()
    if not docs:
        print("[!] No documents found in public.civic_documents table.")
        return

    counts = Counter([d.get("collection") for d in docs])
    print(f"\n[+] Total Stored Records in Supabase: {len(docs)} documents\n")
    print(f"{'Collection Name':<25} | {'Stored Row Count':<15}")
    print("-" * 45)
    for col, count in counts.most_common():
        print(f"{col:<25} | {count:<15}")
    print("-" * 45)

    # Show sample complaints
    complaints = [d.get("data") for d in docs if d.get("collection") == "complaints" and d.get("data")]
    if complaints:
        print(f"\n[+] Sample Recent Complaints Stored in Supabase (Total: {len(complaints)}):")
        print("=" * 75)
        for i, c in enumerate(complaints[:5], 1):
            tid = c.get("tracking_id", "N/A")
            title = c.get("title", "No Title")
            dept = c.get("department", "General")
            prio = c.get("priority", "MEDIUM")
            status = c.get("status", "NEW")
            created = c.get("created_at", "")
            print(f"[{i}] {tid:15} | Dept: {dept:22} | Prio: {prio:8} | Status: {status:10}")
            print(f"    Title: \"{title}\" (Created: {created})")
            print("-" * 75)

    # Export to local CSV for easy review
    flat_complaints = []
    for c in complaints:
        loc = c.get("location") or {}
        ai_pred = c.get("ai_prediction") or {}
        flat_complaints.append({
            "id": c.get("id"),
            "tracking_id": c.get("tracking_id"),
            "title": c.get("title"),
            "description": c.get("description"),
            "department": c.get("department"),
            "category": c.get("category"),
            "priority": c.get("priority"),
            "status": c.get("status"),
            "ai_confidence": c.get("ai_confidence"),
            "ai_reasoning": ai_pred.get("reasoning"),
            "ward": loc.get("ward"),
            "latitude": loc.get("lat"),
            "longitude": loc.get("lng"),
            "citizen_name": c.get("citizen_name"),
            "created_at": c.get("created_at"),
            "is_demo": c.get("is_demo")
        })

    if flat_complaints:
        df = pd.DataFrame(flat_complaints)
        out_csv = "supabase_exported_complaints.csv"
        df.to_csv(out_csv, index=False)
        print(f"\n[+] Exported {len(df)} live complaints from Supabase to '{out_csv}'.\n")


if __name__ == "__main__":
    main()
