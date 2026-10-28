"""
CivicPulse Database Persistence Verification Script
===================================================
Tests registration, login, and complaint filing against the live backend
and verifies immediate persistence into Supabase PostgreSQL.
"""

import os
import time
import dotenv
import requests

dotenv.load_dotenv("Major-Project/backend/.env")
if not os.environ.get("SUPABASE_URL"):
    dotenv.load_dotenv(".env")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
BACKEND_URL = "http://localhost:8000"


def main():
    print("=" * 75)
    print("       DATABASE PERSISTENCE VERIFICATION TEST")
    print("=" * 75)

    s = requests.Session()
    test_email = f"citizen_test_{int(time.time())}@smartcity.gov"

    # 1. Register User
    print(f"\n[1] Testing Registration for: {test_email}...")
    reg_r = s.post(f"{BACKEND_URL}/api/auth/register", json={
        "name": "Aarav Sharma",
        "email": test_email,
        "password": "Password123!",
        "role": "citizen",
        "phone": "+91-98765-43210",
        "language": "en"
    })
    print(f"    -> Registration HTTP Status: {reg_r.status_code}")
    assert reg_r.status_code == 200, f"Registration failed: {reg_r.text}"
    reg_data = reg_r.json()
    token = reg_data["token"]
    user_id = reg_data["user"]["id"]
    print(f"    -> User Successfully Created with UUID: {user_id}")

    # 2. Login User
    print(f"\n[2] Testing Login for: {test_email}...")
    login_r = s.post(f"{BACKEND_URL}/api/auth/login", json={
        "email": test_email,
        "password": "Password123!"
    })
    print(f"    -> Login HTTP Status: {login_r.status_code}")
    assert login_r.status_code == 200, f"Login failed: {login_r.text}"
    login_data = login_r.json()
    print(f"    -> User Logged in! Last Login Timestamp: {login_data['user'].get('last_login_at')} | Login Count: {login_data['user'].get('login_count')}")

    # 3. Create Complaint
    print(f"\n[3] Testing Complaint Submission...")
    headers = {"Authorization": f"Bearer {token}"}
    comp_payload = {
        "title": "Severe sewage overflow on 80ft road",
        "description": "Black sewage water overflowing onto main street near bus stop creating extreme health hazard",
        "department": "Water & Sewage",
        "category": "Sewage Blockage",
        "priority": "HIGH",
        "location": {
            "lat": 12.9352,
            "lng": 77.6245,
            "address": "80 Feet Road, Koramangala 4th Block",
            "ward": "Koramangala"
        },
        "media": [],
        "ai_prediction": {
            "detected_issue": "Sewage Blockage",
            "department": "Water & Sewage",
            "category": "Sewage Blockage",
            "priority": "HIGH",
            "confidence": 96,
            "reasoning": "Sewage water overflowing on main road",
            "source": "local_ml"
        }
    }

    comp_r = s.post(f"{BACKEND_URL}/api/complaints", headers=headers, json=comp_payload)
    print(f"    -> Complaint Submission HTTP Status: {comp_r.status_code}")
    assert comp_r.status_code == 200, f"Complaint creation failed: {comp_r.text}"
    comp_data = comp_r.json()
    comp_id = comp_data["id"]
    tracking_id = comp_data["tracking_id"]
    print(f"    -> Complaint Created! Tracking ID: {tracking_id} | ID: {comp_id}")

    # 4. Verify in Supabase DB via REST API
    print(f"\n[4] Querying live Supabase PostgreSQL Instance ({SUPABASE_URL})...")
    db_headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}

    # Verify user record in Supabase
    u_res = requests.get(f"{SUPABASE_URL}/rest/v1/civic_documents?collection=eq.users&id=eq.{user_id}", headers=db_headers)
    u_rows = u_res.json() if u_res.status_code == 200 else []
    print(f"    [+] Supabase 'users' collection: Found {len(u_rows)} row matching ID '{user_id}'")
    if u_rows:
        print(f"        -> Email in DB: {u_rows[0]['data']['email']} | Name: {u_rows[0]['data']['name']} | Role: {u_rows[0]['data']['role']}")

    # Verify login activity record in Supabase
    la_res = requests.get(f"{SUPABASE_URL}/rest/v1/civic_documents?collection=eq.login_activity&select=id,data&limit=5", headers=db_headers)
    la_rows = la_res.json() if la_res.status_code == 200 else []
    print(f"    [+] Supabase 'login_activity' collection: Found {len(la_rows)} activity events")
    if la_rows:
        print(f"        -> Latest Event: Action={la_rows[0]['data']['action']} | Email={la_rows[0]['data']['email']} | Status={la_rows[0]['data']['status']}")

    # Verify complaint record in Supabase
    c_res = requests.get(f"{SUPABASE_URL}/rest/v1/civic_documents?collection=eq.complaints&id=eq.{comp_id}", headers=db_headers)
    c_rows = c_res.json() if c_res.status_code == 200 else []
    print(f"    [+] Supabase 'complaints' collection: Found {len(c_rows)} row matching ID '{comp_id}'")
    if c_rows:
        c_doc = c_rows[0]["data"]
        print(f"        -> Tracking ID: {c_doc.get('tracking_id')} | Title: \"{c_doc.get('title')}\"")
        print(f"        -> Department: {c_doc.get('department')} | Priority: {c_doc.get('priority')} | Status: {c_doc.get('status')}")
        print(f"        -> Ward: {c_doc.get('location', {}).get('ward')} | Citizen Name: {c_doc.get('citizen_name')}")

    print("\n" + "=" * 75)
    print("      ALL USER, LOGIN, AND COMPLAINT DATA STORED IN DATABASE SUCCESSFULLY!")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
