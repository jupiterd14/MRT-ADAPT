"""
Offline simulation of the Google OAuth authorize flow.
Run:  python test_google_flow.py
"""
import os
from unittest.mock import patch

os.environ['RATELIMIT_ENABLED'] = '0'  # disable rate limits during test

from app import app
from models import User, db
from datetime import datetime


def run_test(email_to_use):
    print("\n" + "=" * 70)
    print(f"SIMULATING GOOGLE LOGIN FOR: {email_to_use}")
    print("=" * 70)

    # ── 1. Verify the user exists BEFORE
    with app.app_context():
        user_before = User.query.filter_by(username=email_to_use).first()
        if not user_before:
            print(f"❌ FAIL: No User row for {email_to_use}")
            return False
        print(f"✓ Found user id={user_before.id} role={user_before.role}")
        print(f"  BEFORE → google_id={user_before.google_id!r} is_active={user_before.is_active}")

    fake_google_info = {
        'email': email_to_use,
        'name': 'Test User',
        'sub': 'fake-google-sub-1234567890',
    }
    fake_token = {'access_token': 'fake', 'id_token': 'fake'}

    # ── 2. Hit the real route with mocks
    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess.clear()

        with app.app_context():
            google_client = app.config.get('GOOGLE_CLIENT')
            if google_client is None:
                print("❌ FAIL: GOOGLE_CLIENT not configured on app.config")
                return False

            with patch.object(google_client, 'authorize_access_token', return_value=fake_token), \
                 patch.object(google_client, 'parse_id_token', return_value=fake_google_info):
                resp = client.get('/login/google/authorize', follow_redirects=False)

        print(f"\n[HTTP] status={resp.status_code}")
        print(f"[HTTP] location={resp.headers.get('Location')}")
        body = resp.get_data(as_text=True)
        if 'GOOGLE LOGIN FAILED' in body:
            print("[HTTP] ❌ Debug traceback in body:")
            print(body[:1500])
            return False

        with client.session_transaction() as sess:
            print(f"\n[SESSION] user_id={sess.get('user_id')!r}")
            print(f"[SESSION] role={sess.get('role')!r}")
            print(f"[SESSION] username={sess.get('username')!r}")
            print(f"[SESSION] google_user={sess.get('google_user')!r}")
            if sess.get('user_id') is None:
                print("❌ FAIL: Session has no user_id after authorize")
                return False
            if sess.get('role') != 'operator':
                print(f"⚠️  WARN: role is {sess.get('role')!r}, expected 'operator'")

        with app.app_context():
            user_after = User.query.filter_by(username=email_to_use).first()
            print(f"\n[DB] AFTER → google_id={user_after.google_id!r}")
            if user_after.google_id != 'fake-google-sub-1234567890':
                print("❌ FAIL: google_id was NOT linked")
                return False
            print("✓ google_id linked correctly")

        resp2 = client.get('/operator-dashboard', follow_redirects=False)
        print(f"\n[DASHBOARD] status={resp2.status_code} location={resp2.headers.get('Location')}")
        if resp2.status_code == 302 and '/login' in (resp2.headers.get('Location') or ''):
            print("❌ FAIL: Guard bounced the logged-in operator to /login")
            return False
        print("✓ Dashboard accessible after login")

    print("\n" + "=" * 70)
    print("✅ PASS: Full Google login flow simulated successfully")
    print("=" * 70)
    return True


if __name__ == '__main__':
    ok = run_test('junnied0514@gmail.com')
    exit(0 if ok else 1)