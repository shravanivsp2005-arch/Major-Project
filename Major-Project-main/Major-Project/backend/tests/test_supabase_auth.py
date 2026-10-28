from types import SimpleNamespace

from supabase_auth.errors import AuthApiError

import civic_supabase_auth
from civic_supabase_auth import SupabaseAuth


class FakeAdmin:
    def __init__(self):
        self.created = []
        self.updated = []
        self.deleted = []
        self.users = []
        self.create_error = None

    def create_user(self, attributes):
        if self.create_error:
            raise self.create_error
        self.created.append(attributes)
        return SimpleNamespace(user=SimpleNamespace(id='auth-user-1'))

    def update_user_by_id(self, user_id, attributes):
        self.updated.append((user_id, attributes))
        return SimpleNamespace(user=SimpleNamespace(id=user_id))

    def delete_user(self, user_id):
        self.deleted.append(user_id)

    def list_users(self, page, per_page):
        start = (page - 1) * per_page
        return self.users[start:start + per_page]


class FakeAuth:
    def __init__(self, admin=None, user=None):
        self.admin = admin
        self.user = user
        self.credentials = None

    def sign_in_with_password(self, credentials):
        self.credentials = credentials
        return SimpleNamespace(user=self.user)


class FakeClient:
    def __init__(self, auth):
        self.auth = auth


def test_create_user_uses_supabase_auth_with_confirmed_email_and_metadata(monkeypatch):
    admin = FakeAdmin()
    service_client = FakeClient(FakeAuth(admin=admin))
    clients = []

    def fake_create_client(url, key):
        clients.append((url, key))
        return service_client

    monkeypatch.setattr(civic_supabase_auth, 'create_client', fake_create_client)
    auth = SupabaseAuth('https://project.supabase.co', 'anon-key', 'service-key')

    user_id = auth.create_user('PERSON@example.com', 'secret-password', {'name': 'Person'})

    assert user_id == 'auth-user-1'
    assert clients == [('https://project.supabase.co', 'service-key')]
    assert admin.created == [{
        'email': 'person@example.com',
        'password': 'secret-password',
        'email_confirm': True,
        'user_metadata': {'name': 'Person'},
    }]


def test_password_login_uses_anon_key_and_normalizes_email(monkeypatch):
    admin = FakeAdmin()
    auth_client = FakeClient(FakeAuth(user=SimpleNamespace(id='auth-user-2')))
    clients = []

    def fake_create_client(url, key):
        clients.append((url, key))
        return auth_client

    monkeypatch.setattr(civic_supabase_auth, 'create_client', fake_create_client)
    auth = SupabaseAuth('https://project.supabase.co', 'anon-key', 'service-key')

    user_id = auth.authenticate('PERSON@example.com', 'secret-password')

    assert user_id == 'auth-user-2'
    assert clients == [
        ('https://project.supabase.co', 'service-key'),
        ('https://project.supabase.co', 'anon-key'),
    ]
    assert auth_client.auth.credentials == {
        'email': 'person@example.com',
        'password': 'secret-password',
    }


def test_legacy_password_migration_updates_existing_auth_account(monkeypatch):
    admin = FakeAdmin()
    monkeypatch.setattr(
        civic_supabase_auth,
        'create_client',
        lambda *_args: FakeClient(FakeAuth(admin=admin)),
    )
    auth = SupabaseAuth('https://project.supabase.co', 'anon-key', 'service-key')

    user_id = auth.migrate_legacy_password(
        'person@example.com',
        'legacy-password',
        {'name': 'Person'},
        auth_user_id='existing-auth-user',
    )

    assert user_id == 'existing-auth-user'
    assert admin.updated == [(
        'existing-auth-user',
        {
            'password': 'legacy-password',
            'email_confirm': True,
            'user_metadata': {'name': 'Person'},
        },
    )]
    assert admin.created == []


def test_legacy_migration_recovers_existing_auth_account_by_email(monkeypatch):
    admin = FakeAdmin()
    admin.create_error = AuthApiError('User already registered', 422, 'user_already_exists')
    admin.users = [SimpleNamespace(id='existing-auth-user', email='person@example.com')]
    monkeypatch.setattr(
        civic_supabase_auth,
        'create_client',
        lambda *_args: FakeClient(FakeAuth(admin=admin)),
    )
    auth = SupabaseAuth('https://project.supabase.co', 'anon-key', 'service-key')

    user_id = auth.migrate_legacy_password(
        'person@example.com',
        'legacy-password',
        {'name': 'Person'},
    )

    assert user_id == 'existing-auth-user'
    assert admin.updated[0][0] == 'existing-auth-user'
