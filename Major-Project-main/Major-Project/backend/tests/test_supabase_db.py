import asyncio
from types import SimpleNamespace

import pytest

from supabase_db import SupabaseCollection


class FakeQuery:
    def __init__(self, error=None):
        self.error = error
        self.upsert_payload = None

    def select(self, *_args):
        return self

    def eq(self, *_args):
        return self

    def upsert(self, payload, **_kwargs):
        self.upsert_payload = payload
        return self

    def execute(self):
        if self.error:
            raise self.error
        return SimpleNamespace(data=[])


class ExistingRowQuery(FakeQuery):
    def __init__(self):
        super().__init__()
        self.select_data = [{
            'id': 'user-1',
            'data': {'id': 'user-1', 'email': 'person@example.com', 'password_hash': 'legacy'},
        }]

    def execute(self):
        if self.error:
            raise self.error
        return SimpleNamespace(data=self.select_data)


class FakeClient:
    def __init__(self, query):
        self.query = query

    def table(self, _name):
        return self.query


class ClaimQuery:
    def __init__(self):
        self.claim = None
        self.persisted = None
        self.mode = None

    def insert(self, payload):
        self.mode = 'insert'
        self.claim = payload
        return self

    def select(self, *_args):
        self.mode = 'select'
        return self

    def eq(self, *_args):
        return self

    def execute(self):
        if self.mode == 'insert':
            if self.persisted:
                raise RuntimeError('duplicate key')
            self.persisted = self.claim
            return SimpleNamespace(data=[])
        if self.mode == 'select' and self.persisted:
            return SimpleNamespace(data=[{
                'id': self.persisted['id'],
                'data': self.persisted['data'],
            }])
        raise RuntimeError('Unexpected claim query')


def test_read_failure_is_not_hidden_by_an_in_memory_fallback():
    collection = SupabaseCollection(FakeClient(FakeQuery(ConnectionError('Supabase unavailable'))), 'users')

    with pytest.raises(ConnectionError, match='Supabase unavailable'):
        asyncio.run(collection.find_one({'email': 'person@example.com'}))


def test_failed_user_write_does_not_succeed_from_cache():
    collection = SupabaseCollection(FakeClient(FakeQuery(ConnectionError('Supabase unavailable'))), 'users')

    with pytest.raises(ConnectionError, match='Supabase unavailable'):
        asyncio.run(collection.insert_one({
            'id': 'user-1',
            'email': 'person@example.com',
            'password_hash': 'salt$digest',
        }))


def test_user_write_persists_the_password_hash_not_a_plaintext_password():
    query = FakeQuery()
    collection = SupabaseCollection(FakeClient(query), 'users')

    asyncio.run(collection.insert_one({
        'id': 'user-1',
        'email': 'person@example.com',
        'password_hash': 'salt$digest',
    }))

    assert query.upsert_payload['collection'] == 'users'
    assert query.upsert_payload['data']['password_hash'] == 'salt$digest'
    assert 'password' not in query.upsert_payload['data']


def test_update_one_can_remove_legacy_password_hash():
    query = ExistingRowQuery()
    collection = SupabaseCollection(FakeClient(query), 'users')

    asyncio.run(collection.update_one(
        {'id': 'user-1'},
        {'$set': {'auth_user_id': 'auth-user-1'}, '$unset': {'password_hash': ''}},
    ))

    assert query.upsert_payload['data']['auth_user_id'] == 'auth-user-1'
    assert 'password_hash' not in query.upsert_payload['data']


def test_insert_if_absent_claims_unique_row_only_once():
    query = ClaimQuery()
    collection = SupabaseCollection(FakeClient(query), 'bootstrap_state')

    assert asyncio.run(collection.insert_if_absent({
        'id': 'first-admin-bootstrap',
        'claimed_at': 'first-claim',
    }))
    assert not asyncio.run(collection.insert_if_absent({
        'id': 'first-admin-bootstrap',
        'claimed_at': 'second-claim',
    }))
    assert query.persisted['collection'] == 'bootstrap_state'
    assert query.persisted['id'] == 'first-admin-bootstrap'
