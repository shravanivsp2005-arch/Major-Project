"""Small document repository backed by a Supabase JSONB table."""
import re
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional

from supabase import Client, create_client

_MISSING = object()


def _get_value(document: Dict[str, Any], path: str) -> Any:
    value: Any = document
    for part in path.split('.'):
        if not isinstance(value, dict) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _matches(document: Dict[str, Any], query: Dict[str, Any]) -> bool:
    for key, expected in query.items():
        if key == '$or':
            if not any(_matches(document, branch) for branch in expected):
                return False
            continue
        actual = _get_value(document, key)
        if isinstance(expected, dict) and any(str(k).startswith('$') for k in expected):
            for operator, operand in expected.items():
                if operator == '$gte' and (actual is _MISSING or actual < operand):
                    return False
                if operator == '$lte' and (actual is _MISSING or actual > operand):
                    return False
                if operator == '$in' and (actual is _MISSING or actual not in operand):
                    return False
                if operator == '$nin' and actual is not _MISSING and actual in operand:
                    return False
                if operator == '$regex':
                    flags = re.IGNORECASE if expected.get('$options') == 'i' else 0
                    if actual is _MISSING or re.search(operand, str(actual), flags) is None:
                        return False
                if operator == '$options':
                    continue
        elif actual is _MISSING or actual != expected:
            return False
    return True


def _project(document: Dict[str, Any], projection: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    result = deepcopy(document)
    if not projection:
        return result
    excluded = [key for key, include in projection.items() if not include]
    for key in excluded:
        if key == '_id':
            continue
        result.pop(key, None)
    return result


class SupabaseCursor:
    def __init__(self, rows: Iterable[Dict[str, Any]], projection: Optional[Dict[str, Any]] = None):
        self._rows = [_project(row, projection) for row in rows]

    def sort(self, field: str, direction: int):
        self._rows.sort(key=lambda row: _get_value(row, field) if _get_value(row, field) is not _MISSING else '', reverse=direction < 0)
        return self

    async def to_list(self, length: int):
        return self._rows[:length]

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for row in self._rows:
            yield row


class SupabaseCollection:
    def __init__(self, client: Client, name: str):
        self.client = client
        self.name = name

    def _rows(self) -> List[Dict[str, Any]]:
        response = self.client.table('civic_documents').select('id,data').eq('collection', self.name).execute()
        return [dict(row.get('data') or {}, id=row.get('id')) for row in (response.data or [])]

    async def find_one(self, query: Dict[str, Any], projection: Optional[Dict[str, Any]] = None):
        for row in self._rows():
            if _matches(row, query):
                return _project(row, projection)
        return None

    def find(self, query: Optional[Dict[str, Any]] = None, projection: Optional[Dict[str, Any]] = None):
        rows = self._rows()
        matched = [row for row in rows if _matches(row, query or {})]
        return SupabaseCursor(matched, projection)

    async def insert_one(self, document: Dict[str, Any]):
        value = deepcopy(document)
        document_id = str(value.get('id'))
        value.pop('_id', None)
        self.client.table('civic_documents').upsert(
            {'collection': self.name, 'id': document_id, 'data': value},
            on_conflict='collection,id'
        ).execute()

    async def insert_if_absent(self, document: Dict[str, Any]) -> bool:
        value = deepcopy(document)
        document_id = str(value.get('id'))
        value.pop('_id', None)
        try:
            self.client.table('civic_documents').insert(
                {'collection': self.name, 'id': document_id, 'data': value}
            ).execute()
            return True
        except Exception:
            existing = await self.find_one({'id': document_id})
            if existing is not None:
                return existing == value
            raise

    async def delete_one(self, query: Dict[str, Any]):
        row = await self.find_one(query)
        if row is not None:
            self.client.table('civic_documents').delete().eq(
                'collection', self.name
            ).eq('id', str(row['id'])).execute()

    async def update_one(self, query: Dict[str, Any], update: Dict[str, Any], upsert: bool = False):
        row = await self.find_one(query)
        if row is None:
            if not upsert:
                return
            row = deepcopy(query)
        changes = update.get('$set', {})
        row.update(deepcopy(changes))
        for field in update.get('$unset', {}):
            row.pop(field, None)
        document_id = str(row.get('id'))
        self.client.table('civic_documents').upsert(
            {'collection': self.name, 'id': document_id, 'data': row},
            on_conflict='collection,id'
        ).execute()

    async def update_many(self, query: Dict[str, Any], update: Dict[str, Any]):
        rows = [row for row in self._rows() if _matches(row, query)]
        for row in rows:
            row.update(deepcopy(update.get('$set', {})))
            for field in update.get('$unset', {}):
                row.pop(field, None)
            self.client.table('civic_documents').upsert(
                {'collection': self.name, 'id': str(row['id']), 'data': row},
                on_conflict='collection,id'
            ).execute()

    async def count_documents(self, query: Optional[Dict[str, Any]] = None):
        return sum(1 for row in self._rows() if _matches(row, query or {}))

    async def create_index(self, *args, **kwargs):
        return None


class SupabaseDatabase:
    def __init__(self, url: str, key: str):
        self.client = create_client(url, key)

    def __getattr__(self, collection: str) -> SupabaseCollection:
        return SupabaseCollection(self.client, collection)
