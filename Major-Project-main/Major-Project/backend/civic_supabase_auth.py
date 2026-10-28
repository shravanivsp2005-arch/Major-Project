"""Supabase Auth operations for CivicPulse accounts."""
from typing import Any, Dict, Optional

from supabase import Client, create_client
from supabase_auth.errors import AuthApiError


class SupabaseAuth:
    def __init__(self, url: str, anon_key: str, service_role_key: str):
        self.url = url
        self.anon_key = anon_key
        self.admin_client: Client = create_client(url, service_role_key)

    def create_user(self, email: str, password: str, metadata: Dict[str, Any]) -> str:
        response = self.admin_client.auth.admin.create_user({
            'email': email.lower(),
            'password': password,
            'email_confirm': True,
            'user_metadata': metadata,
        })
        return self._user_id(response.user)

    def authenticate(self, email: str, password: str) -> str:
        client = create_client(self.url, self.anon_key)
        response = client.auth.sign_in_with_password({
            'email': email.lower(),
            'password': password,
        })
        if response.user is None:
            raise RuntimeError('Supabase Auth returned no user after successful sign-in')
        return self._user_id(response.user)

    def migrate_legacy_password(
        self,
        email: str,
        password: str,
        metadata: Dict[str, Any],
        auth_user_id: Optional[str] = None,
    ) -> str:
        attributes = {
            'password': password,
            'email_confirm': True,
            'user_metadata': metadata,
        }
        if auth_user_id:
            response = self.admin_client.auth.admin.update_user_by_id(auth_user_id, attributes)
            return self._user_id(response.user)

        try:
            return self.create_user(email, password, metadata)
        except AuthApiError as exc:
            if exc.status != 422:
                raise

        existing_user = self._find_user_by_email(email)
        if existing_user is None:
            raise RuntimeError('Unable to locate the existing Supabase Auth user')
        response = self.admin_client.auth.admin.update_user_by_id(
            self._user_id(existing_user), attributes
        )
        return self._user_id(response.user)

    def delete_user(self, user_id: str) -> None:
        self.admin_client.auth.admin.delete_user(user_id)

    def _find_user_by_email(self, email: str):
        page = 1
        while True:
            users = self.admin_client.auth.admin.list_users(page=page, per_page=100)
            for user in users:
                if (getattr(user, 'email', '') or '').lower() == email.lower():
                    return user
            if len(users) < 100:
                return None
            page += 1

    @staticmethod
    def _user_id(user: Any) -> str:
        user_id = getattr(user, 'id', None)
        if not user_id:
            raise RuntimeError('Supabase Auth returned a user without an ID')
        return str(user_id)
