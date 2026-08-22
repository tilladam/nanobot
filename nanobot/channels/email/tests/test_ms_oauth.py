from __future__ import annotations

import time

import pytest

import nanobot.channels.email.ms_oauth as ms_oauth
from nanobot.channels.email.ms_oauth import (
    MicrosoftOAuthError,
    MSOAuthToken,
    account_fingerprint,
    get_email_oauth_login_status,
    get_email_oauth_token,
    logout_email_oauth,
)


def _use_temp_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(ms_oauth, "get_data_dir", lambda: tmp_path)


def _write_token(monkeypatch: pytest.MonkeyPatch, tmp_path, **overrides) -> MSOAuthToken:
    _use_temp_credentials(monkeypatch, tmp_path)
    fields = dict(
        access="access-1",
        refresh="refresh-1",
        expires=int(time.time() * 1000) + 60 * 60 * 1000,
        scope="offline_access IMAP.AccessAsUser.All SMTP.Send",
        fingerprint=account_fingerprint("tenant-1", "client-1", "bot@example.com"),
    )
    fields.update(overrides)
    token = MSOAuthToken(**fields)
    ms_oauth._write_token(token)  # noqa: SLF001
    return token


def test_account_fingerprint_is_stable_and_discriminates_inputs() -> None:
    a = account_fingerprint("tenant-1", "client-1", "Bot@Example.com")
    b = account_fingerprint("tenant-1", "client-1", "bot@example.com")
    c = account_fingerprint("tenant-2", "client-1", "bot@example.com")
    assert a == b  # case-insensitive mailbox
    assert a != c


def test_get_login_status_returns_none_when_no_token(monkeypatch, tmp_path) -> None:
    _use_temp_credentials(monkeypatch, tmp_path)
    assert get_email_oauth_login_status("tenant-1", "client-1", "bot@example.com") is None


def test_get_login_status_returns_none_on_fingerprint_mismatch(monkeypatch, tmp_path) -> None:
    _write_token(monkeypatch, tmp_path)
    assert get_email_oauth_login_status("tenant-1", "client-1", "someone-else@example.com") is None


def test_get_login_status_returns_token_on_match(monkeypatch, tmp_path) -> None:
    token = _write_token(monkeypatch, tmp_path)
    status = get_email_oauth_login_status("tenant-1", "client-1", "bot@example.com")
    assert status == token


def test_get_token_raises_when_not_signed_in(monkeypatch, tmp_path) -> None:
    _use_temp_credentials(monkeypatch, tmp_path)
    with pytest.raises(MicrosoftOAuthError, match="not signed in"):
        get_email_oauth_token(
            tenant_id="tenant-1",
            client_id="client-1",
            client_secret="secret-1",
            mailbox="bot@example.com",
        )


def test_get_token_returns_fresh_token_without_refresh(monkeypatch, tmp_path) -> None:
    token = _write_token(monkeypatch, tmp_path)

    def fail_refresh(*_a, **_kw):
        raise AssertionError("should not refresh a fresh token")

    monkeypatch.setattr(ms_oauth, "_refresh_token", fail_refresh)

    result = get_email_oauth_token(
        tenant_id="tenant-1",
        client_id="client-1",
        client_secret="secret-1",
        mailbox="bot@example.com",
    )
    assert result == token


def test_get_token_refreshes_stale_token_and_persists_it(monkeypatch, tmp_path) -> None:
    _write_token(monkeypatch, tmp_path, expires=int(time.time() * 1000) - 1000)

    refreshed = MSOAuthToken(
        access="access-2",
        refresh="refresh-2",
        expires=int(time.time() * 1000) + 60 * 60 * 1000,
        scope="offline_access IMAP.AccessAsUser.All SMTP.Send",
        fingerprint=account_fingerprint("tenant-1", "client-1", "bot@example.com"),
    )
    calls: list[str] = []

    def fake_refresh(token, **kwargs):
        calls.append(token.access)
        return refreshed

    monkeypatch.setattr(ms_oauth, "_refresh_token", fake_refresh)

    result = get_email_oauth_token(
        tenant_id="tenant-1",
        client_id="client-1",
        client_secret="secret-1",
        mailbox="bot@example.com",
    )
    assert result == refreshed
    assert calls == ["access-1"]
    assert get_email_oauth_login_status("tenant-1", "client-1", "bot@example.com") == refreshed


def test_get_token_without_refresh_token_raises(monkeypatch, tmp_path) -> None:
    _write_token(monkeypatch, tmp_path, expires=int(time.time() * 1000) - 1000, refresh=None)

    with pytest.raises(MicrosoftOAuthError, match="expired"):
        get_email_oauth_token(
            tenant_id="tenant-1",
            client_id="client-1",
            client_secret="secret-1",
            mailbox="bot@example.com",
        )


def test_logout_removes_token_file(monkeypatch, tmp_path) -> None:
    _write_token(monkeypatch, tmp_path)
    assert logout_email_oauth() is True
    assert get_email_oauth_login_status("tenant-1", "client-1", "bot@example.com") is None
    assert logout_email_oauth() is False
