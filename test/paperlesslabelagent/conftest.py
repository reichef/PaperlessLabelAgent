import pytest


@pytest.fixture(autouse=True)
def mock_paperless_environment(monkeypatch):
    """core/config.py runs load_dotenv(override=True) at import time, which copies the developer's
    real .env (real credentials, DELETE_INPUT_FILES_AFTER_UPLOAD=true) into os.environ. Force mock
    mode and the documented delete default for every test, so no test can reach a real
    Paperless-ngx instance or delete files by accident. Tests that need other values override
    them with their own monkeypatch.setenv calls."""
    monkeypatch.setenv("ACCOUNT", "mock")
    monkeypatch.setenv("PASSWORD", "mock")
    monkeypatch.setenv("API_URL", "")
    monkeypatch.delenv("DELETE_INPUT_FILES_AFTER_UPLOAD", raising=False)
