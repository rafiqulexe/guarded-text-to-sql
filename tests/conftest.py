import os

# Settings require an API key; tests must never need a real one.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
