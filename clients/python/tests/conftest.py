"""Makes `causa_rag_client` importable regardless of which directory
pytest is invoked from (root repo or clients/python/) — this package isn't
pip-installed in the platform's own dev environment, only published
independently (clients/python/pyproject.toml).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
