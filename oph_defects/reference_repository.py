"""Locate an explicitly supplied or adjacent theory checkout for evidence builds."""
import os
from pathlib import Path


def theory_repository():
    explicit = os.environ.get('OPH_THEORY_REPOSITORY')
    if explicit:
        return Path(explicit).expanduser().resolve()
    workspace = Path(__file__).resolve().parents[2]
    candidates = (workspace/'repair-law'/'reverse-engineering-reality',
                  workspace/'reverse-engineering-reality',
                  workspace/'observer-patch-holography')
    return next((p for p in candidates if (p/'model').is_dir()), candidates[1])
