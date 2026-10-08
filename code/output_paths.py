"""Keep newly computed results separate from the published data."""
import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]


def output_directory():
    path = Path(os.environ.get('ROMAN12_OUTPUT_DIR', PACKAGE / 'results')).resolve()
    for name in ('data', 'code', 'tables'):
        protected = PACKAGE / name
        if path == protected or protected in path.parents:
            raise ValueError('Choose a results directory outside code/, data/, and tables/.')
    if path == PACKAGE:
        raise ValueError('Choose a separate results directory.')
    return path
