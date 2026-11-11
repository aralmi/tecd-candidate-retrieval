from pathlib import Path
import hashlib
import json
import pandas as pd

def project_root(start=None):
    start = Path(start or Path.cwd()).resolve()
    for root in (start, *start.parents):
        if (root / 'pyproject.toml').is_file() and (root / 'research').is_dir():
            return root
    raise FileNotFoundError('Run from a folder inside recsys-research')

def read_result(chapter, name):
    path = project_root() / 'research' / chapter / 'results' / name
    return json.loads(path.read_text()) if path.suffix == '.json' else pd.read_csv(path)

def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()
