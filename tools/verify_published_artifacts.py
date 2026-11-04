"""Verify published manifests, hydrated files or locally available LFS objects."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    configured = subprocess.run(['git','config','--get','lfs.storage'],cwd=ROOT,capture_output=True,text=True).stdout.strip()
    common = subprocess.check_output(['git','rev-parse','--git-common-dir'],cwd=ROOT,text=True).strip()
    storage = Path(configured) if configured else ROOT/common/'lfs'
    if not storage.is_absolute(): storage = ROOT/storage
    total = weights = 0
    for folder in ('campaign_20261103','historical_models'):
        directory = ROOT/'artifacts'/folder
        manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        for item in manifest['files']:
            path = directory/item['path']
            if item['storage'] == 'git-lfs':
                if path.stat().st_size < 256:
                    pointer = path.read_text()
                    match = re.fullmatch(r'version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize (\d+)\n',pointer)
                    assert match and match[1] == item['sha256'] and int(match[2]) == item['bytes'],path
                    oid = match[1]
                    path = storage/'objects'/oid[:2]/oid[2:4]/oid
                if item['path'].endswith('.safetensors'): weights += 1
            assert path.stat().st_size == item['bytes'],path
            assert digest(path) == item['sha256'],path
            total += 1
        for relative, expected in manifest.get('dataset_sha256',{}).items():
            assert digest(ROOT/relative) == expected,relative
    assert weights == 42,weights
    print(f'PASS: {total} published files, {weights} weight files, all SHA-256 and sizes match.')

if __name__ == '__main__': main()
