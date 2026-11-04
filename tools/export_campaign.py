"""Publish selected campaign weights via LFS and portable, checksummed evidence.

Lives outside scripts/: the evaluated source snapshot must remain unchanged.
The working tree contains small LFS pointers; original local weights stay intact.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/campaign_20261103'
TARGET = ROOT / 'artifacts/campaign_20261103'

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def scrub(text):
    # Cover JSON escaped paths as well as ordinary logs and forward slashes.
    replacements = [
        (str(SOURCE.resolve()), 'outputs/campaign_20261103'),
        (str(ROOT), '.'),
    ]
    for old, new in replacements:
        for variant in (old.replace('\\', '\\\\'), old.replace('\\', '/'), old):
            text = text.replace(variant, new)
    text = re.sub(r'(?i)NVIDIA GeForce RTX\s+\d+(?:\s+Ti)?', 'CUDA GPU', text)
    text = re.sub(r'(?i)[A-Z]:(?:\\+|/)(?:Users|Anaconda)(?:[^\r\n"\s]*)', '<LOCAL_ENV>', text)
    return text

def clean_lfs(source, target):
    # Materialize the standard content-addressed LFS object without stdin filters.
    storage = subprocess.check_output(['git','config','--get','lfs.storage'],cwd=ROOT,text=True).strip()
    if not storage:
        raise ValueError('Configure LFS storage on a volume with enough space first')
    digest = sha(source)
    obj = Path(storage)/'objects'/digest[:2]/digest[2:4]/digest
    obj.parent.mkdir(parents=True, exist_ok=True)
    if not obj.exists():
        temporary = obj.with_suffix('.exporting')
        shutil.copyfile(source, temporary)
        assert sha(temporary) == digest
        temporary.replace(obj)
    assert obj.stat().st_size == source.stat().st_size and sha(obj) == digest
    target.write_text(f'version https://git-lfs.github.com/spec/v1\noid sha256:{digest}\nsize {obj.stat().st_size}\n',encoding='utf-8',newline='\n')

def main():
    if (TARGET/'manifest.json').exists():
        raise FileExistsError('Never overwrite a completed export')
    summary = json.loads((ROOT/'docs/gpu_campaign_results.json').read_text())
    assert summary['completed_reports'] == summary['expected_reports'] == 99
    TARGET.mkdir(parents=True, exist_ok=True)
    files = []
    weights = 0
    def publish(src, relative, model=False):
        nonlocal weights
        dst = TARGET / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        original = sha(src)
        if src.suffix == '.safetensors':
            clean_lfs(src, dst)
            pointer = dst.read_text()
            assert f'oid sha256:{original}' in pointer
            files.append({'path':relative, 'bytes':src.stat().st_size, 'sha256':original,
                          'source_sha256':original, 'storage':'git-lfs', 'transformation':'byte-identical weights'})
            weights += 1
            print('WEIGHT', weights, relative, flush=True)
            return
        raw = src.read_bytes()
        try:
            text = raw.decode('utf-8')
            exported = scrub(text)
            if src.name == 'adapter_config.json':
                value = json.loads(exported)
                base = value.get('base_model_name_or_path', '').replace('\\', '/')
                if base.startswith('outputs/campaign_20261103/'):
                    value['base_model_name_or_path'] = base.replace('outputs/campaign_20261103/', 'artifacts/campaign_20261103/', 1)
                    exported = json.dumps(value, ensure_ascii=False, indent=2)+'\n'
            raw = exported.encode('utf-8')
        except UnicodeDecodeError:
            pass
        dst.write_bytes(raw)
        exported_hash = sha(dst)
        files.append({'path':relative, 'bytes':len(raw), 'sha256':exported_hash,
                      'source_sha256':original, 'storage':'git',
                      'transformation':'byte-identical' if original == exported_hash else 'machine metadata redacted / adapter base path relocated'})

    def sessions(directory):
        relative = directory.relative_to(SOURCE).as_posix()+'.zip'
        archive = TARGET/relative
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists() and archive.stat().st_size < 256:
            text = archive.read_text()
            match = re.search(r'oid sha256:([0-9a-f]{64})\nsize (\d+)', text)
            if match:
                digest, size = match[1], int(match[2])
                storage = subprocess.check_output(['git','config','--get','lfs.storage'],cwd=ROOT,text=True).strip()
                obj = Path(storage)/'objects'/digest[:2]/digest[2:4]/digest
                assert obj.stat().st_size == size and sha(obj) == digest
                with zipfile.ZipFile(obj) as z:
                    assert z.testzip() is None
                    count = len(z.infolist())
                files.append({'path':relative,'bytes':size,'sha256':digest,'storage':'git-lfs',
                              'members':count,'transformation':'lossless journal archive; transient locks excluded'})
                print('REUSE JOURNAL',relative,count,flush=True)
                return
        count = 0
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for folder, dirs, names in os.walk(directory):
                dirs.sort()
                for name in sorted(names):
                    if name == '.lock': continue
                    src = Path(folder)/name
                    raw = src.read_bytes()
                    text = raw.decode('utf-8')
                    assert scrub(text) == text, f'Private metadata in journal: {src}'
                    z.writestr(src.relative_to(directory).as_posix(), raw)
                    count += 1
        digest, size = sha(archive), archive.stat().st_size
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
        # Stream the archive into LFS before replacing the temporary archive by its pointer.
        pointer = archive.with_suffix('.pointer')
        clean_lfs(archive, pointer)
        for attempt in range(10):
            try:
                archive.write_bytes(pointer.read_bytes())
                pointer.unlink()
                break
            except PermissionError:
                if attempt == 9: raise
                time.sleep(1)
        files.append({'path':relative, 'bytes':size, 'sha256':digest, 'storage':'git-lfs',
                      'members':count, 'transformation':'lossless journal archive; transient locks excluded'})
        print('JOURNAL', relative, count, flush=True)

    for folder, dirs, names in os.walk(SOURCE):
        dirs.sort()
        for excluded in ('__pycache__', 'train_failed_disk'):
            if excluded in dirs: dirs.remove(excluded)
        if 'sessions' in dirs:
            sessions(Path(folder)/'sessions')
            dirs.remove('sessions')
        for name in sorted(names):
            if name == '.lock' or name.endswith('.pyc'): continue
            src = Path(folder)/name
            publish(src, src.relative_to(SOURCE).as_posix())
    publish(ROOT/'outputs/gpu_ledger.json', 'gpu_ledger.json')
    publish(ROOT/'outputs/campaign_cpu_tests.log', 'cpu_tests.log')
    for folder in ('ppnl','scripts'):
        for src in sorted((ROOT/folder).rglob('*.py')):
            publish(src, 'source/'+src.relative_to(ROOT).as_posix())
    data = {p.relative_to(ROOT).as_posix():sha(p) for folder in ('data/derived','data/policy_v3')
            for p in sorted((ROOT/folder).glob('*')) if p.is_file()}
    assert weights == 33, weights
    manifest = {'schema_version':1, 'campaign':'campaign_20261103', 'weight_files':weights,
                'completed_reports':99, 'dataset_sha256':data, 'files':files,
                'notes':['Weight bytes and journal members are preserved.',
                         'Machine-specific metadata is redacted; original and exported checksums are recorded.',
                         'Archived locks document the original run; relocated/redacted copies require a new dev freeze before new testing.',
                         'Original local weights and evidence remain untouched.']}
    (TARGET/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'files':len(files),'weights':weights,'bytes':sum(x['bytes'] for x in files)},indent=2),flush=True)

if __name__ == '__main__': main()
