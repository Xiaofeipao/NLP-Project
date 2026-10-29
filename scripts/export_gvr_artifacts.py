"""Export a portable, checksummed GVR checkpoint and evidence bundle."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppnl.task_adapters import checkpoint_hash


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    source = Path('outputs/gvr')
    target = Path('artifacts/gvr')
    if target.exists():
        raise FileExistsError('Export requires a fresh artifacts/gvr directory')
    identity = json.loads((source / 'verification.json').read_text(encoding='utf-8'))['identity']
    files = []

    def record(src, dst, transformation='byte-identical'):
        files.append({'path': dst.relative_to(target).as_posix(), 'bytes': dst.stat().st_size,
                      'sha256': digest(dst), 'source_sha256': digest(src),
                      'transformation': transformation})

    def copy(src, relative):
        dst = target / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        record(src, dst)

    def metadata(src, relative, transform=lambda value: value):
        value = json.loads(src.read_text(encoding='utf-8'))
        value = transform(value)
        dst = target / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        record(src, dst, 'portable metadata; machine device field omitted and relative paths normalized')

    def training(value):
        value.pop('gpu', None)
        value['source_checkpoint'] = value['source_checkpoint'].replace('\\', '/')
        return value

    for stage in ('generation', 'verification', 'repair'):
        model = source / stage / 'train/best'
        if stage != 'generation':
            model /= stage
        for src in sorted(model.iterdir()):
            if not src.is_file() or src.name == 'README.md':
                continue
            relative = f'models/{stage}/{src.name}'
            if src.name == 'adapter_config.json':
                metadata(src, relative, lambda value: {**value, 'base_model_name_or_path':
                                                       'artifacts/gvr/models/generation'})
            else:
                copy(src, relative)
        for phase in ('pilot', 'train'):
            metadata(source / stage / phase / 'training.json', f'training/{stage}/{phase}.json', training)
        expected = identity['base_sha256'] if stage == 'generation' else identity['adapters_sha256'][stage]
        if checkpoint_hash(target / 'models' / stage) != expected:
            raise ValueError(f'Export changed {stage} weights')

    for split in ('dev', 'test_id', 'test_ood'):
        directory = source / 'evaluation' / split
        for name in ('metrics.json', 'metrics_recomputed.json', 'candidate_traces.jsonl',
                     'pipeline_traces.jsonl', 'case_studies.json'):
            copy(directory / name, f'evaluation/{split}/{name}')
        archive = target / 'evaluation' / split / 'sessions.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
            for src in sorted((directory / 'sessions').rglob('*')):
                if src.is_file() and src.name != '.lock':
                    raw = src.read_bytes()
                    if re.search(rb'(?i)(?<![a-z0-9])[a-z]:[\\/]|NVIDIA GeForce|RTX\s+\d+', raw):
                        raise ValueError(f'Nonportable journal metadata: {src}')
                    handle.writestr(src.relative_to(directory / 'sessions').as_posix(), raw)
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                original = directory / 'sessions' / member.filename
                if hashlib.sha256(handle.read(member)).hexdigest() != digest(original):
                    raise ValueError('Archive changed journal bytes')
        files.append({'path': archive.relative_to(target).as_posix(), 'bytes': archive.stat().st_size,
                      'sha256': digest(archive), 'transformation': 'lossless journal archive; lock files excluded'})

    copy(source / 'verification.json', 'verification.json')
    copy(source / 'source_changes.patch', 'historical_run.patch')
    copy(Path('outputs/gpu_ledger.json'), 'gpu_ledger.json')
    copy(Path('docs/gvr_gpu_run.json'), 'historical_summary.json')
    copy(source / 'config.json', 'config.json')

    def protocol(value):
        value.update(generation_base='artifacts/gvr/models/generation',
                     verification_adapter='artifacts/gvr/models/verification',
                     repair_adapter='artifacts/gvr/models/repair', output_root='outputs/gvr_replay')
        value['evidence'] = {key: False for key in value['evidence']}
        return value

    metadata(source / 'protocol.json', 'protocol.json', protocol)
    manifest = {'format_version': 1, 'training_source_commit': '96b9320',
                'historical_run_patch_sha256': digest(source / 'source_changes.patch'),
                'identity': identity, 'files': sorted(files, key=lambda value: value['path']),
                'notes': ['Original run used the historical evaluator and controller.',
                          'metrics_recomputed.json uses corrected task subsets without new model calls.',
                          'Weight bytes and archived journals are unchanged; metadata transformations are listed.',
                          'The shared historical GPU ledger is preserved; it must not be reset.']}
    (target / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'files': len(files), 'bytes': sum(item['bytes'] for item in files),
                      'identity_verified': True}, indent=2))


if __name__ == '__main__':
    main()
