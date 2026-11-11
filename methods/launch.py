"""Dispatch to existing commands without importing CUDA or changing defaults."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def load_spec(method):
    if method not in ('baseline','gvr','harness','lora','rl','prefix','policy'):
        raise ValueError('Unknown method')
    return json.loads((ROOT/'methods'/method/'manifest.json').read_text(encoding='utf-8'))

def build_command(method, command, arguments):
    spec = load_spec(method)
    if command not in spec['commands']:
        raise ValueError(f'Unknown command: {command}')
    return [sys.executable, *spec['commands'][command], *arguments]

def main(method):
    spec = load_spec(method)
    parser = argparse.ArgumentParser(description=spec['description'])
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('command', nargs='?', choices=spec['commands'])
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.describe:
        print(json.dumps(spec, ensure_ascii=False, indent=2))
        return
    if not args.command:
        parser.print_help()
        return
    forwarded = args.arguments
    if forwarded[:1] == ['--']:
        forwarded = forwarded[1:]
    result = subprocess.run(build_command(method,args.command,forwarded),cwd=ROOT)
    raise SystemExit(result.returncode)
