"""Run the published computations in fresh directories, with no cached results."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parents[1]
CODE = PACKAGE / 'code'
JOBS = {
    'verification': [('verify_coloring.py', '--milp'),
                     ('verify_coloring.py', '--milp', '--bipartite'),
                     ('verify_coloring.py', '--milp', '--unit'),
                     ('verify_split_algorithms.py',), ('verify_approximation.py',)],
    'benchmark': [('benchmark_split_optimization.py',)],
    'scaling': [('benchmark_scaling.py',)],
}


def reference_subset(reference, actual, path=''):
    """Compare all reported scientific values, excluding run-dependent metadata."""
    if isinstance(reference, dict):
        for key, value in reference.items():
            if key in {'seconds', 'message', 'hardware'}:
                continue
            if key not in actual:
                raise RuntimeError(f'Missing result: {path}/{key}')
            reference_subset(value, actual[key], f'{path}/{key}')
    elif isinstance(reference, list):
        if len(reference) != len(actual):
            raise RuntimeError(f'Length differs: {path}')
        for i, (left, right) in enumerate(zip(reference, actual)):
            reference_subset(left, right, f'{path}/{i}')
    elif reference != actual:
        raise RuntimeError(f'Result differs at {path}: reference={reference}, new={actual}')


def compare(group, folder):
    if group == 'verification':
        names = ['coloring-checks.json', 'coloring-checks-bipartite.json',
                 'coloring-checks-unit.json', 'split-algorithm-checks.json',
                 'approximation-checks.json']
        for name in names:
            ref = json.loads((PACKAGE / 'data' / name).read_text(encoding='utf-8'))
            new = json.loads((folder / name).read_text(encoding='utf-8'))
            if any(row.get('unverified') for row in new.get('milp', [])):
                raise RuntimeError(f'An integer-program check is unverified in {name}.')
            reference_subset(ref, new, name)
    else:
        name = 'split-benchmark-cbc.json' if group == 'benchmark' else 'split-scaling.json'
        ref = json.loads((PACKAGE / 'data' / name).read_text(encoding='utf-8'))
        new = json.loads((folder / name).read_text(encoding='utf-8'))
        if not new.get('complete'):
            raise RuntimeError('Run is incomplete.')
        reference_subset(ref['instances'], new['instances'], 'generated inputs')
        if group == 'benchmark':
            fields = ['id', 'family', 'p', 'N', 'seed', 'approx_value', 'baseline_value',
                      'structural_value', 'structural_lower', 'structural_route',
                      'structural_certified', 'lower', 'upper', 'optimum_certified',
                      'approx_ratio_or_certificate', 'baseline_ratio_or_certificate']
            rows = 'results'
        else:
            fields = ['family', 'p', 'N', 'optimum', 'route', 'approx_ratio']
            rows = 'rows'
        reference_subset([{k: row[k] for k in fields} for row in ref[rows]],
                         [{k: row[k] for k in fields} for row in new[rows]], group)
    print(f'PASS: {group} scientific results match the published data.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['verification', 'benchmark', 'scaling', 'all'])
    parser.add_argument('--output', type=Path, default=PACKAGE / 'results',
                        help='fresh run root; group subdirectories are created here')
    args = parser.parse_args()
    root = args.output.resolve()
    groups = list(JOBS) if args.task == 'all' else [args.task]
    if root == PACKAGE or any(root == PACKAGE / n or PACKAGE / n in root.parents
                              for n in ('code', 'data', 'tables')):
        parser.error('Output must be separate from the package source and published data.')
    for group in groups:
        folder = root / group
        if folder.exists() and any(folder.iterdir()):
            parser.error(f'{folder} is not empty. Use a new --output directory.')
    for group in groups:
        folder = root / group
        folder.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, ROMAN12_OUTPUT_DIR=str(folder),
                   PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1')
        for job in JOBS[group]:
            print('Running ' + ' '.join(job), flush=True)
            subprocess.run([sys.executable, str(CODE / job[0]), *job[1:]],
                           env=env, check=True)
        compare(group, folder)


if __name__ == '__main__':
    main()
