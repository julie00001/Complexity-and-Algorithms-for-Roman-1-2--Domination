"""Audit the archive and reconstruct manuscript tables using only Python's standard library."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

PACKAGE = Path(__file__).resolve().parents[1]
DATA = PACKAGE / 'data'
FAMILIES = {'bipartite': 'Bipartite', 'cubic_single': 'Cubic, one copy',
            'cubic_double': 'Cubic, two copies', 'general_pairs': 'Random pairs',
            'matching_stress': 'Matching stress'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(name):
    return json.loads((DATA / name).read_text(encoding='utf-8'))


def check_data():
    main, large = read('split-benchmark-cbc.json'), read('split-scaling.json')
    require(main.get('complete') and large.get('complete'), 'Incomplete recorded experiment.')
    require(len(main['instances']) == len(main['results']) == 39, 'Expected 39 main instances.')
    require(len(large['instances']) == len(large['rows']) == 8, 'Expected eight scaling instances.')
    source = {x['id']: x for x in main['instances']}
    require(len(source) == 39, 'Duplicate instance IDs.')
    expected = {(f, p, s) for f in list(FAMILIES)[:4] for p in (24, 48, 96)
                for s in (20261011, 20261012, 20261013)}
    expected |= {('matching_stress', p, 20261011) for p in (24, 48, 96)}
    require({(r['family'], r['p'], r['seed']) for r in main['results']} == expected,
            'The main experiment does not match the stated design.')
    for r in main['results']:
        x = source[r['id']]
        require(all(x[k] == r[k] for k in ('family', 'p', 'seed')), 'Input/result ID mismatch.')
        masks = x['neighborhood_masks']
        require(r['N'] == r['p'] + len(masks), 'Incorrect target order.')
        require(all(type(m) is int and 0 <= m < 1 << r['p'] and m.bit_count() == 2
                    for m in masks), 'Invalid degree-two neighborhood.')
        require(r['lower'] == r['upper'] == r['structural_value'] == r['structural_lower'],
                'Reference optimum is not certified.')
        require(r['optimum_certified'] and r['structural_certified'], 'Missing certificate flag.')
        opt = r['lower']
        require(opt <= r['approx_value'] <= r['baseline_value'], 'Invalid approximation values.')
        require(3 * r['approx_value'] <= 4 * opt, 'The 4/3 bound fails.')
        for key, numerator in [('approx_ratio_or_certificate', 'approx_value'),
                               ('baseline_ratio_or_certificate', 'baseline_value')]:
            require(math.isclose(r[key], r[numerator] / opt, rel_tol=1e-13), 'Ratio mismatch.')
        require(r['full_lower'] <= opt, 'Invalid direct-model lower bound.')
        require(r['full_value'] is None or r['full_value'] >= opt, 'Invalid direct incumbent.')
        require(r['full_status'] == (0 if r['full_certified'] else 1), 'Unexpected solver status.')
        if r['full_certified']:
            require(r['full_value'] == r['full_lower'] == opt, 'Direct optimum mismatch.')
        if r['family'] == 'cubic_single':
            require(r['approx_value'] == opt, 'One-copy exactness failed.')
        if r['family'] == 'matching_stress':
            require(3 * r['approx_value'] == 4 * opt, 'Stress ratio is not 4/3.')
    require(sum(r['full_certified'] for r in main['results']) == 19,
            'Published direct-model count differs.')
    with (DATA / 'split-benchmark-cbc.csv').open(encoding='utf-8', newline='') as handle:
        csv_rows = list(csv.DictReader(handle))
    require(len(csv_rows) == 39, 'CSV record count differs.')
    for r, c in zip(main['results'], csv_rows):
        require(set(r) == set(c), 'CSV columns differ.')
        for k, v in r.items():
            require(c[k] == ('' if v is None else str(v)), f'CSV/JSON mismatch: {r["id"]}/{k}')
    expected_large = {(f, p) for f in ('bipartite', 'cubic_single') for p in (192, 384, 768, 1024)}
    require({(r['family'], r['p']) for r in large['rows']} == expected_large, 'Scaling design differs.')
    for x, r in zip(large['instances'], large['rows']):
        require((x['family'], x['p']) == (r['family'], r['p']), 'Scaling input/result mismatch.')
        require(r['N'] == r['p'] + len(x['neighborhood_masks']), 'Scaling target order differs.')
        require(all(type(m) is int and 0 <= m < 1 << r['p'] and m.bit_count() == 2
                    for m in x['neighborhood_masks']), 'Scaling neighborhood invalid.')
        require(1 <= r['approx_ratio'] <= 4 / 3, 'Scaling ratio outside guarantee.')
    require(max(r['N'] for r in large['rows']) == 10078, 'Largest order differs.')
    for name in ('coloring-checks.json', 'coloring-checks-bipartite.json', 'coloring-checks-unit.json'):
        c = read(name)
        require((c['local_pairs'], c['accepted_local_pairs']) == (36, 12), 'Local count differs.')
        require((c['selector_graphs'], c['selector_vectors'], c['accepted_selector_vectors'])
                == (49, 1161216, 18138), 'Selector count differs.')
        require(sum(v['total'] for v in c['atlas'].values()) == 1249
                and c['forward_assignments'] == 829, 'Atlas count differs.')
        require(len(c['milp']) == 6 and all(r['status'] in (0, 2) and not r.get('unverified')
                                          for r in c['milp']), 'Unverified coloring check.')
    s, a = read('split-algorithm-checks.json'), read('approximation-checks.json')
    require(s['all_passed'] and a['all_passed'], 'A finite verification failed.')
    require((s['exhaustive_cases'], s['matching_cases'], s['exact_two_cross3_cases'],
             s['small_clique_cases'], s['larger_random_cases']) == (6540, 6359, 1015, 153, 32),
            'Split verification counts differ.')
    t = a['transfer']
    require((t['all_cover_transfer_checks'], t['dense_transfer_checks'],
             t['additional_dense_inputs'], t['dense_third_branch_inputs'],
             t['dense_independent_milp_checks']) == (54682, 1077, 60, 52, 10),
            'Transfer verification counts differ.')
    require(a['apx']['decoder_checks'] == 16693
            and sum(a['apx']['decoder_branches'].values()) == 16693, 'Decoder count differs.')
    require(all(r['milp_optimum'] == r['p'] + r['tau'] for r in a['apx']['instances']),
            'Cubic reduction objective identity differs.')
    return main, large, s, a


def csv_text(rows):
    handle = io.StringIO(newline='')
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def make_tables():
    main, large, split, approximation = check_data()
    runtime, quality, design, mapping = [], [], [], []
    purposes = {
        'bipartite': 'Exact matching subclass; comparison and scalability',
        'cubic_single': 'Degree-two/cross-degree-three exact subclass; exactness and scalability',
        'cubic_double': 'APX-complete class; auxiliary integer optimization and approximation quality',
        'general_pairs': 'General degree-two instances outside guaranteed polynomial subclasses; comparison and quality',
        'matching_stress': 'Constructed tight 4/3 examples; polynomial exact reference',
    }
    for family, label in FAMILIES.items():
        group = [r for r in main['results'] if r['family'] == family]
        design.append({'family': label, 'main_instances': len(group),
                       'main_clique_orders': '24;48;96',
                       'main_seeds': '20261011' if family == 'matching_stress' else '20261011;20261012;20261013',
                       'scaling_instances': sum(r['family'] == family for r in large['rows']),
                       'purpose': purposes[family]})
        for p in (24, 48, 96):
            block = [r for r in group if r['p'] == p]
            low, high = min(r['N'] for r in block), max(r['N'] for r in block)
            runtime.append({'family': label, 'p': p,
                            'target_order': str(low) if low == high else f'{low}--{high}',
                            'structural_seconds_median': f'{statistics.median(r["structural_seconds"] for r in block):.4f}',
                            'direct_seconds_median': f'{statistics.median(r["full_seconds"] for r in block):.3f}',
                            'direct_solved': f'{sum(r["full_certified"] for r in block)}/{len(block)}'})
        ratios = [r['approx_value'] / r['lower'] for r in group]
        baselines = [r['baseline_value'] / r['lower'] for r in group]
        quality.append({'family': label, 'instances': len(group),
                        'approximation_mean': f'{statistics.mean(ratios):.3f}',
                        'approximation_maximum': f'{max(ratios):.3f}',
                        'baseline_mean': f'{statistics.mean(baselines):.3f}',
                        'baseline_maximum': f'{max(baselines):.3f}'})
    scaling = [{'family': FAMILIES[r['family']], 'p': r['p'], 'target_order': r['N'],
                'optimum': r['optimum'], 'structural_seconds_median': f'{r["structural_seconds"]:.3f}',
                'approximation_ratio': f'{r["approx_ratio"]:.3f}'} for r in large['rows']]
    for r in main['results']:
        mapping.append({'dataset': 'split-benchmark-cbc.json', 'instance_key': r['id'],
                        'runtime_row': f'{FAMILIES[r["family"]]}; p={r["p"]}',
                        'quality_row': FAMILIES[r['family']], 'scaling_row': ''})
    for r in large['rows']:
        mapping.append({'dataset': 'split-scaling.json',
                        'instance_key': f'{r["family"]};p={r["p"]};seed=20261011',
                        'runtime_row': '', 'quality_row': '',
                        'scaling_row': f'{FAMILIES[r["family"]]}; p={r["p"]}'})
    coloring = []
    for variant, name in [('amplified_split', 'coloring-checks.json'),
                          ('amplified_bipartite', 'coloring-checks-bipartite.json'),
                          ('unit_split', 'coloring-checks-unit.json')]:
        for r in read(name)['milp']:
            coloring.append({'variant': variant, 'source': r['source'], 'n': r['n'],
                             'source_edges': r['m'], 'M': r['M'], 'target_order': r['N'],
                             'result': 'optimal' if r['status'] == 0 else 'infeasible_at_threshold',
                             'optimum': r.get('optimum', ''),
                             'threshold': 4 * r['n']})
    return {'runtime.csv': runtime, 'quality.csv': quality, 'scaling.csv': scaling,
            'experimental_design.csv': design, 'instance_table_map.csv': mapping,
            'appendix_coloring.csv': coloring,
            'appendix_cubic_reduction.csv': approximation['apx']['instances']}


def audit():
    check_data()
    manifest = PACKAGE / 'MANIFEST.sha256'
    require(manifest.exists(), 'MANIFEST.sha256 is missing.')
    count = 0
    for line in manifest.read_text(encoding='utf-8').splitlines():
        digest, relative = line.split('  ', 1)
        path = (PACKAGE / relative).resolve()
        require(PACKAGE in path.parents, 'Unsafe manifest path.')
        require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                f'Checksum mismatch or missing file: {relative}')
        count += 1
    for name, rows in make_tables().items():
        require((PACKAGE / 'tables' / name).read_text(encoding='utf-8') == csv_text(rows),
                f'Included table differs from reconstruction: {name}')
    print(f'PASS: {count} checksums; 39 main and eight scaling inputs; all recorded finite checks; seven reconstructed tables/maps.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['audit', 'tables'])
    parser.add_argument('--output', type=Path, default=PACKAGE / 'results/tables')
    args = parser.parse_args()
    if args.task == 'audit':
        audit()
    else:
        folder = args.output.resolve()
        require(folder != DATA and DATA not in folder.parents and folder != PACKAGE / 'code'
                and PACKAGE / 'code' not in folder.parents, 'Do not write tables over data or code.')
        folder.mkdir(parents=True, exist_ok=True)
        for name, rows in make_tables().items():
            (folder / name).write_text(csv_text(rows), encoding='utf-8', newline='\n')
        print(f'Wrote seven CSV tables/maps to {folder}. No optimization was run.')


if __name__ == '__main__':
    main()
