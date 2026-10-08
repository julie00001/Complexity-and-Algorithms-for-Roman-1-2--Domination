"""Reproducible synthetic benchmark for the split-graph optimization results.

Each MILP receives a 10-second, single-thread budget. Instance generation is
excluded from timings; preprocessing/model construction is included. All
completed rows and generated neighborhoods are retained, including time limits.
Solver process failures are reported as errors.
"""
import csv
import json
import math
import platform
import random
import statistics
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import pulp
import re
import subprocess
import tempfile
import networkx as nx
import numpy as np
import scipy
from scipy.sparse import coo_matrix
from verify_split_algorithms import canonical, graph_masks, valid

from output_paths import output_directory

HERE = output_directory()
LIMIT = 10.0
FAMILIES = ['bipartite', 'cubic_single', 'cubic_double', 'general_pairs']
SIZES = [24, 48, 96]
SEEDS = [20261011, 20261012, 20261013]
warnings.filterwarnings('ignore', message='Unrecognized options detected:.*', category=RuntimeWarning)


def solve(c, a, lb, ub, seed=0):
    """CBC 2.10.3 with common settings, through the binary bundled by PuLP 3.3.0."""
    executable = pulp.PULP_CBC_CMD().path
    a = a.tocsr()
    def expression(indices, values):
        return ' '.join(f'{float(v):+.16g} x{int(i)}' for i, v in zip(indices, values) if v) or '0 x0'
    lines = ['Minimize', ' obj: ' + expression(range(len(c)), c), 'Subject To']
    for r in range(a.shape[0]):
        expr = expression(a.indices[a.indptr[r]:a.indptr[r+1]], a.data[a.indptr[r]:a.indptr[r+1]])
        if math.isfinite(lb[r]):
            lines.append(f' l{r}: {expr} >= {float(lb[r]):.16g}')
        if math.isfinite(ub[r]):
            lines.append(f' u{r}: {expr} <= {float(ub[r]):.16g}')
    lines += ['Binary'] + [f' x{i}' for i in range(len(c))] + ['End']
    with tempfile.TemporaryDirectory(prefix='roman12-') as directory:
        folder = Path(directory)
        model, solution = folder/'model.lp', folder/'solution.sol'
        model.write_text('\n'.join(lines), encoding='ascii')
        result = subprocess.run([executable, str(model), '-seconds', str(LIMIT),
                                 '-threads', '0', '-ratio', '0', '-allowableGap', '0',
                                 '-randomSeed', str(seed), '-randomCbcSeed', str(seed),
                                 '-timeMode', 'elapsed', '-solve', '-solu', str(solution)],
                                capture_output=True, text=True, cwd=directory,
                                timeout=LIMIT+20,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if result.returncode or not solution.exists():
            raise RuntimeError(f'CBC failed with code {result.returncode}.')
        sol = solution.read_text(encoding='utf-8').splitlines()
    optimal = sol[0].startswith('Optimal')
    assert optimal or sol[0].startswith('Stopped on time'), sol[0]
    x = None
    if 'objective value' in sol[0] and 'no integer solution' not in sol[0].lower():
        x = np.zeros(len(c))
        for line in sol[1:]:
            tokens = line.split()
            if len(tokens) >= 3 and re.fullmatch(r'x\d+', tokens[1]):
                x[int(tokens[1][1:])] = float(tokens[2])
        assert np.max(np.abs(x-np.round(x))) < 1e-5
    objective = float(np.dot(c,x)) if x is not None else None
    match = re.search(r'Lower bound:\s*([-+\d.eE]+)', result.stdout)
    # CBC prints this summary to three decimals. Subtract .001 before rounding
    # the bound up to an integer, to avoid overstating it through log rounding.
    lower = objective if optimal else float(match[1])-.001 if match else 0
    return SimpleNamespace(x=x, status=0 if optimal else 1,
                           message=sol[0], mip_dual_bound=lower,
                           mip_gap=(objective-lower)/objective if objective else None)


def generate(family, p, seed):
    rng = random.Random(seed)
    if family == 'bipartite':
        h = nx.bipartite.random_graph(p//2, p//2, min(1, 12/p), seed=seed)
        copies = 3
    elif family in ('cubic_single', 'cubic_double'):
        h = nx.random_regular_graph(3, p, seed=seed)
        copies = 1 if family == 'cubic_single' else 2
    elif family == 'general_pairs':
        h = nx.gnp_random_graph(p, min(1, 6/(p-1)), seed=seed)
        copies = 3
    elif family == 'matching_stress':
        h = nx.Graph()
        h.add_nodes_from(range(p))
        h.add_edges_from((v, v+1) for v in range(0, p, 2))
        copies = 8
    else:
        raise ValueError(family)
    edges = sorted((min(u, v), max(u, v)) for u, v in h.edges())
    neighborhoods = [((1 << u) | (1 << v)) for u, v in edges for _ in range(copies)]
    return neighborhoods


def preprocess(p, tests):
    q = sum(bool(e) for e in tests)
    z = len(tests)-q
    b = [0] * p
    counts = {}
    forced = 0
    for e in tests:
        if not e:
            continue
        verts = []
        remaining = e
        while remaining:
            bit = remaining & -remaining
            verts.append(bit.bit_length()-1)
            remaining ^= bit
        for v in verts:
            b[v] += 1
        if len(verts) == 1:
            forced |= e
        else:
            pair = tuple(verts)
            counts[pair] = counts.get(pair, 0) + 1
    v = max(range(p), key=lambda x: b[x])
    one = (z + 2 + q - b[v], 1 << v)
    two = min((z + 4 + q - b[u] - b[v] + counts.get((u, v), 0), (1 << u) | (1 << v))
              for u in range(p) for v in range(u+1, p))
    h = nx.Graph()
    h.add_nodes_from(v for v in range(p) if not forced >> v & 1)
    h.add_edges_from((u, v) for u, v in counts if not forced & ((1 << u) | (1 << v)))
    return z, b, forced, h, min(one, two)


def fast_approx(p, tests):
    z, b, forced, h, small = preprocess(p, tests)
    matching = nx.maximal_matching(h)
    cover = forced | sum(1 << v for e in matching for v in e)
    for v in range(p):
        if cover.bit_count() >= 3:
            break
        cover |= 1 << v
    value, support = min(small, (z+p+cover.bit_count(), cover))
    # A certified lower bound from the disjoint edges of the matching.
    lb = min(small[0], z+p+max(3, forced.bit_count()+len(matching)))
    baseline = min(small[0], z+2*p)
    return value, support, lb, baseline


def structural(p, tests, seed):
    start = time.perf_counter()
    z, b, forced, h, small = preprocess(p, tests)
    if all(not e or e.bit_count() == 2 for e in tests) and max(b, default=0) <= 3:
        val, support = small
        return {'value': val, 'lower': val, 'support': support, 'route': 'cross3',
                'seconds': time.perf_counter()-start, 'certified': True}
    if nx.is_bipartite(h):
        colors = nx.bipartite.color(h)
        left = {v for v in h if colors[v] == 0}
        matching = nx.bipartite.maximum_matching(h, top_nodes=left)
        cover_set = nx.bipartite.to_vertex_cover(h, matching, top_nodes=left)
        low_tau = len(cover_set)
        route = 'matching'
    else:
        vertices = sorted(h)
        index = {v: i for i, v in enumerate(vertices)}
        rr, cc = [], []
        for j, (u, v) in enumerate(h.edges()):
            rr.extend((j, j)); cc.extend((index[u], index[v]))
        mat = coo_matrix((np.ones(len(rr)), (rr, cc)), shape=(len(h.edges()), len(vertices))).tocsc()
        result = solve(np.ones(len(vertices)), mat, np.ones(len(h.edges())), np.full(len(h.edges()), np.inf), seed)
        if result.x is None:
            cover_set = set(v for e in nx.maximal_matching(h) for v in e)
        else:
            cover_set = {v for v, x in zip(vertices, result.x) if x > .5}
        assert all(u in cover_set or v in cover_set for u, v in h.edges())
        dual = getattr(result, 'mip_dual_bound', None)
        low_tau = max(len(nx.maximal_matching(h)), math.ceil(dual-1e-6) if dual is not None and math.isfinite(dual) else 0)
        route = 'cover_milp'
    cover = forced | sum(1 << v for v in cover_set)
    for v in range(p):
        if cover.bit_count() >= 3:
            break
        cover |= 1 << v
    value, support = min(small, (z+p+cover.bit_count(), cover))
    low = min(small[0], z+p+max(3, forced.bit_count()+low_tau))
    assert low <= value
    return {'value': value, 'lower': low, 'support': support, 'route': route,
            'seconds': time.perf_counter()-start, 'certified': low == value}


def full_milp(adj, seed):
    start = time.perf_counter()
    n = len(adj)
    rr, cc, dd = [], [], []
    lower = np.full(3*n, -np.inf)
    upper = np.full(3*n, np.inf)
    for v in range(n):
        d = adj[v].bit_count()
        for row, val in ((3*v, 1), (3*v+1, 1), (3*v+2, -max(0, d-2))):
            rr.extend((row, row)); cc.extend((v, n+v)); dd.extend((val, val))
        upper[3*v] = 1
        lower[3*v+1] = 1
        upper[3*v+2] = 2
        mask = adj[v]
        while mask:
            bit = mask & -mask
            u = bit.bit_length()-1
            rr.extend((3*v+1, 3*v+2)); cc.extend((n+u, n+u)); dd.extend((1, 1))
            mask ^= bit
    matrix = coo_matrix((dd, (rr, cc)), shape=(3*n, 2*n)).tocsc()
    result = solve([1]*n+[2]*n, matrix, lower, upper, seed)
    elapsed = time.perf_counter()-start
    value = None
    if result.x is not None:
        f = [int(round(result.x[v]))+2*int(round(result.x[n+v])) for v in range(n)]
        assert valid(adj, f)
        value = sum(f)
    dual = getattr(result, 'mip_dual_bound', None)
    lower = math.ceil(dual-1e-6) if dual is not None and math.isfinite(dual) else 0
    return {'value': value, 'lower': lower, 'seconds': elapsed,
            'certified': result.status == 0, 'status': int(result.status),
            'message': str(result.message), 'solver_gap': getattr(result, 'mip_gap', None)}


def hardware():
    info = {'python': platform.python_version(), 'platform': platform.platform(),
            'processor': platform.processor(), 'numpy': np.__version__,
            'scipy': scipy.__version__, 'networkx': nx.__version__}
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            info['cpu_name'] = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
    except (ImportError, OSError):
        pass
    return info


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    metadata = {'hardware': hardware(), 'solver_time_limit_seconds': LIMIT,
                'solver_threads': 'serial (CBC threads=0)', 'solver': 'CBC 2.10.3 bundled with PuLP 3.3.0',
                'upper_constraint_M': 'max(0, degree-2)',
                'sizes': SIZES, 'seeds': SEEDS,
                'fast_timing_repetitions': 20,
                'timing_scope': 'instance generation excluded; preprocessing and model construction included',
                'families': FAMILIES, 'instances': [], 'results': []}
    path = HERE / 'split-benchmark-cbc.json'
    if path.exists():
        previous = json.loads(path.read_text(encoding='utf-8'))
        metadata['instances'] = previous['instances']
        metadata['results'] = previous['results']
    # Warm up the solver once outside measured runs.
    solve([1], coo_matrix([[1.]]).tocsc(), [1], [np.inf])
    jobs = [(family, p, seed) for family in FAMILIES for p in SIZES for seed in SEEDS]
    jobs += [('matching_stress', p, SEEDS[0]) for p in SIZES]
    done = {row['id'] for row in metadata['results']}
    for i, (family, p, seed) in enumerate(jobs):
        ident = f'{family}_p{p}_s{seed}'
        if ident in done:
            continue
        print(json.dumps({'starting': ident}), flush=True)
        tests = generate(family, p, seed)
        adj = graph_masks(p, tests)
        metadata['instances'].append({'id': ident, 'family': family, 'p': p, 'seed': seed, 'neighborhood_masks': tests})
        timings = []
        for _ in range(20):
            start = time.perf_counter()
            approx, support, approx_lb, baseline = fast_approx(p, tests)
            timings.append(time.perf_counter()-start)
        f = canonical(adj, support)
        assert valid(adj, f) and sum(f) == approx
        reduced = structural(p, tests, seed)
        f = canonical(adj, reduced['support'])
        assert valid(adj, f) and sum(f) == reduced['value']
        full = full_milp(adj, seed)
        lb = max(approx_lb, reduced['lower'], full['lower'])
        ub = min(approx, reduced['value'], full['value'] if full['value'] is not None else len(adj))
        assert lb <= ub and 3*approx <= 4*lb
        if full['certified']:
            assert full['value'] == full['lower'] and full['value'] == lb
        if reduced['certified']:
            assert reduced['value'] == lb
        row = {'id': ident, 'family': family, 'p': p, 'N': len(adj), 'seed': seed,
               'approx_value': approx, 'approx_seconds': statistics.median(timings),
               'baseline_value': baseline, 'lower': lb, 'upper': ub, 'optimum_certified': lb == ub,
               'approx_ratio_or_certificate': approx/lb, 'baseline_ratio_or_certificate': baseline/lb,
               'structural_value': reduced['value'], 'structural_lower': reduced['lower'],
               'structural_route': reduced['route'], 'structural_seconds': reduced['seconds'],
               'structural_certified': reduced['certified'],
               'full_value': full['value'], 'full_lower': full['lower'],
               'full_seconds': full['seconds'], 'full_certified': full['certified'],
               'full_status': full['status'], 'full_gap': full['solver_gap']}
        metadata['results'].append(row)
        path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        with (HERE / 'split-benchmark-cbc.csv').open('w', newline='', encoding='utf-8') as out:
            writer = csv.DictWriter(out, fieldnames=list(row))
            writer.writeheader(); writer.writerows(metadata['results'])
        print(json.dumps({'progress': f'{i+1}/{len(jobs)}', 'id': ident, 'N': len(adj),
                          'full_s': round(full['seconds'], 3), 'full_opt': full['certified'],
                          'structural_s': round(reduced['seconds'], 4), 'structural_opt': reduced['certified'],
                          'approx_ratio': round(approx/lb, 4)}), flush=True)
    metadata['complete'] = True
    metadata['timing_note'] = 'Sequential runs; MILP timings include model construction. The CLI startup and model/solution file I/O are included for both MILPs.'
    path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    with (HERE / 'split-benchmark-cbc.csv').open('w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=list(metadata['results'][0]))
        writer.writeheader(); writer.writerows(metadata['results'])


if __name__ == '__main__':
    main()
