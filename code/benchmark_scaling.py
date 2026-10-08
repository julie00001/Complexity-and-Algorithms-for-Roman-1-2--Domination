"""Larger exact-subclass instances; no general integer program is timed here."""
import json
import statistics
import time
from pathlib import Path
from benchmark_split_optimization import generate, fast_approx, structural, hardware
from verify_split_algorithms import graph_masks, canonical, valid

from output_paths import output_directory

HERE = output_directory()

def main():
    HERE.mkdir(parents=True, exist_ok=True)
    rows, instances = [], []
    data = {'hardware': hardware(), 'seed': 20261011, 'repetitions': 3,
            'timing_scope': 'preprocessing included; generation and output-label validation excluded',
            'rows': rows, 'instances': instances}
    for family in ['bipartite', 'cubic_single']:
        for p in [192, 384, 768, 1024]:
            tests = generate(family, p, data['seed'])
            timings, approx_times = [], []
            for _ in range(data['repetitions']):
                reduced = structural(p, tests, data['seed'])
                timings.append(reduced['seconds'])
                start = time.perf_counter()
                approx, support, _, _ = fast_approx(p, tests)
                approx_times.append(time.perf_counter()-start)
            adj = graph_masks(p, tests)
            f = canonical(adj, reduced['support'])
            assert reduced['certified'] and valid(adj, f) and sum(f) == reduced['value']
            f = canonical(adj, support)
            assert valid(adj,f) and sum(f)==approx and 3*approx <= 4*reduced['value']
            row = {'family': family, 'p': p, 'N': len(adj), 'optimum': reduced['value'],
                   'route': reduced['route'], 'structural_seconds': statistics.median(timings),
                   'approx_seconds': statistics.median(approx_times),
                   'approx_ratio': approx/reduced['value']}
            rows.append(row)
            instances.append({'family': family, 'p': p, 'neighborhood_masks':tests})
            (HERE/'split-scaling.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
            print(json.dumps(row),flush=True)
    data['complete']=True
    (HERE/'split-scaling.json').write_text(json.dumps(data,indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
