"""Finite checks for approximation transfer and the cubic-cover L-reduction.

Dependencies: networkx, numpy, scipy.  Run from any working directory.
The mathematical proofs are in the manuscript; these checks are independent
consistency tests, not performance benchmarks or substitutes for proofs.
"""
import itertools
import json
import random
import time
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
from verify_split_algorithms import graph_masks, canonical, valid, oracle

from output_paths import output_directory

HERE = output_directory()


def auxiliary(p, tests):
    forced = 0
    for mask in tests:
        if mask.bit_count() == 1:
            forced |= mask
    edges = sorted(set(mask for mask in tests if mask.bit_count() == 2 and not mask & forced))
    return forced, edges


def vc_exact(p, edges):
    return min((s for s in range(1 << p) if all(s & e for e in edges)), key=int.bit_count)


def small_candidates(p, tests, adj):
    sets = [1 << v for v in range(p)]
    sets += [(1 << u) | (1 << v) for u, v in itertools.combinations(range(p), 2)]
    return min((canonical(adj, s) for s in sets), key=sum)


def candidate_from_cover(p, adj, forced, cover):
    s = forced | cover
    for v in range(p):
        if s.bit_count() >= 3:
            break
        s |= 1 << v
    return canonical(adj, s)


def transfer_checks():
    cases = covers_checked = third_optimal = 0
    worst = (1.0, None)
    for p in (3, 4, 5):
        options = [s for s in range(1 << p) if s.bit_count() <= 2]
        for q in range(5):
            for tpl in itertools.combinations_with_replacement(options, q):
                tests = list(tpl)
                adj = graph_masks(p, tests)
                optimum, _ = oracle(adj)
                forced, edges = auxiliary(p, tests)
                tau = vc_exact(p, edges).bit_count()
                small = small_candidates(p, tests, adj)
                allowed = ((1 << p) - 1) ^ forced
                cover = 0
                for e in edges:
                    if not e & cover:
                        cover |= e
                f = min([small, candidate_from_cover(p, adj, forced, cover)], key=sum)
                assert valid(adj, f) and 3 * sum(f) <= 4 * optimum
                ratio = sum(f) / optimum
                if ratio > worst[0]:
                    worst = (ratio, {'p': p, 'neighborhoods': tests})
                # Test the general transfer with every auxiliary vertex cover,
                # using its actual ratio alpha, not just maximal matching.
                for cover in range(1 << p):
                    if cover & ~allowed or not all(cover & e for e in edges):
                        continue
                    if tau == 0 and cover:
                        continue  # A multiplicative approximation must return 0.
                    numerator = max(tau, cover.bit_count())
                    denominator = tau or 1
                    if tau == 0:
                        numerator = 1
                    f = min([small, candidate_from_cover(p, adj, forced, cover)], key=sum)
                    assert valid(adj, f)
                    assert (numerator + denominator) * sum(f) <= 2 * numerator * optimum
                    covers_checked += 1
                cases += 1
                third_optimal += sum(small) > optimum
        print(json.dumps({'transfer_p': p, 'cumulative_cases': cases}), flush=True)
    # An infinite family attains 4/3 for the unpruned matching-endpoint rule.
    # Six vertices and six copies on each of three disjoint auxiliary edges.
    p = 6
    tests = [3, 12, 48] * 6
    adj = graph_masks(p, tests)
    small = small_candidates(p, tests, adj)
    full = candidate_from_cover(p, adj, 0, (1 << p) - 1)
    exact = sum(canonical(adj, 1 | 4 | 16))
    assert exact == 9 and min(sum(small), sum(full)) == 10
    # Larger multiplicity suppresses the pair candidate and attains 4/3.
    tests = [3, 12, 48] * 8
    adj = graph_masks(p, tests)
    result = min(sum(small_candidates(p, tests, adj)), sum(candidate_from_cover(p, adj, 0, 63)))
    assert result == 12 and sum(canonical(adj, 21)) == 9
    # Dense neighborhood multisets exercise the third branch, which is never
    # strictly optimal in the small exhaustive family above.
    rng = random.Random(20261008)
    large_cases = large_third = large_covers = large_milp = 0
    for p in range(6, 11):
        pair_options = [(1 << u) | (1 << v) for u, v in itertools.combinations(range(p), 2)]
        for index in range(12):
            tests = [rng.choice(pair_options) for _ in range(rng.randint(3*p, 7*p))]
            tests += [0] * rng.randrange(4)
            if index % 3 == 0:
                tests += [1 << rng.randrange(p)] * rng.randrange(1, 4)
            adj = graph_masks(p, tests)
            forced, edges = auxiliary(p, tests)
            tau_cover = vc_exact(p, edges)
            tau = tau_cover.bit_count()
            small = small_candidates(p, tests, adj)
            exact_f = min([small, candidate_from_cover(p, adj, forced, tau_cover)], key=sum)
            optimum = sum(exact_f)
            assert valid(adj, exact_f)
            if index < 2:
                val, _ = ip_optimum(adj)
                assert val == optimum
                large_milp += 1
            large_third += sum(small) > optimum
            allowed = ((1 << p) - 1) ^ forced
            matched = 0
            for e in edges:
                if not e & matched:
                    matched |= e
            f = min([small, candidate_from_cover(p, adj, forced, matched)], key=sum)
            assert valid(adj, f) and 3 * sum(f) <= 4 * optimum
            if sum(f) / optimum > worst[0]:
                worst = (sum(f) / optimum, {'p': p, 'neighborhoods': tests})
            for cover in range(1 << p):
                if cover & ~allowed or not all(cover & e for e in edges) or (tau == 0 and cover):
                    continue
                num, den = (cover.bit_count(), tau) if tau else (1, 1)
                f = min([small, candidate_from_cover(p, adj, forced, cover)], key=sum)
                assert valid(adj, f) and (num + den) * sum(f) <= 2 * num * optimum
                large_covers += 1
            large_cases += 1
    return {'exhaustive_inputs': cases, 'all_cover_transfer_checks': covers_checked,
            'third_branch_strictly_optimal_inputs': third_optimal,
            'additional_dense_inputs': large_cases, 'dense_third_branch_inputs': large_third,
            'dense_transfer_checks': large_covers, 'dense_independent_milp_checks': large_milp,
            'worst_sampled_ratio': worst, 'tight_matching_example': {'p': 6, 'q': 24, 'algorithm': 12, 'optimum': 9}}


def ip_optimum(adj):
    n = len(adj)
    a = lil_matrix((3 * n, 2 * n), dtype=float)
    lower = np.full(3 * n, -np.inf)
    upper = np.full(3 * n, np.inf)
    for v in range(n):
        # Binary x_v and y_v denote labels 1 and 2.
        a[3*v, v] = a[3*v, n+v] = 1
        upper[3*v] = 1
        a[3*v+1, v] = a[3*v+1, n+v] = 1
        lower[3*v+1] = 1
        d = adj[v].bit_count()
        a[3*v+2, v] = a[3*v+2, n+v] = -d
        upper[3*v+2] = 2
        for u in range(n):
            if adj[v] >> u & 1:
                a[3*v+1, n+u] = 1
                a[3*v+2, n+u] = 1
    costs = np.array([1.] * n + [2.] * n)
    sol = milp(costs, integrality=np.ones(2*n), bounds=Bounds(0, 1),
               constraints=LinearConstraint(a.tocsr(), lower, upper),
               options={'time_limit': 45, 'mip_rel_gap': 0})
    assert sol.success, sol.message
    f = [int(round(sol.x[v])) + 2 * int(round(sol.x[n+v])) for v in range(n)]
    assert valid(adj, f) and abs(sum(f) - sol.fun) < 1e-6
    return sum(f), f


def decode(p, edges, adj, f):
    assert valid(adj, f)
    w = sum(f)
    if w >= 2 * p:
        return (1 << p) - 1, 'large_weight'
    s = sum(1 << v for v in range(p) if f[v] == 2)
    assert s.bit_count() >= 3
    normalized = canonical(adj, s)
    assert valid(adj, normalized) and sum(normalized) <= w
    cover = s
    for e in edges:
        if not e & cover:
            cover |= e & -e
    assert all(cover & e for e in edges)
    assert cover.bit_count() <= w - p
    return cover, 'normalized'


def apx_checks():
    rng = random.Random(20261007)
    instances = [
        ('cube', nx.cubical_graph()),
        ('two_K4', nx.disjoint_union(nx.complete_graph(4), nx.complete_graph(4))),
        ('Petersen', nx.petersen_graph()),
        ('pentagonal_prism', nx.circular_ladder_graph(5)),
        ('K33_plus_K4', nx.disjoint_union(nx.complete_bipartite_graph(3, 3), nx.complete_graph(4))),
        ('Frucht', nx.frucht_graph()),
        ('hexagonal_prism', nx.circular_ladder_graph(6)),
        ('random_14', nx.random_regular_graph(3, 14, seed=20261007)),
        ('random_16', nx.random_regular_graph(3, 16, seed=20261008)),
    ]
    results = []
    total_decode = 0
    branches = {'large_weight': 0, 'normalized': 0}
    for name, source in instances:
        source = nx.convert_node_labels_to_integers(source)
        p = len(source)
        assert p >= 8 and all(source.degree[v] == 3 for v in source)
        edges = sorted((1 << u) | (1 << v) for u, v in source.edges())
        tests = [e for e in edges for _ in range(2)]
        adj = graph_masks(p, tests)
        tau_mask = vc_exact(p, edges)
        tau = tau_mask.bit_count()
        optimum, optimum_f = ip_optimum(adj)
        assert len(adj) == 4 * p and optimum == p + tau and optimum <= 3 * tau
        assert all(sum(bool((1 << v) & e) for e in tests) == 6 for v in range(p))
        samples = [optimum_f]
        # Exhaust every clique support set (including noncovers).
        if p <= 12:
            samples.extend(canonical(adj, s) for s in range(1 << p))
        # Also test sets that include independent vertices labeled 2.
        for _ in range(512):
            s = rng.getrandbits(len(adj))
            f = canonical(adj, s)
            f = [1 if x == 0 and rng.random() < .25 else x for x in f]
            samples.append(f)
        # Positive changes at independent vertices probe non-normal low-weight solutions.
        for x in range(p, len(adj)):
            f = optimum_f.copy()
            f[x] = 2
            if valid(adj, f):
                samples.append(f)
        for f in samples:
            cover, branch = decode(p, edges, adj, f)
            assert cover.bit_count() - tau <= sum(f) - optimum
            branches[branch] += 1
            total_decode += 1
        row = {'source': name, 'p': p, 'target_order': len(adj), 'tau': tau,
               'milp_optimum': optimum, 'decoder_checks': len(samples)}
        results.append(row)
        print(json.dumps(row), flush=True)
    return {'instances': results, 'decoder_checks': total_decode, 'decoder_branches': branches,
            'L_constants': [3, 1]}


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    report = {'transfer': transfer_checks(), 'apx': apx_checks(), 'all_passed': True}
    report['seconds'] = round(time.monotonic() - start, 2)
    (HERE / 'approximation-checks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
