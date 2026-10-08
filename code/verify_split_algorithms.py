"""Exhaustive checks of the split-graph formula against an independent oracle."""
import itertools, json, random, time
from pathlib import Path
import networkx as nx

from output_paths import output_directory

HERE = output_directory()

def graph_masks(p,neighborhoods):
    n=p+len(neighborhoods)
    adj=[((1<<p)-1) ^ (1<<v) for v in range(p)] + list(neighborhoods)
    for j,b in enumerate(neighborhoods,p):
        for v in range(p):
            if b>>v&1: adj[v]|=1<<j
    return adj

def oracle(adj):
    # Enumerates only the vertices assigned 2. All remaining labels are then
    # optimized directly from the definition, with no split-graph assumption.
    n=len(adj);best=n;best_s=0
    for s in range(1<<n):
        val=2*s.bit_count()
        if val>=best:continue
        for v in range(n):
            if not (s>>v&1) and (adj[v]&s).bit_count() not in (1,2):val+=1
        if val<best:best,best_s=val,s
    return best,best_s

def canonical(adj,s):
    return [2 if (s>>v)&1 else (0 if (adj[v]&s).bit_count() in (1,2) else 1) for v in range(len(adj))]

def valid(adj,f):
    s=sum(1<<v for v,x in enumerate(f) if x==2)
    return all(x or (adj[v]&s).bit_count() in (1,2) for v,x in enumerate(f))

def formula(p,neighborhoods):
    zero=neighborhoods.count(0)
    tests=[b for b in neighborhoods if b]
    q=len(tests)
    bests1=max((sum(bool(b&(1<<v)) for b in tests),1<<v) for v in range(p))
    bests2=max((sum(bool(b&s) for b in tests),s) for s in ((1<<u)|(1<<v) for u,v in itertools.combinations(range(p),2)))
    d1,s1=bests1;d2,s2=bests2
    covers=[s for s in range(1<<p) if all(b&s for b in tests)]
    cover=min(covers,key=int.bit_count);tau=cover.bit_count()
    s3=cover
    for v in range(p):
        if s3.bit_count()>=3:break
        s3|=1<<v
    values=[zero+2+q-d1,zero+4+q-d2,zero+p+max(3,tau)]
    which=min(range(3),key=values.__getitem__)
    adj=graph_masks(p,neighborhoods)
    f=canonical(adj,[s1,s2,s3][which])
    assert valid(adj,f) and sum(f)==values[which]
    # Polynomial matching algorithm, when the residual pair graph is bipartite.
    forced=0
    for b in tests:
        if b.bit_count()==1:forced|=b
    residual=nx.Graph()
    residual.add_nodes_from(v for v in range(p) if not forced>>v&1)
    for b in tests:
        if b.bit_count()==2 and not b&forced:
            residual.add_edge(*(v for v in range(p) if b>>v&1))
    matching_checked=False
    if nx.is_bipartite(residual):
        sides=nx.bipartite.color(residual)
        left={v for v in residual if sides[v]==0}
        matching=nx.bipartite.maximum_matching(residual,top_nodes=left)
        kcover=nx.bipartite.to_vertex_cover(residual,matching,top_nodes=left)
        full=forced|sum(1<<v for v in kcover)
        assert full.bit_count()==tau
        assert all(b&full for b in tests)
        matching_checked=True
    cross3=False
    if all(b.bit_count()==2 for b in neighborhoods):
        delta=max((sum(bool(b&(1<<v)) for b in neighborhoods) for v in range(p)),default=0)
        if delta<=3:
            assert min(values[:2])==min(values)
            cross3=True
    approx=min(values[0],values[1],zero+2*p)
    assert min(values)<=approx<=2*min(values)
    return min(values),matching_checked,cross3

def main():
    HERE.mkdir(parents=True, exist_ok=True)
    start=time.monotonic();count=0;matching_count=0;cross_count=0;branches=[0,0,0]
    by_p={}
    for p in (3,4,5):
        options=[s for s in range(1<<p) if s.bit_count()<=2]
        subtotal=0
        for q in range(5):
            for tests in itertools.combinations_with_replacement(options,q):
                neighborhoods=list(tests)
                predicted,mc,cc=formula(p,neighborhoods)
                actual,s=oracle(graph_masks(p,neighborhoods))
                assert actual==predicted,(p,tests,predicted,actual,s)
                count+=1;subtotal+=1;matching_count+=mc;cross_count+=cc
        by_p[p]=subtotal
        print(json.dumps({'p':p,'cases':subtotal,'cumulative':count}),flush=True)
    # Larger examples probe branches obscured by tiny-instance optima.
    rng=random.Random(20261007);larger=[]
    for p,q in [(5,9),(6,8),(7,7),(8,6)]:
        options=[s for s in range(1<<p) if s.bit_count()<=2]
        for _ in range(8):
            tests=[rng.choice(options) for _ in range(q)]
            val,mc,cc=formula(p,tests)
            actual,_=oracle(graph_masks(p,tests))
            assert val==actual
            larger.append({'p':p,'q':q,'optimum':val})
    special=[]
    for name,p,tests in [
        ('three_disjoint_pairs',6,[3,12,48]*6),
        ('three_private_leaves_each',5,[1<<v for v in range(5) for _ in range(3)]),
        ('triangle_auxiliary',6,[3,5,6]*4),
        ('bipartite_auxiliary_K33',6,[(1<<u)|(1<<v) for u in range(3) for v in range(3,6)]),
        ('all_clique_no_tests',5,[]),
    ]:
        val,mc,cc=formula(p,tests)
        actual,_=oracle(graph_masks(p,tests))
        assert val==actual
        special.append({'name':name,'p':p,'q':len(tests),'optimum':val,'matching_applicable':mc,'cross3_applicable':cc})
    small_count=0
    for p in (0,1,2):
        options=list(range(1<<p))
        for q in range(6):
            for tests in itertools.combinations_with_replacement(options,q):
                adj=graph_masks(p,list(tests))
                val=min([len(adj)]+[sum(canonical(adj,s)) for s in range(1,1<<p)])
                actual,_=oracle(adj)
                assert val==actual,(p,tests,val,actual)
                small_count+=1
    report={'exhaustive_cases':count,'small_clique_cases':small_count,'cases_by_clique_order':by_p,
            'matching_cases':matching_count,'exact_two_cross3_cases':cross_count,
            'larger_random_cases':len(larger),'special_cases':special,
            'all_passed':True,'seconds':round(time.monotonic()-start,2)}
    (HERE/'split-algorithm-checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
