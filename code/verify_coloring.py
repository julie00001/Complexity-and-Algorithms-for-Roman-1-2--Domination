"""Independent finite checks for the proposed direct 3-coloring reduction."""
import itertools, json, sys, time
from pathlib import Path
import networkx as nx

from output_paths import output_directory

HERE = output_directory()

def color3(h):
    order = sorted(h, key=lambda v: -h.degree[v])
    coloring = {}
    def go(i):
        if i == len(order):
            return dict(coloring)
        v = order[i]
        forbidden = {coloring[u] for u in h[v] if u in coloring}
        for c in range(3):
            if c not in forbidden:
                coloring[v] = c
                ans = go(i+1)
                if ans is not None:
                    return ans
                del coloring[v]
        return None
    return go(0)

def edge_ok(a,b):
    return all(1 <= (a & ~(1<<c)).bit_count() + (b & ~(1<<c)).bit_count() <= 2 for c in range(3))

def build(h, mult, split=True):
    g = nx.Graph()
    clique = [(v,c) for v in h for c in range(3)]
    g.add_nodes_from(clique)
    if split:
        g.add_edges_from(itertools.combinations(clique,2))
    groups = []
    for v in h:
        groups.append(('A',v,[(v,c) for c in range(3)]))
    for u,v in h.edges:
        for c in range(3):
            groups.append(('B',(u,v,c),[(z,d) for z in (u,v) for d in range(3) if d != c]))
    for typ,key,neighbors in groups:
        for j in range(mult):
            t = (typ,key,j)
            g.add_node(t)
            g.add_edges_from((t,x) for x in neighbors)
    return g,clique

def feasible(g, f):
    return all(f[v] != 0 or 1 <= sum(f[u] == 2 for u in g[v]) <= 2 for v in g)

def main():
    HERE.mkdir(parents=True, exist_ok=True)
    start=time.monotonic()
    masks=[i for i in range(8) if 1<=i.bit_count()<=2]
    truth=[]
    for a,b in itertools.product(masks,repeat=2):
        ok=edge_ok(a,b)
        assert ok == (a & b == 0), (a,b)
        truth.append({'a':a,'b':b,'accepted':ok})
    atlas=[h for h in nx.graph_atlas_g() if 3<=len(h)<=7]
    counts={}
    checked_forward=0
    for h in atlas:
        n=len(h)
        col=color3(h)
        counts.setdefault(n,{'total':0,'colorable':0})
        counts[n]['total']+=1
        counts[n]['colorable']+=col is not None
        if col is not None:
            # M=1 is enough to validate adjacency and the forward assignment.
            # The gap theorem itself requires M>4n.
            g,c=build(h,1)
            f={x:0 for x in g}
            f.update({(v,d):(2 if col[v]==d else 1) for v in h for d in range(3)})
            assert feasible(g,f)
            assert sum(f.values())==4*n
            checked_forward+=1
    # Exhaustive assignments of 0..3 selected colors per source vertex for n<=5.
    # This is a constraint check, not enumeration of all functions on G_M(H).
    selector_vectors=0
    selector_graphs=0
    accepted_vectors=0
    for h in atlas:
        if len(h)>5: continue
        n=len(h); found=False
        for s in itertools.product(range(8),repeat=n):
            selector_vectors+=1
            if not all(1<=a.bit_count()<=2 for a in s): continue
            if not all(edge_ok(s[u],s[v]) for u,v in h.edges): continue
            found=True;accepted_vectors+=1
            assert all(s[u]&s[v]==0 for u,v in h.edges)
            chosen=[(a & -a).bit_length()-1 for a in s]
            assert all(chosen[u]!=chosen[v] for u,v in h.edges)
            assert 3*n+sum(a.bit_count() for a in s)>=4*n
        assert found == (color3(h) is not None)
        selector_graphs+=1
    # Full binary MILP: no prescribed labels, no twin normalization assumption.
    # x_v=1 means f(v)=1; y_v=1 means f(v)=2.
    milp_results=[]
    if '--milp' in sys.argv:
        import numpy as np
        from scipy.optimize import milp, Bounds, LinearConstraint
        from scipy.sparse import lil_matrix
        examples=[('empty3',nx.empty_graph(3)),('triangle',nx.complete_graph(3)),
                  ('cycle5',nx.cycle_graph(5)),('K4',nx.complete_graph(4)),
                  ('wheel6',nx.wheel_graph(6)),('K5',nx.complete_graph(5))]
        if '--unit' in sys.argv:
            examples=[('two_K33',nx.disjoint_union(nx.complete_bipartite_graph(3,3),nx.complete_bipartite_graph(3,3))),
                      ('K4_plus_K33',nx.disjoint_union(nx.complete_graph(4),nx.complete_bipartite_graph(3,3))),
                      ('K5_plus_K33',nx.disjoint_union(nx.complete_graph(5),nx.complete_bipartite_graph(3,3))),
                      ('square_C12',nx.power(nx.cycle_graph(12),2)),
                      ('square_C10',nx.power(nx.cycle_graph(10),2)),
                      ('rook33_plus_K33',nx.disjoint_union(nx.convert_node_labels_to_integers(nx.line_graph(nx.complete_bipartite_graph(3,3))),nx.complete_bipartite_graph(3,3)))]
        for name,h in examples:
            n=len(h);mult=1 if '--unit' in sys.argv else 4*n+1
            split='--bipartite' not in sys.argv
            g,clique=build(h,mult,split)
            if '--unit' in sys.argv:
                assert n>=10 and min(dict(h.degree()).values())>=3 and max(dict(h.degree()).values())<=4
                indep=set(g)-set(clique)
                assert all(g.degree[v] in (3,4) for v in indep)
                assert all(len(set(g[v]) & indep)<=9 for v in clique)
                assert len({frozenset(g[v]) for v in g})==len(g) # no false twins
                assert len({frozenset(set(g[v])|{v}) for v in g})==len(g) # no true twins
            verts=list(g);idx={v:i for i,v in enumerate(verts)};N=len(verts)
            a=lil_matrix((3*N+1,2*N),dtype=float)
            lo=np.full(3*N+1,-np.inf);hi=np.full(3*N+1,np.inf)
            for v,i in idx.items():
                a[3*i,i]=a[3*i,N+i]=1;hi[3*i]=1
                a[3*i+1,i]=a[3*i+1,N+i]=1;lo[3*i+1]=1
                d=g.degree[v]
                a[3*i+2,i]=a[3*i+2,N+i]=-d;hi[3*i+2]=2
                for u in g[v]:
                    a[3*i+1,N+idx[u]]=1
                    a[3*i+2,N+idx[u]]=1
            obj=np.r_[np.ones(N),np.full(N,2.)]
            a[-1,:]=obj;hi[-1]=4*n
            res=milp(obj,integrality=np.ones(2*N),bounds=Bounds(0,1),
                     constraints=LinearConstraint(a.tocsc(),lo,hi),
                     options={'time_limit':45,'mip_rel_gap':0.0})
            record={'source':name,'n':n,'m':h.number_of_edges(),'M':mult,'N':N,
                    'status':int(res.status),'message':str(res.message)}
            if res.status==0:
                labels=np.rint(res.x[:N]+2*res.x[N:]).astype(int)
                assert feasible(g,dict(zip(verts,labels)))
                assert int(sum(labels))<=4*n
                if split: assert int(sum(labels))==4*n
                assert color3(h) is not None
                record['optimum']=int(sum(labels))
            elif res.status==2:
                assert color3(h) is None
                record['infeasible_at_threshold']=4*n
            else:
                record['unverified']=True
            milp_results.append(record)
            print(json.dumps(record),flush=True)
    structure=[]
    if '--bipartite' in sys.argv:
        for name,h in [('path3',nx.path_graph(3)),('triangle',nx.complete_graph(3)),('K4',nx.complete_graph(4))]:
            g,c=build(h,3,False)
            assert nx.is_bipartite(g)
            assert min(dict(g.degree()).values())==3
            assert max(nx.core_number(g).values())<=4
            kappa=nx.node_connectivity(g)
            assert kappa==3
            structure.append({'source':name,'M':3,'N':len(g),'connectivity':kappa,'degeneracy':max(nx.core_number(g).values())})
    report={'variant':'bipartite' if '--bipartite' in sys.argv else 'split',
            'local_pairs':len(truth),'accepted_local_pairs':sum(t['accepted'] for t in truth),
            'atlas':counts,'forward_assignments':checked_forward,
            'selector_graphs':selector_graphs,'selector_vectors':selector_vectors,
            'accepted_selector_vectors':accepted_vectors,'milp':milp_results,
            'structure':structure,'seconds':round(time.monotonic()-start,2)}
    filename='coloring-checks-unit.json' if '--unit' in sys.argv else ('coloring-checks-bipartite.json' if '--bipartite' in sys.argv else 'coloring-checks.json')
    (HERE/filename).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
