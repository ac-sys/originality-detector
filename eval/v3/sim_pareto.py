__file__='/workspace/detector/eval/v3/sim_search.py'; src=open(__file__).read().split('out = {"v2_dev"')[0]
exec(src)
import json
g=json.load(open('data/simeval/cache/grid.json'))
for lim in (0.0182,0.016,0.014,0.012,0.010):
    ok=[x for x in g if x[0][1]<=lim]
    if not ok: print(lim,"none"); continue
    b=max(ok,key=lambda x:x[0][0]); p=b[1]
    r={k:score(F[k],p) for k in ("orig_test","hard_test")}
    print(lim,b, {k:(v["f1"],v["clean_sentence_fpr"],v["recall_by_type"]) for k,v in r.items()})
