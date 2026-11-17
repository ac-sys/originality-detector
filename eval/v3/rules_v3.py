"""Compare doc-level decision rules (thresholds fit on calib humans + esl_calib only; nothing tuned on test/esl/academic).
-> eval/v3/rules_results.json"""
import sys, pathlib, json, numpy as np
sys.argv = ["x", "fk,dk", "fk", "tmp"]; B = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(B / "eval/v3"))
import importlib.util
spec = importlib.util.spec_from_file_location("f", B / "eval/v3/fit_v3.py")
src = open(B / "eval/v3/fit_v3.py").read().split("# ---------- window")[0]
g = {"__file__": str(B / "eval/v3/fit_v3.py"), "__name__": "x"}; exec(src.split("res = {")[0], g)
load_docs, thr_at, z, C, V2D = g["load_docs"], g["thr_at"], g["z"], g["C"], g["V2D"]
cal = load_docs("calib", ["fk", "dk"], 800); esc = load_docs("esl_calib", ["fk", "dk"])
pool = [d for d in cal if d[0]["label"] == 0] + esc
T = {m: {f: thr_at([d[1][m] for d in pool], f) for f in (.01, .02)} for m in ("fk", "dk")}
rules = {"v2": lambda d, f: d[2] > V2D["fpr_1pct" if f == .01 else "fpr_2pct"],
         "fusion": lambda d, f: z(d) > C["thresholds"]["fpr_1pct" if f == .01 else "fpr_2pct"],
         "and(fk,dk)": lambda d, f: d[1]["fk"] > T["fk"][f] and d[1]["dk"] > T["dk"][f],
         "and(fusion,fk,dk)": lambda d, f: z(d) > C["thresholds"]["fpr_1pct" if f == .01 else "fpr_2pct"] and d[1]["fk"] > T["fk"][f] and d[1]["dk"] > T["dk"][f]}
out = {"per_model_thresholds": T}
for sp in ("test", "esl", "academic"):
    docs = load_docs(sp, ["fk", "dk"], 600 if sp == "test" else None)
    for name, r in rules.items():
        for f in (.01, .02):
            res = {}
            for lab in (1, 0, -1):
                xs = [r(d, f) for d in docs if d[0]["label"] == lab]
                if xs: res[{1: "tpr", 0: "fpr", -1: "gpt4polished_flag"}[lab]] = round(float(np.mean(xs)), 3)
            srcs = sorted({d[0].get("source") for d in docs if d[0]["label"] == 0})
            res["human_by_source"] = {s: round(float(np.mean([r(d, f) for d in docs if d[0].get("source") == s])), 3) for s in srcs}
            out[f"{sp}|{name}|{int(f*100)}pct"] = res
json.dump(out, open(B / "eval/v3/rules_results.json", "w"), indent=1)
for k, v in out.items(): print(k, v)
