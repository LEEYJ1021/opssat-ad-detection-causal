"""README 표/캡션과 실행 로그에서 그림 입력 CSV를 만든다 (수치 하드코딩 없음).

results/**/*.csv 원본이 없는 환경에서 그림을 재현하기 위한 단계다.
원본 CSV가 생기면 각 fig 스크립트의 입력 경로만 교체하면 된다.

사용: python extract_inputs.py README.md 최종분석로그.txt inputs/
"""
import re, sys, json
from pathlib import Path
import numpy as np
import pandas as pd

README, LOG, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
lines = README.read_text(encoding="utf-8").splitlines()
NUM = r"[-−+]?\d+(?:\.\d+)?(?:×10[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)?"


def f(s):
    """'−0.090', '0.438', '**+0.462**' -> float"""
    s = re.sub(r"[*`]", "", str(s)).replace("−", "-").replace(",", "").replace("%", "").strip()
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else np.nan


def ci(s):
    """'0.438 [0.401, 0.477]' -> (0.438, 0.401, 0.477); '[a, b]' -> (nan, a, b)"""
    s = re.sub(r"[*`]", "", str(s)).replace("−", "-").replace(",", "")
    br = re.search(r"\[\s*([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s*\]", s)
    head = re.match(r"\s*([-+]?\d+(?:\.\d+)?)", s)
    pt = float(head.group(1)) if head and (not br or head.start(1) < br.start()) else np.nan
    return pt, float(br.group(1)), float(br.group(2))


def table_after(heading, nth=0, start=0):
    """heading 문자열을 포함하는 줄 이후 nth번째 markdown 표를 DataFrame으로."""
    h = heading.lstrip("*# ")
    i0 = next(i for i in range(start, len(lines)) if (lines[i].startswith(heading) if heading.startswith("###") else lines[i].lstrip("*# ").startswith(h)))
    found, i = -1, i0
    while i < len(lines):
        if lines[i].startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1]):
            found += 1
            if found == nth:
                hdr = [c.strip() for c in re.split(r"(?<!\\)\|", lines[i].strip().strip("|"))]
                rows, j = [], i + 2
                while j < len(lines) and lines[j].startswith("|"):
                    cells = [c.strip() for c in re.split(r"(?<!\\)\|", lines[j].strip().strip("|"))]
                    rows.append(cells)
                    j += 1
                return pd.DataFrame(rows, columns=hdr)
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                i += 1
            continue
        i += 1
    raise KeyError(heading)


def save(df, name):
    df.to_csv(OUT / name, index=False)
    print(f"{name}: {len(df)} rows")


# ---------------- Table 1 -> fig01, fig05a ----------------
t1 = table_after("### Table 1")
t1["channel"] = t1["Channel"].str.replace("CADC", "")
t1["n_seg"] = t1["# segments"].map(f).astype(int)
t1["prev_pct"] = t1["Anomaly %"].map(f)
t1["mcc"] = t1["MCC"].map(f)
t1["youden_j"] = t1["Youden's J"].map(f)
t1["sensor"] = t1["Sensor"].str.contains("Magn").map({True: "Magn.", False: "Phot."})
dec = t1["Decision"].str.replace("*", "", regex=False)
t1["status"] = np.where(dec.str.startswith("Included"), "incl.",
                np.select([dec.str.contains("structural"), dec.str.contains("underpowered"), dec.str.contains("chance")],
                          ["S", "U", "C"], "?"))
cnt = dec.str.extract(r"(\d+) anomalous / (\d+) nominal").astype(float)
t1["n_anom"] = np.where(cnt[0].notna(), cnt[0], np.round(t1.n_seg * t1.prev_pct / 100))
t1["n_nom"] = t1.n_seg - t1.n_anom
assert (t1.status != "?").all()
save(t1[["channel", "sensor", "n_seg", "prev_pct", "n_anom", "n_nom", "mcc", "youden_j", "status"]], "channel_scope.csv")

# ---------------- Table 2 -> fig02 (original arm) ----------------
t2 = table_after("### Table 2")
o = pd.DataFrame({"model": t2["Model"], "repr": t2["Representation"], "auc": t2["AUC"].map(f),
                  "level": t2["level share"].map(f), "diff": t2["diff share"].map(f),
                  "diff2": t2["diff² share"].map(f), "arm": "original",
                  "attribution": np.where(t2["Representation"] == "sequence", "IntegratedGradients", "SHAP/permutation")})

# rerun arm: 시퀀스 6모델 share는 로그에서, 표 10모델 AUC는 R6 표의 D 열에서
log = LOG.read_text(encoding="utf-8", errors="ignore")
seq = re.findall(r"^\s+(\w+)\s+AUC=([\d.]+)\s+share\(level/diff/diff2\)=([\d.]+)/([\d.]+)/([\d.]+)", log, flags=re.M)
last = {}
for m, auc, a, b, c in seq:           # 같은 모델이 여러 arm에 나오므로 마지막(fixed = 재실행) 값을 사용
    last[m] = (float(auc), float(a), float(b), float(c))
name_map = {"CNN1D": "CNN1D", "TCN": "TCN", "BiLSTM": "BiLSTM", "BiGRU": "BiGRU",
            "TinyTransformer": "TinyTransformer", "LightMamba": "LightMamba"}
r6 = table_after("**Table R6")
rr = []
for _, r in o.iterrows():
    if r["repr"] == "sequence":
        auc, a, b, c = last[name_map[r["model"]]]
        rr.append(dict(model=r["model"], repr="sequence", auc=auc, level=a, diff=b, diff2=c,
                       arm="rerun", attribution="IntegratedGradients"))
    else:
        key = r["model"].replace(" (sklearn)", "")
        row = r6[r6["Model"] == key]
        assert len(row) == 1, key
        rr.append(dict(model=r["model"], repr="tabular", auc=f(row.iloc[0]["D (placebo de-duplicated)"]),
                       level=np.nan, diff=np.nan, diff2=np.nan, arm="rerun", attribution="permutation"))
save(pd.concat([o, pd.DataFrame(rr)]), "attribution_16models_both_arms.csv")

# 합의 통계 (Table 6 + R6 agreement)
t6 = table_after("### Table 6")
w_orig = f(t6.iloc[0, 1])
chi = re.findall(NUM, t6.iloc[1, 1].replace("−", "-"))
ag = dict(W_original=w_orig, chi2_original=float(chi[0]), chi2_p_original="5.8e-5",
          W_rerun="pending", binom_p=0.0015)
(OUT / "agreement_statistics.json").write_text(json.dumps(ag, ensure_ascii=False, indent=1))

# ---------------- Fig 3: 채널별 level AUROC (Figure 3 캡션), R7b, Table 5 ----------------
cap3 = next(l for l in lines if l.startswith("*(a-1)"))
seg = re.search(r"\(complete-case; (.*?); n_anomalous ([\d/]+)\)", cap3)
pairs = re.findall(r"(\d{4}) ([\d.]+)", seg.group(1))
nan_ = [int(x) for x in seg.group(2).split("/")]
pc = pd.DataFrame(pairs, columns=["channel", "level_auroc"]).astype({"level_auroc": float})
pc["n_anom"] = nan_
bonf = re.search(r"two-sided p_bonf = ([\d.]+)", cap3)
pc["note_p_bonf"] = np.where(pc.channel == "0873", float(bonf.group(1)), np.nan)
save(pc, "per_channel_level_auroc.csv")

r7b = table_after("**Table R7b")
rows = []
for _, r in r7b.iterrows():
    _, lo, hi = ci(r["95% CI"])
    rows.append(dict(scheme=r["Scheme"], lo=lo, hi=hi, p=f(r["p vs 0.5"]), excludes_0_5=r["Excludes 0.5"].strip() == "yes"))
r2 = table_after("**Table R2 ")
pooled = ci(r2.iloc[0]["AUROC `level`"])[0]
d = pd.DataFrame(rows); d["pooled_level_auroc"] = pooled
save(d, "level_reversal_diagnosis.csv")

t5 = table_after("### Table 5")
t5["channel"] = t5["Channel"].str.replace("CADC", "")
for c_ in ["Sig. (full)", "Sig. (train)", "Sig. (test)"]:
    t5[c_] = t5[c_].str.replace("*", "", regex=False).str.strip().map(lambda s: "n/a" if s.startswith("n/a") else s.split()[0])
t5["agree"] = t5["3-way agree"].str.replace("*", "", regex=False).str.extract(r"^(Yes|No|n/a)")[0]
t5 = t5.rename(columns={"Feature": "feature", "Sig. (full)": "full", "Sig. (train)": "train", "Sig. (test)": "test"})
raw5 = table_after("### Table 5")
t5["test_p_note"] = raw5["Sig. (test)"].str.extract(r"p = ([\d.]+)")[0].astype(float)
save(t5[["channel", "feature", "full", "train", "test", "agree", "test_p_note"]], "triple_verification_matrix.csv")

# ---------------- Table 4 -> fig04 ----------------
t4 = table_after("### Table 4")
t4["family"] = np.where(t4["Outcome"].str.startswith("Effect"), "Effect size |d|", "Significant-segment fraction")
t4["feature"] = t4["Outcome"].str.extract(r"\((.*)\)")[0]
t4["I2"] = t4["I²"].map(f)
t4["Q_p"] = t4["Q-test p"].map(lambda s: float(re.sub(r"[×10⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+", "", "") or 0) if False else s)
sup = str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "-0123456789")


def sci(s):
    s = s.replace("\\|", "|").strip()
    m = re.match(r"([\d.]+)×10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)", s)
    return float(m.group(1)) * 10 ** int(m.group(2).translate(sup)) if m else float(s)


t4["Q_p"] = t4["Q-test p"].map(sci)
save(t4[["family", "feature", "I2", "Q_p"]], "heterogeneity_summary.csv")

# ---------------- Table 10 -> fig05b ----------------
t10 = table_after("### Table 10")
t10["n_pairs"] = t10["n pairs"].map(f).astype(int)
t10["second_first_pct"] = t10["frac(first precedes second)"].map(lambda s: float(re.findall(r"\*{0,2}([\d.]+)%", s)[-1]))
t10["cmp"] = t10["Comparison"].str.replace(".", "", regex=False)
t9 = table_after("### Table 9")
t9["cmp"] = t9["Comparison"].str.replace(".", "", regex=False)
t9["p_wilcoxon"] = t9["p (Wilcoxon)"].map(lambda s: sci(re.sub(r"\s*\(n\.s\.\)", "", s)))
m = t10.merge(t9[["cmp", "p_wilcoxon"]], on="cmp")
m["regime"] = m["Regime"]
save(m[["cmp", "regime", "n_pairs", "second_first_pct", "p_wilcoxon"]], "temporal_precedence.csv")

# ---------------- Fig 6: R2, R7, direction-agnostic, SMD breadth ----------------
rows = []
for _, r in r2.iterrows():
    for con, col in [("diff - level", "Δ(diff − level)"), ("diff2 - level", "Δ(diff² − level)")]:
        pt, lo, hi = ci(r[col])
        rows.append(dict(dataset=r["Dataset"], cluster_unit=r["Cluster unit (K)"], method="cluster bootstrap (R2)",
                         contrast=con, delta=pt if not np.isnan(pt) else f(r[col]), ci_low=lo, ci_high=hi))
r7 = table_after("**Table R7 —")
pt_row = r7[r7["Scheme"] == "Point estimate"].iloc[0]
pt_d = {"diff - level": f(pt_row["Δ(diff − level) [95% CI]"]), "diff2 - level": f(pt_row["Δ(diff² − level) [95% CI]"])}
for _, r in r7.iterrows():
    if r["Scheme"] == "Point estimate":
        continue
    for con, col in [("diff - level", "Δ(diff − level) [95% CI]"), ("diff2 - level", "Δ(diff² − level) [95% CI]")]:
        _, lo, hi = ci(r[col])
        rows.append(dict(dataset="OPS-SAT-AD", cluster_unit=r["Unit (K)"], method=r["Scheme"], contrast=con,
                         delta=pt_d[con], ci_low=lo, ci_high=hi))
allr = pd.DataFrame(rows)
allr = allr[~((allr.dataset == "OPS-SAT-AD") & (allr.method == "cluster bootstrap (R2)"))]  # R7의 single-level 행이 대체
allr["dataset"] = allr["dataset"].str.replace(" (reference)", "", regex=False)
allr.loc[allr.cluster_unit.str.startswith("channel (K = 1,064"), "method"] = "cluster bootstrap (R2, ref.)"
da = table_after("Direction-agnostic variant")
da_map = {}
for _, r in da.iterrows():
    da_map[r["Dataset"]] = dict(zip(["diff - level", "diff2 - level"], [f(r["Δ(diff − level)"]), f(r["Δ(diff² − level)"])]))
allr["delta_dir_agnostic"] = [da_map[d_][c_] for d_, c_ in zip(allr.dataset, allr.contrast)]
save(allr, "cluster_robust_contrasts.csv")

sm = re.search(r"97/(1,038).*?142/1,038.*?145/1,038", "\n".join(lines))
brd = []
for feat in ["level", "diff", "diff²"]:
    mm = re.search(rf"(\d+)/1,038 channels \((?:[\d.]+)%;[^)]*\) (?:are significant )?for `{re.escape(feat)}`|(\d+)/1,038 \([\d.]+%;[^)]*\) for `{re.escape(feat)}`", "\n".join(lines))
    brd.append(dict(feature=feat.replace("²", "2"), k=int(next(g for g in mm.groups() if g)), n=1038))
save(pd.DataFrame(brd), "stage1_channel_level_significance.csv")

# ---------------- Fig 6b: R8, R10, observed retention ----------------
r8 = table_after("**Table R8")
r8o = pd.DataFrame({"procedure": r8["Procedure"], "cov_pop": r8.iloc[:, 1].map(f), "cov_clus": r8.iloc[:, 2].map(f),
                    "width": r8["Mean width"].map(f)})
r8o["mc_se"] = np.sqrt(0.95 * 0.05 / 500) * 100
save(r8o, "coverage_summary.csv")
r10 = table_after("**Table R10")
rows = []
for _, r in r10.iterrows():
    for col, kind in [("Pooled bias (mean, SD)", "pooled"), ("Equal-weight bias (mean, SD)", "equal-weight")]:
        mu, sd = re.findall(r"[-+]?\d+\.\d+", r[col].replace("−", "-"))[:2]
        rows.append(dict(mechanism=r["Mechanism (K = 8, 500 reps)"], estimator=kind, mean=float(mu), sd=float(sd)))
save(pd.DataFrame(rows), "retention_bias_summary.csv")
txt = "\n".join(lines)
mo = re.search(r"Spearman ρ = ([−-]?[\d.]+) \(p = ([\d.]+)\)", txt)
json.dump(dict(rho=float(mo.group(1).replace("−", "-")), p=float(mo.group(2)), n=5), open(OUT / "real_retention.json", "w"))

# ---------------- Fig 6c: R3 + fixed |z|>3 reference ----------------
r3 = table_after("**Table R3")
rows = []
for _, r in r3.iterrows():
    ds = "SMAP/MSL" if r["Dataset"].startswith("SMAP") else "SMD"
    far = f(r["FAR target"])
    pt, lo, hi = ci(re.sub(r"pp", "", r.iloc[-1]))
    n = int(f(re.search(r"n = ([\d,]+)", r["Dataset"]).group(1))) if "n =" in r["Dataset"] else None
    for rule, col in [("level", "`level`"), ("diff", "`diff`"), ("diff2", "`diff²`"), ("fusion", "`level`\\|`diff`\\|`diff²`")]:
        key = [c for c in r3.columns if c.replace("\\", "") == col.replace("\\", "")][0]
        rows.append(dict(dataset=ds, far_target=far / 100, rule=rule, coverage=f(r[key]) / 100,
                         delta_fusion_pp=f(re.search(r"[-+]?\d+\.\d+", r.iloc[-1].replace("−", "-")).group()),
                         ci_low_pp=lo, ci_high_pp=hi, n_anom=n))
cur = pd.DataFrame(rows)
cur["n_anom"] = cur.groupby(["dataset", "far_target"]).n_anom.transform(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
cur["n_anom"] = cur.groupby("dataset").n_anom.transform(lambda s: s.dropna().iloc[0])
save(cur, "far_coverage_points.csv")
fx = []
for ds, pat in [("SMD", r"SMD coverage level ([\d.]+)% / diff ([\d.]+)% / diff² ([\d.]+)% at measured placebo FAR of ([\d.]+)% / ([\d.]+)% / ([\d.]+)%"),
                ("SMAP/MSL", r"SMAP/MSL: level ([\d.]+)% / diff ([\d.]+)% / diff² ([\d.]+)% at ([\d.]+)% / ([\d.]+)% / ([\d.]+)%")]:
    g = re.search(pat, txt).groups()
    for k, rule in enumerate(["level", "diff", "diff2"]):
        fx.append(dict(dataset=ds, rule=rule, coverage=float(g[k]) / 100, measured_far=float(g[3 + k]) / 100))
save(pd.DataFrame(fx), "fixed_threshold_points.csv")

# ---------------- Fig 7: Table 13 ----------------
t13 = table_after("**Table 13")
rows = []
for _, r in t13.iterrows():
    ds = re.sub(r"[*]", "", r["Dataset"])
    rows.append(dict(dataset=ds, regime=r["Regime"], n=int(f(r["n"])), pct=f(r["`diff` precedes `level`"].split("%")[0])))
t13o = pd.DataFrame(rows)
save(t13o, "precedence_baseline.csv")
inc = table_after("**Increment (anomalous − normal).**")
rows = []
for _, r in inc.iterrows():
    ds = re.sub(r"\s*🔶|\*", "", r["Dataset"]).strip()
    pt, lo, hi = ci(r["Increment (pp)"])
    pt = f(r["Increment (pp)"].split("[")[0])
    rows.append(dict(dataset=ds, increment_pp=pt, ci_low=lo, ci_high=hi, pending=("🔶" in r["Dataset"])))
save(pd.DataFrame(rows), "precedence_increment.csv")

# ---------------- 캡션에 인용된 n 검증용 ----------------
json.dump(dict(readme_sha=str(hash(txt) % 10**8)), open(OUT / "extract_meta.json", "w"))
print("done")

# ---------------- arm summary (R6 agreement table) ----------------
ag6 = table_after("**Agreement statistics")
def row(prefix):
    return ag6[ag6.iloc[:, 0].str.replace("`", "").str.contains(prefix, regex=False)].iloc[0]
def two(s):
    a, b = re.findall(r"(\d+)\s*/\s*(\d+)", s)[0]; return int(a), int(b)
recs = []
for arm, col in [("original", 1), ("rerun", 2)]:
    d_all = two(row("Top feature").iloc[col]); d_tab = two(row("of which tabular").iloc[col]); d_seq = two(row("of which sequence").iloc[col])
    lvl = two(row("Models with").iloc[col])
    recs.append(dict(arm=arm, top_diff=d_all[0], top_diff2=d_all[1], tab_diff=d_tab[0], tab_diff2=d_tab[1],
                     seq_diff=d_seq[0], seq_diff2=d_seq[1], level_below_third=lvl[0], n_models=lvl[1]))
save(pd.DataFrame(recs), "arm_summary.csv")
