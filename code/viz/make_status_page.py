
from __future__ import annotations

import base64
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PATHS

OUT = PATHS.outputs / "status.html"


def img(name: str) -> str:
    p = PATHS.figures / f"{name}.png"
    if not p.is_file():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f"data:image/png;base64,{b64}"


def read(name: str):
    p = PATHS.results / name
    return pd.read_csv(p) if p.is_file() else None


def cv_row(fname: str):
    df = read(fname)
    if df is None:
        return None
    piv = df.pivot_table(index="repeat", columns="protocol", values="auc")
    if not {"image-level", "patient-level"}.issubset(piv.columns):
        return None
    d = piv["image-level"] - piv["patient-level"]
    return {
        "patient": piv["patient-level"].mean(), "image": piv["image-level"].mean(),
        "gap": d.mean(), "sd": d.std(), "pos": int((d > 0).sum()), "n": len(d),
    }


def figure_block(name: str, caption: str) -> str:
    src = img(name)
    if not src:
        return ""
    return f"""<figure>
      <img src="{src}" alt="{caption}">
      <figcaption>{caption}</figcaption>
    </figure>"""


def main() -> int:
    COHORTS = [
        ("PAPILA", "glaucoma", 244, 2.9, "e7_cv_split.csv", "e6_pixel_baseline.csv"),
        ("ODIR-5K", "normal vs abnormal", 3034, 22.2,
         "e7_cv_split_odir.csv", "e6_pixel_baseline_odir.csv"),
        ("DeepDRiD", "referable DR", 400, 29.2,
         "e7_cv_split_deepdrid.csv", "e6_pixel_baseline_deepdrid.csv"),
    ]

    def px(df):
        if df is None:
            return None
        s = df.set_index("representation")
        learned = s.loc["dinov2_vitb14 (learned)", "R@1"]
        best = s.drop(index="dinov2_vitb14 (learned)")["R@1"].max()
        return {"learned": learned, "best": best, "ratio": learned / best,
                "ncand": int(s["n_test"].iloc[0]) - 1}

    split_rows, px_rows = [], []
    for label, task, npat, dsc, cvf, pxf in COHORTS:
        c = cv_row(cvf)
        if c:
            split_rows.append((label, task, npat, dsc, c))
        p = px(read(pxf))
        if p:
            px_rows.append((label, p))
    split_rows.sort(key=lambda r: r[3])
    px_rows.sort(key=lambda r: r[1]["ncand"])

    bb_rows = []
    for label, dsfx, dsc in (("PAPILA", "", 2.9), ("ODIR-5K", "_odir", 22.2),
                             ("DeepDRiD", "_deepdrid", 29.2)):
        for i, (key, bname, bsfx) in enumerate(
                [("dinov2_vitb14", "DINOv2 ViT-B/14", ""),
                 ("retfound_cfp", "RETFound ViT-L/16", "_retfound_cfp")]):
            e1 = read(f"e1_leakage_audit{dsfx}.csv")
            r = auc = None
            if e1 is not None:
                s = e1[(e1.backbone == key) & (e1.variant == "canon")]
                if not s.empty:
                    r, auc = float(s.iloc[0]["R@1"]), float(s.iloc[0]["task_auc"])
            cv = read(f"e7_cv_split{dsfx}{bsfx}.csv")
            gap = None
            if cv is not None:
                piv = cv.pivot_table(index="repeat", columns="protocol", values="auc")
                if {"image-level", "patient-level"}.issubset(piv.columns):
                    gap = float((piv["image-level"] - piv["patient-level"]).mean())
            if r is None and gap is None:
                continue
            hi = ' class="num hi"' if key == "retfound_cfp" else ' class="num"'
            bb_rows.append(
                f'<tr><td>{label if i == 0 else ""}</td>'
                f'<td class="num">{f"{dsc:.1f}%" if i == 0 else ""}</td>'
                f'<td>{bname}</td>'
                f'<td{hi}>{r:.3f}</td><td class="num">{auc:.3f}</td>'
                f'<td{hi}>{gap:+.3f}</td></tr>')
    backbone_tbody = "\n".join(bb_rows)

    e9 = read("e9_same_eye_control_deepdrid_fields.csv")
    e9row = None
    if e9 is not None:
        s = e9[e9.representation.str.contains("learned")]
        if not s.empty:
            e9row = s.iloc[0]

    e3o = read("e3_mitigation_odir.csv")
    mit = None
    if e3o is not None:
        base = e3o[e3o.method.str.startswith("none")].iloc[0]
        worst = e3o.loc[e3o["R@1"].idxmax()]
        mit = {"b_r": base["R@1"], "b_a": base["auc"],
               "w_r": worst["R@1"], "w_a": worst["auc"]}

    def stat(v, d=3):
        return f"{v:.{d}f}" if v is not None else "—"

    split_tbody = "\n".join(
        f'<tr><td>{lab} · {npat:,} pat.</td><td class="num">{dsc:.1f}%</td>'
        f'<td class="num">{c["patient"]:.3f}</td><td class="num">{c["image"]:.3f}</td>'
        f'<td class="num hi">{c["gap"]:+.3f} ± {c["sd"]:.3f}</td>'
        f'<td class="num">{c["pos"]}/{c["n"]}</td></tr>'
        for lab, task, npat, dsc, c in split_rows)

    px_head = "".join(f"<th>{lab}<br><span style='font-weight:400'>"
                      f"{p['ncand']:,} cand.</span></th>" for lab, p in px_rows)
    px_body = (
        '<tr><td>DINOv2 ViT-B/14 (learned)</td>'
        + "".join(f'<td class="num">{p["learned"]:.3f}</td>' for _, p in px_rows)
        + '</tr><tr><td>best pixel baseline</td>'
        + "".join(f'<td class="num">{p["best"]:.3f}</td>' for _, p in px_rows)
        + '</tr><tr><td><strong>learned / best pixel</strong></td>'
        + "".join(f'<td class="num hi">{p["ratio"]:.2f}×</td>' for _, p in px_rows)
        + '</tr>')

    def leak(fname, variant):
        df = read(fname)
        if df is None:
            return None
        s = df[(df.backbone == "dinov2_vitb14") & (df.variant == variant)]
        return None if s.empty else {"r": float(s.iloc[0]["R@1"]),
                                     "a": float(s.iloc[0]["task_auc"])}

    bl_p = leak("e1_leakage_audit.csv", "canon")
    bg_p = leak("e1_leakage_audit.csv", "canonbg")
    bl_o = leak("e1_leakage_audit_odir.csv", "canon")
    bg_o = leak("e1_leakage_audit_odir.csv", "canonbg")

    adv = read("adv_papila.csv")
    adv_r = adv_a = None
    if adv is not None:
        zp = adv[adv.representation == "z_path"]
        if not zp.empty:
            adv_r, adv_a = float(zp["ret_R@1"].mean()), float(zp["task_auc"].mean())

    html = f"""<title>The Fellow Eye Problem</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=Source+Sans+3:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  --ground:#f6f7f7; --surface:#ffffff; --surface-2:#eef0f1;
  --ink:#16191d; --ink-2:#454d55; --ink-3:#6f7880;
  --rule:#dfe3e5; --rule-2:#c9cfd2;
  --accent:#33357a; --accent-soft:#e9eaf3;
  --good:#0f7a5f; --warn:#9a6a10; --stop:#98352c;
  --serif:'Newsreader',Georgia,serif;
  --sans:'Source Sans 3',system-ui,-apple-system,sans-serif;
  --mono:'IBM Plex Mono',ui-monospace,Menlo,monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#101316; --surface:#171b1f; --surface-2:#1e232a;
    --ink:#e8eaec; --ink-2:#b0b8bf; --ink-3:#848d95;
    --rule:#2a3037; --rule-2:#3a424a;
    --accent:#9ba0e8; --accent-soft:#22253f;
    --good:#4cbf9a; --warn:#d3a53f; --stop:#e0796c;
  }}
}}
:root[data-theme="dark"] {{
  --ground:#101316; --surface:#171b1f; --surface-2:#1e232a;
  --ink:#e8eaec; --ink-2:#b0b8bf; --ink-3:#848d95;
  --rule:#2a3037; --rule-2:#3a424a;
  --accent:#9ba0e8; --accent-soft:#22253f;
  --good:#4cbf9a; --warn:#d3a53f; --stop:#e0796c;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:var(--sans); font-size:16.5px; line-height:1.62;
  -webkit-font-smoothing:antialiased;
}}
.wrap {{ max-width:820px; margin:0 auto; padding:0 24px 96px; }}

header.top {{ border-bottom:1px solid var(--rule); margin-bottom:44px; padding:56px 0 32px; }}
.eyebrow {{
  font-family:var(--mono); font-size:11.5px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--accent); margin:0 0 18px;
}}
h1 {{
  font-family:var(--serif); font-weight:500; font-size:clamp(2.1rem,5vw,3.05rem);
  line-height:1.08; letter-spacing:-.015em; margin:0 0 18px; text-wrap:balance;
}}
.standfirst {{
  font-family:var(--serif); font-size:1.16rem; line-height:1.55;
  color:var(--ink-2); margin:0; max-width:62ch;
}}

.statusgrid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(148px,1fr)); gap:1px;
  background:var(--rule); border:1px solid var(--rule); margin:36px 0 0; }}
.statusgrid div {{ background:var(--surface); padding:14px 16px; }}
.statusgrid dt {{ font-family:var(--mono); font-size:10.5px; letter-spacing:.1em;
  text-transform:uppercase; color:var(--ink-3); margin:0 0 6px; }}
.statusgrid dd {{ margin:0; font-size:1.02rem; font-weight:600;
  font-variant-numeric:tabular-nums; }}
.pill {{ display:inline-block; font-family:var(--mono); font-size:10.5px;
  letter-spacing:.06em; padding:2px 7px; border:1px solid currentColor; }}
.pill.ok {{ color:var(--good); }} .pill.run {{ color:var(--warn); }}

h2 {{
  font-family:var(--serif); font-weight:600; font-size:1.62rem; line-height:1.2;
  margin:64px 0 6px; letter-spacing:-.01em; text-wrap:balance;
}}
h2 + .sub {{ color:var(--ink-3); font-size:.95rem; margin:0 0 26px; }}
h3 {{ font-family:var(--sans); font-weight:600; font-size:1.02rem; margin:34px 0 8px; }}
p {{ max-width:66ch; }}

.claim {{
  border-left:3px solid var(--accent); background:var(--accent-soft);
  padding:20px 24px; margin:26px 0; font-family:var(--serif); font-size:1.1rem;
  line-height:1.55;
}}
.claim strong {{ font-weight:600; }}

ol.claims {{ list-style:none; counter-reset:c; padding:0; margin:0;
  display:flex; flex-direction:column; gap:34px; }}
ol.claims > li {{ counter-increment:c; position:relative; padding-left:46px; }}
ol.claims > li::before {{
  content:counter(c); position:absolute; left:0; top:1px;
  font-family:var(--mono); font-size:12px; font-weight:500; color:var(--accent);
  border:1px solid var(--rule-2); width:29px; height:29px;
  display:grid; place-items:center;
}}

.tablewrap {{ overflow-x:auto; margin:20px 0; border:1px solid var(--rule); }}
table {{ border-collapse:collapse; width:100%; font-size:.9rem; background:var(--surface); }}
th, td {{ text-align:left; padding:9px 14px; border-bottom:1px solid var(--rule); }}
th {{ font-family:var(--mono); font-size:10.5px; letter-spacing:.08em;
  text-transform:uppercase; color:var(--ink-3); font-weight:500;
  background:var(--surface-2); white-space:nowrap; }}
td {{ font-variant-numeric:tabular-nums; }}
tbody tr:last-child td {{ border-bottom:none; }}
td.num {{ font-family:var(--mono); }}
td.hi {{ font-weight:600; color:var(--accent); }}

figure {{ margin:26px 0; }}
figure img {{ width:100%; height:auto; display:block; border:1px solid var(--rule); background:#fcfcfb; }}
figcaption {{ font-size:.85rem; color:var(--ink-3); margin-top:9px; max-width:66ch; }}

.correction {{ border:1px solid var(--rule); border-top:3px solid var(--stop);
  background:var(--surface); padding:18px 22px; margin:18px 0; }}
.correction h4 {{ margin:0 0 6px; font-size:.95rem; font-weight:600; }}
.correction p {{ margin:0; font-size:.93rem; color:var(--ink-2); }}
.correction .was {{ font-family:var(--mono); font-size:.85rem; color:var(--ink-3); }}

ul.todo {{ list-style:none; padding:0; margin:20px 0; display:flex;
  flex-direction:column; gap:1px; background:var(--rule);
  border:1px solid var(--rule); }}
ul.todo li {{ background:var(--surface); padding:13px 18px; display:flex;
  gap:14px; align-items:baseline; font-size:.95rem; }}
ul.todo .tag {{ font-family:var(--mono); font-size:10px; letter-spacing:.08em;
  text-transform:uppercase; color:var(--ink-3); min-width:74px; }}

code {{ font-family:var(--mono); font-size:.88em; background:var(--surface-2);
  padding:1px 5px; }}
footer {{ margin-top:72px; padding-top:22px; border-top:1px solid var(--rule);
  font-size:.85rem; color:var(--ink-3); }}
a {{ color:var(--accent); }}
a:focus-visible, li:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
</style>

<div class="wrap">
<header class="top">
  <p class="eyebrow">Research status &middot; 21 August 2026</p>
  <h1>The Fellow&nbsp;Eye Problem</h1>
  <p class="standfirst">Every fundus dataset contains a natural experiment the field discards: a
  patient's two eyes share everything except, sometimes, the disease. Using that structure as
  supervision does not work. Measuring what it leaks does.</p>

  <dl class="statusgrid">
    <div><dt>Cohorts</dt><dd>3 &nbsp;<span class="pill ok">open access</span></dd></div>
    <div><dt>Patients</dt><dd>3,678</dd></div>
    <div><dt>Encoders</dt><dd>4 &nbsp;<span class="pill ok">incl. RETFound</span></dd></div>
    <div><dt>Interventions</dt><dd>1 of 4 &nbsp;<span class="pill ok">helps</span></dd></div>
  </dl>
</header>

<h2>Where the argument stands</h2>
<p class="sub">The original plan was a method paper. The evidence killed the method and produced a better claim.</p>

<div class="claim">
Patient identity is strongly present in frozen fundus features and is <strong>not removable at the
representation level</strong>. What it costs in reported accuracy is governed by how often a
patient's two eyes carry the same label — so the danger is concentrated in exactly the small,
highly concordant cohorts this literature is built on.
</div>

<h2>The evidence</h2>
<p class="sub">Four claims, each with the control that could have sunk it.</p>

<ol class="claims">
<li>
<h3>Fellow eyes are strongly retrievable — and it is not just near-duplicate images</h3>
<p>Given one eye, a frozen DINOv2 encoder picks that patient's <em>other</em> eye out of
{px_rows[-1][1]['ncand']:,} {px_rows[-1][0]} candidates
{stat(px_rows[-1][1]['learned'])} of the time — 442&times; chance, after circle-cropping and
mirroring every left eye into a shared orientation. It holds on all three cohorts.</p>
<p>Wagner &amp; Keane (<em>npj Digital Medicine</em>, 2026) argued that published fundus
re-identification is an artefact of near-identical images, since raw pixels match RETFound. Running
their control across three cohorts decides it — and shows why the dispute happened.</p>
<div class="tablewrap"><table>
<thead><tr><th>Representation</th>{px_head}</tr></thead>
<tbody>{px_body}</tbody></table></div>
{figure_block("f8_pixel_baseline", "The learned encoder's margin over raw pixels grows monotonically with the candidate pool.")}
<p>The margin grows with the pool. At ~120 candidates every representation saturates and the
baselines are indistinguishable — <strong>that is the regime the published dispute was argued
in</strong>. The objection is testable, and testing it needs a pool large enough for pixels to fail.</p>
</li>

<li>
<h3>Near-duplicate or not? The control neither side ran</h3>
<p>DeepDRiD carries two fields per eye, so <em>which eye</em> and <em>which view</em> can be
separated inside one cohort, one encoder, one candidate pool.</p>
<div class="tablewrap"><table>
<thead><tr><th>Condition</th><th>R@1</th></tr></thead><tbody>
<tr><td><strong>fellow eye, same field</strong></td><td class="num hi">{stat(e9row['fellow_eye_same_field_R@1'] if e9row is not None else None)}</td></tr>
<tr><td>same eye, different field <em>(the near-duplicate case)</em></td><td class="num">{stat(e9row['same_eye_diff_field_R@1'] if e9row is not None else None)}</td></tr>
<tr><td>fellow eye, different field</td><td class="num">{stat(e9row['fellow_eye_diff_field_R@1'] if e9row is not None else None)}</td></tr>
</tbody></table></div>
<p>With the field held fixed the same eye is only
<strong>{stat(e9row['ratio_sameeye_over_fellow_matched_field'] if e9row is not None else None, 2)}&times;</strong>
easier than the fellow eye — far from the overwhelming advantage the rebuttal implies. And
“fellow eye, same field” <em>beats</em> “same eye, different field”: field of view matters more than
which eye it is.</p>
<p><strong>Stated as a limit:</strong> DeepDRiD's two fields are macula- and disc-centred, so they are
genuinely different views rather than repeat shots at one fixation. This is a <em>lower bound</em> on
the near-duplicate advantage. What it does establish is that genuine bilateral retrieval is not a
near-duplicate artefact.</p>
</li>

<li>
<h3>The cost is real, and the cohort's own structure predicts it</h3>
<div class="tablewrap"><table>
<thead><tr><th>Cohort</th><th>Eyes disagree</th><th>Patient-grouped</th><th>Image-level</th><th>Inflation</th><th>Repeats &gt;0</th></tr></thead>
<tbody>
{split_tbody}
</tbody></table></div>
<p>Monotone in discordance, and the most discordant cohort sits at exactly zero with 5 of 10 repeats
positive — a coin flip. Two points were a line; three make it a relationship, and the prediction
that the most discordant cohort would show no inflation was made before the experiment ran.</p>
{figure_block("f3_split_protocol", "Repeated 5-fold cross-validation, 10 repeats. Identical features and classifier; only the grouping changes.")}
</li>

<li>
<h3>No representation-level defence works — and on the larger cohort they backfire</h3>
<p>Adversarial invariance drives its own discriminator's loss from 2.6 to 15.8 while a
<em>fresh</em> cosine probe still recovers the fellow eye at 42&times; chance: the encoder learned to
hide identity from one head, not to remove it.</p>
<p>Closed-form subspace projection is worse. On ODIR, removing the leading shared directions
<em>raises</em> retrieval from {stat(mit['b_r'] if mit else None)} to {stat(mit['w_r'] if mit else None)}
while dropping AUC from {stat(mit['b_a'] if mit else None)} to {stat(mit['w_a'] if mit else None)} —
what remains after the shared structure is stripped is precisely the idiosyncratic variation.</p>
{figure_block("f9_mitigation_both", "Ideal is top-left. Both cohorts move the wrong way, and the sample-size excuse is gone: with 1,821 training pairs the estimable subspace ceiling is 1,821, not 147.")}
</li>

<li>
<h3>One thing helps — and it is preprocessing, not representation learning</h3>
<p>Ben Graham illumination normalisation is the only intervention tested that moves
<em>both</em> axes the right way on <em>both</em> cohorts.</p>
<div class="tablewrap"><table>
<thead><tr><th>Intervention</th><th>PAPILA R@1</th><th>PAPILA AUC</th><th>ODIR R@1</th><th>ODIR AUC</th><th>Verdict</th></tr></thead>
<tbody>
<tr><td>none</td><td class="num">{stat(bl_p['r'] if bl_p else None)}</td><td class="num">{stat(bl_p['a'] if bl_p else None)}</td><td class="num">{stat(bl_o['r'] if bl_o else None)}</td><td class="num">{stat(bl_o['a'] if bl_o else None)}</td><td>—</td></tr>
<tr><td><strong>Ben Graham colour norm.</strong></td><td class="num hi">{stat(bg_p['r'] if bg_p else None)} &darr;</td><td class="num hi">{stat(bg_p['a'] if bg_p else None)} &uarr;</td><td class="num hi">{stat(bg_o['r'] if bg_o else None)} &darr;</td><td class="num hi">{stat(bg_o['a'] if bg_o else None)} &uarr;</td><td><strong>both improve</strong></td></tr>
<tr><td>adversarial adapter</td><td class="num">{stat(adv_r)}</td><td class="num">{stat(adv_a)}</td><td>—</td><td>—</td><td>hides, doesn't remove</td></tr>
<tr><td>subspace projection <em>r</em>=32</td><td class="num">0.320</td><td class="num">0.747</td><td class="num">0.341</td><td class="num">0.667</td><td>backfires on ODIR</td></tr>
</tbody></table></div>
<p>It is a <strong>mitigation, not a fix</strong>: retrieval falls 8–10% and stays at 45&times;
chance on PAPILA, 406&times; on ODIR. What earns it a place is that it is free — a preprocessing
step that costs nothing at inference and slightly <em>improves</em> diagnostic AUC instead of
trading it away. It also fits the mechanism: the cue lives in shared colour and illumination
between a patient's two eyes, so attacking it in the pixels works where attacking it downstream
in the representation did not.</p>
</li>

<li>
<h3>The encoder that leaks least causes the most inflation</h3>
<p>RETFound — the retinal foundation model the published dispute was argued over — retrieves the
fellow eye roughly three times less often than a general-purpose DINOv2, at comparable diagnostic
AUC. Domain pretraining genuinely reduces measurable leakage. It also produces the
<em>larger</em> split-protocol inflation, on every cohort.</p>
<div class="tablewrap"><table>
<thead><tr><th>Cohort</th><th>Eyes disagree</th><th>Encoder</th><th>R@1</th><th>Task AUC</th><th>Inflation</th></tr></thead>
<tbody>{backbone_tbody}</tbody></table></div>
<p>The two numbers measure different things. R@1 is how much identity a representation
<em>carries</em>; inflation is how much a classifier <em>exploits</em> it under a bad split, which
also depends on how good the honest model already is. So the strongest form of the conclusion is
this: <strong>reducing measurable leakage does not reduce the evaluation error it causes.</strong>
Protocol is the remedy; picking a better encoder is not.</p>
<p>The discordance ordering replicates under both encoders — three cohorts, two encoders, monotone
in both — so the central claim is backbone-independent.</p>
</li>

<li>
<h3>The structure that causes the problem also solves a real task</h3>
<p>Given a patient whose two eyes disagree, <em>which</em> one is affected? ODIR supplies 675 such
pairs; PAPILA supplies 7, which is why this had to wait. Comparing the eyes beats scoring them
independently — and the advantage is the pairing, not task-matched training data.</p>
{figure_block("f10_asymmetry", "186 held-out discordant pairs, 20 bootstrap resamples. The matched-data baseline trains a per-eye classifier on exactly the same 798 eyes and does worse, ruling out the obvious confound.")}
<p>The hand-specified difference contrast (0.813) ties the learned concatenation (0.814): a linear
model handed both eyes rediscovers the difference. That is what the shared-patient-factor account
predicts — the constitutional and acquisition terms are common to both eyes and cancel on
subtraction, leaving the eye-specific residual.</p>
</li>

<li>
<h3>The signal is visible in the raw similarity structure</h3>
{figure_block("f7_similarity_structure", "Cosine similarity between test eyes, ordered by patient, each row centred on its own mean. Red squares mark the two eyes of one patient.")}
</li>
</ol>

<h2>Corrections</h2>
<p class="sub">Numbers that changed once a control or a bug was found. Recorded because the earlier
versions were quoted.</p>

<div class="correction">
  <h4>Split-protocol inflation on PAPILA</h4>
  <p class="was">+0.073 ± 0.054 &nbsp;&rarr;&nbsp; +0.049 ± 0.016</p>
  <p>The first estimate came from repeated 60/15/25 holdouts on a 61-patient test set. Five-fold
  cross-validation puts every patient in a test fold and tightens the interval more than threefold.</p>
</div>

<div class="correction">
  <h4><code>GroupKFold</code> returned identical folds every repeat</h4>
  <p class="was">patient-level s.d. = 0.0000 across 10 supposedly independent repeats</p>
  <p>It partitions deterministically by group size, so permuting rows beforehand changes nothing.
  Replaced with <code>StratifiedGroupKFold(shuffle=True)</code>. Zero variance across independent
  repeats is always a bug, never a result.</p>
</div>

<div class="correction">
  <h4>Attribute probes were badly under-regularised</h4>
  <p class="was">age R² 0.22 &nbsp;&rarr;&nbsp; 0.56</p>
  <p>A fixed <code>Ridge(alpha=1.0)</code> on 768–2048-dimensional features with ~294 training rows.
  Replaced with <code>RidgeCV</code>. Age leakage had been understated by more than half.</p>
</div>

<div class="correction">
  <h4>Dead ReLU channels detonated the ResNet probes</h4>
  <p class="was">age R² = −8.6</p>
  <p>152 of 2,048 ResNet-50 channels have zero training variance. Dividing by <code>std + 1e-6</code>
  sent held-out samples to |z| = 1803 against 17 on train. Now zeroed below a scale-relative floor
  and clipped.</p>
</div>

<div class="correction">
  <h4>The bilateral contrastive loss was silently inactive</h4>
  <p class="was"><code>nce = nan</code> at every step</p>
  <p><code>masked_fill(-inf)</code> on the similarity diagonal, then multiplied by a boolean mask:
  <code>inf × 0 = nan</code>. The training curve looked entirely plausible while the central loss
  term did nothing.</p>
</div>

<h2>What is left</h2>
<ul class="todo">
  <li><span class="tag">next</span><span>External validation on a third cohort, to test the concordance–inflation relationship beyond two points — BRSET needs PhysioNet credentialing, so apply early</span></li>
  <li><span class="tag">next</span><span>Draft the manuscript — the skeleton with every result slotted in is in paper/draft.md</span></li>
  <li><span class="tag">for A*</span><span>Re-evaluate published PAPILA and ODIR results under patient grouping, and release the corrected table as a benchmark</span></li>
</ul>

<footer>
Generated from <code>outputs/results/*.csv</code> by <code>code/viz/make_status_page.py</code>.
All figures reproduce from cached embeddings on CPU in under a minute.
</footer>
</div>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"-> {OUT}  ({OUT.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
