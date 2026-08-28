# Your Other Eye

Code for ***Your Other Eye: measuring patient-identity leakage in retinal
representations***.

---

## What this code does

Every fundus cohort contains two eyes per patient, and most pipelines treat
those eyes as independent samples. This repository is the measurement apparatus
for what that assumption costs.

It does four things:

**1. Measures how much patient identity a representation carries.** Given one
eye, it asks a frozen encoder to retrieve that patient's *other* eye from the
whole test split, and reports Recall@1 and MRR against an explicit chance rate.
It also fits linear probes for patient attributes (age, sex) and a patient-ID
classifier.

**2. Measures what an image-level split costs.** It trains the same classifier
on the same features twice — once with folds grouped by patient, once with
images shuffled freely — and reports the difference under repeated
cross-validation.

**3. Tests whether the identity signal can be removed.** Three families:
an adversarial disentanglement adapter with a conditional fellow-eye
discriminator, a closed-form conditional pair-mean subspace projection, and
illumination normalisation at the preprocessing stage. Each is scored on both
axes at once — identity removed against task accuracy retained.

**4. Uses the bilateral structure deliberately.** On patients whose two eyes
carry different labels, it tests whether comparing the eyes identifies the
affected one better than scoring them independently.

Two controls are built in throughout. Raw-pixel and colour-histogram baselines
run through the identical retrieval protocol, so any claim about a learned
encoder is stated against what trivial image similarity already achieves. And
where a cohort provides two fields per eye, *which eye* and *which view* are
separated rather than confounded.

Everything runs on a CPU. The backbone is frozen and its output cached to disk,
so the only heavy step happens once per (dataset, encoder, preprocessing
variant); every experiment afterwards reads those cached vectors.

---

## Where the data comes from

None of it is redistributed here. The scripts download it, and each dataset
keeps its own licence and citation requirements.

| Dataset | Source | Licence | Credentials |
|---|---|---|---|
| **PAPILA** | figshare `10.6084/m9.figshare.14798004` — Kovalyk et al., *Scientific Data* 2022 | GPL-3.0+ | none |
| **ODIR-5K** | HF `bumbledeep/odir`, a mirror of the Kaggle release — Peking University / Shanggong Medical | MIT (mirror) | none |
| **DeepDRiD** | `github.com/deepdrdoc/DeepDRiD` — Liu et al., *Patterns* 2022 | CC BY-SA 4.0 | none |
| **RETFound** (weights) | HF `bitfount/RETFound_MAE` — Zhou et al., *Nature* 2023 | CC BY-NC 4.0 | none |

All four are public and need no application or API token. Roughly 2.3 GB of
images in total, plus 1.2 GB for the RETFound checkpoint.

`code/dataio/download_odir.py` is an alternative route to ODIR-5K that pulls
from Kaggle directly. It needs an API token at `~/.kaggle/kaggle.json` and is
not required — the Hugging Face mirror is the default path.

Please cite the original dataset papers if you use them.

---

## Running it

### Install

**Python 3.11.** PyTorch has no wheels for 3.12+ at the pinned version, and
3.10 or earlier lacks the typing syntax used here.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

No GPU is required or used.

### Smallest complete run

Four commands, about fifteen minutes on a laptop:

```bash
python code/dataio/download_papila.py
python code/dataio/build_manifest.py --dataset papila
python code/features/extract_embeddings.py --dataset papila --variant canon
python code/experiments/run_experiments.py --dataset papila --exp E1 E7
```

### Step 1 — download the cohorts

```bash
python code/dataio/download_papila.py
python code/dataio/download_odir_hf.py
python code/dataio/download_deepdrid.py
```

### Step 2 — build the manifests

```bash
python code/dataio/build_manifest.py
```

Flattens every cohort into one schema — `patient_id`, `eye`, `image_path`,
`pair_id`, `y_*` labels, `split` — and assigns splits **by patient**. It also
emits `split_image`, an i.i.d. split over images, which exists only so the
experiments can measure what that alternative costs. Writes to
`data/manifests/`.

### Step 3 — extract features

The one expensive step. Run once per configuration; everything downstream reads
the cache in `outputs/embeddings/`.

```bash
# main configuration
python code/features/extract_embeddings.py \
    --dataset papila odir deepdrid_fields --backbone dinov2_vitb14 --variant canon

# preprocessing ablation
python code/features/extract_embeddings.py --dataset papila \
    --variant raw crop canon canonbg

# encoder ablation
python code/features/extract_embeddings.py --dataset papila \
    --backbone dinov2_vits14 resnet50_in1k retfound_cfp --variant canon
```

Runtime on the reference machine (Intel i7-1355U, 10 threads, no GPU):

| Backbone | dim | img/s | PAPILA (488) | DeepDRiD (1,600) | ODIR (6,068) |
|---|---|---|---|---|---|
| DINOv2 ViT-B/14 | 768 | 1.2 | 7 min | 22 min | 84 min |
| DINOv2 ViT-S/14 | 384 | 2.8 | 3 min | 10 min | 36 min |
| ResNet-50 | 2048 | 2.8 | 3 min | 10 min | 36 min |
| RETFound ViT-L/16 | 1024 | 0.95 | 9 min | 28 min | 106 min |

RETFound weights download automatically on first use.

**DeepDRiD note.** It has two fields per eye. Extract on `deepdrid_fields` (all
1,600 images), then copy the cache so the one-field manifest can share it:

```bash
cp outputs/embeddings/deepdrid_fields__dinov2_vitb14__canon.npz \
   outputs/embeddings/deepdrid__dinov2_vitb14__canon.npz
```

### Step 4 — run the experiments

```bash
python code/experiments/run_experiments.py --dataset papila   --exp all
python code/experiments/run_experiments.py --dataset odir     --exp all
python code/experiments/run_experiments.py --dataset deepdrid --exp all
python code/experiments/run_experiments.py --dataset deepdrid_fields --exp E9
python code/experiments/run_experiments.py --dataset papila \
    --backbone retfound_cfp --exp E1 E7
```

| | Experiment | What it measures |
|---|---|---|
| E1 | leakage audit | fellow-eye retrieval, patient-ID probe, attribute probes, task AUC |
| E2 | split protocol | image- vs patient-level splits, repeated holdout |
| E3 | mitigation | identity-vs-accuracy trade-off for subspace projection |
| E4 | dimensionality | how many directions must be destroyed to remove identity |
| E5 | low label | task AUC against the labelled-patient budget |
| E6 | pixel baseline | learned features vs raw pixels and colour histograms |
| E7 | cv split | split-protocol difference under repeated cross-validation |
| E8 | asymmetry | which of a patient's two eyes is the affected one |
| E9 | same-eye control | same-eye vs fellow-eye retrieval, field of view held fixed |

Results land in `outputs/results/` as one CSV per experiment.

The adversarial adapter is trained separately:

```bash
python code/experiments/train_adapter.py --dataset papila --label y_glaucoma \
    --epochs 40 --seeds 0 1 2 --tag adv_papila
```

### Step 5 — figures and tables

```bash
python code/viz/make_figures.py     # PNG + vector PDF, into outputs/figures/
python code/viz/make_tables.py      # Markdown + LaTeX booktabs, into outputs/tables/
```

`code/viz/verify_draft_numbers.py` re-derives every number quoted in the
manuscript from the result CSVs and fails if any has drifted. Run it after
changing anything upstream.

---

## Layout

```
code/
├── config.py                        paths, encoder registry, preprocessing variants
├── dataio/
│   ├── download_papila.py           figshare
│   ├── download_odir_hf.py          Hugging Face mirror (default)
│   ├── download_odir.py             Kaggle route (needs a token; optional)
│   ├── download_deepdrid.py         GitHub
│   ├── preprocess.py                circle crop, laterality canonicalisation, Ben Graham
│   ├── build_manifest.py            unified schema, patient-grouped splits
│   └── embeddings.py                cached-feature store, pair-aware batch sampler
├── features/
│   └── extract_embeddings.py        frozen-encoder extraction, cached to .npz
├── models/
│   ├── bilateral.py                 adapter, gradient reversal, conditional adversary
│   └── subspace.py                  conditional pair-mean projection, INLP baseline
├── experiments/
│   ├── probes.py                    retrieval / attribute / identity probes, bootstrap CIs
│   ├── run_experiments.py           E1–E9
│   └── train_adapter.py             adversarial training loop
└── viz/
    ├── style.py                     shared plotting style
    ├── make_figures.py              all figures
    ├── make_tables.py               all tables
    └── verify_draft_numbers.py      manuscript number check

data/       created by the download scripts   (git-ignored)
outputs/    embeddings, results, figures, tables   (git-ignored)
```

---

## Protocol choices worth knowing before reusing this

**Splits are by patient.** `build_manifest.py` emits both `split`
(patient-grouped) and `split_image` (i.i.d. over images). Use `split`.
`split_image` is instrumentation, not an option.

**Laterality is canonicalised before anything is measured.** Left eyes are
mirrored so all eyes share one anatomical orientation, and the black frame
around the retinal disc is cropped away. Both are side- or camera-specific cues
that would otherwise dominate any retrieval result.

**Encoder normalisation is per-backbone.** RETFound expects 0.5/0.5 statistics,
not ImageNet's. Feeding it ImageNet-normalised input degrades every feature it
returns without raising an error, so the statistics live in the encoder registry
in `config.py` rather than being hardcoded in the preprocessing.

**Retrieval numbers need their candidate-pool size.** R@1 is not comparable
across test splits of different size, because every representation saturates on
a small pool. `probes.fellow_eye_retrieval` returns the chance rate and the lift
alongside R@1 for that reason.

**Features are z-scored with training-split statistics**, with degenerate
dimensions zeroed rather than divided through — post-ReLU backbones carry dead
channels whose zero training variance would otherwise send held-out samples to
absurd values.

---
