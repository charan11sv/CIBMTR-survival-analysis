# CIBMTR Survival — Multi-Approach Ensemble (Better v2)
*Notebook:* `cibmtr version 2.ipynb`  
*Scope:* Three complementary pipelines + final rank ensembling  
*Outputs:* `submission1.csv`, `submission2.csv`, `submission3.csv`, `submission.csv` (final)

Unlike the first one this notebook is a copy of another kaggle user of which i made a few changes

This notebook is a **stronger, more comprehensive v2** of the earlier single-target NA+LightGBM approach. It tries three different pipelines aligned with the leaderboard’s *ranking* objective and *equity* emphasis, then blends them with **rank-based ensembling**:

1. **Event-masked Pairwise Ranking Neural Network (PRL-NN)**  
2. **“Yunbase”–style tree ensemble with subgroup-aware target & weights**  
3. **Multi-target survival ensemble** (Cox partial hazard, KM survival, NA cumulative hazard, signed time) with CatBoost/LightGBM/**CatBoost-Cox**  
4. **Final rank averaging** across the three submissions

---

## Why this notebook is better than the previous one

**The previous notebook** fit a single **LightGBM regressor** on **\(-\tilde H(t)\)** (Nelson–Aalen) and submitted one risk score.

**This notebook improves on that in four ways:**

1. **Objective closer to the leaderboard metric.**  
   - Uses a **pairwise hinge ranking loss** with **censoring-aware masks** (PRL-NN), which directly optimizes *ordering* (C-index proxy), rather than a squared error to a surrogate target.

2. **Multiple survival signals, not just one.**  
   - Learns from **four complementary targets**:  
     - **Cox partial hazard** (semi-parametric ranking signal),  
     - **KM survival \(S(t)\)** (non-parametric survival),  
     - **NA \(-H(t)\)** (non-parametric hazard),  
     - **Signed time** (positive=evident event time, negative=censored).  
   - Adds **CatBoost with Cox loss** to model proportional hazards directly.

3. **Fairness & subgroup robustness baked in.**  
   - **PRL-NN** adds a **race-wise variance penalty** and an **auxiliary loss** masked to events.  
   - **Yunbase variant** adjusts **weights by race_group** and builds subgroup-aware targets (KM within subgroup).

4. **Rank-based ensembling across diverse learners.**  
   - Converts all model outputs to **ranks** and blends them with tuned weights → robust to calibration differences, aligned with a rank metric.

---

## Contents
- [Data & Files](#data--files)
- [Approach A — Event-masked PRL-NN](#approach-a--event-masked-prl-nn)
- [Approach B — Yunbase Variant](#approach-b--yunbase-variant)
- [Approach C — Multi-Target Tree Ensemble](#approach-c--multi-target-tree-ensemble)
- [Final Ensemble](#final-ensemble)
- [How to Run](#how-to-run)
- [Reproducibility & Environment](#reproducibility--environment)
- [Limitations & Notes](#limitations--notes)

---

## Data & Files

**Inputs** (Kaggle paths expected):
- `train.csv`, `test.csv`, `sample_submission.csv` from the competition
- Prebuilt wheels for `lifelines` (installed in-notebook)
- (For Yunbase block) a bundled `baseline.py` copied into working dir

**Outputs**
- `submission1.csv` — Yunbase variant  
- `submission2.csv` — Event-masked PRL-NN  
- `submission3.csv` — Multi-target tree ensemble  
- `submission.csv` — Final rank-averaged submission from the three above

---

## Approach A — Event-masked PRL-NN
**Goal:** learn a **censoring-aware ranking score** using a neural network trained with a **pairwise hinge loss**, then **boost event-likely cases** via a separate classifier “mask”.

### Pipeline
1. **Event classifier (mask):**  
   - 5-fold **XGBoost Classifier** on `efs` to estimate \(P(\text{event})\).  
   - Converts probs → hard mask; used to **bump** NN risk scores for likely events *(+margin)*.

2. **Feature handling:**  
   - Categorical label encoding + learned embeddings; numeric features standardized.

3. **Pairwise Ranking NN (PyTorch Lightning)**
   - **Architecture:** categorical embeddings → concat with scaled numerics → MLP head; also a small **auxiliary head** predicting time (used only on **events**).  
   - **Loss \(\mathcal{L}\):**
     - **Main:** pairwise hinge **ranking** loss on valid pairs with at least one event;  
       invalid pairs masked when censoring prevents known ordering.  
     - **Auxiliary:** MSE on event rows (predict time), **masked** to `efs=1`.  
     - **Fairness reg.:** adds a **race-group variance penalty** on pairwise loss to reduce subgroup disparity.
   - **Training:** PL trainer with early LR scheduling, optional SWA.

4. **Scoring & submission**
   - Computes CV metric (competition C-index style).  
   - Applies **event mask margin** to OOF & test predictions.  
   - Writes **`submission2.csv`**.

**Why it helps:** directly optimizes **ordering**, uses **censoring logic** in the loss, and incorporates **subgroup robustness**.

---

## Approach B — Yunbase Variant
**Goal:** a strong, lightweight **tabular tree ensemble** with **subgroup-aware target shaping** and **race-aware weights**.

### Key choices
- **Subgroup KM target:** for each **race_group**, compute **KM survival** at each subject’s `efs_time` → a continuous **survival score**.  
  - Apply a small **downshift** for censored rows within each subgroup to separate ties.
- **Race-aware weighting:** row weights scale by a **race_group mapping** and event/censor mix (to reduce subgroup skew during training).
- **Models:** **LightGBM** + **CatBoost** regressors; simple 50–50 blend.
- **Metric parity check:** a local **`score()`** function replicates the competition’s stratified C-index evaluation.

**Outputs**
- Saves OOF predictions for audit, prints per-model and blended CV.  
- Writes **`submission1.csv`**.

**Why it helps:** the **target is explicitly subgroup-aware**, and the **weights** nudge learning to keep **equity** while remaining fast and stable on tabular data.

> *Note:* this block contains minor transformations (e.g., time transforms) from the source it was adapted from. Keep them if they improved CV; otherwise consider replacing with the cleaner targets used in Approach C.

---

## Approach C — Multi-Target Tree Ensemble
**Goal:** capture **complementary survival signals** and blend them with **ranked ensembling**.

### 1) Feature engineering & typing
- **HLA match aggregation:** combines multiple HLA low/high-res match flags into **summary counts** (e.g., across A/B/C/DRB1/… loci).  
- Cast **categoricals** to `category`; leave numerics as numeric.

### 2) Targets (built with 5-fold CV inside the notebook)
For each fold, compute out-of-fold (OOF) labels:

- **`target1`: Cox partial hazard**  
  - Fit **Lifelines CoxPH** on train fold (one-hot cats; constant-col guards).  
  - Predict partial hazard on valid fold → OOF vector.

- **`target2`: Kaplan–Meier survival \(S(t)\)**  
  - Fit KM on train fold and evaluate at each valid row’s `efs_time`.

- **`target3`: Nelson–Aalen \(-H(t)\)**  
  - Fit NA on train fold and return **negative cumulative hazard** at valid times.

- **`target4`: Signed time**  
  - `+efs_time` for events, `-efs_time` for censored (simple ordering surrogate used by Cox-aware models).

### 3) Base models (per target, 5-fold)
- **CatBoostRegressor** (RMSE) on `target1/2/3`  
- **LightGBMRegressor** (RMSE) on `target1/2/3`  
- **CatBoostRegressor with `loss_function="Cox"`** on `target4` (two variants with different depth/grow policies)

This yields **8 model streams** with OOF preds and test preds.

### 4) Rank-weighted ensemble
- Convert each model’s predictions to **ranks** (handles scale/calibration mismatches).  
- Blend with tuned weights, e.g.  
