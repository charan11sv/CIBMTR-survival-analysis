# CIBMTR Survival Analysis — Theory-Backed Kaggle Submission
*Notebook:* `cibmtr version 1.ipynb`  
*Competition:* CIBMTR — Equity in post-HCT Survival Predictions. The leaderboard focused on ranking patients by risk using a concordance-style objective with equity considerations across subgroups. :contentReference[oaicite:0]{index=0}

---

## TL;DR (What this notebook does)
1) Audits data and **imputes missing values** with **Predictive Mean Matching (PMM)** via Iterative Imputer (MICE).  
2) Builds a **non-parametric survival target** using the **Nelson–Aalen cumulative hazard** \(\tilde H(t)\) and uses **\(-\tilde H(t)\)** as a **regression label**.  
3) Trains a **LightGBM regressor** on mixed numeric/categorical features to produce a **monotone risk score** suitable for **C-index ranking** (any monotone transform preserves pairwise order).  
4) Writes `submission.csv` with columns: `ID, prediction`.

---

## 1) Problem Framing

### Goal
Rank patients by post-HCT **event-free survival (EFS)** risk, where **events** include EFS-defined adverse outcomes; **censoring** indicates incomplete follow-up. The competition’s leaderboard rewarded **good ranking of risk** and **equitable performance** across subgroups, hence we use a **monotonic risk score** with careful handling of censored data. :contentReference[oaicite:1]{index=1}

### Notation & Risk Metrics (Survival Primer)
Given time-to-event \(T\), event indicator \(\delta\in\{0,1\}\), and time \(t\ge 0\):

- **Survival function**: \(S(t)=P(T>t)\).  
- **Density**: \(f(t)=\frac{d}{dt}\,[1-S(t)]\).  
- **Hazard**: \(h(t)=\lim_{\Delta t\to 0}\frac{P(t\le T<t+\Delta t\mid T\ge t)}{\Delta t}=\frac{f(t)}{S(t)}\).  
- **Cumulative hazard**: \(H(t)=\int_0^t h(u)\,du\).  
- **Link**: \(S(t)=\exp[-H(t)] \iff \log S(t)=-H(t)\).

**Nelson–Aalen estimator** provides a **non-parametric** estimate of \(H(t)\) that is robust to right-censoring:  
\[
\tilde H(t)=\sum_{t_i\le t}\frac{d_i}{n_i},
\]
where \(d_i\) is the number of events and \(n_i\) the number at risk at time \(t_i\). :contentReference[oaicite:2]{index=2}

> **Why cumulative hazard?**  
> - It directly accounts for censoring via risk sets \((n_i)\).  
> - It connects cleanly to survival via \(S(t)=e^{-\tilde H(t)}\).  
> - It is **monotone in risk**: higher \(\tilde H(t)\) ⇒ higher risk by time \(t\).

---

## 2) Why These Choices

### 2.1 Nelson–Aalen → \(-\tilde H(t)\) as the Learning Target
- **Leaderboard favors ranking (C-index)**: any **strictly monotone transform** of a risk score preserves pairwise order. Using **\(-\tilde H(t)\)** is equivalent (up to monotone transform) to modeling \(\log S(t)\), making a regressor’s outputs naturally interpretable as a **pseudo log-survival** signal. This aligns with a **ranking metric** without committing to proportional hazards or a specific parametric form. :contentReference[oaicite:3]{index=3}
- **Non-parametric baseline**: Nelson–Aalen does not assume proportional hazards or a specified distribution (contrast with Cox PH or AFT). It is robust and easy to compute. :contentReference[oaicite:4]{index=4}
- **Censoring tweak**: a small negative offset for censored rows (e.g., \(-0.15\)) creates a **margin** between observed events vs. censored at the same follow-up, which can sharpen ranking in a regression fit (pragmatic margin-based learning).

> Alternatives considered:  
> - **Cox PH** (semi-parametric): directly optimizes partial likelihood for ranking but assumes proportional hazards.  
> - **AFT** (parametric/semi-parametric): models \(\log T\), strong distributional choices.  
> - **Discrete-time survival** or **gradient boosting survival (Cox objectives)**: excellent options, but the NA-label + general regressor is fast, flexible, and performed competitively in community write-ups. :contentReference[oaicite:5]{index=5}

### 2.2 PMM Imputation (Iterative/MICE)
- **Predictive Mean Matching** draws **plausible** donor values from observed data instead of purely model-generated numbers, which helps **preserve distributions and category coherence**, especially for skewed clinical features.  
- **Iterative chain (MICE)** uses conditional models per feature, propagating multivariate structure.  
- **Why not mean/median?** They distort variance and tails—harmful for risk ranking. **PMM** mitigates this while being simpler than fully Bayesian imputation.

### 2.3 LightGBM Regressor (tabular, mixed types)
- **Handles nonlinearities & interactions** without heavy feature engineering.  
- **Efficient** with large/mixed (numeric + categorical) tabular data.  
- **Robust ranking**: although trained with RMSE on \(-\tilde H\), the **induced ordering** often aligns with C-index; additional pairwise/ranking losses are optional but not required to get strong baselines in practice. Community solutions explored NA labels with gradient boosting variants successfully. :contentReference[oaicite:6]{index=6}

### 2.4 Equity Considerations
The competition emphasized **equitable performance** (e.g., not degrading ranking quality for specific subgroups). While this notebook trains a single global model, its target choice (NA) and monotone risk formulation are compatible with:  
- **Group-wise C-index** monitoring, and  
- **Fairness-aware ensembling or reweighting** at training or calibration time. :contentReference[oaicite:7]{index=7}

---

## 3) End-to-End Pipeline (What each step does)

1. **Load & Audit**  
   - Load `train.csv` / `test.csv` (index `ID`).  
   - Compute per-column missingness; (in the shared notebook threshold was set high so that columns aren’t dropped).

2. **PMM Imputation (MICE)**  
   - Detect **categoricals** (object dtype) and **numerics**.  
   - Drop constant/near-constant columns.  
   - **Label-encode** categoricals temporarily for imputation; **standardize** numerics.  
   - Run **Iterative Imputer** with `sample_posterior=True` (PMM-like stochastic draws).  
   - **Inverse-transform** back to original scales and decode categories.

3. **Target Engineering (Nelson–Aalen)**  
   - Fit NA on `(efs_time, efs)` in a training slice.  
   - Compute label: `naf_label = - H̃(efs_time)`; apply **small censoring margin** (subtract a constant for \(\delta=0\)).

4. **Modeling (LightGBM Regressor)**  
   - Cast categoricals to `category`; align train/test columns.  
   - Train with moderate depth/leaves, conservative learning rate (as in the notebook).  
   - Output **`prediction`** per `ID` (higher = “more negative log-survival”, i.e., more risk).

5. **Submission**  
   - Save `submission.csv` with `ID, prediction`.  
   - Designed to be **monotone risk scores** compatible with **C-index ranking**.

---

## 4) Theoretical Details (Deeper dive)

### 4.1 From Survival to a Regressable Label
- We do **not** directly predict time-to-event. Instead, we estimate \(H(t)\) non-parametrically and train on **\(-H(t)\)**. This creates a **continuous target**, smoothly increasing with risk at the observed \(t\), and—crucially—  
  \[
  \underbrace{-H(t)}_{\text{label}} \quad \text{is monotone in}\quad \underbrace{1-S(t)}_{\text{risk by time }t}.
  \]
- Because the leaderboard’s objective is **ranking** (C-index) rather than strict calibration, **any strictly increasing transform** of a true risk score is acceptable. \(-H(t)\) is a principled, smooth proxy. :contentReference[oaicite:8]{index=8}

### 4.2 Why Nelson–Aalen (vs. KM / Cox / AFT)
- **KM** estimates \(S(t)\) stepwise; \(-\log S(t)\) is analogous to \(H(t)\) but NA works **natively** in hazard space and often shows nicer additivity for learning.  
- **Cox PH** optimizes a ranking-like partial likelihood but assumes **proportional hazards**; violations can hurt.  
- **AFT** posits a **parametric** form for \(\log T\); mis-specification risks bias.  
- **NA** is **assumption-light**, robust to censoring, and easy to compute in a single pass. :contentReference[oaicite:9]{index=9}

### 4.3 Handling Censoring
- At time \(t_i\), NA’s increment \(d_i/n_i\) uses only **events**; censored observations reduce the **risk set** thereafter.  
- The notebook adds a **small margin** to separate censored vs. event cases at equal times, making the regression **margin-aware** (useful when many ties exist).

### 4.4 What the Leaderboard Measured (and why our target fits)
- **Concordance Index (C-index)** computes the probability that among comparable pairs, the patient who fails **earlier** gets a **higher risk score**. The competition highlighted **equity**, i.e., consistent discrimination across subgroups; your risk score should **rank well overall** and **not fail on subpopulations**. :contentReference[oaicite:10]{index=10}

> **Practical upshot:** Optimize a **smooth, monotone** risk proxy (like \(-\tilde H\)) using a **strong tabular learner**; then check **subgroup C-indices** (fairness).

---

## 5) Modeling Details

### 5.1 Features & Types
- **Numeric**: standardized for imputation; raw scale restored for modeling.  
- **Categorical**: label-encoded only for imputation; in modeling, cast to `category` so LightGBM uses native categorical splits (no one-hot explosion).

### 5.2 LightGBM (typical parameters used)
A compact, robust configuration (as in the notebook) might look like:
```python
lgbm_naf_params = {
  "objective": "regression", "metric": "rmse",
  "learning_rate": 0.05, "num_leaves": 10, "max_depth": 5,
  "min_data_in_leaf": 10, "min_sum_hessian_in_leaf": 1e-3,
  "feature_fraction": 1.0, "bagging_fraction": 1.0, "bagging_freq": 1,
  "lambda_l1": 0.0, "lambda_l2": 0.0,
  "seed": 53, "num_threads": 4
}
# trained for ~1000 boosting rounds
