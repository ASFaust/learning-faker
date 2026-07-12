# Task selection: normalization anchor, failure modes, and inclusion criteria

Reference for which HPO tasks belong in the learning-curve simulator and why some
are excluded. Written as we add sources (LCBench, PD1, FCNet, TaskSet) and hit
recurring pathologies; criteria are converging, so this is the checklist for
vetting any *new* source.

## 1. Datasets currently included

| source | time axis | first logged step | true init? | notes |
|---|---|---|---|---|
| LCBench | epoch | 0 | ~yes | 35 tasks, SGD + cosine |
| PD1 | global_step | **1000** | **no** | 19 tasks; first eval already ~1000 steps in |
| FCNet | epoch | 0 | yes | 4 tasks, 4 seed replicas (aleatoric signal) |
| TaskSet | step | 0 | yes | diverse RNN/CNN/MLP/flow/VAE; HPs from seeds |

All four log **both val and train loss, co-sampled on the same time grid** (verified);
we use the checkpoint-evaluated train loss, not the noisy minibatch loss.

## 2. Normalization anchor = median initial loss

Targets are `y = log(loss / ref_task)`. `ref_task = median over the task's runs of the
val loss at the FIRST observation (index 0)`.

**Why the initial loss, and why the median:**
- The initial loss is **divergence-immune**: at (or just after) initialization no config
  has diverged yet, so configs cluster tightly at the architecture's init loss
  (within-task max/min ≈ 1.0 for healthy tasks). Contrast the earlier `p25 of loss at
  t_rel≥5%` anchor, which got *contaminated* on tasks where many configs diverge by 5%
  (the RNN failure: a p25 ref of 1.9e4).
- The **median** is robust to the diverged minority: even PD1 (no true step-0, first
  obs ≈ step 1000) and FCNet (MSE regression, some configs blow up fast) give a sane
  median anchor because it ignores the tail.
- Sanity check: healthy init medians match modality expectations — the RNN classifier
  sits at **0.69 ≈ ln 2**, word-LMs at ~9 ≈ ln(vocab), VAEs at ~2e4 nats.

**Inference protocol.** For an unseen task, obtain the anchor by evaluating a sensible
config at/near init on the val set — cheap (≈no training) and config-independent for
healthy tasks. Then train a bit for the partial curve and invert the task embedding.
(PD1-style tasks were trained with a step-1000 anchor, a mild train/inference definition
mismatch that embedding-inversion absorbs as a constant offset.)

## 3. Failure modes → exclusion criteria

### 3a. Synthetic optimization toys — excluded by family
Convex bowls / synthetic loss surfaces used for optimizer meta-learning, **not real NN
training**. They converge to machine precision (val loss ~1e-18..1e-22, far below any
real-NN floor) and/or are pathologically bimodal, so no reference is sane.

- **Excluded families:** `quadratic_family`, `TwoD_Bowl`, `losg_tasks_family`.
- **Detector:** `scripts/audit_families.py` — per-run convergence depth. Flag if
  `abs_min < 1e-6` (deepest real task, fcnet/naval regression, only reaches ~4.8e-5) or
  median per-run depth `log(loss/ref) < -12`.

### 3b. Majority-diverging tasks — excluded by behavior
Real architectures, but a **majority of the sampled HP configs diverge before the first
observation** (catastrophic LR blows the loss up in the first step). The median init is
then itself a diverged value, so the per-task anchor is unusable and every config —
including the healthy minority — gets garbage targets.

- **Detector:** `scripts/find_diverging_tasks.py` — flag a task if
  `median-init > 10 × family-median-init` (modality-relative, so legit high-baseline
  families like VAEs aren't caught, only genuine outliers).
- **Excluded (21 TaskSet tasks):** rnn_text_classification `seed58/70/96`;
  mlp_ae `seed39/48/62`; mlp `seed39/43/49/78`; conv_fc `seed36/38/40/70`;
  conv_pooling `seed91`; char_rnn_language_model `seed38/39`;
  word_rnn_language_model `seed4/8/39`; nvp `seed75`.

## 4. Inclusion checklist for a NEW task / source

A task should satisfy all of:
1. **Real NN training**, not a synthetic optimization test function.
2. **Both val + train loss**, co-sampled on the same time grid.
3. **Sane init anchor**: median initial val loss ≈ the modality norm (≈ ln(#classes) for
   classifiers, reconstruction/NLL scale for AE/flow/VAE); configs cluster at init
   (within-task max/min near 1).
4. **Realistic convergence floor**: min loss above ~1e-5 (not machine precision).
5. **Not majority-diverging**: median-init not ≫ its family/modality norm.

**Vetting procedure:** run `scripts/audit_families.py` (catches 3a via convergence depth)
and `scripts/find_diverging_tasks.py` (catches 3b via median-init outliers); add flagged
families/tasks to the exclusions in `lcfaker/data/taskset.py`.

## 5. Deferred refinement

The 21 majority-diverging tasks are *dropped* wholesale, which discards their healthy
config minority (and the divergence-boundary signal it carries). A future option:
anchor on a **low percentile of the initial loss** (≈ p10 of index-0) instead of the
median — this estimates the true architecture init from the well-behaved configs and
would rescue tasks whose *best* configs are sane, dropping only those where even the
minimum init is diverged (e.g. rnn seed58, min-init ~5e5). Not done yet.
