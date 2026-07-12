"""Overnight hyperparameter optimization for the learning-curve surrogate.

    python -m lcfaker.hpo --gpus 1,3,5 --source joint4 --timeout 36000

Spawns one worker process per --gpus entry (indices in PCI-bus order, matching
nvidia-smi and train.py's --gpu). All workers share a single Optuna study via
file-based JournalStorage and coordinate ASHA pruning through it: weak trials are
killed after a couple of epochs, strong ones run to --max-epochs. The study is on
disk, so a killed/resumed run continues where it left off (reuse --study-name).

Objective = minimize held-out val pinball, on a fixed config-split seed so every
trial is scored on the same unseen curves. When it finishes it writes
best_config.json, which `python -m lcfaker.train --config <that file>` consumes to
train the winner to convergence.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import subprocess
import time
from pathlib import Path

# Match nvidia-smi / train.py device numbering; must precede any torch/CUDA init.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

SOURCES = ["lcbench", "pd1", "fcnet", "taskset", "joint", "joint3", "joint4", "synthetic"]


def suggest(trial):
    """The 9-D search space -> (ModelConfig kwargs, optimizer hp)."""
    cfg = dict(
        d_model=trial.suggest_categorical("d_model", [16, 24, 32, 48, 64]),
        hidden_dim=trial.suggest_categorical("hidden_dim", [64, 128, 192, 256]),
        num_freq_bands=trial.suggest_int("num_freq_bands", 4, 10),
        emb_dropout=trial.suggest_float("emb_dropout", 0.0, 0.4),
        dropout=trial.suggest_float("dropout", 0.0, 0.6),
        block=trial.suggest_categorical("block", ["gated", "silu"]),
        n_layers=trial.suggest_categorical("n_layers", [1, 2, 3]),
    )
    hp = dict(
        lr=trial.suggest_float("lr", 3e-4, 1e-2, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
    )
    return cfg, hp


def make_pruner(args):
    import optuna
    # ASHA: async successive halving. Resource = epochs reported via trial.report;
    # a trial can first be pruned at min_resource, then survivors thin by 1/rf each rung.
    return optuna.pruners.SuccessiveHalvingPruner(
        min_resource=args.min_resource, reduction_factor=args.reduction_factor)


def make_storage(journal_path):
    import optuna
    from optuna.storages.journal import JournalFileBackend
    # File journal (not SQLite): safe under concurrent writes from the GPU workers.
    return optuna.storages.JournalStorage(JournalFileBackend(str(journal_path)))


def query_gpu_mem():
    """{index: used_MiB} in PCI-bus order (nvidia-smi's own enumeration)."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True)
    return {int(i): int(m) for i, m in (ln.split(",") for ln in out.strip().splitlines())}


def check_gpus(gpus, allow_busy):
    """Validate the requested indices and abort (unless --allow-busy) if any is
    already occupied, so an HPO run can't collide with another job on a GPU."""
    try:
        used = query_gpu_mem()
    except Exception as e:  # nvidia-smi missing/odd output: skip the check, don't block
        print(f"WARNING: could not query nvidia-smi ({e}); skipping GPU availability check")
        return
    for g in gpus:
        if g not in used:
            raise SystemExit(f"--gpus: index {g} not found (visible: {sorted(used)})")
    busy = [g for g in gpus if used[g] > 2000]
    if busy:
        msg = ", ".join(f"gpu{g}={used[g]}MiB" for g in busy)
        if allow_busy:
            print(f"WARNING: selected GPUs already in use ({msg}); continuing (--allow-busy)")
        else:
            raise SystemExit(f"selected GPUs already in use ({msg}). "
                             f"Pick free ones or pass --allow-busy to override.")


def worker(gpu, args, journal_path):
    """One process pinned to one GPU: build data once, then pull trials from the
    shared study until the budget runs out."""
    import optuna
    import torch

    from .model import ModelConfig
    from .train import build_source, config_split, fit, make_loaders

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(gpu)
    device = f"cuda:{gpu}"
    tag = f"[gpu{gpu}]"
    print(f"{tag} {torch.cuda.get_device_name(gpu)}: loading source={args.source} ...", flush=True)

    # data + split built ONCE per worker and reused across all its trials
    ds, vocab = build_source(args.source)
    train_ds, val_ds = config_split(ds, args.val_frac, args.seed)
    train_dl, val_dl = make_loaders(train_ds, val_ds, args.batch_size, device)
    print(f"{tag} ready (train={len(train_ds)} val={len(val_ds)} tasks={vocab.n_tasks})", flush=True)

    study = optuna.load_study(
        study_name=args.study_name, storage=make_storage(journal_path),
        # seed the sampler per-GPU so workers explore different startup points;
        # they still coordinate through the shared storage once TPE kicks in.
        sampler=optuna.samplers.TPESampler(seed=args.seed + gpu, n_startup_trials=16),
        pruner=make_pruner(args))

    def objective(trial):
        cfg_kw, hp = suggest(trial)
        cfg = ModelConfig(**cfg_kw)

        def on_epoch(ep, record):
            trial.report(record["val_pinball"], ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

        best_val, best_ep = fit(cfg, hp, device, vocab, train_dl, val_dl,
                                epochs=args.max_epochs, on_epoch=on_epoch, log=False)
        trial.set_user_attr("best_epoch", best_ep)
        # a diverged config yields non-finite loss: return a finite penalty so TPE
        # learns to avoid that region rather than crash on NaN.
        return best_val if math.isfinite(best_val) else 1e3

    def log_cb(study, trial):
        try:
            best = f"{study.best_value:.4f}"
        except ValueError:
            best = "  -  "
        val = f"{trial.value:.4f}" if trial.value is not None else "  -  "
        print(f"{tag} trial {trial.number:4d} {trial.state.name.lower():8s} "
              f"val={val} ep={trial.user_attrs.get('best_epoch', '-')} | study best={best}",
              flush=True)

    study.optimize(objective, timeout=args.timeout, n_trials=args.n_trials,
                   callbacks=[log_cb],
                   # keep the overnight run alive through the odd OOM / numerical blowup
                   catch=(RuntimeError, ValueError, FloatingPointError))
    print(f"{tag} worker done", flush=True)


def report(study, out_dir, args, elapsed):
    states = [t.state.name for t in study.trials]
    n_done = states.count("COMPLETE")
    n_pruned = states.count("PRUNED")
    print(f"\n=== HPO finished in {elapsed / 3600:.2f}h: "
          f"{n_done} complete, {n_pruned} pruned, {len(states)} total ===")
    if n_done == 0:
        print("no completed trials -- nothing to report.")
        return
    best = study.best_trial
    print(f"best val pinball = {best.value:.4f}  "
          f"(trial {best.number}, best_epoch={best.user_attrs.get('best_epoch', '?')})")
    for k, v in best.params.items():
        print(f"    {k:16s} = {v}")
    best_path = out_dir / "best_config.json"
    best_path.write_text(json.dumps(
        {"value": best.value, "best_epoch": best.user_attrs.get("best_epoch"),
         "params": best.params}, indent=2))
    print(f"\nwrote {best_path}")
    print("train the winner to convergence with:")
    print(f"  python -m lcfaker.train --source {args.source} "
          f"--config {best_path} --epochs {max(args.max_epochs * 3, 30)}")


def main():
    ap = argparse.ArgumentParser(description="Overnight ASHA hyperparameter search.")
    ap.add_argument("--gpus", required=True,
                    help="comma-separated GPU indices in PCI-bus order, e.g. 1,3,5")
    ap.add_argument("--source", default="joint4", choices=SOURCES)
    ap.add_argument("--max-epochs", type=int, default=25,
                    help="max epochs per trial (ASHA prunes weak ones early, so few run this long)")
    ap.add_argument("--min-resource", type=int, default=2,
                    help="ASHA: epochs a trial must run before it can be pruned")
    ap.add_argument("--reduction-factor", type=int, default=3,
                    help="ASHA: keep top 1/factor of trials at each rung")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=None,
                    help="per-worker wall-clock budget in seconds (e.g. 36000 = 10h)")
    ap.add_argument("--n-trials", type=int, default=None,
                    help="optional per-worker trial cap (default: run until --timeout)")
    ap.add_argument("--study-name", default=None,
                    help="reuse a name to resume an interrupted study")
    ap.add_argument("--out", default=None, help="study dir (default: hpo/<study-name>)")
    ap.add_argument("--allow-busy", action="store_true",
                    help="run even if a selected GPU already has >2GB in use")
    args = ap.parse_args()

    if args.timeout is None and args.n_trials is None:
        raise SystemExit("give a budget: --timeout SECONDS and/or --n-trials N")

    gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    if not gpus:
        raise SystemExit("--gpus is empty")
    check_gpus(gpus, args.allow_busy)

    if not args.study_name:
        args.study_name = f"hpo_{args.source}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.out or f"hpo/{args.study_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    journal_path = out_dir / "study.journal"

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=args.study_name, storage=make_storage(journal_path),
        direction="minimize", pruner=make_pruner(args), load_if_exists=True)
    (out_dir / "hpo_args.json").write_text(json.dumps(vars(args), indent=2))

    print(f"study '{args.study_name}' -> {out_dir}")
    print(f"objective: minimize val pinball | source={args.source} max_epochs={args.max_epochs} "
          f"ASHA(min={args.min_resource}, rf={args.reduction_factor})")
    print(f"workers: {len(gpus)} on GPUs {gpus} | budget: timeout={args.timeout}s "
          f"n_trials={args.n_trials}")

    ctx = mp.get_context("spawn")  # required for CUDA in child processes
    procs = [ctx.Process(target=worker, args=(g, args, journal_path), name=f"gpu{g}")
             for g in gpus]
    t0 = time.time()
    for p in procs:
        p.start()
    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        print("\ninterrupt -> terminating workers (study is on disk and resumable)")
        for p in procs:
            p.terminate()
        for p in procs:
            p.join()

    report(study, out_dir, args, time.time() - t0)


if __name__ == "__main__":
    main()
