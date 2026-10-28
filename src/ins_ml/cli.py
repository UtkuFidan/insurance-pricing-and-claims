"""Command line interface: ``ins-ml pricing | reserving | fraud | all``."""

from __future__ import annotations

import argparse

import pandas as pd

from .data import build_policy_table, load_openml, synthetic_portfolio


def _load(args):
    if args.synthetic:
        freq, sev = synthetic_portfolio(args.n, seed=0)
    else:
        freq, sev = load_openml(args.data)
    return build_policy_table(freq, sev), sev, freq


def _reconcile(freq, sev, df, out):
    """Write how the claim counts in the frequency table line up with the severity amounts."""
    import json
    from pathlib import Path

    from .data import amount_share_by_exposure, reconcile_claims

    tbl, info = reconcile_claims(freq, sev)
    Path(out).mkdir(parents=True, exist_ok=True)
    tbl.to_csv(Path(out) / "claims_reconciliation.csv", index=False, float_format="%.5g")
    amount_share_by_exposure(df).to_csv(
        Path(out) / "amount_share_by_exposure.csv", index=False, float_format="%.5g"
    )
    (Path(out) / "claims_reconciliation.json").write_text(json.dumps(info, indent=2))
    return tbl, info


def _french_context(args, out):
    """French results for the cross-market table: from this run or from files written earlier."""
    from pathlib import Path

    names = ("frequency_cv.csv", "tuned_params.json", "shap_importance.csv")
    freq_csv, hp_json, shap_csv = (Path(out) / n for n in names)
    if not (freq_csv.exists() and hp_json.exists() and shap_csv.exists()):
        return None
    import json

    df, _, _ = _load(args)
    return {
        "frequency_cv": pd.read_csv(freq_csv),
        "hps": json.loads(hp_json.read_text()),
        "shap": pd.read_csv(shap_csv),
        "policies": df,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ins-ml", description=__doc__)
    p.add_argument(
        "command", choices=["reconcile", "pricing", "spain", "reserving", "fraud", "all"]
    )
    p.add_argument(
        "--synthetic", action="store_true", help="use a seeded synthetic portfolio (offline)"
    )
    p.add_argument("--n", type=int, default=20000, help="synthetic portfolio size")
    p.add_argument("--data", default="data/openml", help="cache directory for OpenML downloads")
    p.add_argument("--spain-data", default="data/mendeley", help="cache for the Mendeley download")
    p.add_argument("--out", default="results", help="directory for CSV/JSON tables")
    p.add_argument("--figs", default="docs", help="directory for PNG figures")
    p.add_argument("--sims", type=int, default=2000, help="bootstrap ODP simulations")
    p.add_argument("--rounds", type=int, default=1200, help="max boosting rounds")
    p.add_argument("--folds", type=int, default=5, help="cross-validation folds")
    p.add_argument("--trials", type=int, default=25, help="Optuna trials per boosted model")
    p.add_argument(
        "--reuse-params", action="store_true", help="reuse results/tuned_params.json, skip tuning"
    )
    args = p.parse_args(argv)
    pd.set_option("display.width", 140)

    if args.command in ("reconcile", "pricing", "all"):
        df, sev, freq = _load(args)
        print(_reconcile(freq, sev, df, args.out)[0])
    if args.command in ("pricing", "all"):
        from .pricing import run_pricing

        res = run_pricing(
            df,
            args.out,
            args.figs,
            folds=args.folds,
            trials=args.trials,
            rounds=args.rounds,
            reuse_params=args.reuse_params,
        )
        print(
            res["meta"], "\n", res["frequency_cv"].round(4), "\n", res["pure_premium_cv"].round(4)
        )
    if args.command in ("spain", "all"):
        from .spain import run_spain

        res = run_spain(
            args.out,
            args.figs,
            synthetic=args.synthetic,
            data_dir=args.spain_data,
            folds=args.folds,
            trials=args.trials,
            rounds=args.rounds,
            reuse_params=args.reuse_params,
            french=None if args.synthetic else _french_context(args, args.out),
        )
        print(res["meta"], "\n", res["comparison"])
    if args.command in ("reserving", "all"):
        from .reserving import run_reserving

        res = run_reserving(args.out, args.figs, n_sims=args.sims)
        print(res["mack"].round(0), "\n", res["summary"])
    if args.command in ("fraud", "all"):
        from .fraud import run_fraud

        if args.command == "fraud":
            df, sev, _ = _load(args)
        res = run_fraud(df, sev, args.out)
        print(res["summary"], "\n", res["reasons"][["reason", "claims"]], "\n", res["top"])


if __name__ == "__main__":
    main(argv=None)
