"""1 つの run_id 分のスコアリング実行器（学習は一切行わない）。

このファイルは**指標を一切計算しない**。jacob_cov のスコア (predicted_scores) と
ベンチマークの公表精度 (reference_scores) を書き出すだけで、順位相関などの指標は
外部の固定された評価層 airas-eval が `make evaluate` 経由で算出する。

各アーキテクチャに対して行うのは未学習ネットワークの順伝播・逆伝播 各 1 回のみ。

出力:
  {results_dir}/{run_id}/eval_inputs/nas_pre_training.json  -- airas-eval への入力
  {results_dir}/{run_id}/scores.json                        -- 生スコア（監査用）
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.model import build_network
from src.preprocess import (
    CONDITIONS,
    build_input,
    input_statistics,
    load_reference_table,
    select_archs,
    true_accuracies,
)

# プロキシが数値的に定義できないアーキテクチャに与える番兵値 (前研究と同一)。
SENTINEL_SCORE = -1.0e8


def _finite(value: float) -> float:
    return value if math.isfinite(value) else SENTINEL_SCORE


def jacob_cov(net: torch.nn.Module, inputs: torch.Tensor) -> float:
    """Jacobian covariance (Mellor et al. 2021; Abdelfattah et al. 2021 の jacob_cov)。

    ミニバッチ内の各入力に対する出力の入力ヤコビアンを取り、その相関行列の
    固有値から -sum(log(v+k) + 1/(v+k)) を計算する。前研究と同一の実装。
    """
    net.zero_grad(set_to_none=True)
    x = inputs.clone().requires_grad_(True)
    y = net(x)
    y.backward(torch.ones_like(y))
    if x.grad is None:
        return SENTINEL_SCORE

    jacob = x.grad.detach().reshape(x.shape[0], -1).cpu().double().numpy()
    if not np.all(np.isfinite(jacob)):
        return SENTINEL_SCORE
    if np.any(jacob.std(axis=1) == 0.0):
        return SENTINEL_SCORE

    corrs = np.corrcoef(jacob)
    if not np.all(np.isfinite(corrs)):
        return SENTINEL_SCORE
    eigenvalues = np.linalg.eigvalsh(corrs)
    k = 1e-5
    shifted = eigenvalues + k
    if np.any(shifted <= 0):
        return SENTINEL_SCORE
    return _finite(float(-np.sum(np.log(shifted) + 1.0 / shifted)))


def score_architectures(
    archs: list[str],
    condition: str,
    seeds: list[int],
    npz_path: str,
    batch_size: int,
    device: torch.device,
) -> tuple[list[float], dict[str, Any]]:
    """各アーキテクチャのスコアを、指定シードにわたる平均として返す。

    シードは重み初期化と入力の構成 (バッチ選択・位相・置換・乱数) の両方を決める。
    """
    per_seed: list[list[float]] = []
    stats: dict[str, Any] = {}
    for seed in seeds:
        inputs = build_input(condition, npz_path, batch_size, seed).to(device)
        stats[f"seed{seed}"] = input_statistics(inputs)
        scores: list[float] = []
        t0 = time.time()
        for arch in archs:
            net = build_network(arch, seed=seed, use_bn=True).to(device)
            net.train()
            try:
                scores.append(float(jacob_cov(net, inputs)))
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                scores.append(SENTINEL_SCORE)
            del net
        per_seed.append(scores)
        print(
            f"  seed={seed} scored {len(scores)} architectures in {time.time()-t0:.1f}s",
            flush=True,
        )

    matrix = np.asarray(per_seed, dtype=np.float64)
    return [float(v) for v in matrix.mean(axis=0)], stats


def write_outputs(
    out_dir: Path,
    predicted: list[float],
    reference: list[float],
    archs: list[str],
    extra: dict[str, Any],
) -> None:
    eval_dir = out_dir / "eval_inputs"
    eval_dir.mkdir(parents=True, exist_ok=True)
    payload = {"predicted_scores": predicted, "reference_scores": reference}
    with (eval_dir / "nas_pre_training.json").open("w") as f:
        json.dump(payload, f)

    # 監査用。どのアーキテクチャ集合を採点したかと入力統計だけを残す。
    # 実行時の設定（condition, seeds …）は書かない: run が使った条件は
    # 実行基盤が記録した dispatch から取る。
    audit = {"architectures": archs, **extra}
    with (out_dir / "scores.json").open("w") as f:
        json.dump(audit, f, indent=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--n-archs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--reference-table", required=True)
    parser.add_argument("--cifar-batches", required=True)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    device = torch.device("cpu")
    torch.set_num_threads(max(1, torch.get_num_threads()))
    print(
        f"device={device} threads={torch.get_num_threads()} "
        f"condition={args.condition} seeds={seeds}",
        flush=True,
    )

    table = load_reference_table(args.reference_table)
    archs = select_archs(table, args.n_archs)
    print(f"committed table={len(table)} scored={len(archs)}", flush=True)

    out_dir = Path(args.results_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    predicted, stats = score_architectures(
        archs,
        condition=args.condition,
        seeds=seeds,
        npz_path=args.cifar_batches,
        batch_size=args.batch_size,
        device=device,
    )
    reference = true_accuracies(table, archs)
    write_outputs(
        out_dir,
        predicted=predicted,
        reference=reference,
        archs=archs,
        extra={"input_statistics": stats},
    )

    n_sentinel = sum(1 for v in predicted if v == SENTINEL_SCORE)
    print(f"SENTINEL_SCORES: {n_sentinel}/{len(archs)}")
    print(f"INPUT_STATISTICS: {json.dumps(stats)}")
    print(f"wrote {out_dir/'eval_inputs'/'nas_pre_training.json'}")


if __name__ == "__main__":
    main()
