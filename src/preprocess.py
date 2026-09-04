"""入力条件の構成と参照表の読み込み。

本研究の介入は「jacob_cov に渡す入力テンソルの統計構造」だけを変える。
すべての条件は同一シードの実 CIFAR-10 ミニバッチ (128 枚) から派生し、
形状 (128, 3, 32, 32) と dtype (float32) を共有する。

  cifar10   : 実ミニバッチ。標準的な per-channel 正規化のみ。
  phasescr  : 位相スクランブル。各画像の 2 次元フーリエ振幅スペクトルを保ち、
              位相を乱数に置き換える (Hermite 対称性は実ノイズの位相を借りて保証)。
              二次統計 (パワースペクトル = 空間自己相関) は画像ごとに厳密に保存され、
              意味内容 (物体・エッジの配置) は破壊される。
  pixshuf   : 画素シャッフル。画像ごとに 1 つの空間置換を 3 チャネル共通に適用する。
              画素値の周辺分布とチャネル間相関は厳密に保存され、空間相関は破壊される。
  randinput : 同形状の N(0,1) 乱数テンソル (前研究の条件をそのまま再現)。

データはリポジトリに同梱した固定抽出物 (data/) から読む。クラスタのキャッシュや
ダウンロードには依存しない。ラベルは一切使わない。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
IMAGE_SHAPE = (3, 32, 32)
CONDITIONS = ("cifar10", "phasescr", "pixshuf", "randinput")


def load_reference_table(path: str | Path) -> dict[str, float]:
    """同梱した NAS-Bench-201 の部分表 (arch -> CIFAR-10 テスト精度 %) を読む。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"reference table not found: {path}")
    payload = json.loads(path.read_text())
    table = payload.get("accuracies")
    if not isinstance(table, dict) or not table:
        raise ValueError(f"unexpected reference table format in {path}")
    return {str(k): float(v) for k, v in table.items()}


def select_archs(table: dict[str, float], n_archs: int) -> list[str]:
    """同梱表の (ソート済み) 先頭 n 件。full では 1,000 件すべて。

    表自体が固定シード 2026 で抽出済みの集合なので、ここでは再抽出しない。
    sanity / pilot は同じ集合の接頭辞を使う。
    """
    archs = sorted(table.keys())
    if n_archs > len(archs):
        raise ValueError(f"n_archs={n_archs} exceeds the committed table ({len(archs)})")
    return archs[:n_archs]


def true_accuracies(table: dict[str, float], archs: list[str]) -> list[float]:
    return [table[a] for a in archs]


def load_cifar_batch(npz_path: str | Path, batch_size: int, seed: int) -> np.ndarray:
    """同梱バッチ (uint8) を読み、正規化した float32 (B,3,32,32) を返す。"""
    npz_path = Path(npz_path)
    if not npz_path.is_file():
        raise FileNotFoundError(f"CIFAR-10 batches not found: {npz_path}")
    with np.load(npz_path) as z:
        key = f"seed{seed}"
        if key not in z:
            raise KeyError(f"no committed batch for seed {seed} in {npz_path}")
        raw = z[key]
    if raw.shape[0] < batch_size:
        raise ValueError(f"committed batch has {raw.shape[0]} images < {batch_size}")
    images = raw[:batch_size].astype(np.float32) / 255.0
    mean = np.array(CIFAR10_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array(CIFAR10_STD, dtype=np.float32).reshape(1, 3, 1, 1)
    return (images - mean) / std


def phase_scramble(images: np.ndarray, seed: int) -> np.ndarray:
    """振幅スペクトルを保ち位相を乱す。画像ごとに 1 つの位相場を 3 チャネルで共有。

    実ノイズ場 n の FFT の位相は Hermite 対称なので、|F(x)| exp(i arg F(n)) の
    逆変換は実数になる。DC 成分の位相は原画像のものを保ち、平均を保存する。
    """
    rng = np.random.default_rng(seed)
    out = np.empty_like(images)
    for b in range(images.shape[0]):
        noise = rng.standard_normal(images.shape[2:]).astype(np.float64)
        noise_phase = np.angle(np.fft.fft2(noise))
        for c in range(images.shape[1]):
            spectrum = np.fft.fft2(images[b, c].astype(np.float64))
            phase = noise_phase.copy()
            phase[0, 0] = np.angle(spectrum[0, 0])
            scrambled = np.fft.ifft2(np.abs(spectrum) * np.exp(1j * phase))
            out[b, c] = np.real(scrambled).astype(np.float32)
    return out


def pixel_shuffle(images: np.ndarray, seed: int) -> np.ndarray:
    """画像ごとに 1 つの空間置換を全チャネルに適用する。"""
    rng = np.random.default_rng(seed)
    b, c, h, w = images.shape
    flat = images.reshape(b, c, h * w)
    out = np.empty_like(flat)
    for i in range(b):
        perm = rng.permutation(h * w)
        out[i] = flat[i][:, perm]
    return out.reshape(b, c, h, w)


def make_random_input(batch_size: int, seed: int) -> np.ndarray:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(
        (batch_size, *IMAGE_SHAPE), generator=generator, dtype=torch.float32
    ).numpy()


def build_input(
    condition: str, npz_path: str | Path, batch_size: int, seed: int
) -> torch.Tensor:
    """介入の本体: 条件名から入力テンソルを 1 つ作る。"""
    if condition == "randinput":
        return torch.from_numpy(make_random_input(batch_size, seed))
    real = load_cifar_batch(npz_path, batch_size, seed)
    if condition == "cifar10":
        return torch.from_numpy(real)
    if condition == "phasescr":
        return torch.from_numpy(phase_scramble(real, seed))
    if condition == "pixshuf":
        return torch.from_numpy(pixel_shuffle(real, seed))
    raise ValueError(f"unknown input condition: {condition!r}")


def input_statistics(x: torch.Tensor) -> dict[str, float]:
    """監査用の入力統計 (指標ではない)。平均・標準偏差・隣接画素相関 (lag 1)。"""
    a = x.detach().cpu().double().numpy()
    centered = a - a.mean(axis=(2, 3), keepdims=True)
    var = (centered**2).mean()
    lag_h = (centered[:, :, :, 1:] * centered[:, :, :, :-1]).mean()
    lag_v = (centered[:, :, 1:, :] * centered[:, :, :-1, :]).mean()
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "lag1_autocorr": float(0.5 * (lag_h + lag_v) / var) if var > 0 else 0.0,
    }
