from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "atlas_gdp_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_forecast_vs_actual(df: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(df["date"], df["actual"], label="actual")
    plt.plot(df["date"], df["forecast"], label="forecast")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_fan_chart(df: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(df["date"], df["point"], label="point")
    if {"q10", "q90"}.issubset(df.columns):
        plt.fill_between(df["date"], df["q10"], df["q90"], alpha=0.2)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_coverage_calibration(df: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", label="ideal")
    plt.plot(df["nominal"], df["observed"], marker="o", color="#2563eb", label="observed")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Nominal coverage")
    plt.ylabel("Observed coverage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
