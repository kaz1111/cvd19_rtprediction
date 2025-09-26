"""PoC script to estimate COVID-19 effective reproduction number (Rt) in Python.

This script mirrors the original R workflow by
1. downloading the publicly available Tokyo COVID-19 case and recovery data,
2. preparing the cumulative counts expected by the existing `SIR.stan` model, and
3. running the Stan model via `cmdstanpy` to obtain posterior draws for Rt-related
   parameters.

Usage
-----
$ python python/poc_rt_estimation.py

Prerequisites
-------------
- pandas
- numpy
- cmdstanpy (and a configured CmdStan toolchain)

These dependencies can be installed with:
$ pip install pandas numpy cmdstanpy

To install CmdStan itself (needed by cmdstanpy), run:
$ python -m cmdstanpy.install_cmdstan
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

DATA_URL = "https://dl.dropboxusercontent.com/s/6mztoeb6xf78g5w/COVID-19.csv"
TOKYO_PREF_NAME = "東京都"
DEFAULT_POPULATION = 13_942_856  # 東京都の人口
DEFAULT_RECOVERY_LAG = 21  # days


@dataclass
class StanInput:
    """Container for the Stan data dictionary and the prepared DataFrame."""

    stan_data: Dict[str, np.ndarray]
    features: pd.DataFrame


def fetch_tokyo_timeseries() -> pd.DataFrame:
    """Download and aggregate daily confirmed and recovered case counts."""

    df = pd.read_csv(DATA_URL)
    tokyo = df[df["居住都道府県"] == TOKYO_PREF_NAME].copy()
    tokyo["確定日"] = pd.to_datetime(tokyo["確定日"], format="%m/%d/%Y")

    grouped = (
        tokyo.groupby("確定日")[["人数", "退院数"]]
        .sum()
        .rename(columns={"人数": "陽性人数", "退院数": "退院人数"})
    )

    full_index = pd.date_range(grouped.index.min(), grouped.index.max(), freq="D")
    daily = grouped.reindex(full_index, fill_value=0)
    daily.index.name = "date"

    daily["陽性累積"] = daily["陽性人数"].cumsum()
    daily["退院累積"] = daily["退院人数"].cumsum()
    daily["前週陽性"] = daily["陽性人数"].shift(7, fill_value=0)

    # Map Monday=0..Sunday=6 to lubridate::wday output (Sunday=1).
    daily["Wd"] = ((daily.index.weekday + 1) % 7) + 1

    return daily.reset_index().rename(columns={"index": "確定日"})


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series while avoiding division by zero."""

    result = numerator / denominator.replace({0: np.nan})
    return result.fillna(0.0)


def prepare_features(population: int = DEFAULT_POPULATION) -> pd.DataFrame:
    """Compute the SIR features used in the original R workflow."""

    daily = fetch_tokyo_timeseries()

    susceptible = population - daily["陽性累積"]
    beta = safe_divide(daily["陽性人数"] * population, daily["陽性累積"] * susceptible)
    gamma = safe_divide(daily["退院人数"], daily["前週陽性"])
    r0 = safe_divide(beta * susceptible, gamma * population)

    daily = daily.assign(
        population=population,
        beta=beta,
        gamma=gamma,
        R0=r0,
    )
    return daily


def build_stan_input(
    population: int = DEFAULT_POPULATION, recovery_lag: int = DEFAULT_RECOVERY_LAG
) -> StanInput:
    """Create the Stan data dictionary for CmdStanPy."""

    features = prepare_features(population)

    stan_data = {
        "n_sample": len(features),
        "I_obs": features["陽性累積"].astype(int).to_numpy(),
        "R_obs": features["退院累積"].astype(int).to_numpy(),
        "S0": int(population),
        "R_lag": int(recovery_lag),
        "Wd": features["Wd"].astype(int).to_numpy(),
    }

    return StanInput(stan_data=stan_data, features=features)


def run_model(stan_data: StanInput, *, chains: int = 2) -> None:
    """Compile the Stan model and run sampling with CmdStanPy."""

    stan_file = Path(__file__).resolve().parents[1] / "SIR.stan"
    model = CmdStanModel(stan_file=str(stan_file))

    fit = model.sample(
        data=stan_data.stan_data,
        chains=chains,
        iter_sampling=200,
        iter_warmup=200,
        thin=2,
        seed=123,
        show_progress=True,
    )

    summary = fit.summary()
    rt_columns = [col for col in summary.index if col.startswith("R_param")]
    print("Posterior summary for Rt (first 5 entries):")
    print(summary.loc[rt_columns[:5], ["Mean", "5%", "95%"]])


def main() -> None:
    stan_input = build_stan_input()
    print(
        f"Prepared data with {stan_input.stan_data['n_sample']} days of observations."
    )
    run_model(stan_input)


if __name__ == "__main__":
    main()
