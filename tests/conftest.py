"""Fixtures partilhadas pelos testes (dataset tiny) e ajudas para criar datasets de teste."""
import os
import shutil
import sys

import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from model import Problem, load_data, load_rules, build_travel   # noqa: E402

TINY = os.path.join(ROOT, "data_tiny")
SMALL = os.path.join(ROOT, "data_small")


@pytest.fixture(scope="session")
def data():
    return load_data(TINY)


@pytest.fixture(scope="session")
def rules(data):
    return load_rules(data)


@pytest.fixture(scope="session")
def travel_km(data):
    return build_travel(data)[0]


@pytest.fixture(scope="session")
def travel_min(data):
    return build_travel(data)[1]


@pytest.fixture
def problem():
    return Problem(TINY)


def copy_dataset(src, dst, edit=None):
    """Copia um dataset para dst; edit(tables) pode alterar as tabelas (dict nome → DataFrame)."""
    shutil.copytree(src, dst)
    if edit:
        tables = {f[:-4]: pd.read_csv(os.path.join(dst, f)) for f in os.listdir(dst) if f.endswith(".csv")}
        edit(tables)
        for name, df in tables.items():
            df.to_csv(os.path.join(dst, f"{name}.csv"), index=False)
    return dst
