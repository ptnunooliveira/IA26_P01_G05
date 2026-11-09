"""Testes com datasets diferentes: o código não pode depender dos nomes nem das quantidades do tiny."""
import os
import subprocess
import sys

import pytest

from conftest import TINY, SMALL, ROOT, copy_dataset
from model import Problem, load_data, validate_data, simulate_plan
from search import solve, triangle_inequality_holds
from local_search import hill_climbing
from csp_solver import solve_csp

needs_small = pytest.mark.skipif(not os.path.isdir(SMALL), reason="data_small não está na pasta")


def test_tiny_is_valid(data):
    assert validate_data(data) == []


@needs_small
def test_small_loads_with_its_own_sizes():
    p = Problem.from_csv(SMALL)
    assert len(p.requests) == 80 and len(p.vehicles) == 2 and p.warehouse_id == 7
    assert p.days == [0, 1]
    assert not triangle_inequality_holds(p)          # falha por 0,01 km: a H1 usa caminhos mais curtos


@needs_small
def test_alternating_datasets_gives_same_results():
    """tiny → small → tiny → small no MESMO processo: os resultados não podem mudar."""
    runs = []
    for path in (TINY, SMALL, TINY, SMALL):
        _, result, stats = solve(Problem(path), "greedy", "entry")
        runs.append((round(result["distance_km"], 2), stats["expanded"]))
    assert runs[0] == runs[2] and runs[1] == runs[3] and runs[0] != runs[1]
    # e num processo separado
    code = ("import sys; sys.path.insert(0, %r); from model import Problem; from search import solve; "
            "print(round(solve(Problem(%r), 'greedy', 'entry')[1]['distance_km'], 2))" % (ROOT, SMALL))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert float(out.stdout) == runs[1][0]


@pytest.mark.parametrize("breaker, message", [
    (lambda t: t["order_items"].drop(columns=["weight_kg"], inplace=True), "faltam as colunas"),
    (lambda t: t["order_items"].loc.__setitem__((0, "product_id"), 999), "referências"),
    (lambda t: t["distance_matrix"].drop(index=[0, 1], inplace=True), "faltam 2 pares"),
    (lambda t: t["orders"].loc.__setitem__((1, "order_id"), 1), "repetido"),
    (lambda t: t["rules"].drop(index=t["rules"].index[t["rules"]["rule"] == "stop_service_min"],
                               inplace=True), "faltam as regras"),
])
def test_invalid_dataset_is_rejected(tmp_path, breaker, message):
    path = copy_dataset(TINY, tmp_path / "invalid", breaker)
    with pytest.raises(ValueError, match=message):
        Problem(path)


def test_missing_file_is_rejected(tmp_path):
    path = copy_dataset(TINY, tmp_path / "missing")
    os.remove(os.path.join(path, "vehicles.csv"))
    with pytest.raises(FileNotFoundError, match="vehicles.csv"):
        Problem(path)


def test_dataset_with_different_parameters(tmp_path):
    """3 carrinhas, outras capacidades e 5 min de serviço: tudo é lido dos CSVs."""
    def change(t):
        v = t["vehicles"]
        v.loc[len(v)] = [3, "AA-00-AA", 1, 120.0, 0, 45.0]
        v["capacity_kg"] = v["capacity_kg"] + 10
        t["rules"].loc[t["rules"]["rule"] == "stop_service_min", "value"] = "5"
    p = Problem(copy_dataset(TINY, tmp_path / "params", change))
    assert len(p.vehicles) == 3 and p.rules["stop_service_min"] == 5
    plan, result, stats = solve(p, "greedy", "entry")
    assert stats["status"] == "ok" and result["feasible"]
    _, cost, _, hc = hill_climbing(p, plan)
    assert cost <= result["distance_km"] and hc["status"] in ("local_optimum", "limit")


def test_csp_and_local_search_on_reduced_instance(tmp_path):
    def keep(t):
        t["orders"] = t["orders"][t["orders"]["order_id"].isin([1, 2, 3, 4])]
        t["order_items"] = t["order_items"][t["order_items"]["order_id"].isin([1, 2, 3, 4])]
    p = Problem(copy_dataset(TINY, tmp_path / "reduced", keep))
    plan, result, stats = solve_csp(p)
    assert stats["status"] == "ok" and simulate_plan(p, plan)["feasible"]
