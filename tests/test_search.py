"""Testes das heurísticas e dos algoritmos de pesquisa (search.py)."""
import os
import subprocess
import sys

import pytest

from conftest import TINY, ROOT, copy_dataset
import search
from model import Problem, State, VanState, simulate_plan, plan_cost
from search import (DayTask, HEURISTICS, best_first_search, possible_moves, solve,
                    heuristic01, heuristic02)


def reduced_dataset(tmp_path, orders):
    """O tiny só com algumas encomendas do MESMO dia (instância pequena, ótimo calculável por UCS)."""
    def keep(t):
        t["orders"] = t["orders"][t["orders"]["order_id"].isin(orders)]
        t["order_items"] = t["order_items"][t["order_items"]["order_id"].isin(orders)]
    return copy_dataset(TINY, tmp_path / "reduced", keep)


def ucs(task, state):
    """
    Custo ótimo a partir de `state`, no mesmo espaço de ações (UCS = A* com h = 0) e SEM a poda
    de becos sem saída — o valor de referência não depende das podas que queremos testar.
    """
    task.initial_state = lambda: state
    task.dead_end = lambda s: False
    _, cost, _, status = best_first_search(task, lambda t, s: 0.0, lambda g, h: g)
    assert status in ("ok", "no_solution")
    return cost


# ---------------------------------------------------------------- heurísticas

def test_h2_cache_does_not_leak_between_datasets(tmp_path):
    """
    Reproduz o erro antigo: a cache global _MIN_ENTRY (indexada só pelo location_id) fazia a H2
    de um dataset usar as distâncias de outro. Aqui o dataset "x2" é o tiny com as distâncias a
    dobrar: a H2 tem de dobrar, qualquer que seja a ordem de execução.
    """
    def double(t):
        t["distance_matrix"]["distance_km"] *= 2
    x2 = copy_dataset(TINY, tmp_path / "x2", double)
    h = lambda path: heuristic02(DayTask(Problem(path), 0, {}), DayTask(Problem(path), 0, {}).initial_state())
    tiny_first, x2_after = h(TINY), h(x2)
    x2_first, tiny_after = h(x2), h(TINY)
    assert x2_after == pytest.approx(2 * tiny_first)
    assert (tiny_first, x2_after) == (tiny_after, x2_first)
    # e num processo separado (sem nada em memória) dá o mesmo valor
    code = ("import sys; sys.path.insert(0, %r); from model import Problem; from search import *; "
            "t = DayTask(Problem(%r), 0, {}); print(heuristic02(t, t.initial_state()))" % (ROOT, str(x2)))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert float(out.stdout) == pytest.approx(x2_after)


@pytest.mark.parametrize("name", ["farthest", "entry"])
def test_heuristic_is_zero_at_goal_and_non_negative(problem, name):
    h = HEURISTICS[name]
    task = DayTask(problem, 0, {})
    goal = State(tuple(VanState(task.W, 900.0, frozenset(), frozenset(), True) for _ in task.vans),
                 (), frozenset(), frozenset())
    assert task.is_goal(goal) and h(task, goal) == 0
    values = []                                         # todos os estados avaliados numa pesquisa
    recording = lambda t, s: values.append(h(t, s)) or values[-1]
    assert best_first_search(task, recording, lambda g, v: v)[3] == "ok"
    assert len(values) > 10 and min(values) >= 0


@pytest.mark.parametrize("name", ["farthest", "entry"])
def test_heuristics_are_admissible_on_reduced_instance(tmp_path, name):
    """h(s) ≤ custo ótimo restante C*(s), para vários estados s (C* calculado por UCS)."""
    problem = Problem(reduced_dataset(tmp_path, [4, 10]))          # só o dia 0
    h = HEURISTICS[name]
    for day in problem.days:
        task = DayTask(problem, day, {})
        frontier, checked = [task.initial_state()], 0
        while frontier and checked < 25:
            state = frontier.pop(0)
            optimum = ucs(DayTask(problem, day, {}), state)
            if optimum < float("inf"):
                assert h(task, state) <= optimum + 1e-9
                checked += 1
            frontier += [m[2] for m in possible_moves(task, state)][:3]
        assert checked > 0


@pytest.mark.parametrize("orders", [[4, 10], [1, 5, 6]])
@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_astar_with_pruning_matches_ucs_without_it(tmp_path, orders, alpha):
    """O A* (com a poda de becos sem saída) dá o mesmo custo que a UCS sem essa poda."""
    problem = Problem(reduced_dataset(tmp_path, orders))
    for day in problem.days:
        optimum = ucs(DayTask(problem, day, {}, alpha), DayTask(problem, day, {}, alpha).initial_state())
        for name in HEURISTICS:
            task = DayTask(problem, day, {}, alpha)
            _, cost, _, status = best_first_search(task, HEURISTICS[name], lambda g, h: g + h)
            assert status == "ok" and cost == pytest.approx(optimum)


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("algorithm, heuristic", [("greedy", "entry"), ("weighted_astar", "farthest")])
def test_search_cost_equals_simulated_soft_cost(problem, algorithm, heuristic, alpha):
    """
    Regressão: o custo que a pesquisa otimiza (km + alpha × espera) tem de ser IGUAL ao que a
    simulação mede no plano final, dia a dia. Antes diferia 10–20 min de espera: a pesquisa contava
    duas visitas seguidas ao mesmo produtor como duas paragens (dois serviços de 10 min).
    """
    plan, result, stats = solve(problem, algorithm, heuristic, alpha=alpha)
    assert stats["status"] == "ok"
    for day in problem.days:
        routes = [r for (d, _), r in result["routes"].items() if d == day]
        simulated = sum(r["distance_km"] + alpha * r["wait_min"] for r in routes)
        assert stats["cost"][day] == pytest.approx(simulated)
    assert sum(stats["cost"].values()) == pytest.approx(plan_cost(result, "distance_km", alpha))


def test_astar_soft_cost_matches_simulation_regression(problem):
    """O caso exato em que a diferença aparecia: A* H1 com alpha = 1 no tiny."""
    plan, result, stats = solve(problem, "astar", "farthest", alpha=1.0)
    assert sum(stats["cost"].values()) == pytest.approx(plan_cost(result, "distance_km", 1.0))


@pytest.mark.parametrize("heuristic", ["farthest", "entry"])
def test_astar_does_not_lose_the_optimum_of_day_1(problem, heuristic):
    """
    Regressão: a antiga poda por "dominância" cortava o ótimo do dia 1 do tiny (306,09 km em vez
    de 300,69). Com a deteção de estados repetidos exatos, as duas heurísticas dão o mesmo ótimo.
    """
    plan, result, stats = solve(problem, "astar", heuristic, max_iterations=400_000)   # H2: 224 mil nós no dia 1
    assert stats["cost"][1] == pytest.approx(300.69) and round(result["distance_km"], 2) == 533.09


# ---------------------------------------------------------------- algoritmos

@pytest.mark.parametrize("algorithm, heuristic, km", [("greedy", "farthest", 623.41),
                                                       ("greedy", "entry", 840.37),
                                                       ("weighted_astar", "entry", 533.09)])
def test_solve_tiny_complete_and_feasible(problem, algorithm, heuristic, km):
    plan, result, stats = solve(problem, algorithm, heuristic)
    assert stats["status"] == "ok" and result["feasible"]
    assert round(result["distance_km"], 2) == km
    delivered = set().union(*(r["delivered"] for r in result["routes"].values()))
    assert delivered == set(problem.requests.index)                     # todos os itens
    assert {v for _, v in plan} <= set(problem.vehicles)                # só carrinhas que existem
    # o custo da pesquisa (soma dos km das ações) é o mesmo que a simulação mede
    assert sum(stats["cost"].values()) == pytest.approx(result["distance_km"])


def test_solve_is_reproducible(problem):
    first = solve(problem, "greedy", "entry")[0]
    assert solve(problem, "greedy", "entry")[0] == first


def test_limit_is_not_reported_as_no_solution(problem):
    plan, result, stats = solve(problem, "astar", "farthest", max_iterations=50)
    assert plan is None and result is None and stats["status"].startswith("limit")


def test_infeasible_instance_reports_no_solution(tmp_path):
    """Um item com janela de entrega impossível (antes de dar para passar pelo armazém)."""
    def impossible(t):
        oi = t["order_items"]
        oi.loc[oi["order_item_id"] == 1, "delivery_end"] = "2026-10-31 07:50:00"
        oi.loc[oi["order_item_id"] == 1, "delivery_start"] = "2026-10-31 07:40:00"
    problem = Problem(copy_dataset(TINY, tmp_path / "impossible", impossible))
    plan, result, stats = solve(problem, "greedy", "entry")
    assert plan is None and stats["status"].startswith("no_solution")
