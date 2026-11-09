"""Testes do CSP (csp_solver.py) e do Hill Climbing (local_search.py) no dataset tiny."""
import pytest

from model import simulate_plan
from search import solve
from local_search import hill_climbing
from csp_solver import solve_csp, build_day_csp


def test_csp_tiny_feasible_and_validated(problem):
    plan, result, stats = solve_csp(problem)
    assert stats["status"] == "ok" and result["feasible"]
    assert stats["backtracks"] == {0: 82, 1: 1178}
    assert result == simulate_plan(problem, plan)        # validado pelo mesmo simulador


@pytest.mark.parametrize("windows_as", ["inset", "function"])
def test_unary_constraints_are_equivalent_to_domain_cut(problem, windows_as):
    """A biblioteca transforma restrições unárias em cortes no domínio (preProcess)."""
    _, base, s0 = solve_csp(problem)
    _, other, s1 = solve_csp(problem, windows_as=windows_as)
    assert s1["backtracks"] == s0["backtracks"] and other["distance_km"] == base["distance_km"]


def test_csp_time_limit_reports_limit(problem):
    _, _, stats = solve_csp(problem, variable_order="input", time_limit=0.01)
    assert stats["status"].startswith("limit")


def test_capacity_constraint_class_matches_function(problem):
    csp1, _ = build_day_csp(problem, 0, {})
    csp2, _ = build_day_csp(problem, 0, {}, capacity_as="function")
    assert len(csp1._constraints) == len(csp2._constraints)


def test_hill_climbing_never_worsens_and_stays_feasible(problem):
    plan, result, _ = solve(problem, "greedy", "farthest")
    new_plan, cost, history, stats = hill_climbing(problem, plan)
    assert history == sorted(history, reverse=True) and history[0] == pytest.approx(result["distance_km"])
    assert simulate_plan(problem, new_plan)["feasible"] and stats["status"] == "local_optimum"
    assert round(cost, 2) == 601.97 and stats["evaluations"] == 1052
    # sem vizinhos repetidos nem iguais ao plano
    from local_search import generate_neighbors, _signature
    neighbors = generate_neighbors(problem, plan)
    assert len({_signature(n) for n in neighbors}) == len(neighbors)
    assert _signature(plan) not in {_signature(n) for n in neighbors}
