"""
Testes do model.py com rotas escritas à mão (entregável da fase de Modelação).

Correr no terminal:   python -m pytest tests            (todos os testes)
                      python tests/test_model.py        (só estes, com mensagens passo a passo)

Os valores esperados foram calculados à mão a partir da matriz de tempos (dataset tiny).
Se algum teste falhar, aparece um AssertionError com a explicação.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from model import (Problem, Vehicle, load_data, load_rules, build_travel, DATE_COLUMNS,
                   stop_location, stop_window, stop_service,
                   current_load, deliveries_until_next_warehouse,
                   simulate_route, simulate_plan, plan_cost)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data_tiny")


# ---------------------------------------------------------------
# PASSO 1 — Leitura dos CSVs
# ---------------------------------------------------------------
def test_load_data(data):
    expected = {"collection_centers": 1, "customers": 12, "delivery_locations": 5,
                "distance_matrix": 72, "localities": 49, "locations": 9, "order_items": 40,
                "orders": 12, "producers": 3, "products": 15, "rules": 10, "vehicles": 2}
    assert len(data) == len(expected), f"esperava {len(expected)} ficheiros, li {len(data)}"
    for name, n in expected.items():
        assert name in data, f"falta o ficheiro {name}"
        assert len(data[name]) == n, f"{name}: esperava {n}, li {len(data[name])}"
    for col in DATE_COLUMNS:
        assert is_datetime64_any_dtype(data["order_items"][col]), \
            f"order_items.{col} devia ser datetime, mas é {data['order_items'][col].dtype}"
    print("Passo 1  OK — leitura dos CSVs")


# ---------------------------------------------------------------
# PASSO 2 — Regras e matriz
# ---------------------------------------------------------------
def test_rules(rules):
    assert rules["stop_service_min"] == 10
    assert rules["warehouse_handling_min"] == 15
    assert rules["max_warehouse_dwell_h"] == 24
    assert rules["max_warehouse_dwell_min"] == 1440
    assert rules["stop_service_min"] + 5 == 15            # é número, não texto
    assert rules["flow"] == "producer>warehouse>delivery_location"
    assert rules["window_semantics"] == "service_start"
    assert rules["start_date"] == pd.Timestamp("2026-10-30")
    print("Passo 2a OK — regras")


def test_travel(data, travel_km, travel_min):
    dm = data["distance_matrix"]
    assert travel_min[2][4] == 50.2 and travel_min[4][7] == 31.4 and travel_km[2][4] == 37.66
    n_origins = dm["origin_location_id"].nunique()
    assert len(travel_min) == n_origins and len(travel_km) == n_origins
    assert sum(len(d) for d in travel_min.values()) == len(dm)
    for o in travel_min:
        assert o not in travel_min[o], f"existe o par ({o}, {o})"
        for d in travel_min[o]:
            assert travel_min[o][d] == travel_min[d][o], f"tempo não simétrico ({o},{d})"
            assert travel_km[o][d] == travel_km[d][o], f"km não simétricos ({o},{d})"
    assert travel_min[2][4] > travel_km[2][4], "minutos e km trocados?"
    print("Passo 2b OK — matriz de tempos e km")


# ---------------------------------------------------------------
# PASSO 3 e 4 — Tabela de pedidos e minutos
# ---------------------------------------------------------------
def test_requests(problem):
    req = problem.requests
    locations = load_data(DATA_PATH)["locations"]
    assert len(req) == 40 and req.index.is_unique
    assert "location_id" not in req.columns
    assert not any(c.endswith(("_x", "_y")) for c in req.columns)
    assert req.loc[1, "origin_id"] == 2 and req.loc[1, "destination_id"] == 7
    assert req.loc[1, "product_name"] == "Curgete"
    assert req.loc[1, "origin_name"] == "Campos Leite" and req.loc[1, "destination_name"] == "Casa 3"
    producers = set(locations[locations["location_type"] == "producer"]["location_id"])
    deliveries = set(locations[locations["location_type"] == "delivery_location"]["location_id"])
    assert set(req["origin_id"]) <= producers, "há origens que não são produtores"
    assert set(req["destination_id"]) <= deliveries, "há destinos que não são locais de entrega"
    assert round(req["weight_kg"].sum(), 1) == 270.0
    print("Passo 3  OK — tabela de pedidos")

    assert req.loc[1, "pick_start"] == 1890 and req.loc[1, "pick_end"] == 2070
    assert req.loc[1, "deliv_start"] == 2070 and req.loc[1, "deliv_end"] == 2205
    for col in ["pick_start", "pick_end", "deliv_start", "deliv_end"]:
        assert (req[col] >= 0).all(), f"valores negativos em {col}"
    assert (req["pick_start"] < req["pick_end"]).all()
    assert (req["deliv_start"] < req["deliv_end"]).all()
    print("Passo 4  OK — janelas em minutos")


def test_problem(problem):
    assert problem.warehouse_id == 4
    assert {v: problem.vehicles[v].capacity_kg for v in problem.vehicles} == {1: 150.0, 2: 200.0}
    assert isinstance(problem.vehicles[1], Vehicle) and problem.request_by_id[1].origin_id == 2
    assert problem.travel_minutes[2][4] == 50.2
    print("Passo 6  OK — classe Problem")


# ---------------------------------------------------------------
# PASSO 5 — Auxiliares de uma paragem
# ---------------------------------------------------------------
def test_stop_helpers(problem):
    assert stop_location(problem, "pickup", 1) == 2
    assert stop_location(problem, "delivery", 1) == 7
    assert stop_location(problem, "warehouse", None) == 4
    assert stop_window(problem, "pickup", 1) == (1890, 2070)
    assert stop_window(problem, "delivery", 1) == (2070, 2205)
    assert stop_window(problem, "warehouse", None) == (0, float("inf"))
    assert stop_service(problem, "pickup") == 10 and stop_service(problem, "warehouse") == 30
    for func, args in [(stop_location, ("recolha", 1)), (stop_window, ("recolha", 1)),
                       (stop_service, ("recolha",))]:
        try:
            func(problem, *args)
            assert False, f"{func.__name__} aceitou um tipo de paragem inválido"
        except ValueError:
            pass
    print("Passo 5a OK — auxiliares de uma paragem")


# ---------------------------------------------------------------
# PASSO 5b — Tempo e janelas (regra service_start)
# ---------------------------------------------------------------
def test_simulate_route_time(problem):
    route_1 = [("pickup", 1), ("warehouse", None), ("delivery", 1)]

    # Parte à meia-noite → chega cedo e espera duas vezes
    #   pickup 1 (loc 2): chega 50.2 → espera até 1890 → acaba 1900
    #   armazém  (loc 4): chega 1950.2 → acaba 1980.2
    #   entrega 1 (loc 7): chega 2011.6 → espera até 2070 → acaba 2080 → volta 2111.4
    r = simulate_route(problem, route_1, 0, 1)
    assert r["feasible"], r["reason"]
    assert round(r["end_time"], 1) == 2111.4
    t = r["timeline"]
    assert round(t[0]["arrival"], 1) == 50.2 and t[0]["start"] == 1890 and t[0]["end"] == 1900
    assert round(t[1]["arrival"], 1) == 1950.2 and round(t[1]["end"], 1) == 1980.2
    assert round(t[2]["arrival"], 1) == 2011.6 and t[2]["start"] == 2070 and t[2]["end"] == 2080

    # Parte tarde → chega à recolha depois de fechar
    r = simulate_route(problem, route_1, 2100, 1)
    assert not r["feasible"] and "pickup" in r["reason"] and len(r["timeline"]) == 0

    # Caso limite de service_start: chega às 2065 (fecha às 2070), acaba às 2075 → VÁLIDO
    r = simulate_route(problem, route_1, 2014.8, 1)
    assert r["feasible"], r["reason"]
    assert round(r["end_time"], 1) == 2228.0

    # Chega às 2075, depois de fechar → inviável
    assert not simulate_route(problem, route_1, 2024.8, 1)["feasible"]

    # Duas entregas seguidas no MESMO sítio → uma só paragem → o serviço conta uma vez
    #   entrega 1 começa 2070 (sem serviço próprio) · entrega 2 começa 2070, acaba 2080 · volta 2111.4
    route_2 = [("pickup", 1), ("pickup", 2), ("warehouse", None), ("delivery", 1), ("delivery", 2)]
    r = simulate_route(problem, route_2, 0, 1)
    assert r["feasible"], r["reason"]
    assert round(r["end_time"], 1) == 2111.4
    assert r["timeline"][4]["arrival"] == r["timeline"][3]["end"], "no mesmo sítio não há viagem"

    # Rota vazia
    r = simulate_route(problem, [], 500, 1)
    assert r["feasible"] and r["end_time"] == 500
    print("Passo 5b OK — tempo e janelas")


# ---------------------------------------------------------------
# PASSO 5c-1 — Carga e estado dos itens
# ---------------------------------------------------------------
def test_simulate_route_load(problem):
    assert current_load(problem, set(), set()) == 0
    assert current_load(problem, {1}, set()) == 6.3
    assert round(current_load(problem, {1}, {2}), 1) == 10.0

    route = [("pickup", 1), ("pickup", 2), ("warehouse", None), ("delivery", 1),
             ("delivery", 2), ("pickup", 3), ("warehouse", None), ("delivery", 3)]
    assert deliveries_until_next_warehouse(route, 2) == [1, 2]
    assert deliveries_until_next_warehouse(route, 6) == [3]
    assert deliveries_until_next_warehouse(route, 7) == []

    assert not simulate_route(problem, [("delivery", 1)], 0, 1)["feasible"]
    assert not simulate_route(problem, [("pickup", 1), ("delivery", 1)], 0, 1)["feasible"]
    assert not simulate_route(problem, [("pickup", 1), ("pickup", 1)], 0, 1)["feasible"]
    assert not simulate_route(problem, [("pickup", 1)], 0, 1)["feasible"]

    r = simulate_route(problem, [("pickup", 1), ("warehouse", None)], 0, 1)
    assert r["feasible"] and list(r["warehouse_out"]) == [1]
    assert round(r["warehouse_out"][1], 1) == 1965.2

    route_2 = [("pickup", 1), ("pickup", 2), ("warehouse", None), ("delivery", 1), ("delivery", 2)]
    problem.vehicles[99] = Vehicle(99, 8.0, False)
    try:
        r = simulate_route(problem, route_2, 0, 99)
        assert not r["feasible"] and "capacity" in r["reason"]
        assert simulate_route(problem, route_2, 0, 1)["feasible"]
    finally:
        del problem.vehicles[99]
    print("Passo 5c-1 OK — carga, capacidade e fluxo")


# ---------------------------------------------------------------
# PASSO 5c-2 — Armazém entre dias + regra das 24 h
# ---------------------------------------------------------------
def test_warehouse_between_days(problem):
    day0 = [("pickup", 5), ("warehouse", None)]
    day1 = [("warehouse", None), ("delivery", 5)]

    r0 = simulate_route(problem, day0, 0, 1)
    assert r0["feasible"], r0["reason"]
    assert round(r0["warehouse_out"][5], 1) == 500.8
    assert round(r0["end_time"], 1) == 515.8
    assert round(r0["wait_min"], 1) == 394.2
    assert round(r0["distance_km"], 2) == 61.2

    r1 = simulate_route(problem, day1, 1850, 1, r0["warehouse_out"])       # 1364.2 min ≤ 1440
    assert r1["feasible"] and r1["delivered"] == {5} and r1["warehouse_out"] == {}
    assert round(r1["end_time"], 1) == 1947.0

    r1 = simulate_route(problem, day1, 1950, 1, r0["warehouse_out"])       # 1464.2 min > 1440
    assert not r1["feasible"] and "warehouse" in r1["reason"]

    w = {5: 500.8}
    simulate_route(problem, day1, 1850, 1, w)
    assert w == {5: 500.8}, "simulate_route alterou o warehouse_in de quem a chamou"
    print("Passo 5c-2 OK — armazém entre dias e regra das 24 h")


# ---------------------------------------------------------------
# PASSO 5c-3 — Plano completo, trocas entre carrinhas e custo
# ---------------------------------------------------------------
def test_simulate_plan(problem):
    day0 = [("pickup", 5), ("warehouse", None)]
    day1 = [("warehouse", None), ("delivery", 5)]

    plan = {(0, 1): (0, day0), (1, 1): (1850, day1)}
    r = simulate_plan(problem, plan, require_all=False)
    assert r["feasible"] and r["warehouse_end"] == {}
    assert round(r["distance_km"], 2) == 103.98
    assert round(r["duration_min"], 1) == 612.8
    assert round(r["wait_min"], 1) == 394.2

    r = simulate_plan(problem, plan, require_all=True)
    assert not r["feasible"] and "not delivered" in r["reason"]

    # Outra carrinha no dia seguinte
    assert simulate_plan(problem, {(0, 1): (0, day0), (1, 2): (1850, day1)}, require_all=False)["feasible"]

    # TROCA NO MESMO DIA: a carrinha 2 espera no armazém pelo item que a carrinha 1 descarrega
    r = simulate_plan(problem, {(0, 1): (0, day0), (0, 2): (0, day1)}, require_all=False)
    assert r["feasible"], r["reason"]
    assert round(r["routes"][(0, 2)]["timeline"][0]["end"], 1) == 515.8    # carregou depois de chegar

    # Ninguém traz o item → a carrinha ficaria à espera para sempre → inviável
    r = simulate_plan(problem, {(0, 2): (0, day1)}, require_all=False)
    assert not r["feasible"] and "never arrive" in r["reason"]

    # Troca nos dois sentidos: cada carrinha entrega um item que a outra recolheu
    a = [("pickup", 31), ("warehouse", None), ("delivery", 33)]
    b = [("pickup", 33), ("warehouse", None), ("delivery", 31)]
    assert simulate_plan(problem, {(0, 1): (400, a), (0, 2): (400, b)}, require_all=False)["feasible"]

    # Se ninguém o leva, o item fica no armazém no fim
    r = simulate_plan(problem, {(0, 1): (0, day0)}, require_all=False)
    assert r["feasible"] and list(r["warehouse_end"]) == [5]

    # Rota que começa fora do seu dia
    assert not simulate_plan(problem, {(1, 1): (500, day0)}, require_all=False)["feasible"]

    # Custo
    r = simulate_plan(problem, {(0, 1): (0, day0), (1, 1): (1850, day1)}, require_all=False)
    assert round(plan_cost(r), 2) == 103.98
    assert round(plan_cost(r, "duration_min"), 1) == 612.8
    assert plan_cost(simulate_plan(problem, {(0, 1): (0, day0)})) == float("inf")
    print("Passo 5c-3 OK — plano completo, trocas entre carrinhas e custo")


# ---------------------------------------------------------------
def run_all(data_path=DATA_PATH):
    data = load_data(data_path)
    test_load_data(data)
    rules = load_rules(data)
    test_rules(rules)
    travel_km, travel_min = build_travel(data)
    test_travel(data, travel_km, travel_min)
    problem = Problem(data_path)
    test_requests(problem)
    test_problem(problem)
    test_stop_helpers(problem)
    test_simulate_route_time(problem)
    test_simulate_route_load(problem)
    test_warehouse_between_days(problem)
    test_simulate_plan(problem)
    print("\nTodos os testes passaram.")


if __name__ == "__main__":
    run_all()
