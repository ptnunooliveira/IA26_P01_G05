from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import pandas as pd

MINUTES_PER_DAY = 1440

# region Aux Arrays

# Files name
FILES_NAME = ["collection_centers", "customers", "delivery_locations",
              "distance_matrix", "localities", "locations", "order_items",
              "orders", "producers", "products", "rules", "vehicles"]

# Columns from order_items.csv
DATE_COLUMNS = ["availability_start", "availability_end", "delivery_start", "delivery_end"]

# Numeric rules to be convered into integers or floats
NUMERIC_RULES = ["horizon_days", "stop_service_min", "warehouse_handling_min", "max_warehouse_dwell_h",
                 "avg_speed_kmh", "route_factor", "generator_seed"]


MINUTE_COLUMNS = {
    "availability_start": "pick_start",
    "availability_end": "pick_end",
    "delivery_start": "deliv_start",
    "delivery_end": "deliv_end"
}

# endregion


# region Aux Functions

# Convert columns belong to DataFrame in Dates
def convert_dates(df, columns):
    for name in columns:
        df[name] = pd.to_datetime(df[name])

    return df

# Add new time columns to requests table (in minutes order from start_time) for algorithms purposes
def add_minutes(requests, start_time):
    for date_col, min_col in MINUTE_COLUMNS.items():
        requests[min_col] = (requests[date_col] - start_time).dt.total_seconds() / 60

    return requests

# endregion


# region Load Data

# Load data from csv files 
def load_data(data_path):
    missing = [f"{name}.csv" for name in FILES_NAME if not (Path(data_path) / f"{name}.csv").exists()]
    if missing:
        raise FileNotFoundError(f"{data_path}: faltam os ficheiros {missing}")
    data = {}
    for name in FILES_NAME:
        data[name] = pd.read_csv(f"{data_path}/{name}.csv")

    # Add here as needed
    data["order_items"] = convert_dates(data["order_items"], DATE_COLUMNS)

    return data

# endregion 


# region Validate Data

# Colunas que o modelo usa (as outras podem existir, mas não são obrigatórias)
REQUIRED_COLUMNS = {
    "collection_centers": ["center_id", "capacity_kg"],
    "delivery_locations": ["delivery_location_id"],
    "distance_matrix": ["origin_location_id", "destination_location_id", "distance_km", "duration_min"],
    "locations": ["location_id", "location_type", "ref_id", "name"],
    "order_items": ["order_item_id", "order_id", "product_id", "weight_kg"] + DATE_COLUMNS,
    "orders": ["order_id", "customer_id", "delivery_location_id", "status"],
    "producers": ["producer_id"],
    "products": ["product_id", "producer_id", "name"],
    "rules": ["rule", "value"],
    "vehicles": ["vehicle_id", "capacity_kg"],
}
REQUIRED_RULES = ["start_date", "flow", "window_semantics"] + NUMERIC_RULES
PRIMARY_KEYS = {"collection_centers": "center_id", "customers": "customer_id",
                "delivery_locations": "delivery_location_id", "locations": "location_id",
                "order_items": "order_item_id", "orders": "order_id", "producers": "producer_id",
                "products": "product_id", "vehicles": "vehicle_id"}


def validate_data(data):
    """
    Verifica um dataset já lido com load_data. Devolve a lista de problemas encontrados
    (vazia = dataset válido). As mesmas verificações aparecem na EDA.
    """
    problems = []
    for name, cols in REQUIRED_COLUMNS.items():
        missing = [c for c in cols if c not in data[name].columns]
        if missing:
            problems.append(f"{name}: faltam as colunas {missing}")
        elif data[name][cols].isna().any().any():
            problems.append(f"{name}: há valores nulos em colunas obrigatórias")
    if problems:
        return problems                                   # sem colunas não vale a pena continuar

    for name, key in PRIMARY_KEYS.items():
        if key in data[name] and not data[name][key].is_unique:
            problems.append(f"{name}: {key} repetido")

    oi, prod, orders, loc = data["order_items"], data["products"], data["orders"], data["locations"]
    checks = [("order_items.order_id", oi["order_id"], orders["order_id"]),
              ("order_items.product_id", oi["product_id"], prod["product_id"]),
              ("products.producer_id", prod["producer_id"], data["producers"]["producer_id"]),
              ("orders.delivery_location_id", orders["delivery_location_id"],
               data["delivery_locations"]["delivery_location_id"])]
    for label, values, keys in checks:
        if not values.isin(keys).all():
            problems.append(f"{label}: há referências que não existem")
    for kind, table, key in [("producer", "producers", "producer_id"),
                             ("delivery_location", "delivery_locations", "delivery_location_id"),
                             ("collection_center", "collection_centers", "center_id")]:
        if set(loc.loc[loc["location_type"] == kind, "ref_id"]) != set(data[table][key]):
            problems.append(f"locations: os sítios do tipo {kind} não correspondem a {table}")
    if (loc["location_type"] == "collection_center").sum() != 1:
        problems.append("o modelo suporta exatamente um centro de recolha")

    if (oi["availability_start"] >= oi["availability_end"]).any():
        problems.append("order_items: janela de recolha com início ≥ fim")
    if (oi["delivery_start"] >= oi["delivery_end"]).any():
        problems.append("order_items: janela de entrega com início ≥ fim")
    if (oi["delivery_end"] <= oi["availability_start"]).any():
        problems.append("order_items: entrega termina antes de a recolha abrir")
    if (oi["weight_kg"] <= 0).any() or (data["vehicles"]["capacity_kg"] <= 0).any():
        problems.append("pesos e capacidades têm de ser positivos")

    dm, ids = data["distance_matrix"], set(loc["location_id"])
    pairs = set(zip(dm["origin_location_id"], dm["destination_location_id"]))
    expected = {(a, b) for a in ids for b in ids if a != b}
    if not expected <= pairs:
        problems.append(f"distance_matrix: faltam {len(expected - pairs)} pares de sítios")
    if (dm["distance_km"] < 0).any() or (dm["duration_min"] < 0).any():
        problems.append("distance_matrix: distâncias ou durações negativas")

    rules = set(data["rules"]["rule"])
    missing_rules = [r for r in REQUIRED_RULES if r not in rules]
    if missing_rules:
        problems.append(f"rules: faltam as regras {missing_rules}")
    return problems

# endregion


# region Rules + Matrix

# Load rules into a dictionary
# Example: rules = {
#       "start_date": "2026-10-30",
#       "horizon_days": "2"
#       }
def load_rules(data):
    rules = dict(zip(data["rules"]["rule"], data["rules"]["value"]))
    for name in NUMERIC_RULES:
        rules[name] = float(rules[name])

    rules["max_warehouse_dwell_min"] = rules["max_warehouse_dwell_h"] * 60
    rules["start_date"] = pd.to_datetime(rules["start_date"])
    return rules


# Load travel information (km and minutes) into a dictionary of dicitionaries
# in order to have easy access 
# travel_km[i][j] gives back the distance in kilometers between i and j, i beeing the origin and j the destination
# travel_minute[i][j] gives back the distance in minutes between i and j, i beeing the origin and j the destination
def build_travel(data):
    bt = data["distance_matrix"]
    travel_minute = {}
    travel_km = {}

    for orig, dest, km, time in zip(bt["origin_location_id"], bt["destination_location_id"], bt["distance_km"], bt["duration_min"]):
        if orig not in travel_km:
            travel_km[orig] = {}
            travel_minute[orig] = {}
        travel_km[orig][dest] = km
        travel_minute[orig][dest] = time

    return travel_km, travel_minute


# endregion


# region Building Orders Table


def build_requests(data):

    # Origin Branch
    items = data["order_items"]
    products = data["products"]
    locations = data["locations"]

    request = pd.merge(items, products, on="product_id")
    request = request.drop(columns=["unit", "unit_weight_kg", "category", "quantity"])

    mask = locations["location_type"] == "producer"
    producers_locations = locations[mask]

    request = pd.merge(request, producers_locations, left_on="producer_id", right_on="ref_id")
    request = request.drop(columns=["ref_id", "municipality", "location_type", "producer_id"])
    request = request.rename(columns={"location_id": "origin_id", 
                                      "name_x": "product_name", 
                                      "name_y": "origin_name",
                                      "latitude": "origin_latitude",
                                      "longitude": "origin_longitude"})

    # Destination Branch
    orders = data["orders"]
    orders = orders[orders["status"] == "confirmed"]       # só as encomendas confirmadas são servidas

    request = pd.merge(request, orders, on="order_id")
    request = request.drop(columns=["order_datetime", "status"])

    mask = locations["location_type"] == "delivery_location"
    destination_location = locations[mask]

    request = pd.merge(request, destination_location, left_on="delivery_location_id", right_on="ref_id")
    request = request.drop(columns=["ref_id", "municipality", "location_type", "delivery_location_id", "customer_id"])
    request = request.rename(columns={"location_id":"destination_id",
                                      "name": "destination_name",
                                      "latitude": "destination_latitude",
                                      "longitude": "destination_longitude"})


    return request


# endregion


# region Prepare data for a route simulation
def get_warehouse_id(data):
    locations = data["locations"]
    mask = locations["location_type"] == "collection_center"
    warehouses = locations[mask]
    warehouse_id = warehouses["location_id"].iloc[0]

    return warehouse_id


def load_vehicles(data):
    vehicles = {}
    for row in data["vehicles"].itertuples():
        vehicles[row.vehicle_id] = Vehicle(row.vehicle_id, float(row.capacity_kg),
                                           bool(getattr(row, "refrigerated", False)))
    return vehicles

# endregion


# region Entities (Request, Vehicle, State)

@dataclass(frozen=True)
class Request:
    """Um item de encomenda a levar do produtor (origin) ao local de entrega (destination)."""
    item_id: int
    order_id: int
    product: str
    weight_kg: float
    origin_id: int
    destination_id: int
    pick_start: float          # janelas em minutos desde start_date
    pick_end: float
    deliv_start: float
    deliv_end: float


@dataclass(frozen=True)
class Vehicle:
    vehicle_id: int
    capacity_kg: float
    refrigerated: bool         # lido mas não usado: nenhum produto exige frio (ver EDA)


class VanState(NamedTuple):
    """Uma carrinha num estado da pesquisa."""
    location: int
    time: float
    carrying_in: frozenset     # recolhidos, a caminho do armazém
    carrying_out: frozenset    # carregados no armazém, a caminho do destino
    finished: bool


class State(NamedTuple):
    """
    Estado da pesquisa de UM dia. É um tuplo (imutável e "hashable"), por isso pode ser
    usado diretamente no conjunto de visitados e na fila de prioridade.
    """
    vans: tuple                # um VanState por carrinha
    warehouse: tuple           # ((item, hora de chegada ao armazém), ...), ordenado
    to_pick: frozenset         # itens do dia ainda por recolher
    to_deliver: frozenset      # itens do dia ainda por entregar

# endregion


# region Problem Class

# Put all things together
class Problem: 
    """Uma instância do problema, lida de uma pasta com os 12 CSVs (qualquer dataset)."""

    def __init__(self, data_path):
        data = load_data(data_path)
        problems = validate_data(data)
        if problems:
            raise ValueError(f"{data_path}: dataset inválido:\n  - " + "\n  - ".join(problems))
        self.name = Path(data_path).name
        self.rules = load_rules(data)
        self.travel_km, self.travel_minutes = build_travel(data)

        self.requests = build_requests(data)
        self.requests = add_minutes(self.requests, self.rules["start_date"])
        self.requests = self.requests.set_index("order_item_id")

        self.warehouse_id = get_warehouse_id(data)
        self.warehouse_capacity = float(data["collection_centers"]["capacity_kg"].iloc[0])
        self.vehicles = load_vehicles(data)
        self.request_by_id = {i: Request(i, r.order_id, r.product_name, float(r.weight_kg),
                                         r.origin_id, r.destination_id, r.pick_start, r.pick_end,
                                         r.deliv_start, r.deliv_end)
                              for i, r in self.requests.iterrows()}
        days = (self.requests[["pick_start", "deliv_start"]] // MINUTES_PER_DAY).astype(int)
        self.days = sorted(set(days["pick_start"]) | set(days["deliv_start"]))

    @classmethod
    def from_csv(cls, data_path):
        return cls(data_path)


#endregion


# region Aux Route Functions

# region Answer questions about ONE stop

def stop_location(problem, stop_type, item):

    if stop_type == "pickup":
        return problem.request_by_id[item].origin_id
    elif stop_type == "delivery":
        return problem.request_by_id[item].destination_id
    elif stop_type == "warehouse": 
        return problem.warehouse_id
    else:
        raise ValueError(f"Stop type unknown: {stop_type}")


def stop_window(problem, stop_type, item):

    if stop_type == "pickup":
        r = problem.request_by_id[item]
        return r.pick_start, r.pick_end
    elif stop_type == "delivery":
        r = problem.request_by_id[item]
        return r.deliv_start, r.deliv_end
    elif stop_type == "warehouse":
        return 0, float("inf")
    else: 
        raise ValueError(f"Stop type unknown: {stop_type}")
        

def stop_service(problem, stop_type):

    if stop_type == "pickup" or stop_type == "delivery":
        return problem.rules["stop_service_min"]
    elif stop_type == "warehouse":
        return problem.rules["warehouse_handling_min"] * 2
    else: 
        raise ValueError(f"Stop type unknown: {stop_type}")

# endregion

# region Check load weight & Check next deliveries 

def current_load(problem, carrying_in, carrying_out):

    total_load = 0
    for item in carrying_in | carrying_out:
        total_load += problem.request_by_id[item].weight_kg

    return total_load

# Route Example
# route = [
#     ("pickup",    1),      # recolher o item 1 (no origin_id do item 1)
#     ("pickup",    2),      # recoher o item 2
#     ("warehouse", None),   # passar pelo armazém
#     ("delivery",  1),      # entregar o item 1 (no destination_id do item 1)
#     ("delivery",  2),      # entregar o item 2
# ]
def deliveries_until_next_warehouse(route, vehicle_position):

    items = []
    for stop_type, item in route[vehicle_position + 1:]:
        if stop_type == "delivery":
            items.append(item)
        elif stop_type == "warehouse":
            break

    return items

# endregion

# endregion


# region Route Simulation

# Route Example
# route = [
#     ("pickup",    1),      # recolher o item 1 (no origin_id do item 1)
#     ("pickup",    2),      # recoher o item 2
#     ("warehouse", None),   # passar pelo armazém
#     ("delivery",  1),      # entregar o item 1 (no destination_id do item 1)
#     ("delivery",  2),      # entregar o item 2
# ]
#
# COMO FUNCIONA A SIMULAÇÃO (versão com trocas entre carrinhas)
# -------------------------------------------------------------
# O serviço (10 min) é por PARAGEM, ou seja, por visita a um sítio: vários itens recolhidos
# ou entregues seguidos no mesmo sítio partilham os mesmos 10 min (ver _service_time).
#
# As carrinhas de um dia são simuladas EM PARALELO, por ordem de horas: avança sempre a
# carrinha que está mais atrasada no relógio. Assim, quando uma carrinha chega ao armazém,
# já aconteceu tudo o que as outras fizeram antes dessa hora.
#
# No armazém há duas fases:
#   1. DESCARREGAR (15 min): o que a carrinha recolheu entra no armazém (fica disponível
#      para QUALQUER carrinha, incluindo as outras do mesmo dia).
#   2. CARREGAR (15 min): carrega as entregas que faz até ao próximo armazém. Se um desses
#      itens ainda não chegou (vem noutra carrinha), a carrinha ESPERA no armazém até ele
#      chegar. Se nunca chega (nenhuma carrinha o traz), o plano é inviável.
def _service_time(problem, route, position, location):
   
    stop_type = route[position][0]
    if position + 1 < len(route):
        next_type, next_item = route[position + 1]
        if next_type == stop_type and stop_location(problem, next_type, next_item) == location:
            return 0.0                      # a visita continua: o serviço conta no fim
    return stop_service(problem, stop_type)


# O estado de UMA carrinha durante a simulação.
class _VanRun:

    def __init__(self, problem, vehicle_id, route, start_time):
        self.vehicle_id = vehicle_id
        self.route = route
        self.capacity = problem.vehicles[vehicle_id].capacity_kg
        self.time = start_time
        self.start_time = start_time
        self.location = problem.warehouse_id
        self.position = 0                # próxima paragem da rota
        self.carrying_in = set()         # na carrinha, a caminho do armazém (recolhidos)
        self.carrying_out = set()        # na carrinha, a caminho do destino (carregados no armazém)
        self.delivered = set()
        self.timeline = []
        self.distance_km = 0.0
        self.wait_min = 0.0
        self.pending_load = None         # carregamento à espera de itens (fase 2 do armazém)
        self.finished = False


def _step(problem, van, shared):
    """
    Executa a próxima paragem de uma carrinha.
    Devolve "ok", "blocked" (à espera de itens no armazém) ou ("fail", motivo).
    shared = {"warehouse": {item: hora de chegada}, "picked": set(), "delivered": set()}
    """
    handling = problem.rules["warehouse_handling_min"]
    max_dwell = problem.rules["max_warehouse_dwell_min"]
    warehouse = shared["warehouse"]

    # --- fase 2 de um armazém que ficou à espera
    if van.pending_load is not None:
        entry, unload_end, needed = van.pending_load
        missing = [it for it in needed if it not in warehouse]
        if missing:
            return "blocked"
        load_start = max([unload_end] + [warehouse[it] for it in needed])   # espera pelo último
        for it in needed:
            dwell = load_start - warehouse[it]                               # regra das 24 h
            if dwell > max_dwell:
                return ("fail", f"warehouse: item {it} stayed {dwell} min in the warehouse (max {max_dwell})")
        for it in needed:
            del warehouse[it]
            van.carrying_out.add(it)
        load = current_load(problem, van.carrying_in, van.carrying_out)
        if load > van.capacity:
            return ("fail", f"warehouse: load {load} kg exceeds capacity {van.capacity} kg")
        van.wait_min += load_start - unload_end
        van.time = load_start + handling
        entry["end"] = van.time
        entry["load"] = load
        van.pending_load = None
        van.position += 1
        return "ok"

    # --- fim da rota: voltar ao armazém
    if van.position >= len(van.route):
        if van.carrying_in:
            return ("fail", f"route ends with items not taken to the warehouse: {sorted(van.carrying_in)}")
        if van.location != problem.warehouse_id:
            van.time += problem.travel_minutes[van.location][problem.warehouse_id]
            van.distance_km += problem.travel_km[van.location][problem.warehouse_id]
            van.location = problem.warehouse_id
        van.finished = True
        return "ok"

    stop_type, item = van.route[van.position]

    # 1. ONDE
    destination = stop_location(problem, stop_type, item)

    # 2. VIAGEM (se já estou no sítio, a viagem é 0)
    if destination != van.location:
        van.time += problem.travel_minutes[van.location][destination]
        van.distance_km += problem.travel_km[van.location][destination]
    arrival = van.time

    # 3. JANELA — regra service_start (rules.csv): a janela controla o INÍCIO do serviço
    open_window, close_window = stop_window(problem, stop_type, item)
    if van.time > close_window:
        return ("fail", f"{stop_type} of item {item}: arrived at {van.time}, window closed at {close_window}")
    if van.time < open_window:
        van.wait_min += open_window - van.time        # chegou cedo → espera
        van.time = open_window
    service_start = van.time

    entry = {"stop": stop_type, "item": item, "location": van.location,
             "destination": destination, "arrival": arrival, "start": service_start}
    van.timeline.append(entry)
    van.location = destination

    # 4. e 5. SERVIÇO e CARGA
    if stop_type == "pickup":
        if item in shared["picked"]:
            return ("fail", f"pickup of item {item}: item was already picked up")
        shared["picked"].add(item)
        van.carrying_in.add(item)
        van.time += _service_time(problem, van.route, van.position, destination)
        load = current_load(problem, van.carrying_in, van.carrying_out)
        if load > van.capacity:
            return ("fail", f"pickup of item {item}: load {load} kg exceeds capacity {van.capacity} kg")
        entry["end"], entry["load"] = van.time, load
        van.position += 1
        return "ok"

    if stop_type == "delivery":
        if item not in van.carrying_out:
            return ("fail", f"delivery of item {item}: item is not on board from the warehouse")
        van.carrying_out.remove(item)
        van.delivered.add(item)
        shared["delivered"].add(item)
        van.time += _service_time(problem, van.route, van.position, destination)
        entry["end"] = van.time
        entry["load"] = current_load(problem, van.carrying_in, van.carrying_out)
        van.position += 1
        return "ok"

    # stop_type == "warehouse"
    # FASE 1 — DESCARREGAR: o item chega ao armazém no fim da descarga
    unload_end = service_start + handling
    for it in van.carrying_in:
        warehouse[it] = unload_end
    van.carrying_in.clear()
    stored = sum(problem.request_by_id[it].weight_kg for it in warehouse)
    if stored > problem.warehouse_capacity:
        return ("fail", f"warehouse: {stored:.1f} kg stored exceeds capacity {problem.warehouse_capacity} kg")
    # FASE 2 — CARREGAR (pode ficar à espera de itens que outra carrinha ainda vai trazer)
    needed = deliveries_until_next_warehouse(van.route, van.position)
    van.pending_load = (entry, unload_end, needed)
    van.time = unload_end
    return _step(problem, van, shared)


def simulate_day(problem, day_routes, warehouse_in=None):
    """
    Simula TODAS as carrinhas de um dia, em paralelo.
    day_routes   = {vehicle_id: (start_time, route)}
    warehouse_in = {item: hora de chegada} do que está no armazém no início do dia.

    Devolve: feasible, reason, routes ({vehicle_id: resultado da carrinha}),
             warehouse_out, delivered, distance_km, duration_min, wait_min
    """
    shared = {"warehouse": dict(warehouse_in) if warehouse_in else {},   # CÓPIA
              "picked": set(), "delivered": set()}
    vans = [_VanRun(problem, v, route, start) for v, (start, route) in sorted(day_routes.items())]

    def van_result(van):
        return {"feasible": True, "reason": None, "end_time": van.time,
                "timeline": van.timeline, "delivered": van.delivered,
                "distance_km": van.distance_km, "duration_min": van.time - van.start_time,
                "wait_min": van.wait_min}

    def fail(reason, van=None):
        prefix = f"vehicle {van.vehicle_id}: " if van is not None and len(vans) > 1 else ""
        return {"feasible": False, "reason": prefix + reason,
                "end_time": van.time if van else None,
                "timeline": van.timeline if van else [],
                "routes": {v.vehicle_id: van_result(v) for v in vans},
                "warehouse_out": None, "delivered": None,
                "distance_km": None, "duration_min": None, "wait_min": None}

    while True:
        active = sorted((v for v in vans if not v.finished), key=lambda v: (v.time, v.vehicle_id))
        if not active:
            break
        progressed = False
        for van in active:                    # tenta primeiro a carrinha mais atrasada
            status = _step(problem, van, shared)
            if status == "ok":
                progressed = True
                break
            if status != "blocked":
                return fail(status[1], van)
        if not progressed:                    # todas à espera de itens que nunca chegam
            waiting = {v.vehicle_id: [it for it in v.pending_load[2] if it not in shared["warehouse"]]
                       for v in active}
            van = active[0]
            return fail(f"warehouse: items never arrive at the warehouse {waiting}", van)

    return {"feasible": True, "reason": None,
            "routes": {v.vehicle_id: van_result(v) for v in vans},
            "warehouse_out": shared["warehouse"],
            "delivered": shared["delivered"],
            "distance_km": sum(v.distance_km for v in vans),
            "duration_min": sum(v.time - v.start_time for v in vans),
            "wait_min": sum(v.wait_min for v in vans)}


def simulate_route(problem, route, start_time, vehicle_id, warehouse_in=None):
    """
    Simula UMA rota, de UMA carrinha (é o simulate_day com uma só carrinha).

    warehouse_in : {item: hora de chegada} do que já está no armazém quando a rota começa.

    Devolve um dicionário:
      feasible, reason, end_time, timeline
      warehouse_out : {item: hora de chegada} do que fica no armazém no fim
      delivered     : conjunto dos itens entregues nesta rota
      distance_km, duration_min, wait_min
    """
    r = simulate_day(problem, {vehicle_id: (start_time, route)}, warehouse_in)
    van = r["routes"][vehicle_id]
    if not r["feasible"]:
        return {"feasible": False, "reason": r["reason"], "end_time": r["end_time"],
                "timeline": r["timeline"], "warehouse_out": None, "delivered": None,
                "distance_km": None, "duration_min": None, "wait_min": None}
    return {"feasible": True, "reason": None, "end_time": van["end_time"],
            "timeline": van["timeline"], "warehouse_out": r["warehouse_out"],
            "delivered": van["delivered"], "distance_km": van["distance_km"],
            "duration_min": van["duration_min"], "wait_min": van["wait_min"]}

# endregion


# region Plan Simulation + Cost

def simulate_plan(problem, plan, require_all=True):
    """
    Simula um plano completo: as rotas de todos os dias e de todas as carrinhas.

    plan        : {(day, vehicle_id): (start_time, route)}   day 0 = start_date
    require_all : True → todos os itens têm de ser entregues (uma solução completa);
                  False → aceita planos parciais (útil para testes)

    Dentro de um dia, as carrinhas são simuladas em paralelo (simulate_day): uma carrinha
    pode levar itens que outra descarregou, desde que já tenham chegado ao armazém.
    O que fica no armazém no fim de um dia passa para o dia seguinte.

    Devolve: feasible, reason, distance_km, duration_min, wait_min,
             routes ({(day, vehicle): resultado da carrinha}), warehouse_end
    """
    stock = {}
    delivered = set()
    totals = {"distance_km": 0.0, "duration_min": 0.0, "wait_min": 0.0}
    routes = {}

    def fail(reason):
        return {"feasible": False, "reason": reason, **{k: None for k in totals},
                "routes": routes, "warehouse_end": None}

    for day in sorted({d for d, _ in plan}):
        day_routes = {}
        for (d, vehicle_id), (start_time, route) in plan.items():
            if d != day:
                continue
            if not (day * MINUTES_PER_DAY <= start_time < (day + 1) * MINUTES_PER_DAY):
                return fail(f"day {day}, vehicle {vehicle_id}: start_time {start_time} is outside the day")
            day_routes[vehicle_id] = (start_time, route)

        r = simulate_day(problem, day_routes, stock)
        if not r["feasible"]:
            return fail(f"day {day}: {r['reason']}")

        repeated = delivered & r["delivered"]
        if repeated:
            return fail(f"day {day}: items delivered twice {sorted(repeated)}")
        delivered |= r["delivered"]
        for k in totals:
            totals[k] += r[k]
        for v, res in r["routes"].items():
            routes[(day, v)] = res
        stock = r["warehouse_out"]

    if require_all:
        missing = set(problem.requests.index) - delivered
        if missing:
            return fail(f"items not delivered: {sorted(missing)}")

    return {"feasible": True, "reason": None, **totals, "routes": routes, "warehouse_end": stock}


def plan_cost(result, metric="distance_km", alpha=0.0):
    """
    Custo de um plano já simulado. Um plano inviável tem custo infinito.
    metric: "distance_km" | "duration_min" | "wait_min"
    alpha:  soft constraint — custo de 1 minuto de espera, nas unidades de `metric`
            (ex.: metric="distance_km", alpha=0.5 → custo = km + 0,5 × minutos de espera)
    """
    if not result["feasible"]:
        return float("inf")
    return result[metric] + alpha * result["wait_min"]

# endregion
