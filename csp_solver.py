"""
FASE 6 — CSP com a biblioteca python-constraint.

O código segue a estrutura dos Notebooks 04 e 91 (Course Timetabling):
    - from constraint import *  /  Problem()  /  addVariable  /  addConstraint  /  getSolution
    - as restrições UNÁRIAS (janelas, regra das 24 h) são aplicadas diretamente ao DOMÍNIO
      antes de criar as variáveis (como o lesson_domain do Notebook 91);
    - restrições entre variáveis com lambda / FunctionConstraint;
    - a solução é mostrada numa tabela pandas (como o display_timetable).

ATENÇÃO: "from constraint import *" traz uma classe Problem da biblioteca. Este ficheiro nunca
usa o Problem do model.py pelo nome (recebe o objeto `problem` já criado), por isso não há
conflito. No notebook, NÃO se faz "from constraint import *": importa-se só daqui.

FORMULAÇÃO (um CSP por dia). Os itens são agrupados: itens da mesma encomenda e do mesmo
produtor são recolhidos juntos; itens da mesma encomenda são entregues juntos.
    Variáveis, para cada grupo g:
        V_g  carrinha                {1, 2}
        P_g  hora da recolha         slots de 15 min dentro da janela de recolha
        D_g  hora da entrega         slots de 15 min dentro da janela de entrega
    Restrições:
        R1 janelas de tempo                          (unária → domínio)
        R2 regra das 24 h                            (unária → domínio)
        R3 entrega depois de recolha + armazém       (binária)
        R4 a mesma carrinha não está em dois sítios  (4-ária, com as carrinhas)
        R5 capacidade                                (n-ária, verificação parcial)

PARA AS COMPARAÇÕES DO NOTEBOOK (secção 9):
    solve_csp(problem, solver="backtracking" | "recursive" | "min_conflicts",
              variable_order=..., value_order=..., windows_as=..., capacity_as=..., slot=...,
              max_solutions=K, alpha=...)
O plano final é SEMPRE validado pelo simulate_plan (o mesmo validador dos outros algoritmos).
"""
import math
import random
import time

import pandas as pd
from constraint import *

from model import simulate_plan, MINUTES_PER_DAY

SLOT = 15     # minutos


# =====================================================================
# Solver: o BacktrackingSolver da biblioteca, a contar os backtracks
# =====================================================================

class CountingBacktrackingSolver(BacktrackingSolver):
    """
    O BacktrackingSolver da biblioteca (versão 1.4.0), com três acrescentos:
      1. CONTA os backtracks (uma variável fica sem valores e o solver RECUA) e as atribuições
         experimentadas — a biblioteca não expõe estes números;
      2. LIMITES de tempo e de backtracks: ao atingi-los pára com status = "limit"
         (≠ "sem solução", que só se conclui quando a árvore é esgotada);
      3. ESTRATÉGIAS configuráveis. A biblioteca fixa "Degree, depois MRV" e a ordem do domínio;
         para as comparar foi preciso copiar o ciclo do getSolutionIter (não há opção na API):
           variable_order = "degree_mrv" (o da biblioteca) | "mrv_degree" | "mrv" | "input"
                            | "chronological" (grupo que abre mais cedo primeiro; V antes de P e D)
           value_order    = "domain" (a ordem do domínio = as nossas preferências) | "lcv"
    O forward checking é o da biblioteca (forwardcheck=True).
    """

    def __init__(self, forwardcheck=True, variable_order="degree_mrv", value_order="domain",
                 max_backtracks=None, time_limit=None):
        super().__init__(forwardcheck)
        self.variable_order, self.value_order = variable_order, value_order
        self.max_backtracks, self.time_limit = max_backtracks, time_limit
        self.backtracks = self.assignments = 0
        self.when = {}                       # hora de abertura de cada variável (build_day_csp)
        self.status = "running"

    def _key(self, variable, domains, vconstraints, position):
        degree, size = len(vconstraints[variable]), len(domains[variable])
        return {"degree_mrv": (-degree, size), "mrv_degree": (size, -degree),
                "mrv": (size,), "input": (position[variable],),
                "chronological": (self.when.get(variable, 0),)}[self.variable_order]   # empate: nome

    def _lcv(self, variable, values, domains, vconstraints, assignments, pushdomains):
        """Least Constraining Value: primeiro o valor que deixa MAIS valores aos vizinhos."""
        def room(value):
            assignments[variable] = value
            for d in pushdomains:
                d.pushState()
            ok = all(c(v, domains, assignments, pushdomains) for c, v in vconstraints[variable])
            left = sum(len(d) for d in pushdomains) if ok else -1
            for d in pushdomains:
                d.popState()
            del assignments[variable]
            return left
        scores = {value: room(value) for value in values}
        return sorted(values, key=lambda value: scores[value])      # pop() tira o maior

    def _over_limit(self):
        if self.max_backtracks is not None and self.backtracks > self.max_backtracks:
            return True
        return self.time_limit is not None and time.perf_counter() - self._start > self.time_limit

    def getSolutionIter(self, domains, constraints, vconstraints):
        self._start = time.perf_counter()
        position = {v: i for i, v in enumerate(domains)}             # ordem de criação
        forwardcheck = self._forwardcheck
        assignments = {}
        queue = []
        while True:
            lst = sorted((self._key(v, domains, vconstraints, position), v) for v in domains
                         if v not in assignments)
            if lst:
                variable = lst[0][-1]
                values = domains[variable][:]
                if forwardcheck:
                    pushdomains = [domains[x] for x in domains if x not in assignments and x != variable]
                else:
                    pushdomains = None
                if self.value_order == "lcv" and pushdomains:
                    values = self._lcv(variable, values, domains, vconstraints, assignments, pushdomains)
            else:
                self.status = "ok"
                yield assignments.copy()
                if not queue:
                    self.status = "exhausted"
                    return
                variable, values, pushdomains = queue.pop()
                if pushdomains:
                    for domain in pushdomains:
                        domain.popState()
            while True:
                if not values:
                    self.backtracks += 1                      # ← RECUAR: um backtrack
                    if self._over_limit():
                        self.status = "limit"
                        return
                    del assignments[variable]
                    while queue:
                        variable, values, pushdomains = queue.pop()
                        if pushdomains:
                            for domain in pushdomains:
                                domain.popState()
                        if values:
                            break
                        del assignments[variable]
                    else:
                        self.status = "no_solution" if self.status == "running" else "exhausted"
                        return
                assignments[variable] = values.pop()
                self.assignments += 1
                if pushdomains:
                    for domain in pushdomains:
                        domain.pushState()
                for cons, variables in vconstraints[variable]:
                    if not cons(variables, domains, assignments, pushdomains):
                        break
                else:
                    break
                if pushdomains:
                    for domain in pushdomains:
                        domain.popState()
            queue.append((variable, values, pushdomains))


# =====================================================================
# R5: capacidade com verificação PARCIAL
# =====================================================================

class CapacityConstraint(Constraint):
    """
    Peso total dos grupos de cada carrinha ≤ capacidade.

    Como diz o comentário do Notebook 91: uma restrição n-ária com lambda só é verificada
    quando TODAS as suas variáveis estão atribuídas, e o solver não consegue podar cedo.
    Esta classe soma logo o peso dos grupos JÁ atribuídos e rejeita assim que passa a capacidade.
    """

    def __init__(self, weights, capacities):
        self.weights = weights            # {"V_g": peso do grupo}
        self.capacities = capacities      # {carrinha: capacidade}

    def __call__(self, variables, domains, assignments, forwardcheck=False):
        load = {v: 0.0 for v in self.capacities}
        for var in variables:
            if var in assignments:
                load[assignments[var]] += self.weights[var]
        return all(load[v] <= self.capacities[v] for v in load)


# =====================================================================
# Dados do CSP de um dia (como o parsed_data do Notebook 91)
# =====================================================================

def day_groups(problem, day):
    """Os grupos de itens do dia, com o sítio, o peso e as janelas comuns a todos os itens."""
    info = problem.request_by_id
    day_of = lambda m: int(m // MINUTES_PER_DAY)
    groups = {}
    for item, r in info.items():
        pday, dday = day_of(r.pick_start), day_of(r.deliv_start)
        if pday == day and dday == day:
            groups.setdefault(("same", r.order_id, r.origin_id), []).append(item)
        elif pday == day:
            groups.setdefault(("pick", r.order_id, r.origin_id), []).append(item)
        elif dday == day:
            groups.setdefault(("deliv", r.order_id, None), []).append(item)

    result = []
    for (kind, order, origin), items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        g = {"name": f"{kind}_o{order}" + (f"_p{origin}" if origin is not None else ""),
             "kind": kind, "items": sorted(items),
             "weight": sum(info[i].weight_kg for i in items)}
        if kind in ("same", "pick"):
            g["origin"] = info[items[0]].origin_id
            g["pick_window"] = (max(info[i].pick_start for i in items),
                                min(info[i].pick_end for i in items))
        if kind in ("same", "deliv"):
            g["dest"] = info[items[0]].destination_id
            g["deliv_window"] = (max(info[i].deliv_start for i in items),
                                 min(info[i].deliv_end for i in items))
        result.append(g)
    return result


def slots(window, latest=None, slot=SLOT):
    """
    Os slots de 15 min dentro da janela, por ordem DECRESCENTE.
    O solver experimenta primeiro o ÚLTIMO valor da lista (values.pop()), por isso a ordem
    decrescente faz com que experimente primeiro a hora MAIS CEDO (heurística soft).
    """
    start, end = window
    if latest is not None:
        end = min(end, latest)
    first = math.ceil(start / slot) * slot
    return list(range(int(end // slot) * slot, first - 1, -slot))


# =====================================================================
# Constraint functions (como as do Notebook 91)
# =====================================================================

def make_separation(kind1, place1, kind2, place2, travel, S, H, W):
    """
    R4 entre dois eventos da MESMA carrinha. Devolve uma função (t1, t2) → True/False,
    ou None se os dois podem acontecer na mesma visita (mesmo sítio, mesmo tipo).
    """
    if kind1 == kind2:                                       # duas recolhas ou duas entregas
        if place1 == place2:
            return None
        gap = S + travel(place1, place2)
        return lambda a, b: abs(a - b) >= gap
    # uma recolha (em place1) e uma entrega (em place2)
    go = S + travel(place1, W) + 2 * H + travel(W, place2)   # recolha → armazém → entrega
    back = S + travel(place2, place1)                         # entrega → recolha
    return lambda a, b: (b - a >= go) if a <= b else (a - b >= back)


# =====================================================================
# Construir e resolver o CSP de um dia
# =====================================================================

def build_day_csp(problem, day, stock, solver=None, windows_as="domain", capacity_as="class", slot=SLOT):
    """
    Constrói o CSP de um dia. Os dois últimos parâmetros só servem para COMPARAR formas
    equivalentes de escrever as mesmas restrições (secção 9 do notebook):
      windows_as  = "domain"   → R1/R2 cortam o domínio antes de criar a variável (Notebook 91)
                  = "inset"    → domínio = todos os slots do dia + InSetConstraint(slots permitidos)
                  = "function" → domínio = todos os slots do dia + FunctionConstraint(lo ≤ t ≤ hi)
      capacity_as = "class"    → CapacityConstraint (verifica atribuições parciais)
                  = "function" → FunctionConstraint n-ária (só verifica com TODAS atribuídas)
    """
    travel = lambda a, b: 0.0 if a == b else problem.travel_minutes[a][b]
    W = problem.warehouse_id
    S = problem.rules["stop_service_min"]
    H = problem.rules["warehouse_handling_min"]
    max_dwell = problem.rules["max_warehouse_dwell_min"]
    vans = sorted(problem.vehicles)
    groups = day_groups(problem, day)

    # Initiate the solver
    csp_problem = Problem(solver or CountingBacktrackingSolver())
    whole_day = slots((day * MINUTES_PER_DAY, (day + 1) * MINUTES_PER_DAY - slot), slot=slot)

    def add_time_variable(name, allowed):
        """R1/R2: só os slots `allowed`, da forma escolhida em windows_as."""
        if windows_as == "domain":
            csp_problem.addVariable(name, allowed)
        elif windows_as == "inset":
            csp_problem.addVariable(name, whole_day)
            csp_problem.addConstraint(InSetConstraint(set(allowed)), [name])
        else:
            lo, hi = min(allowed), max(allowed)
            csp_problem.addVariable(name, whole_day)
            csp_problem.addConstraint(FunctionConstraint(lambda t, lo=lo, hi=hi: lo <= t <= hi), [name])

    # ---- Add variables and their domain
    # As restrições unárias (R1 janelas, R2 regra das 24 h) vão diretamente ao domínio.
    for g in groups:
        # Soft 1 — carrinhas: as entregas do que dormiu no armazém (de manhã cedo, janelas
        # apertadas) experimentam primeiro a carrinha MAIOR; o resto, a menor.
        # (No dia 31: de 1,2 milhões de backtracks para cerca de 1 200.)
        van_order = sorted(vans, key=lambda v: problem.vehicles[v].capacity_kg)          # menor → maior
        if g["kind"] == "deliv":
            van_order = van_order[::-1]
        csp_problem.addVariable(f"V_{g['name']}", van_order[::-1])           # pop() → 1.º da lista

        if "pick_window" in g:                                               # R1
            pick_domain = slots(g["pick_window"], slot=slot)
            if not pick_domain:
                raise ValueError(f"grupo {g['name']}: a janela de recolha não tem nenhum slot")
            add_time_variable(f"P_{g['name']}", pick_domain)

        if "deliv_window" in g:                                              # R1 + R2
            latest = None
            if g["kind"] == "deliv":
                # R2: carregar até 24 h depois da chegada do item mais antigo; a entrega é,
                # no máximo, 15 min de carga + a viagem depois disso
                arrival = min(stock[i] for i in g["items"])
                latest = arrival + max_dwell + H + travel(W, g["dest"])
            deliv_domain = slots(g["deliv_window"], latest, slot)
            if not deliv_domain:
                raise ValueError(f"grupo {g['name']}: nenhum slot de entrega respeita a janela e as 24 h")
            add_time_variable(f"D_{g['name']}", deliv_domain)

    # hora de abertura de cada variável (só usada pela ordem "chronological")
    when = {}
    for g in groups:
        opens = [g[w][0] for w in ("pick_window", "deliv_window") if w in g]
        when[f"V_{g['name']}"] = min(opens) - 0.5
        if "pick_window" in g:
            when[f"P_{g['name']}"] = g["pick_window"][0]
        if "deliv_window" in g:
            when[f"D_{g['name']}"] = g["deliv_window"][0]
    if isinstance(csp_problem._solver, CountingBacktrackingSolver):
        csp_problem._solver.when = when

    # ---- Constraints
    # R3: a entrega de um grupo do mesmo dia é depois da recolha + armazém
    for g in groups:
        if g["kind"] == "same":
            gap = S + travel(g["origin"], W) + 2 * H + travel(W, g["dest"])
            csp_problem.addConstraint(lambda p, d, gap=gap: d >= p + gap,
                                      [f"P_{g['name']}", f"D_{g['name']}"])

    # R4: a mesma carrinha não está em dois sítios ao mesmo tempo
    def events(g):
        ev = []
        if "pick_window" in g:
            ev.append(("P", g["origin"]))
        if "deliv_window" in g:
            ev.append(("D", g["dest"]))
        return ev

    for x in range(len(groups)):
        for y in range(x + 1, len(groups)):
            g, h = groups[x], groups[y]
            for k1, p1 in events(g):
                for k2, p2 in events(h):
                    if k1 == "D" and k2 == "P":                   # a recolha vem sempre primeiro
                        a_var, b_var = f"P_{h['name']}", f"D_{g['name']}"
                        rule = make_separation("P", p2, "D", p1, travel, S, H, W)
                    else:
                        a_var, b_var = f"{k1}_{g['name']}", f"{k2}_{h['name']}"
                        rule = make_separation(k1, p1, k2, p2, travel, S, H, W)
                    if rule is None:
                        continue
                    csp_problem.addConstraint(
                        lambda vg, vh, a, b, rule=rule: vg != vh or rule(a, b),
                        [f"V_{g['name']}", f"V_{h['name']}", a_var, b_var])

    # R5: capacidade (restrição própria, com verificação parcial)
    weights = {f"V_{g['name']}": g["weight"] for g in groups}
    capacities = {v: problem.vehicles[v].capacity_kg for v in problem.vehicles}
    if capacity_as == "class":
        csp_problem.addConstraint(CapacityConstraint(weights, capacities), list(weights))
    else:
        names = list(weights)
        def fits(*vans_chosen):
            load = {v: 0.0 for v in capacities}
            for name, v in zip(names, vans_chosen):
                load[v] += weights[name]
            return all(load[v] <= capacities[v] for v in load)
        csp_problem.addConstraint(FunctionConstraint(fits), names)

    return csp_problem, groups


def assignment_to_routes(problem, day, groups, solution):
    """
    Transforma a atribuição em rotas (o formato do simulate_plan): para cada carrinha, ordena os
    eventos pela hora, junta os seguidos no mesmo sítio numa visita e mete uma passagem no
    armazém sempre que passa de recolhas para entregas (e no fim, se acabar com recolhas).
    """
    info = problem.request_by_id
    travel = lambda a, b: 0.0 if a == b else problem.travel_minutes[a][b]
    W = problem.warehouse_id
    H = problem.rules["warehouse_handling_min"]
    plan = {}
    for v in sorted(problem.vehicles):
        ev = []
        for g in groups:
            if solution[f"V_{g['name']}"] != v:
                continue
            if "pick_window" in g:
                ev.append((solution[f"P_{g['name']}"], 0, "pickup", g["origin"], g["items"]))
            if "deliv_window" in g:
                ev.append((solution[f"D_{g['name']}"], 1, "delivery", g["dest"], g["items"]))
        if not ev:
            continue
        ev.sort(key=lambda e: (e[0], e[1], e[3]))

        route, previous = [], None
        for time, _, kind, place, items in ev:
            if kind == "delivery" and previous != "delivery":
                route.append(("warehouse", None))
            key = "pick_start" if kind == "pickup" else "deliv_start"
            route += [(kind, i) for i in sorted(items, key=lambda i: (getattr(info[i], key), i))]
            previous = kind
        if previous == "pickup":
            route.append(("warehouse", None))

        first_time, _, first_kind, first_place, _ = ev[0]
        start = first_time - travel(W, first_place) - (2 * H if first_kind == "delivery" else 0)
        plan[(day, v)] = (max(day * MINUTES_PER_DAY, start), route)
    return plan


class TimedRecursiveSolver(RecursiveBacktrackingSolver):
    """O RecursiveBacktrackingSolver da biblioteca com um limite de tempo e um contador de atribuições."""

    def __init__(self, time_limit=None):
        super().__init__(forwardcheck=True)
        self.time_limit, self.assignments, self.status = time_limit, 0, "running"

    def recursiveBacktracking(self, solutions, domains, vconstraints, assignments, single):
        if self.assignments == 0:
            self._start = time.perf_counter()
        self.assignments += 1
        if self.time_limit and time.perf_counter() - self._start > self.time_limit:
            raise TimeoutError
        return super().recursiveBacktracking(solutions, domains, vconstraints, assignments, single)


SOLVERS = {
    "backtracking": lambda limit: CountingBacktrackingSolver(time_limit=limit),
    "recursive": lambda limit: TimedRecursiveSolver(time_limit=limit),
    "min_conflicts": lambda limit: MinConflictsSolver(steps=10_000),
}


def solve_csp(problem, solver="backtracking", time_limit=120, max_solutions=1, alpha=0.0, **options):
    """
    Resolve o CSP dia a dia. O armazém de cada dia (e as horas de chegada, para a R2) vem da
    SIMULAÇÃO do plano do dia anterior.

    solver        = "backtracking" | "recursive" | "min_conflicts", ou uma função limit → solver
    max_solutions = 1 → a primeira atribuição viável (satisfação: as preferências só ORDENAM
                    os valores); K > 1 → percorre até K atribuições e escolhe a de menor
                    custo km + alpha × espera (otimização explícita, mas limitada a K)
    options       = variable_order / value_order (só "backtracking"), windows_as, capacity_as

    Devolve (plan, result, stats); stats["status"] = "ok" | "limit" | "no_solution" | "invalid".
    """
    random.seed(1)                                          # o MinConflicts é aleatório
    factory = SOLVERS.get(solver, solver)
    order = {k: options.pop(k) for k in ("variable_order", "value_order") if k in options}
    plan, stock = {}, {}
    stats = {"status": "ok", "backtracks": {}, "assignments": {}, "variables": {}, "constraints": {},
             "assignment": {}, "groups": {}, "solutions_seen": {}}
    t0 = time.perf_counter()
    for day in problem.days:
        engine = factory(time_limit)
        for k, v in order.items():
            setattr(engine, k, v)
        csp_problem, groups = build_day_csp(problem, day, stock, engine, **options)
        stats["variables"][day] = len(csp_problem._variables)
        stats["constraints"][day] = len(csp_problem._constraints)

        best, best_cost, seen = None, float("inf"), 0
        try:
            if max_solutions == 1:
                candidates = [csp_problem.getSolution()]       # Get ONE solution
            else:
                candidates = csp_problem.getSolutionIter()
            for solution in candidates:
                if solution is None:
                    break
                seen += 1
                day_plan = {**plan, **assignment_to_routes(problem, day, groups, solution)}
                r = simulate_plan(problem, {k: v for k, v in day_plan.items() if k[0] <= day}, require_all=False)
                cost = r["distance_km"] + alpha * r["wait_min"] if r["feasible"] else float("inf")
                if best is None or cost < best_cost:
                    best, best_cost, best_plan, best_r = solution, cost, day_plan, r
                if seen >= max_solutions:
                    break
        except TimeoutError:
            engine.status = "limit"
        stats["backtracks"][day] = getattr(engine, "backtracks", None)
        stats["assignments"][day] = getattr(engine, "assignments", None)
        stats["solutions_seen"][day] = seen
        if best is None:
            status = getattr(engine, "status", "no_solution")
            stats["status"] = f"{'limit' if status == 'limit' else 'no_solution'} (dia {day})"
            break
        if not best_r["feasible"]:
            stats["status"] = f"invalid (dia {day}): {best_r['reason']}"
            break
        stats["assignment"][day], stats["groups"][day] = best, groups
        plan, stock = best_plan, best_r["warehouse_end"]

    known = [b for b in stats["backtracks"].values() if b is not None]
    stats["backtracks_total"] = sum(known) if known else None
    stats["seconds"] = time.perf_counter() - t0
    if stats["status"] != "ok":
        return None, None, stats
    return plan, simulate_plan(problem, plan), stats


# =====================================================================
# Print the solution (como o display_timetable do Notebook 91)
# =====================================================================

def display_assignment(problem, groups, solution):
    """A atribuição de um dia numa tabela pandas: um grupo por linha, por carrinha e hora."""
    def hhmm(m):
        d, r = divmod(int(m), MINUTES_PER_DAY)
        return f"d{d} {r // 60:02d}:{r % 60:02d}"

    names = problem.requests["product_name"].to_dict()
    rows = []
    for g in groups:
        n = g["name"]
        rows.append({
            "grupo": n,
            "carrinha": solution[f"V_{n}"],
            "itens": ", ".join(f"{i} {names[i]}" for i in g["items"]),
            "recolha": hhmm(solution[f"P_{n}"]) if f"P_{n}" in solution else "—",
            "entrega": hhmm(solution[f"D_{n}"]) if f"D_{n}" in solution else "—",
            "peso (kg)": round(g["weight"], 1),
        })
    return (pd.DataFrame(rows)
            .sort_values(["carrinha", "recolha", "entrega"])
            .reset_index(drop=True))
