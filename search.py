"""
FASE 4 — Pesquisa informada: Greedy Search, A* Search e Weighted A* Search.

Os três algoritmos seguem a estrutura do Notebook 02 (Informed Search) e só diferem na
PRIORIDADE com que um nó entra na fila (Russell & Norvig, 2021, cap. 3):
    Greedy:      f(n) = h(n)
    A*:          f(n) = g(n) + h(n)
    Weighted A*: f(n) = g(n) + w · h(n)
Por isso o ciclo (fila de prioridade, visitados, "Generate possible moves") está escrito
uma só vez, em best_first_search.

FORMULAÇÃO (classe DayTask). O plano é feito dia a dia; em cada dia as carrinhas são
planeadas em conjunto (uma pode precisar de um item que a outra descarrega no armazém).
    estado = State(vans, warehouse, to_pick, to_deliver)   (ver model.py)
    ações  = macro-ações da carrinha mais atrasada no relógio (possible_moves)
    g(n)   = km percorridos (+ alpha × minutos de espera, na experiência com soft constraints)

IMPORTANTE — o que o A* garante: as macro-ações (recolher tudo ou só o que já abriu num
produtor, entregar tudo o que vai a bordo para um destino, ordem EDF dentro de uma visita,
armazém só com a carrinha sem entregas a bordo) RESTRINGEM o espaço de rotas. Com uma
heurística admissível e as duas podas (estados repetidos exatos e becos sem saída, ambas
seguras), o A* devolve o custo mínimo de CADA DIA, dado o stock do dia anterior, DENTRO deste
espaço de ações — não o ótimo de todas as rotas possíveis, nem do plano de dois dias.

No fim, o plano é sempre verificado pelo simulate_plan do model.py.
"""
import heapq
import itertools
import time

from model import simulate_plan, MINUTES_PER_DAY, State, VanState


# =====================================================================
# Formulação do problema de UM dia (estado inicial, ações, objetivo)
# =====================================================================

class DayTask:
    """Tudo o que os algoritmos precisam de saber sobre um dia."""

    SOON = 30   # recolha parcial: só os itens que abrem até 30 min depois de a carrinha chegar

    def __init__(self, problem, day, stock, alpha=0.0):
        self.problem = problem
        self.req = problem.request_by_id
        self.day = day
        self.W = problem.warehouse_id
        self.vans = sorted(problem.vehicles)
        self.capacity = {v: problem.vehicles[v].capacity_kg for v in self.vans}
        self.service = problem.rules["stop_service_min"]
        self.handling = problem.rules["warehouse_handling_min"]
        self.max_dwell = problem.rules["max_warehouse_dwell_min"]
        self.alpha = alpha                  # peso da espera no custo (0 = só km)

        day_of = lambda m: int(m // MINUTES_PER_DAY)
        self.to_pick = frozenset(i for i, r in self.req.items() if day_of(r.pick_start) == day)
        self.to_deliver = frozenset(i for i, r in self.req.items() if day_of(r.deliv_start) == day)
        self.stock = dict(stock)            # o que ficou no armazém do dia anterior
        self.start_time = self.choose_start_time()
        # a ligação mais curta que chega a cada sítio (para a H2). É calculada aqui, POR
        # instância: antes era um dicionário global e misturava datasets no mesmo processo.
        sites = list(problem.travel_km)
        self.min_entry = {Z: min(self.km(Y, Z) for Y in sites if Y != Z) for Z in sites}
        # distâncias e tempos de CAMINHO MAIS CURTO: são limites inferiores válidos mesmo que a
        # matriz não respeite a desigualdade triangular (no small falha por 0,01 km)
        self.sp_km = shortest_paths(problem.travel_km)
        self.sp_min = shortest_paths(problem.travel_minutes)

    # --- distâncias, com a diagonal tratada (a matriz não tem pares (x, x))
    def km(self, a, b):
        return 0.0 if a == b else self.problem.travel_km[a][b]

    def minutes(self, a, b):
        return 0.0 if a == b else self.problem.travel_minutes[a][b]

    def weight(self, items):
        return sum(self.req[i].weight_kg for i in items)

    def choose_start_time(self):
        """As carrinhas saem à hora mais tardia que lhes permite chegar à 1.ª paragem quando abre."""
        candidates = [self.req[i].pick_start - self.minutes(self.W, self.req[i].origin_id)
                      for i in self.to_pick]
        candidates += [self.req[j].deliv_start - 2 * self.handling
                       - self.minutes(self.W, self.req[j].destination_id)
                       for j in self.stock if j in self.to_deliver]
        day_start = self.day * MINUTES_PER_DAY
        return max(day_start, min(candidates)) if candidates else day_start

    def initial_state(self):
        vans = tuple(VanState(self.W, self.start_time, frozenset(), frozenset(), False) for _ in self.vans)
        return State(vans, tuple(sorted(self.stock.items())), self.to_pick, self.to_deliver)

    @staticmethod
    def is_goal(state):
        """Objetivo: nada por recolher nem por entregar, e as carrinhas de volta ao armazém."""
        return all(v.finished for v in state.vans) and not state.to_pick and not state.to_deliver

    def serve_visit(self, time, loc, dest, items, stop_type):
        """
        Vai a `dest` e serve `items` numa só visita (o serviço de 10 min conta UMA vez).
        Ordem: entre os itens já abertos, primeiro o que FECHA mais cedo (EDF).
        Devolve (hora no fim, paragens, minutos de espera) ou None se alguma janela já fechou.
        """
        is_pick = stop_type == "pickup"
        opens = lambda i: self.req[i].pick_start if is_pick else self.req[i].deliv_start
        closes = lambda i: self.req[i].pick_end if is_pick else self.req[i].deliv_end
        t = time + self.minutes(loc, dest)
        pending, stops, wait = set(items), [], 0.0
        while pending:
            open_now = [i for i in pending if opens(i) <= t]
            if not open_now:
                t_open = min(opens(i) for i in pending)          # esperar
                wait += t_open - t
                t = t_open
                continue
            i = min(open_now, key=lambda i: (closes(i), i))
            if t > closes(i):
                return None
            stops.append((stop_type, i))
            pending.remove(i)
        return t + self.service, stops, wait

    def dead_end(self, state):
        """
        Poda segura: True se algum item por tratar JÁ não pode cumprir a sua janela.
        Usa limites INFERIORES da hora a que uma carrinha lá pode chegar (a carrinha ativa
        mais próxima, sem desvios, sem esperas), por isso nunca corta um estado com solução.
        Sem esta poda, a pesquisa no dataset small perdia-se em becos sem saída.
        """
        active = [v for v in state.vans if not v.finished]
        if not active:
            return bool(state.to_pick or state.to_deliver)
        m, W = (lambda a, b: self.sp_min[a][b]), self.W
        reach = lambda X: min(v.time + m(v.location, X) for v in active)
        on_board = {j: v for v in active for j in v.carrying_out}
        picked = {i for i, _ in state.warehouse} | {i for v in active for i in v.carrying_in}
        for i in state.to_pick:
            if reach(self.req[i].origin_id) > self.req[i].pick_end:
                return True
        at_w = reach(W)
        for j in state.to_deliver:
            r = self.req[j]
            if j in on_board:
                v = on_board[j]
                t = v.time + m(v.location, r.destination_id)
            elif j in picked:                                    # no armazém ou a caminho dele
                t = at_w + self.handling + m(W, r.destination_id)
            else:                                                # ainda por recolher
                t = (max(reach(r.origin_id), r.pick_start) + self.service + m(r.origin_id, W)
                     + 2 * self.handling + m(W, r.destination_id))
            if t > r.deliv_end:
                return True
        return False


def possible_moves(task, state):
    """
    As ações possíveis no estado (o "Generate possible moves" do Notebook 02).
    Mexe só a carrinha mais atrasada no relógio. Cada ação devolve
    (carrinha, paragens que acrescenta à rota, novo estado, custo da ação).
    O custo da ação é km + alpha × minutos de espera (alpha = 0 → só km).

    Macro-ações: cada ação trata de um GRUPO de itens no mesmo sítio. Com uma ação por item,
    8 itens no mesmo produtor davam 8! ordens diferentes, todas com os mesmos km.
    Os sucessores em que alguma janela já é impossível (task.dead_end) são descartados.
    """
    vans, wh, to_pick, to_deliver = state
    active = [k for k, v in enumerate(vans) if not v.finished]
    if not active:
        return []
    k = min(active, key=lambda k: (vans[k].time, k))
    loc, time, cin, cout, _ = vans[k]
    wh = dict(wh)
    cap = task.capacity[task.vans[k]]
    load_now = task.weight(cin | cout)
    moves = []

    def add(stops, van, km, wait=0.0, warehouse=None, pick=to_pick, deliver=to_deliver):
        v = list(vans)
        v[k] = van
        new = State(tuple(v), tuple(sorted((wh if warehouse is None else warehouse).items())), pick, deliver)
        if not task.dead_end(new):
            moves.append((k, stops, new, km + task.alpha * wait))

    # Move 1 — recolher num produtor: (a) tudo o que falta lá; (b) só o que já abriu.
    #          Não há segunda visita ao produtor onde a carrinha JÁ está: a simulação junta
    #          paragens seguidas no mesmo sítio numa só (um só serviço de 10 min), por isso essa
    #          sequência é exatamente a opção (a) da visita anterior — e contá-la como duas
    #          visitas punha a pesquisa a contar mais 10 min de serviço do que a simulação.
    by_origin = {}
    for i in to_pick:
        by_origin.setdefault(task.req[i].origin_id, set()).add(i)
    for dest, items in by_origin.items():
        if dest == loc:
            continue
        arrival = time + task.minutes(loc, dest)
        options = [frozenset(items)]
        soon = frozenset(i for i in items if task.req[i].pick_start <= arrival + task.SOON)
        if soon and soon != options[0]:
            options.append(soon)
        for batch in options:
            if load_now + task.weight(batch) > cap:
                continue
            served = task.serve_visit(time, loc, dest, batch, "pickup")
            if served is None:
                continue
            t, stops, wait = served
            add(stops, VanState(dest, t, cin | batch, cout, False), task.km(loc, dest), wait,
                pick=to_pick - batch)

    # Move 2 — entregar num destino: tudo o que vai a bordo para lá
    by_dest = {}
    for j in cout:
        by_dest.setdefault(task.req[j].destination_id, set()).add(j)
    for dest, items in by_dest.items():
        served = task.serve_visit(time, loc, dest, items, "delivery")
        if served is None:
            continue
        t, stops, wait = served
        batch = frozenset(items)
        add(stops, VanState(dest, t, cin, cout - batch, False), task.km(loc, dest), wait,
            deliver=to_deliver - batch)

    # Move 3 — ir ao armazém (só com carrying_out vazio): descarrega tudo e carrega um
    #          subconjunto de destinos (uma ação por escolha)
    if not cout:
        arrival = time + task.minutes(loc, task.W)
        unload_end = arrival + task.handling
        warehouse = dict(wh)
        for i in cin:
            warehouse[i] = unload_end
        ready = [j for j in to_deliver if j in warehouse and warehouse[j] <= unload_end]
        fits_warehouse = task.weight(warehouse) <= task.problem.warehouse_capacity
        if fits_warehouse and all(unload_end - warehouse[j] <= task.max_dwell for j in ready):  # 24 h
            groups = {}
            for j in ready:
                groups.setdefault(task.req[j].destination_id, set()).add(j)
            dests = sorted(groups)
            for r in range(len(dests), -1, -1):
                for chosen in itertools.combinations(dests, r):
                    loaded = set().union(*(groups[d] for d in chosen)) if chosen else set()
                    if task.weight(loaded) > cap or (not cin and not loaded):
                        continue
                    rest = {i: t for i, t in warehouse.items() if i not in loaded}
                    add([("warehouse", None)],
                        VanState(task.W, arrival + 2 * task.handling, frozenset(), frozenset(loaded), False),
                        task.km(loc, task.W), warehouse=rest)

        # Move 4 — esperar no armazém pela outra carrinha (que traz itens recolhidos)
        if loc == task.W and not cin:
            others = [vans[o].time for o in active if o != k and vans[o].carrying_in]
            if others and min(others) > time:
                add([], VanState(loc, min(others), cin, cout, False), 0.0, wait=min(others) - time)

    # Move 5 — terminar: voltar ao armazém
    if not cin and not cout:
        add([], VanState(task.W, time + task.minutes(loc, task.W), cin, cout, True), task.km(loc, task.W))
    return moves


def already_visited(visited, state, path_cost):
    """
    O "if state in visited" do Notebook 02: descarta o estado se o MESMO estado (incluindo as
    horas de cada carrinha e de cada item no armazém) já foi expandido com custo ≤.
    Dois estados iguais têm exatamente as mesmas continuações, por isso esta poda nunca corta o
    caminho ótimo; se o mesmo estado voltar com custo MENOR, é expandido outra vez (reabertura).
    (Uma versão anterior também descartava estados "dominados" — iguais mas com horas mais
    tardias. Não era segura: no tiny cortava o ótimo do dia 1 (306,09 km em vez de 300,69),
    porque mudar as horas muda a ordem em que as carrinhas se mexem.)
    """
    if state in visited and visited[state] <= path_cost:
        return True
    visited[state] = path_cost
    return False


# =====================================================================
# Heurísticas (estimativas dos km que AINDA faltam)
# =====================================================================
# Admissibilidade (nunca sobrestimam os km que faltam) — demonstração no notebook, secção 5:
#   H1 usa distâncias de caminho mais curto (sp_km), que respeitam sempre a desigualdade
#      triangular e nunca excedem a matriz; com a matriz direta só seria válida se a própria
#      matriz a respeitasse (falha no small por arredondamentos de 0,01 km).
#   H2 só usa "cada sítio por visitar recebe pelo menos uma viagem", que vale para qualquer matriz.
# Como a espera tem custo ≥ 0, também são admissíveis para o custo km + alpha × espera.

def heuristic01(task, state):
    """
    H1 — "o sítio mais longe que falta".
    Cada carrinha tem de voltar ao armazém; além disso, o sítio por visitar que obriga ao
    MAIOR desvio (ir lá e voltar, em vez de voltar direto). O pior desvio é um limite inferior.
    """
    active = [v for v in state.vans if not v.finished]
    if not active:
        return 0.0
    km, W = (lambda a, b: task.sp_km[a][b]), task.W     # caminho mais curto (ver triangle_inequality_holds)
    base = sum(km(v.location, W) for v in active)

    def detour(X, through_warehouse=False):
        if through_warehouse:     # o item ainda tem de passar pelo armazém antes de ir para X
            return km(W, X) + km(X, W)
        return min(km(v.location, X) + km(X, W) - km(v.location, W) for v in active)

    on_board = set().union(*(v.carrying_out for v in active))
    worst = 0.0
    for i in state.to_pick:
        worst = max(worst, detour(task.req[i].origin_id))
    for j in state.to_deliver:
        worst = max(worst, detour(task.req[j].destination_id, through_warehouse=j not in on_board))
    return base + worst


def heuristic02(task, state):
    """
    H2 — "soma das chegadas mínimas".
    Cada sítio por visitar tem de receber pelo menos UMA viagem, que não pode ser mais curta
    do que a ligação mais curta até ele; soma-se para todos (+ a volta das carrinhas).
    Ao contrário da H1, DESCE sempre que um sítio fica tratado — guia muito melhor o Greedy.
    """
    active = [v for v in state.vans if not v.finished]
    if not active:
        return 0.0
    here = {v.location for v in active}
    places = ({task.req[i].origin_id for i in state.to_pick}
              | {task.req[j].destination_id for j in state.to_deliver})
    total = sum(task.min_entry[X] for X in places if X not in here)
    total += sum(1 for v in active if v.location != task.W) * task.min_entry[task.W]
    return total


HEURISTICS = {"farthest": heuristic01, "entry": heuristic02}


def shortest_paths(matrix):
    """Floyd–Warshall: a menor soma de ligações entre cada par de sítios (0 de um sítio a si próprio)."""
    ids = list(matrix)
    d = {a: {b: (0.0 if a == b else matrix[a][b]) for b in ids} for a in ids}
    for k in ids:
        for a in ids:
            for b in ids:
                if d[a][k] + d[k][b] < d[a][b]:
                    d[a][b] = d[a][k] + d[k][b]
    return d


def triangle_inequality_holds(problem):
    """d(a,c) ≤ d(a,b) + d(b,c) para todos os sítios? (condição da admissibilidade da H1)"""
    d = lambda a, b: 0.0 if a == b else problem.travel_km[a][b]
    ids = list(problem.travel_km)
    return all(d(a, c) <= d(a, b) + d(b, c) + 1e-9 for a in ids for b in ids for c in ids)


# =====================================================================
# Best-first search (Notebook 02) — o ciclo comum aos três algoritmos
# =====================================================================

def best_first_search(task, heuristic, priority, max_iterations=150_000, time_limit=None):
    """
    Performs a best-first search for one day; priority(g, h) gives the value of f(n).
    Returns (routes, cost, nr_iterations, status):
      status = "ok"          → encontrou um estado objetivo (routes = uma lista de paragens por carrinha)
             = "no_solution" → a fila esvaziou: não há plano neste espaço de ações
             = "limit"       → parou por max_iterations ou time_limit (NÃO quer dizer que não haja solução)
    """
    counter = itertools.count()              # desempate no heap (os estados não se comparam)
    start = time.perf_counter()
    # Cada nó guarda só (pai, carrinha, paragens da ação); o caminho é reconstruído no fim.
    # Guardar o caminho inteiro em cada nó esgotava a memória no dataset small.
    nodes = [(None, None, [])]

    def routes_of(node):
        routes = [[] for _ in task.vans]
        while node is not None:
            parent, vehicle, stops = nodes[node]
            if vehicle is not None:
                routes[vehicle][:0] = stops
            node = parent
        return routes

    # Initializes queue and visited states set
    initial_state = task.initial_state()
    h0 = heuristic(task, initial_state)
    priority_queue = [(priority(0.0, h0), h0, next(counter), initial_state, 0, 0.0)]
    visited = {}
    nr_iterations = 0

    while priority_queue:
        _, _, _, current_state, node, path_cost = heapq.heappop(priority_queue)

        if task.is_goal(current_state):
            return routes_of(node), path_cost, nr_iterations, "ok"

        if already_visited(visited, current_state, path_cost):
            continue
        nr_iterations += 1  # counts expanded nodes
        if nr_iterations > max_iterations or (time_limit and time.perf_counter() - start > time_limit):
            return None, float("inf"), nr_iterations, "limit"

        # Generate possible moves
        for vehicle, stops, new_state, move_cost in possible_moves(task, current_state):
            nodes.append((node, vehicle, stops))
            new_cost = path_cost + move_cost
            h = heuristic(task, new_state)
            heapq.heappush(priority_queue, (priority(new_cost, h), h, next(counter), new_state,
                                            len(nodes) - 1, new_cost))

    return None, float("inf"), nr_iterations, "no_solution"


def greedy_search(task, heuristic, **limits):
    """Greedy best-first: f(n) = h(n). Rápido, sem garantia de qualidade."""
    return best_first_search(task, heuristic, lambda g, h: h, **limits)


def a_star_search(task, heuristic, **limits):
    """A*: f(n) = g(n) + h(n). Com h admissível, devolve o custo mínimo no espaço de ações."""
    return best_first_search(task, heuristic, lambda g, h: g + h, **limits)


def weighted_a_star_search(task, heuristic, weight=1.5, **limits):
    """
    Weighted A*: f(n) = g(n) + w · h(n), w > 1. Expande menos nós; com h admissível (e a
    reabertura de estados de best_first_search) o custo do dia fica, no máximo, w × o mínimo do
    mesmo espaço de ações (Pohl, 1970).
    """
    return best_first_search(task, heuristic, lambda g, h: g + weight * h, **limits)


ALGORITHMS = {"greedy": greedy_search, "astar": a_star_search, "weighted_astar": weighted_a_star_search}


# =====================================================================
# Do plano por dias ao plano completo
# =====================================================================

def solve(problem, algorithm="astar", heuristic="farthest", alpha=0.0, **kwargs):
    """
    Constrói um plano completo, dia a dia, com o algoritmo escolhido.
    O armazém de cada dia vem da SIMULAÇÃO do dia anterior (horas reais de chegada).

    algorithm: "greedy" | "astar" | "weighted_astar"   (kwargs: weight, max_iterations, time_limit)
    heuristic: "farthest" (H1) | "entry" (H2)
    alpha:     custo de 1 minuto de espera, em km (0 = só km; ver soft constraints)

    Devolve (plan, result, stats):
      plan   = {(dia, carrinha): (start_time, rota)}, ou None se algum dia falhou
      result = simulate_plan(problem, plan), ou None
      stats  = {"status": "ok" | "limit" | "no_solution" | "invalid", "expanded": {dia: nós},
                "cost": {dia: custo da pesquisa}, "expanded_total": ..., "seconds": ...}
    """
    search = ALGORITHMS[algorithm]
    h = HEURISTICS[heuristic]
    start = time.perf_counter()
    plan, stock = {}, {}
    stats = {"status": "ok", "expanded": {}, "cost": {}}
    for day in problem.days:
        task = DayTask(problem, day, stock, alpha)
        routes, cost, nr_iterations, status = search(task, h, **kwargs)
        stats["expanded"][day], stats["cost"][day] = nr_iterations, cost
        if status != "ok":
            stats["status"] = f"{status} (dia {day})"
            break
        for k, route in enumerate(routes):
            if route:
                plan[(day, task.vans[k])] = (task.start_time, route)
        partial = simulate_plan(problem, {key: plan[key] for key in plan if key[0] <= day}, require_all=False)
        if not partial["feasible"]:          # não devia acontecer: a pesquisa usa as mesmas regras
            stats["status"] = f"invalid (dia {day}): {partial['reason']}"
            break
        stock = partial["warehouse_end"]

    stats["expanded_total"] = sum(stats["expanded"].values())
    stats["seconds"] = time.perf_counter() - start
    if stats["status"] != "ok":
        return None, None, stats
    return plan, simulate_plan(problem, plan), stats


# =====================================================================
# Print (como o print_solution do Notebook 02)
# =====================================================================

def hhmm(minutes):
    """1890 → 'd1 07:30'"""
    d, r = divmod(int(round(minutes)), MINUTES_PER_DAY)
    return f"d{d} {r // 60:02d}:{r % 60:02d}"


def print_plan(problem, plan, result=None):
    """Prints the plan: for each vehicle and day, each stop with time, item, place and load."""
    if result is None:
        result = simulate_plan(problem, plan)
    if not result["feasible"]:
        print("INVIÁVEL:", result["reason"])
        return
    print(f"km: {result['distance_km']:.2f}  |  duração: {result['duration_min']:.0f} min  |  "
          f"espera: {result['wait_min']:.0f} min")
    for key in sorted(plan):
        start, _ = plan[key]
        r = result["routes"][key]
        print(f"\nDia {key[0]}, carrinha {key[1]}: sai {hhmm(start)}, volta {hhmm(r['end_time'])}, "
              f"{r['distance_km']:.1f} km")
        for t in r["timeline"]:
            item = "" if t["item"] is None else f"item {t['item']}"
            print(f"  {hhmm(t['start'])}  {t['stop']:<9} {item:<8} → local {t['destination']}  "
                  f"(carga {t['load']:.1f} kg)")
