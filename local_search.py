"""
FASE 5 — Pesquisa local: Hill Climbing a partir da solução do Greedy.

O algoritmo segue a estrutura do Notebook 03 (Local Search, 8 rainhas):
    - current_board / current_cost  →  current_plan / current_cost
    - gera TODOS os vizinhos, avalia cada um com evaluate() e guarda o melhor;
    - se o melhor vizinho não é melhor do que o atual → pára (ótimo local);
    - senão, passa para o melhor vizinho e repete.

Diferenças em relação às 8 rainhas:
    - a solução inicial é o plano do Greedy (como pede o enunciado), e não uma aleatória;
    - evaluate() devolve os km do plano (via simulate_plan); um plano inviável vale infinito,
      por isso nunca é escolhido (nas rainhas, evaluate() contava os conflitos);
    - procura-se o MÍNIMO de km (nas rainhas, o mínimo de conflitos).

VIZINHANÇA — uma rota é partida em BLOCOS: paragens seguidas do mesmo tipo no mesmo sítio
(ex.: recolher os itens 22, 29 e 35 no produtor 3), e cada passagem pelo armazém é um bloco.
    1. relocate: mudar um bloco de posição na mesma rota
    2. swap:     trocar dois blocos da mesma rota
    3. transfer: passar um bloco para a rota da outra carrinha, no mesmo dia
    4. split:    mudar UM item de sítio (em qualquer rota do dia)  — só na vizinhança "all"
Trocar dois itens recolhidos na mesma visita não muda nada; mexer em blocos evita esses vizinhos inúteis.
"""
import time
import tracemalloc

from model import simulate_plan, plan_cost


# =====================================================================
# Common functions definition (como o evaluate das 8 rainhas)
# =====================================================================

def evaluate(problem, plan, metric="distance_km", alpha=0.0):
    """Calculates the cost of a plan (km [+ alpha × espera]). An infeasible plan costs infinity."""
    return plan_cost(simulate_plan(problem, plan), metric, alpha)


def blocks_of(problem, route):
    """Parte uma rota em blocos (listas de paragens)."""
    blocks = []
    for stop in route:
        stop_type, item = stop
        if stop_type == "warehouse":
            place = problem.warehouse_id
        elif stop_type == "pickup":
            place = problem.request_by_id[item].origin_id
        else:
            place = problem.request_by_id[item].destination_id
        if (blocks and stop_type != "warehouse" and blocks[-1]["type"] == stop_type
                and blocks[-1]["place"] == place):
            blocks[-1]["stops"].append(stop)
        else:
            blocks.append({"type": stop_type, "place": place, "stops": [stop]})
    return [b["stops"] for b in blocks]


def flatten(blocks):
    return [stop for block in blocks for stop in block]


def _signature(plan):
    return tuple(sorted((key, start, tuple(route)) for key, (start, route) in plan.items()))


def generate_neighbors(problem, plan, neighborhood="blocks"):
    """
    Generate all neighbors of the plan (como o ciclo que move uma rainha na sua linha).
    neighborhood = "blocks" → relocate, swap e transfer de blocos
                 = "all"    → também split (mover um só item)
    """
    neighbors, seen = [], {_signature(plan)}

    def add(neighbor):
        """Guarda o vizinho só se for novo (mover um bloco para o lado = trocá-lo com o vizinho)."""
        sig = _signature(neighbor)
        if sig not in seen:
            seen.add(sig)
            neighbors.append(neighbor)

    keys = sorted(plan)
    for key in keys:
        start, route = plan[key]
        blocks = blocks_of(problem, route)
        n = len(blocks)
        # 1. relocate e 2. swap — na mesma rota
        for i in range(n):
            for j in range(n):
                if i != j:
                    new_blocks = list(blocks)
                    block = new_blocks.pop(i)
                    new_blocks.insert(j, block)
                    neighbor = dict(plan)
                    neighbor[key] = (start, flatten(new_blocks))
                    add(neighbor)
                if i < j:
                    new_blocks = list(blocks)
                    new_blocks[i], new_blocks[j] = new_blocks[j], new_blocks[i]
                    neighbor = dict(plan)
                    neighbor[key] = (start, flatten(new_blocks))
                    add(neighbor)
        # 3. transfer — para a outra carrinha, no mesmo dia
        for other in keys:
            if other != key and other[0] == key[0]:
                other_blocks = blocks_of(problem, plan[other][1])
                for i in range(n):
                    for j in range(len(other_blocks) + 1):
                        src = list(blocks)
                        dst = list(other_blocks)
                        dst.insert(j, src.pop(i))
                        neighbor = dict(plan)
                        neighbor[key] = (start, flatten(src))
                        neighbor[other] = (plan[other][0], flatten(dst))
                        add(neighbor)

    # 4. split — um item sozinho para outra posição (de qualquer rota do mesmo dia)
    if neighborhood == "all":
        for key in keys:
            for pos, stop in enumerate(plan[key][1]):
                if stop[0] == "warehouse":
                    continue
                for other in keys:
                    if other[0] != key[0]:
                        continue
                    for j in range(len(plan[other][1]) + 1):
                        src = list(plan[key][1])
                        moved = src.pop(pos)
                        neighbor = dict(plan)
                        if other == key:
                            src.insert(min(j, len(src)), moved)
                            neighbor[key] = (plan[key][0], src)
                        else:
                            dst = list(plan[other][1])
                            dst.insert(j, moved)
                            neighbor[key] = (plan[key][0], src)
                            neighbor[other] = (plan[other][0], dst)
                        add(neighbor)
    return neighbors


# =====================================================================
# Hill-climbing search (Notebook 03)
# =====================================================================

def hill_climbing(problem, initial_plan, neighborhood="blocks", metric="distance_km", alpha=0.0,
                  max_iterations=500, time_limit=None):
    """
    Performs the hill climbing algorithm, starting from initial_plan (the Greedy plan).
    Returns (current_plan, current_cost, history, stats):
      history = the cost after each iteration (point 0 = initial plan) → the
                "cost vs. iteration" curve asked in the project guide.
      stats["status"] = "local_optimum" (nenhum vizinho é melhor) | "limit" (parou por
                max_iterations ou time_limit: o plano devolvido é válido, mas não é ótimo local)
    """
    t0 = time.perf_counter()
    current_plan = dict(initial_plan)
    current_cost = evaluate(problem, current_plan, metric, alpha)
    history = [current_cost]
    evaluations = 0
    status = "local_optimum"

    while True:
        # Generate all neighbors
        neighbors = generate_neighbors(problem, current_plan, neighborhood)

        best_neighbor = None
        best_neighbor_cost = float("inf")

        for neighbor in neighbors:
            neighbor_cost = evaluate(problem, neighbor, metric, alpha)
            evaluations += 1
            if neighbor_cost < best_neighbor_cost:
                best_neighbor_cost = neighbor_cost
                best_neighbor = neighbor

        if best_neighbor is None or best_neighbor_cost >= current_cost:
            break                                                      # local optimum

        current_plan = best_neighbor
        current_cost = best_neighbor_cost
        history.append(current_cost)
        if len(history) - 1 >= max_iterations or (time_limit and time.perf_counter() - t0 > time_limit):
            status = "limit"
            break

    stats = {"status": status, "iterations": len(history) - 1, "evaluations": evaluations,
             "seconds": time.perf_counter() - t0}
    return current_plan, current_cost, history, stats


# =====================================================================
# Medir tempo e memória (como o measure() do Notebook 03)
# =====================================================================

def measure(name, function):
    """
    Corre function() uma vez para medir o TEMPO (sem tracemalloc, que o atrasa) e outra vez
    para medir o PICO DE MEMÓRIA. Devolve (resultado, segundos, KB).
    """
    start = time.perf_counter()
    result = function()
    elapsed = time.perf_counter() - start

    tracemalloc.start()
    baseline, _ = tracemalloc.get_traced_memory()
    function()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_kb = (peak - baseline) / 1024

    print(f"{name}: Time: {elapsed:.2f} s, Peak memory: {peak_kb:.0f} KB")
    return result, elapsed, peak_kb


def plot_history(histories, title="Hill Climbing — custo vs. iteração", ylabel="km"):
    """histories = {"nome": [custos]} → um gráfico com uma linha por execução."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    for name, h in histories.items():
        plt.plot(range(len(h)), h, marker="o", markersize=3, label=name)
    plt.xlabel("iteração")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
