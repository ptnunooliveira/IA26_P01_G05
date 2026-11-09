# IA26_P01_G05 — Agente de rotas para um mercado de produtores locais

Projeto 01 de Inteligência Artificial (LESI, 2026/27). Grupo G05: Marina Silva (31029), Nelson Cruz (31103), Nuno Oliveira (31550), Ricardo Pinto (14865).

Recolha nos produtores → centro de recolha (armazém) → entrega nos locais escolhidos pelos clientes, com janelas de tempo, capacidade, regra das 24 h no armazém e trocas entre carrinhas. Resolvido com Greedy, A\*, Weighted A\* (2 heurísticas), Hill Climbing e CSP (`python-constraint`).

## Estrutura

```
data_tiny/  data_small/      datasets (12 CSVs cada)
01_eda_tiny.ipynb            análise exploratória (DATA_PATH configurável)
IA_P01_G05.ipynb             notebook principal: modelação, algoritmos, experiências, análise
model.py                     leitura + validação dos CSVs, Problem/Request/Vehicle/State, simulação (validador) e custo
search.py                    Greedy, A*, Weighted A* (um ciclo best-first), heurísticas H1 e H2
local_search.py              Hill Climbing (vizinhança sobre blocos, steepest)
csp_solver.py                CSP: variáveis, domínios, restrições R1–R5, solver com contagem de backtracks
tests/                       testes automatizados (pytest)
results/                     resultados guardados pelo notebook (cache)
requirements.txt  readme.txt
```

## Instalação

```bash
python -m pip install -r requirements.txt
```

Testado com Python 3.13 (o código só usa funcionalidades de Python ≥ 3.11). O CSP usa **apenas** `python-constraint` 1.4.0; o Google OR-Tools não é usado.

## Como correr

```bash
python -m pytest tests          # 55 testes (modelo, heurísticas, algoritmos, CSP, datasets diferentes)
jupyter notebook                # abrir 01_eda_tiny.ipynb e IA_P01_G05.ipynb e correr todas as células
```

No notebook principal, `RUN_FRESH = False` reutiliza os resultados de `results/`; cada resultado escreve `[cache]` ou `[calculado]`, para se saber sempre de onde veio. Para recalcular tudo: `RUN_FRESH = True` (ou apagar `results/`).

## Usar outro dataset (ex.: na defesa)

1. Copiar a pasta do dataset (com os 12 CSVs) para junto dos ficheiros, com um nome `data_<nome>`.
2. No notebook principal, mudar `DATA_PATH = "data_<nome>"` e correr tudo. A secção 11 corre automaticamente todos os `data_*` encontrados.
3. Em Python, diretamente:

```python
from model import Problem
from search import solve
from local_search import hill_climbing
from csp_solver import solve_csp

problem = Problem.from_csv("data_<nome>")       # valida; ValueError com a lista de problemas se for inválido
plan, result, stats = solve(problem, "greedy", "entry")
print(stats["status"], result and result["distance_km"])
better, km, history, hc = hill_climbing(problem, plan)
```

Nada no código depende dos nomes ou das quantidades de um dataset: veículos, sítios, dias, capacidades, regras e janelas vêm dos CSVs. Os algoritmos exatos têm limites (`max_iterations`, `time_limit`) e devolvem um estado — `ok`, `limit` (parou no limite: **não** quer dizer que não haja solução) ou `no_solution` — em vez de bloquear ou de devolver planos parciais.

## Decisões principais

- **Validador único**: `simulate_plan` aplica todas as hard constraints e é usado por todos os algoritmos; um plano inviável custa ∞.
- **Pesquisa dia a dia**, com as carrinhas planeadas em conjunto e macro-ações; o A\* dá o custo mínimo **de cada dia, dado o stock do dia anterior, dentro deste espaço de ações** (não o ótimo de todas as rotas nem do plano de dois dias).
- **Heurísticas admissíveis para qualquer dataset**: H1 usa distâncias de caminho mais curto (a matriz do *small* não cumpre a desigualdade triangular por 0,01 km); H2 não depende dessa propriedade.
- **Soft constraint**: tempo de espera, com custo `km + α · espera` (α em km/min); o custo que a pesquisa otimiza é igual ao custo final simulado.
