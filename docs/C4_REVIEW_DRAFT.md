# Черновик архитектурных схем TutorFlow AI (не окончательные 7 C4 преподавателя)

Схемы фиксируют только компоненты, обнаруженные в исправленной копии. Полноценную C4 нотацию, развёртывание и семь отдельных диаграмм нужно согласовать с ТЗ и реальной средой; здесь не утверждается, что требования диаграмм уже выполнены.

## System Context

```mermaid
flowchart LR
    teacher[Репетитор / ученик] --> api[TutorFlow AI]
    api --> ollama[Локальный Ollama]
    api --> db[(SQLite)]
```

## Container (фактически процессы)

```mermaid
flowchart TD
    client[HTTP клиент] --> fastapi[FastAPI]
    fastapi --> graph[LangGraph runtime]
    graph --> worker[Дочерний Python/SymPy worker]
    graph --> models[Ollama HTTP API]
    graph --> sqlite[(SQLite)]
    graph --> logs[JSONL локальные события]
```

## Component: Coordinator & Routes

```mermaid
flowchart TD
  input[Запрос + состояние] --> coordinator[Coordinator]
  coordinator --> profile[Student profile]
  profile --> gate{Тип задачи}
  gate -->|solve_check| checker[Solution checker]
  gate -->|solve_math| solver[Solver agent]
  gate -->|pedagogical/personalize/multi_step/clarify| generator[Generator]
  gate -->|safety| safety[Safety]
```

## Component: Verification & Persistence

```mermaid
flowchart LR
  checker[Checker] --> validator[Validator]
  solver[Solver] --> validator
  generator[Generator] --> math[Symbolic check of supported expression]
  math --> validator
  validator --> save[Save DB]
  safety[Safety] --> save
```

## Memory Layer (actual vs proposed)

```mermaid
flowchart TB
  state[AgentState / Short-term] --> sessions[(Session messages)]
  state --> profile[(Student profiles)]
  sessions --> state
  profile --> state
  semantic[Long-term semantic index — NOT IMPLEMENTED]
```

## Deployment (development only)

```mermaid
flowchart TB
  host[Windows developer workstation]
  subgraph app[Python process]
    api[FastAPI / LangGraph]
  end
  host --> app
  app --> worker[Python worker subprocess]
  app --> ollama[Local Ollama]
  app --> db[(SQLite)]
```

## Dynamic: Check Solution

```mermaid
sequenceDiagram
  participant U as User
  participant A as API
  participant G as Coordinator
  participant S as SymPy worker
  participant V as Validator
  participant D as SQLite
  U->>A: check-solution(prompt)
  A->>G: state(user_prompt)
  G->>S: symbolic solution
  S-->>G: roots/status
  G->>V: verdict + source details
  V-->>G: feedback + validation_status
  G->>D: store if session authorized internally
  G-->>A: result
  A-->>U: feedback
```

**Ограничения текущего состояния:** Нет полноценного контроля доступа к профилям, нет VM/MicroVM, нет готового графического пакета из 7 согласованных C4-диаграмм, нет SLA для событий наблюдаемости и нет автоматической L3 памяти. Диаграмма динамики показывает концептуальный flow, а не результаты реального запущенного сервиса.
