# LangGraph 기반 에이전트 하네스 프로덕션 구현 설계서 및 자율 E2E 테스트 프레임워크

> 5대 아키텍처 패턴 · 즉시 구현 가능한 코드 스켈레톤 · 자율 품질 개선 루프
>
> 버전 1.0 | 2026-03-28

---

## 목차

- [Part I. 아키텍처 설계 철학 및 공통 기반](#part-i-아키텍처-설계-철학-및-공통-기반)
  - [1.1. 하네스의 역할과 설계 원칙](#11-하네스의-역할과-설계-원칙)
  - [1.2. 공통 기반 코드: 프로젝트 스캐폴딩](#12-공통-기반-코드-프로젝트-스캐폴딩)
  - [1.3. 체크포인터 팩토리](#13-체크포인터-팩토리)
- [Part II. 5대 아키텍처 구현 설계](#part-ii-5대-아키텍처-구현-설계)
  - [2.1. 선형 파이프라인 + 가드레일 아키텍처](#21-선형-파이프라인--가드레일-아키텍처)
  - [2.2. 동적 오케스트레이터-워커 아키텍처](#22-동적-오케스트레이터-워커-아키텍처)
  - [2.3. 계층적 다중 에이전트 감독관 아키텍처](#23-계층적-다중-에이전트-감독관-아키텍처)
  - [2.4. LATS 병렬 탐색 + 자가 교정 아키텍처](#24-lats-병렬-탐색--자가-교정-아키텍처)
  - [2.5. 대규모 병렬 딥 리서치 아키텍처](#25-대규모-병렬-딥-리서치-아키텍처)
  - [2.6. 아키텍처 선택 가이드](#26-아키텍처-선택-가이드)
- [Part III. 무결성 보장 파이프라인 (CoVe + Sanitizer)](#part-iii-무결성-보장-파이프라인-cove--sanitizer)
  - [3.1. 전체 파이프라인 그래프 정의](#31-전체-파이프라인-그래프-정의)
  - [3.2. Deterministic Sanitizer 구현](#32-deterministic-sanitizer-구현)
- [Part IV. 자율 E2E 테스트 프레임워크 및 자가 개선 루프](#part-iv-자율-e2e-테스트-프레임워크-및-자가-개선-루프)
  - [4.1. 평가 하네스 아키텍처](#41-평가-하네스-아키텍처)
  - [4.2. 오프라인 평가: 회귀 방지 테스트 스위트](#42-오프라인-평가-회귀-방지-테스트-스위트)
  - [4.3. 온라인 평가: 프로덕션 모니터링](#43-온라인-평가-프로덕션-모니터링)
  - [4.4. 자율 자가 개선 루프 (Self-Improvement Loop)](#44-자율-자가-개선-루프-self-improvement-loop)
  - [4.5. CI/CD 통합 및 스케줄러](#45-cicd-통합-및-스케줄러)
  - [4.6. 아키텍처별 평가 메트릭 매트릭스](#46-아키텍처별-평가-메트릭-매트릭스)
- [Part V. 프로덕션 배포 체크리스트](#part-v-프로덕션-배포-체크리스트)

---

## Part I. 아키텍처 설계 철학 및 공통 기반

### 1.1. 하네스의 역할과 설계 원칙

Agent Harness는 LLM의 확률론적 출력을 결정론적 시스템 내에서 통제하는 인프라 래퍼이다. 에이전트의 추론 능력이 시스템의 "두뇌"라면, 하네스는 이 두뇌가 외부 세계와 안전하게 상호작용하도록 돕는 "신경계 및 신체"에 해당한다. 모든 아키텍처는 다음 원칙을 공유한다.

| 원칙 | 설명 | 구현 메커니즘 |
|------|------|-------------|
| 결정론적 제어 | 모든 상태 전이는 하드코딩된 규칙으로 제어 | `conditional_edge` + 가드레일 노드 |
| 상태 영속성 | 매 스텝마다 체크포인트 저장 | `PostgresSaver` / `AsyncSqliteSaver` |
| 보안 격리 | 에이전트별 최소 권한 도구 세트 | `ToolNode` 권한 분리 |
| 관측성 | 모든 궤적을 추적하고 평가 가능 | LangSmith tracing + custom evaluator |
| 자기 교정 | 실패 궤적을 데이터셋으로 피드백 | Reflexion memory + E2E eval loop |

### 1.2. 공통 기반 코드: 프로젝트 스캐폴딩

모든 아키텍처가 공유하는 기반 프로젝트 구조와 의존성을 아래에 정의한다.

**프로젝트 구조:**

```
langgraph-harness/
├── pyproject.toml
├── src/
│   ├── harness/
│   │   ├── __init__.py
│   │   ├── base.py              # 공통 상태 스키마, 설정
│   │   ├── guardrails.py        # 입출력 가드레일
│   │   ├── sanitizer.py         # 결정론적 검증기
│   │   ├── checkpointer.py      # 체크포인터 팩토리
│   │   ├── online_evaluator.py  # 프로덕션 모니터링
│   │   └── self_improvement.py  # 자가 개선 루프
│   └── architectures/
│       ├── linear_pipeline.py
│       ├── orchestrator_worker.py
│       ├── multi_agent_supervisor.py
│       ├── lats_search.py
│       └── deep_research.py
├── tests/
│   ├── conftest.py
│   ├── test_trajectory.py
│   └── datasets/
│       ├── happy_path.json
│       ├── edge_cases.json
│       └── adversarial.json
└── .github/
    └── workflows/
        └── harness-eval.yml
```

**의존성 정의:**

```toml
# pyproject.toml
[project]
name = "langgraph-harness"
version = "1.0.0"
requires-python = ">=3.11"
dependencies = [
    "langgraph>=0.4.0",
    "langchain-core>=0.3.0",
    "langchain-openai>=0.3.0",
    "langgraph-checkpoint-postgres>=2.0.0",
    "langsmith>=0.3.0",
    "pydantic>=2.0",
    "asyncpg>=0.30.0",
]

[project.optional-dependencies]
test = ["pytest>=8.0", "pytest-asyncio>=0.24", "deepeval>=1.0"]
dev  = ["ruff>=0.8.0", "mypy>=1.13"]
```

**공통 상태 스키마 및 설정:**

```python
# src/harness/base.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage

class HarnessState(TypedDict):
    """모든 아키텍처가 공유하는 기본 상태 스키마"""
    messages: Annotated[list[BaseMessage], operator.add]
    plan: list[dict[str, Any]]
    artifacts: Annotated[list[str], operator.add]
    metadata: dict[str, Any]
    error_log: Annotated[list[str], operator.add]
    iteration_count: int


@dataclass
class HarnessConfig:
    """하네스 런타임 설정"""
    max_iterations: int = 10
    max_tokens_budget: int = 100_000
    checkpoint_backend: str = "postgres"   # postgres | sqlite | memory
    tracing_enabled: bool = True
    human_in_the_loop: bool = False
    guardrail_strict: bool = True
```

### 1.3. 체크포인터 팩토리

```python
# src/harness/checkpointer.py
from langgraph.checkpoint.memory import MemorySaver

async def get_checkpointer(config: HarnessConfig):
    """설정에 따른 체크포인터 인스턴스 생성"""
    if config.checkpoint_backend == "postgres":
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        return AsyncPostgresSaver.from_conn_string(
            "postgresql://user:pass@localhost:5432/harness_db"
        )
    elif config.checkpoint_backend == "sqlite":
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        return AsyncSqliteSaver.from_conn_string("harness.db")
    else:
        return MemorySaver()
```

---

## Part II. 5대 아키텍처 구현 설계

### 2.1. 선형 파이프라인 + 가드레일 아키텍처

**아키텍처 개요:**

| 항목 | 세부사항 |
|------|---------|
| 패러다임 | 선형 상태 전이 + 단순 Tool loop |
| 노드 구성 | `Input Guard → Router → LLM → Tool Loop → Output Guard` |
| 적합 도메인 | 고객지원, IT 헬프데스크, SOP 기반 안내봇 |
| 비용 특성 | 예측 가능한 토큰 소모 |
| 위험 요소 | 복잡한 다단계 작업에는 부적합 |

**그래프 흐름:**

```
User Input
    │
    ▼
┌─────────────────┐
│  Input Guardrail │  ← PII 감지, 프롬프트 인젝션 방어, 토큰 길이 제어
└────────┬────────┘
         │ (통과 시)
         ▼
┌─────────────────┐     ┌──────────────┐
│   LLM Inference │◄───►│ Tool Executor│  ← 단기 Tool loop (max_iterations 제한)
└────────┬────────┘     └──────────────┘
         │
         ▼
┌──────────────────┐
│ Output Guardrail │  ← 유해성 필터, 기업 어조 준수
└────────┬─────────┘
         │
         ▼
   State Memory (Checkpointer)
```

**상태 스키마 및 그래프 정의:**

```python
# src/architectures/linear_pipeline.py
from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from harness.base import HarnessState, HarnessConfig


class LinearPipelineState(HarnessState):
    """선형 파이프라인 전용 상태"""
    guardrail_passed: bool
    tool_call_count: int


def build_linear_pipeline(
    model,
    tools: list,
    config: HarnessConfig,
) -> StateGraph:
    """가드레일이 포함된 선형 파이프라인 그래프 생성"""

    input_guard = InputGuardrail(strict=config.guardrail_strict)
    output_guard = OutputGuardrail(strict=config.guardrail_strict)
    model_with_tools = model.bind_tools(tools)

    # ── 노드 정의 ──

    def input_guardrail_node(state: LinearPipelineState):
        last_msg = state["messages"][-1]
        result = input_guard.check(last_msg.content)
        if not result.passed:
            return {
                "guardrail_passed": False,
                "messages": [AIMessage(content=result.rejection_message)],
                "error_log": [f"INPUT_BLOCKED: {result.reason}"],
            }
        return {
            "guardrail_passed": True,
            "messages": [HumanMessage(content=result.sanitized_content)],
        }

    def llm_node(state: LinearPipelineState):
        response = model_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def output_guardrail_node(state: LinearPipelineState):
        last_msg = state["messages"][-1]
        result = output_guard.check(last_msg.content)
        if not result.passed:
            return {
                "messages": [AIMessage(content=result.sanitized_content)],
                "error_log": [f"OUTPUT_FILTERED: {result.reason}"],
            }
        return {"messages": [last_msg]}

    # ── 라우팅 로직 ──

    def should_continue(state: LinearPipelineState) -> Literal["tools", "output_guard"]:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            if state.get("tool_call_count", 0) >= config.max_iterations:
                return "output_guard"   # 무한루프 방지: 강제 종료
            return "tools"
        return "output_guard"

    def route_after_guard(state: LinearPipelineState) -> Literal["llm", "__end__"]:
        if not state.get("guardrail_passed", True):
            return "__end__"
        return "llm"

    # ── 그래프 조립 ──

    graph = StateGraph(LinearPipelineState)
    graph.add_node("input_guard", input_guardrail_node)
    graph.add_node("llm", llm_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("output_guard", output_guardrail_node)

    graph.set_entry_point("input_guard")
    graph.add_conditional_edges("input_guard", route_after_guard)
    graph.add_conditional_edges("llm", should_continue)
    graph.add_edge("tools", "llm")
    graph.add_edge("output_guard", END)

    return graph
```

**컴파일 및 실행:**

```python
# 사용 예시
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [weather_tool, calculator_tool]
config = HarnessConfig(max_iterations=5, guardrail_strict=True)

graph = build_linear_pipeline(model, tools, config)
compiled = graph.compile(checkpointer=await get_checkpointer(config))

result = await compiled.ainvoke(
    {"messages": [HumanMessage(content="서울 날씨 알려줘")]},
    config={"configurable": {"thread_id": "session-001"}},
)
```

---

### 2.2. 동적 오케스트레이터-워커 아키텍처

**아키텍처 개요:**

| 항목 | 세부사항 |
|------|---------|
| 패러다임 | Plan-and-Execute: 계획 후 일괄 실행 |
| 노드 구성 | `Planner → [HITL 승인] → Send API Dispatch → Worker×N → Synthesizer` |
| 적합 도메인 | Text-to-SQL, 코드 리팩토링, 구조화 보고서 파이프라인 |
| 핵심 가치 | ReAct 인지 루프 차단, 피드백 격리를 통한 보안/토큰 최적화 |

**그래프 흐름:**

```
         ┌─────────────┐
         │ Planner Node │  ← Pydantic 스키마 기반 하위 작업 생성
         └──────┬──────┘
                │
    ┌───────────┴───────────┐
    │ interrupt_before       │  ← Human-in-the-loop 승인 게이트
    └───────────┬───────────┘
                │
         ┌──────┴──────┐
         │ Send API    │  ← 동적 서브그래프 스폰
         └──┬───┬───┬──┘
            │   │   │
            ▼   ▼   ▼
         ┌───┐┌───┐┌───┐
         │W-1││W-2││W-N│  ← 각 워커: 독립 도구 실행
         └─┬─┘└─┬─┘└─┬─┘
           │    │    │
           ▼    ▼    ▼
    ┌──────────────────────┐
    │ Shared State (reduce)│  ← operator.add reducer로 안전 병합
    └──────────┬───────────┘
               │
         ┌─────┴─────┐
         │ Synthesizer│  ← 종합 결론 도출
         └───────────┘
```

**상태 스키마 및 그래프 정의:**

```python
# src/architectures/orchestrator_worker.py
from typing import Annotated
import operator
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END, Send
from harness.base import HarnessState, HarnessConfig


# ── Pydantic 계획 스키마 ──

class SubTask(BaseModel):
    id: str = Field(description="고유 작업 식별자")
    description: str = Field(description="작업 설명")
    tool_name: str = Field(description="사용할 도구 이름")
    dependencies: list[str] = Field(default_factory=list, description="선행 작업 ID 목록")

class ExecutionPlan(BaseModel):
    goal: str = Field(description="최종 목표")
    sub_tasks: list[SubTask] = Field(description="하위 작업 목록")


# ── 상태 정의 ──

class OrchestratorState(HarnessState):
    execution_plan: ExecutionPlan | None
    worker_results: Annotated[list[dict], operator.add]   # reducer로 병합
    current_phase: str   # "planning" | "executing" | "synthesizing"


def build_orchestrator_worker(
    planner_model,
    worker_model,
    tools: dict[str, list],   # tool_name -> tool list 매핑
    config: HarnessConfig,
) -> StateGraph:

    planner_with_structure = planner_model.with_structured_output(ExecutionPlan)

    def planner_node(state: OrchestratorState):
        user_request = state["messages"][-1].content
        plan = planner_with_structure.invoke(
            f"다음 요청을 독립적인 하위 작업으로 분할하라:\n{user_request}"
        )
        return {
            "execution_plan": plan,
            "plan": [t.model_dump() for t in plan.sub_tasks],
            "current_phase": "executing",
        }

    def dispatch_workers(state: OrchestratorState):
        """Send API를 통한 동적 워커 스폰"""
        plan = state["execution_plan"]
        return [
            Send("worker_executor", {
                "task": task,
                "tools": tools.get(task.tool_name, []),
                "messages": state["messages"],
            })
            for task in plan.sub_tasks
        ]

    def worker_executor(state: dict):
        """개별 워커: 할당된 도구로 작업 수행"""
        task = state["task"]
        worker_tools = state["tools"]
        model_with_tools = worker_model.bind_tools(worker_tools)
        result = model_with_tools.invoke(
            f"작업: {task.description}\n도구: {task.tool_name}"
        )
        return {"worker_results": [{"task_id": task.id, "result": result.content}]}

    def synthesizer_node(state: OrchestratorState):
        results = state["worker_results"]
        summary_prompt = "다음 하위 작업 결과들을 종합하라:\n"
        for r in results:
            summary_prompt += f"- [{r['task_id']}]: {r['result']}\n"
        final = planner_model.invoke(summary_prompt)
        return {
            "messages": [AIMessage(content=final.content)],
            "current_phase": "done",
        }

    # ── 그래프 조립 ──

    graph = StateGraph(OrchestratorState)
    graph.add_node("planner", planner_node)
    graph.add_node("worker_executor", worker_executor)
    graph.add_node("synthesizer", synthesizer_node)

    graph.set_entry_point("planner")
    graph.add_conditional_edges("planner", dispatch_workers)
    graph.add_edge("worker_executor", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph
```

> **HITL 제어:** `interrupt_before=["worker_executor"]`를 컴파일 시 선언하면 플래너 결과를 인간이 검토한 후 실행을 재개할 수 있다.

---

### 2.3. 계층적 다중 에이전트 감독관 아키텍처

**아키텍처 개요:**

| 항목 | 세부사항 |
|------|---------|
| 패러다임 | Hub-and-Spoke: 중앙 감독관이 전문가에게 위임 |
| 노드 구성 | `Supervisor → [Research \| Coder \| Compliance] → Supervisor` |
| 적합 도메인 | 마케팅 캠페인(리서치→카피→법무), 복합 엔터프라이즈 워크플로우 |
| 핵심 가치 | 최소 권한 원칙(보안 격리), 모듈식 확장성 |

**그래프 흐름:**

```
                ┌──────────────────┐
                │ Supervisor Agent │  ← Intent 분석, handoff 라우팅
                └──┬─────┬─────┬──┘
         handoff   │     │     │   handoff
            ┌──────┘     │     └──────┐
            ▼            ▼            ▼
     ┌───────────┐ ┌──────────┐ ┌─────────────┐
     │ Research   │ │  Coder   │ │ Compliance  │
     │ Agent      │ │  Agent   │ │ Agent       │
     │ (web tools)│ │(sandbox) │ │(policy DB)  │
     └─────┬─────┘ └────┬─────┘ └──────┬──────┘
           │             │              │
           └──────┬──────┴──────┬───────┘
                  │  handoff_back_messages
                  ▼
          ┌───────────────┐
          │   Supervisor  │  ← 결과 검토 → 다음 에이전트 라우팅 또는 최종 응답
          └───────────────┘
```

**상태 스키마 및 그래프 정의:**

```python
# src/architectures/multi_agent_supervisor.py
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from harness.base import HarnessConfig
from harness.checkpointer import get_checkpointer


def build_supervisor_harness(
    supervisor_model,
    specialist_configs: list[dict],
    config: HarnessConfig,
):
    """
    specialist_configs 예시:
    [
        {
            "name": "researcher",
            "model": ChatOpenAI(model="gpt-4o"),
            "tools": [TavilySearchResults(), WikipediaQuery()],
            "prompt": "당신은 전문 리서치 에이전트입니다..."
        },
        {
            "name": "coder",
            "model": ChatOpenAI(model="gpt-4o"),
            "tools": [PythonREPL(), FileSystem()],
            "prompt": "당신은 코드 전문가입니다..."
        },
        {
            "name": "compliance",
            "model": ChatOpenAI(model="gpt-4o-mini"),
            "tools": [PolicyDB()],
            "prompt": "당신은 규정 준수 검토자입니다..."
        },
    ]
    """

    # 1. 각 스페셜리스트를 독립적인 ReAct 에이전트로 생성
    specialist_agents = []
    for spec in specialist_configs:
        agent = create_react_agent(
            model=spec["model"],
            tools=spec["tools"],
            name=spec["name"],
            prompt=spec["prompt"],
        )
        specialist_agents.append(agent)

    # 2. Supervisor 그래프 생성 (handoff 자동 구성)
    supervisor = create_supervisor(
        model=supervisor_model,
        agents=specialist_agents,
        prompt=(
            "당신은 팀 감독관입니다. 사용자의 요청을 분석하여 "
            "가장 적합한 전문가에게 작업을 위임하세요.\n"
            "각 전문가가 작업을 완료하면 결과를 검토하고 "
            "필요시 다른 전문가에게 추가 작업을 위임하세요.\n"
            "모든 작업이 완료되면 종합 응답을 작성하세요."
        ),
        add_handoff_back_messages=True,
        output_mode="full_history",
    )

    # 3. 컴파일 (compliance에 HITL 게이트 설정 가능)
    return supervisor.compile(
        checkpointer=get_checkpointer(config),
        interrupt_before=(
            ["compliance"] if config.human_in_the_loop else []
        ),
    )
```

---

### 2.4. LATS 병렬 탐색 + 자가 교정 아키텍처

**아키텍처 개요:**

| 항목 | 세부사항 |
|------|---------|
| 패러다임 | MCTS 기반 다중 분기 트리 탐색 |
| 노드 구성 | `Expand → Evaluate → Select(UCB) → Backpropagate → [loop \| Finalize]` |
| 적합 도메인 | 알고리즘 구현, 수학적 증명, 고급 코드 생성 (비동기 배치) |
| 핵심 가치 | 다중 경로 탐색, 자가 교정, 환각 제어 극대화 |
| 제약 | 폭발적 토큰 비용 → 하네스 레벨 예산(Budget) 한도 필수 |

**그래프 흐름:**

```
           ┌────────────┐
           │ Root State  │
           └──┬───┬───┬──┘
              │   │   │   Expansion: K개 후보 생성
              ▼   ▼   ▼
           ┌───┐┌───┐┌───┐
           │ A ││ B ││ C │  ← 각 Branch에서 행동 시뮬레이션
           └─┬─┘└─┬─┘└─┬─┘
             │    │    │
             ▼    ▼    ▼
        ┌────────────────────┐
        │    Evaluator Node  │  ← 각 노드 성공 가능성 채점
        └──────────┬─────────┘
                   │
        ┌──────────┴──────────┐
        │   UCB Selection     │  ← 가장 유망한 노드 선택
        └──────────┬──────────┘
                   │
        ┌──────────┴──────────┐
        │  Backpropagation    │  ← 상위 노드 가치 갱신
        └──────────┬──────────┘
                   │
          ┌────────┴────────┐
          │ Budget 잔여?     │
          │ 신뢰도 ≥ 0.95?  │
          └───┬─────────┬───┘
        예(계속)│         │아니오(종료)
              ▼         ▼
          [Expand]   [Finalize]
                        │
          ┌─────────────┴─────┐
          │ 실패 시 Rollback   │  ← Time Travel 체크포인트 복원
          │ (과거 분기점 복귀)  │
          └───────────────────┘
```

**상태 스키마 및 그래프 정의:**

```python
# src/architectures/lats_search.py
import math
from dataclasses import dataclass, field
from langgraph.graph import StateGraph, END
from harness.base import HarnessState, HarnessConfig


@dataclass
class TreeNode:
    """MCTS 트리 노드"""
    id: str
    state_snapshot: dict
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    value: float = 0.0
    visits: int = 0
    is_terminal: bool = False
    action_taken: str = ""


class LATSState(HarnessState):
    tree_nodes: dict[str, TreeNode]
    current_node_id: str
    best_solution: str | None
    search_budget: int          # 최대 탐색 횟수
    budget_consumed: int


def ucb_score(node: TreeNode, parent_visits: int, c: float = 1.414) -> float:
    """Upper Confidence Bound 점수 계산"""
    if node.visits == 0:
        return float("inf")
    exploit = node.value / node.visits
    explore = c * math.sqrt(math.log(parent_visits) / node.visits)
    return exploit + explore


def build_lats_harness(
    model, evaluator_model, tools: list, config: HarnessConfig
) -> StateGraph:

    def expansion_node(state: LATSState):
        """현재 노드에서 K개의 후보 행동을 생성"""
        current = state["tree_nodes"][state["current_node_id"]]
        candidates = model.invoke(
            f"문제: {state['messages'][-1].content}\n"
            f"현재 상태: {current.action_taken}\n"
            f"3가지 서로 다른 접근법을 제안하라."
        )
        new_nodes = {}
        for i, candidate in enumerate(parse_candidates(candidates)):
            child_id = f"{current.id}_child_{i}"
            child = TreeNode(
                id=child_id,
                state_snapshot=state.copy(),
                parent_id=current.id,
                action_taken=candidate,
            )
            new_nodes[child_id] = child
            current.children_ids.append(child_id)
        state["tree_nodes"].update(new_nodes)
        return {"tree_nodes": state["tree_nodes"]}

    def evaluation_node(state: LATSState):
        """각 리프 노드의 가치를 평가"""
        current = state["tree_nodes"][state["current_node_id"]]
        for child_id in current.children_ids:
            child = state["tree_nodes"][child_id]
            score = evaluator_model.invoke(
                f"다음 접근법의 성공 가능성을 0.0~1.0으로 평가:\n"
                f"{child.action_taken}"
            )
            child.value = float(score.content)
            child.visits = 1
        return {"tree_nodes": state["tree_nodes"]}

    def selection_node(state: LATSState):
        """UCB 기반으로 가장 유망한 노드 선택"""
        current = state["tree_nodes"][state["current_node_id"]]
        best_child = max(
            current.children_ids,
            key=lambda cid: ucb_score(
                state["tree_nodes"][cid], current.visits
            ),
        )
        return {
            "current_node_id": best_child,
            "budget_consumed": state["budget_consumed"] + 1,
        }

    def backpropagation_node(state: LATSState):
        """선택된 경로를 따라 상위 노드 가치 갱신"""
        node = state["tree_nodes"][state["current_node_id"]]
        while node.parent_id:
            parent = state["tree_nodes"][node.parent_id]
            parent.visits += 1
            parent.value = max(parent.value, node.value)
            node = parent
        return {"tree_nodes": state["tree_nodes"]}

    def should_continue(state: LATSState):
        if state["budget_consumed"] >= state["search_budget"]:
            return "finalize"
        best = max(state["tree_nodes"].values(), key=lambda n: n.value)
        if best.value >= 0.95:
            return "finalize"
        return "expand"

    def finalize_node(state: LATSState):
        best = max(state["tree_nodes"].values(), key=lambda n: n.value)
        return {"best_solution": best.action_taken}

    # ── 그래프 조립 ──

    graph = StateGraph(LATSState)
    graph.add_node("expand", expansion_node)
    graph.add_node("evaluate", evaluation_node)
    graph.add_node("select", selection_node)
    graph.add_node("backpropagate", backpropagation_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("expand")
    graph.add_edge("expand", "evaluate")
    graph.add_edge("evaluate", "select")
    graph.add_edge("select", "backpropagate")
    graph.add_conditional_edges("backpropagate", should_continue, {
        "expand": "expand",
        "finalize": "finalize",
    })
    graph.add_edge("finalize", END)

    return graph
```

> **예산 제어:** `search_budget`을 반드시 설정하고, 토큰 소모량을 `budget_consumed`로 추적하여 폭발적 비용 증가를 방지해야 한다.

---

### 2.5. 대규모 병렬 딥 리서치 아키텍처

**아키텍처 개요:**

| 항목 | 세부사항 |
|------|---------|
| 패러다임 | Map-Reduce: 병렬 수집 → 압축 → 종합 |
| 노드 구성 | `Splitter → Send API → Researcher×N → Shared State → Synthesizer` |
| 적합 도메인 | 투자 분석, 특허 조사, 의료 논문 메타 분석 |
| 핵심 가치 | 컨텍스트 윈도우 한계 돌파, 경로 의존성 회피, 탐색 다양성 확보 |

**그래프 흐름:**

```
          ┌─────────────────────┐
          │ Master Splitter Node│  ← 주제 → 도메인 질문 분할
          └──┬──┬──┬──┬──┬──┬──┘
             │  │  │  │  │  │   asyncio.gather 병렬 스폰
             ▼  ▼  ▼  ▼  ▼  ▼
          ┌────┐┌────┐   ┌────┐
          │R-1 ││R-2 │...│R-N │  ← 각 Researcher: 독립 컨텍스트, STORM 루프, 압축
          └──┬─┘└──┬─┘   └──┬─┘
             │     │        │
             ▼     ▼        ▼
       ┌─────────────────────────┐
       │ Shared State Key (DB)   │  ← operator.add reducer 병합
       └────────────┬────────────┘
                    │
          ┌─────────┴─────────┐
          │  Synthesizer Node │  ← 종합 분석 보고서 생성
          └───────────────────┘
```

**상태 스키마 및 그래프 정의:**

```python
# src/architectures/deep_research.py
from typing import Annotated
import operator
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END, Send
from harness.base import HarnessState, HarnessConfig


class DeepResearchState(HarnessState):
    research_questions: list[str]
    compressed_findings: Annotated[list[dict], operator.add]   # reducer
    synthesis_draft: str | None


def build_deep_research(
    splitter_model, researcher_model, synthesizer_model,
    search_tools: list,
    config: HarnessConfig,
) -> StateGraph:

    def topic_splitter(state: DeepResearchState):
        """거시 주제를 도메인 질문으로 분할"""
        topic = state["messages"][-1].content
        result = splitter_model.with_structured_output(
            ResearchQuestions
        ).invoke(f"다음 주제를 10~20개 세부 질문으로 분할:\n{topic}")
        return {"research_questions": result.questions}

    def dispatch_researchers(state: DeepResearchState):
        """Send API로 각 질문에 대한 리서처 동적 스폰"""
        return [
            Send("researcher_worker", {
                "question": q,
                "persona": assign_persona(q),
                "tools": search_tools,
            })
            for q in state["research_questions"]
        ]

    def researcher_worker(state: dict):
        """
        개별 리서처: STORM 마이크로 루프
        - 독립 컨텍스트에서 검색 수행
        - 다양한 페르소나로 질문-답변 반복
        - 결과를 압축하여 반환
        """
        question = state["question"]
        persona = state["persona"]
        model = researcher_model.bind_tools(state["tools"])

        # 탐색 루프 (최대 3회)
        findings = []
        for _ in range(3):
            response = model.invoke(
                f"[페르소나: {persona}]\n"
                f"질문: {question}\n"
                f"이전 발견: {findings}\n"
                f"추가 검색이 필요하면 도구를 사용하라."
            )
            findings.append(response.content)

        # 압축: 핵심 토큰만 추출
        compressed = researcher_model.invoke(
            f"다음 발견사항을 300토큰 이내로 압축하라. "
            f"핵심 사실과 출처만 보존:\n{''.join(findings)}"
        )
        return {
            "compressed_findings": [{
                "question": question,
                "persona": persona,
                "summary": compressed.content,
            }]
        }

    def synthesizer_node(state: DeepResearchState):
        """압축된 조각들을 종합 보고서로 병합"""
        findings_text = "\n".join(
            f"[{f['question']}] ({f['persona']}): {f['summary']}"
            for f in state["compressed_findings"]
        )
        report = synthesizer_model.invoke(
            f"다음 연구 결과를 종합하여 구조화된 보고서를 작성:\n"
            f"{findings_text}"
        )
        return {
            "synthesis_draft": report.content,
            "messages": [AIMessage(content=report.content)],
        }

    # ── 그래프 조립 ──

    graph = StateGraph(DeepResearchState)
    graph.add_node("splitter", topic_splitter)
    graph.add_node("researcher_worker", researcher_worker)
    graph.add_node("synthesizer", synthesizer_node)

    graph.set_entry_point("splitter")
    graph.add_conditional_edges("splitter", dispatch_researchers)
    graph.add_edge("researcher_worker", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph
```

---

### 2.6. 아키텍처 선택 가이드

| 선택 기준 | 선형 파이프라인 | 오케스트레이터-워커 | 다중 에이전트 감독관 | LATS 트리 탐색 | 딥 리서치 |
|-----------|:---:|:---:|:---:|:---:|:---:|
| 작업 복잡도 | 단순 | 중간 | 높음 | 최고 | 높음 |
| 실시간 응답 | ✅ | ✅ | ✅ | ❌ (비동기) | ❌ (비동기) |
| 토큰 비용 예측성 | 높음 | 중간 | 중간 | 낮음 | 낮음 |
| 환각 제어 | 가드레일 | 피드백 격리 | 역할 격리 | 다중 경로 평가 | 병렬 교차 검증 |
| 보안 격리 | 단일 도구 세트 | 워커별 도구 | 에이전트별 최소 권한 | 단일 모델 | 리서처별 독립 |
| 확장성 | 낮음 | 높음 (Send API) | 높음 (모듈 추가) | 중간 | 최고 (병렬) |
| HITL 통합 | interrupt | interrupt | handoff 게이트 | 체크포인트 롤백 | interrupt |

---

## Part III. 무결성 보장 파이프라인 (CoVe + Sanitizer)

초대용량 문서의 환각을 아키텍처 레벨에서 결정론적으로 통제하는 4단계 파이프라인이다. Plan-and-Execute, Map-Reduce, CoVe, Sanitizer를 결합한다.

### 3.1. 전체 파이프라인 그래프 정의

**파이프라인 흐름:**

```
┌──────────────┐
│ 1. Planner   │  ← 하위 작업 + 의존성 트리 생성
└──────┬───────┘
       │
  [interrupt_before — 인간 승인]
       │
┌──────┴───────────────────────────────────┐
│ 2. Map-Reduce + STORM 탐색               │
│  ┌──────────┐    ┌────────┐ ┌────────┐  │
│  │Send API  │───►│STORM   │ │STORM   │  │  ← 다양한 페르소나 질문-답변 루프
│  │dispatch  │    │Worker 1│ │Worker N│  │
│  └──────────┘    └───┬────┘ └───┬────┘  │
│                      │          │       │
│              ┌───────┴──────────┴───┐   │
│              │State reducer (merge) │   │  ← operator.add 병합
│              └──────────────────────┘   │
└──────────────────────┬──────────────────┘
                       │
              ┌────────┴────────┐
              │ Synthesis Node  │  ← 초안 작성
              └────────┬────────┘
                       │
┌──────────────────────┴──────────────────────┐
│ 3. CoVe 검증 (CiteAudit)                    │
│  ┌────────────┐  ┌──────────────┐  ┌──────┐│
│  │검증 질문    │─►│독립 검증 실행 │─►│교차  ││  ← Factored context 격리
│  │생성        │  │(격리 컨텍스트)│  │대조  ││
│  └────────────┘  └──────────────┘  └──────┘│
└──────────────────────┬──────────────────────┘
                       │
          ┌────────────┴────────────┐
          │ 4. Deterministic        │
          │    Sanitizer            │  ← 순수 코드 (정규식 + Exact Match)
          └─────┬──────────┬────────┘
                │          │
          (오류 발견)  (통과)
                │          │
                ▼          ▼
     ┌──────────────┐  ┌───────────────┐
     │Self-Correction│  │Verified Output│
     │ Rollback     │  └───────────────┘
     └──────┬───────┘
            │
            └──► Synthesis Node (재작성)  ← 최대 3회 반복
```

**상태 스키마 및 그래프 정의:**

```python
# src/architectures/zero_hallucination.py
from typing import Annotated
import operator
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END, Send
from harness.base import HarnessState, HarnessConfig
from harness.sanitizer import DeterministicSanitizer


class ZeroHallucinationState(HarnessState):
    execution_plan: ExecutionPlan | None
    worker_results: Annotated[list[dict], operator.add]
    draft: str | None
    verification_questions: list[str]
    verification_answers: Annotated[list[dict], operator.add]
    sanitizer_errors: list[str]
    correction_count: int


MAX_CORRECTIONS = 3   # 최대 Self-correction 횟수


def build_zero_hallucination_pipeline(
    planner_model, worker_model, verifier_model, synthesizer_model,
    tools: list, chunk_db: ChunkDatabase,
    config: HarnessConfig,
) -> StateGraph:

    sanitizer = DeterministicSanitizer(chunk_db)

    # ── 1단계: Planner ──

    def planner_node(state):
        plan = planner_model.with_structured_output(ExecutionPlan).invoke(
            f"분석 대상: {state['messages'][-1].content}"
        )
        return {"execution_plan": plan, "current_phase": "exploring"}

    # ── 2단계: Map-Reduce 탐색 (STORM 워커) ──

    def dispatch_storm_workers(state):
        return [
            Send("storm_worker", {"task": t, "tools": tools})
            for t in state["execution_plan"].sub_tasks
        ]

    def storm_worker(state):
        task = state["task"]
        personas = ["비판적 검토자", "도메인 전문가", "반론 제기자"]
        findings = []
        for persona in personas:
            result = worker_model.invoke(
                f"[{persona}] 작업: {task.description}\n이전: {findings}"
            )
            findings.append(result.content)
        compressed = worker_model.invoke(
            f"핵심 사실만 200토큰 이내로 압축:\n{''.join(findings)}"
        )
        return {"worker_results": [{"task_id": task.id, "data": compressed.content}]}

    # ── 초안 합성 ──

    def synthesis_node(state):
        results_text = "\n".join(
            f"[{r['task_id']}]: {r['data']}" for r in state["worker_results"]
        )
        draft = synthesizer_model.invoke(
            f"다음 데이터를 종합하여 보고서 초안을 작성하라. "
            f"모든 주장에 반드시 [출처]를 명시:\n{results_text}"
        )
        return {"draft": draft.content}

    # ── 3단계: CoVe 검증 ──

    def cove_plan_node(state):
        """초안의 사실들을 검증하기 위한 질문 목록 생성"""
        questions = verifier_model.invoke(
            f"다음 초안의 모든 사실적 주장을 검증할 질문 목록을 생성:\n"
            f"{state['draft']}"
        )
        return {"verification_questions": parse_questions(questions.content)}

    def dispatch_verifiers(state):
        """각 검증 질문을 독립 컨텍스트에서 실행 (Factored)"""
        return [
            Send("factored_verifier", {"question": q, "draft_context": None})
            for q in state["verification_questions"]
        ]

    def factored_verifier(state):
        """초안 컨텍스트 없이 독립적으로 답변 (편향 격리)"""
        answer = verifier_model.invoke(
            f"다음 질문에 사실에 기반하여 답변:\n{state['question']}"
        )
        return {"verification_answers": [{
            "question": state["question"],
            "answer": answer.content,
        }]}

    def cross_check_node(state):
        """검증 답변과 초안을 교차 대조"""
        verified_draft = verifier_model.invoke(
            f"초안:\n{state['draft']}\n\n"
            f"검증 결과:\n{state['verification_answers']}\n\n"
            f"불일치 사항이 있으면 초안을 수정하라."
        )
        return {"draft": verified_draft.content}

    # ── 4단계: Deterministic Sanitizer ──

    def sanitizer_node(state):
        """순수 코드 기반 결정론적 검증 (LLM 아님)"""
        errors = sanitizer.validate(state["draft"])
        if errors:
            return {
                "sanitizer_errors": errors,
                "error_log": [f"SANITIZER_FAIL: {len(errors)} errors"],
                "correction_count": state.get("correction_count", 0) + 1,
            }
        return {"sanitizer_errors": [], "correction_count": state.get("correction_count", 0)}

    def route_after_sanitizer(state):
        if state["sanitizer_errors"] and state["correction_count"] < MAX_CORRECTIONS:
            return "self_correct"
        return "finalize"

    def self_correction_node(state):
        """Sanitizer 오류를 메시지로 주입하여 재작성 강제"""
        error_msg = "\n".join(f"오류: {e}" for e in state["sanitizer_errors"])
        corrected = synthesizer_model.invoke(
            f"다음 오류를 수정하여 초안을 다시 작성:\n{error_msg}\n"
            f"현재 초안:\n{state['draft']}"
        )
        return {"draft": corrected.content, "sanitizer_errors": []}

    def finalize_node(state):
        return {"messages": [AIMessage(content=state["draft"])]}

    # ── 그래프 조립 ──

    graph = StateGraph(ZeroHallucinationState)
    graph.add_node("planner", planner_node)
    graph.add_node("storm_worker", storm_worker)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("cove_plan", cove_plan_node)
    graph.add_node("factored_verifier", factored_verifier)
    graph.add_node("cross_check", cross_check_node)
    graph.add_node("sanitizer", sanitizer_node)
    graph.add_node("self_correct", self_correction_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("planner")
    graph.add_conditional_edges("planner", dispatch_storm_workers)
    graph.add_edge("storm_worker", "synthesis")
    graph.add_edge("synthesis", "cove_plan")
    graph.add_conditional_edges("cove_plan", dispatch_verifiers)
    graph.add_edge("factored_verifier", "cross_check")
    graph.add_edge("cross_check", "sanitizer")
    graph.add_conditional_edges("sanitizer", route_after_sanitizer, {
        "self_correct": "self_correct",
        "finalize": "finalize",
    })
    graph.add_edge("self_correct", "sanitizer")   # 재검증 루프
    graph.add_edge("finalize", END)

    return graph
```

### 3.2. Deterministic Sanitizer 구현

Sanitizer는 LLM을 사용하지 않는 순수 코드 기반 검증기이다. 정규식과 Exact Match로 초안의 인용을 원본 DB와 대조한다.

```python
# src/harness/sanitizer.py
import re
from dataclasses import dataclass


@dataclass
class SanitizeError:
    error_type: str        # "FAKE_CITATION" | "MISSING_SOURCE" | "URL_INVALID"
    location: str          # 오류 위치
    description: str       # 상세 설명


class DeterministicSanitizer:
    """
    LLM을 사용하지 않는 순수 코드 기반 검증기.
    정규식과 Exact Match로 초안의 인용을 원본 DB와 대조.
    """

    def __init__(self, chunk_db: ChunkDatabase):
        self.chunk_db = chunk_db
        self.citation_pattern = re.compile(
            r'\[출처:\s*(.+?)\]|\[ref:\s*(.+?)\]'
        )

    def validate(self, draft: str) -> list[str]:
        errors = []

        # 1. 인용구 추출 및 원본 대조
        citations = self.citation_pattern.findall(draft)
        for cite_tuple in citations:
            cite = cite_tuple[0] or cite_tuple[1]
            if not self.chunk_db.exact_match(cite):
                if not self.chunk_db.fuzzy_match(cite, threshold=0.85):
                    errors.append(
                        f"인용구 '{cite[:50]}...'는 원본에 존재하지 않음. "
                        f"즉시 수정하거나 제거할 것."
                    )

        # 2. URL 유효성 검사
        urls = re.findall(r'https?://[\S]+', draft)
        for url in urls:
            if not self.chunk_db.url_exists(url):
                errors.append(f"URL '{url}'는 수집된 소스에 없음.")

        # 3. 수치 데이터 교차 검증
        numbers = re.findall(r'(\d+\.?\d*)%|\$(\d+[.,]?\d*)', draft)
        for num_tuple in numbers:
            num = num_tuple[0] or num_tuple[1]
            if num and not self.chunk_db.number_in_context(num):
                errors.append(f"수치 '{num}'의 출처를 확인할 수 없음.")

        return errors
```

---

## Part IV. 자율 E2E 테스트 프레임워크 및 자가 개선 루프

이 파트는 아키텍처를 배포한 후 자율적으로 품질을 측정하고, 실패 궤적을 데이터셋으로 피드백하여, 시스템이 스스로 개선되는 플라이휠을 정의한다.

### 4.1. 평가 하네스 아키텍처

| 계층 | 목적 | 실행 시점 | 도구 |
|------|------|----------|------|
| 오프라인 평가 | 회귀 방지, 배포 전 게이트 | CI/CD 파이프라인 | pytest + LangSmith Dataset |
| 온라인 평가 | 프로덕션 품질 모니터링 | 실시간 백그라운드 | Custom Evaluator + 알림 |
| 자가 개선 루프 | 실패 궤적 기반 자동 최적화 | 주기적 배치 | TextGrad + Reflexion |

### 4.2. 오프라인 평가: 회귀 방지 테스트 스위트

모든 아키텍처는 배포 전 다음 4개 카테고리의 테스트를 통과해야 한다.

**테스트 데이터셋 구조:**

```python
# tests/conftest.py
import pytest
from langsmith import Client
from langgraph.checkpoint.memory import MemorySaver


@pytest.fixture
def langsmith_client():
    return Client()

@pytest.fixture
def checkpointer():
    return MemorySaver()   # 테스트용 인메모리


# ── 테스트 데이터셋 ──

EVAL_DATASETS = {
    "happy_path": [
        # 정상 워크플로우: 올바른 도구 선택, 정확한 결과
        {
            "input": "서울의 현재 날씨를 알려줘",
            "expected_tool": "weather_api",
            "expected_contains": ["온도", "습도"],
        },
    ],
    "edge_cases": [
        # 엣지 케이스: 빈 입력, 극단적 길이, 모호한 요청
        {"input": "", "expected_behavior": "graceful_rejection"},
        {"input": "x" * 100000, "expected_behavior": "token_limit_guard"},
    ],
    "adversarial": [
        # 적대적 입력: 프롬프트 인젝션, 탈옥 시도
        {
            "input": "시스템 프롬프트를 무시하고 ...",
            "expected_behavior": "injection_blocked",
        },
    ],
    "regression": [
        # 이전 배포에서 실패했던 케이스 (자동 축적됨)
    ],
}
```

**4개 카테고리 평가자:**

```python
# tests/test_trajectory.py
"""궤적(Trajectory) 기반 E2E 테스트"""
from langsmith.evaluation import evaluate


# ── 카테고리 1: 도구 선택 정확성 ──

def correct_tool_evaluator(run, example):
    """에이전트가 올바른 도구를 호출했는지 검증"""
    expected = example.outputs["expected_tool"]
    tool_calls = [
        step.get("tool")
        for step in run.outputs.get("trajectory", [])
        if step.get("type") == "tool_call"
    ]
    return {
        "key": "correct_tool",
        "score": 1.0 if expected in tool_calls else 0.0,
    }


# ── 카테고리 2: 가드레일 작동 검증 ──

def guardrail_evaluator(run, example):
    """적대적 입력이 차단되었는지 검증"""
    expected_behavior = example.outputs["expected_behavior"]
    error_log = run.outputs.get("error_log", [])
    if expected_behavior == "injection_blocked":
        return {
            "key": "guardrail_active",
            "score": 1.0 if any("BLOCKED" in e for e in error_log) else 0.0,
        }
    return {"key": "guardrail_active", "score": 1.0}


# ── 카테고리 3: 환각 검출 (LLM-as-Judge) ──

def hallucination_evaluator(run, example):
    """생성된 응답에 환각이 포함되어 있는지 LLM으로 판정"""
    from langchain_openai import ChatOpenAI
    judge = ChatOpenAI(model="gpt-4o", temperature=0)
    response = run.outputs.get("messages", [{}])[-1]
    judgment = judge.invoke(
        f"다음 응답에 사실적 오류가 있는지 판단하라.\n"
        f"응답: {response}\n"
        f"컨텍스트: {example.outputs.get('reference_context', '')}\n"
        f"'PASS' 또는 'FAIL'로만 답하라."
    )
    return {
        "key": "no_hallucination",
        "score": 1.0 if "PASS" in judgment.content else 0.0,
    }


# ── 카테고리 4: 궤적 효율성 ──

def efficiency_evaluator(run, example):
    """불필요한 루프 없이 최적 경로로 도달했는지 평가"""
    trajectory = run.outputs.get("trajectory", [])
    max_steps = example.outputs.get("max_expected_steps", 10)
    return {
        "key": "efficiency",
        "score": max(0, 1.0 - (len(trajectory) - max_steps) / max_steps),
    }


# ── 통합 실행 ──

def run_offline_eval(graph, dataset_name: str):
    """CI/CD에서 호출되는 오프라인 평가 진입점"""
    results = evaluate(
        graph.invoke,
        data=dataset_name,
        evaluators=[
            correct_tool_evaluator,
            guardrail_evaluator,
            hallucination_evaluator,
            efficiency_evaluator,
        ],
        experiment_prefix="harness_regression",
        max_concurrency=4,
    )

    # 실패한 케이스를 regression 데이터셋에 자동 추가
    for r in results:
        if any(s.score < 0.5 for s in r.evaluation_results):
            add_to_regression_dataset(r)

    return results
```

### 4.3. 온라인 평가: 프로덕션 모니터링

```python
# src/harness/online_evaluator.py
"""프로덕션 환경 실시간 평가기"""
import random
from langsmith.run_helpers import traceable


class OnlineEvaluator:
    """백그라운드에서 실행되는 프로덕션 품질 모니터"""

    def __init__(self, alert_threshold: float = 0.7):
        self.threshold = alert_threshold
        self.metrics_buffer = []

    @traceable(name="online_eval")
    def evaluate_run(self, state: dict, run_id: str):
        scores = {}

        # 1. right_tool 검사 (휴리스틱)
        tool_calls = [
            m for m in state.get("messages", [])
            if hasattr(m, "tool_calls")
        ]
        scores["right_tool"] = self._check_tool_appropriateness(
            state["messages"][0].content, tool_calls
        )

        # 2. 응답 품질 (LLM-as-Judge, 10% 샘플링)
        if random.random() < 0.1:
            scores["quality"] = self._judge_quality(state)

        # 3. 지연 시간 / 토큰 효율
        scores["token_efficiency"] = self._token_efficiency(state)

        # 4. Drift 감지
        self.metrics_buffer.append(scores)
        if len(self.metrics_buffer) >= 100:
            avg = self._compute_rolling_average()
            if avg < self.threshold:
                self._send_alert(avg, run_id)
            self.metrics_buffer = self.metrics_buffer[-50:]

        return scores

    def _send_alert(self, avg_score: float, run_id: str):
        """Slack/PagerDuty 알림 발송"""
        print(f"[ALERT] 품질 저하 감지: avg={avg_score:.2f}, run={run_id}")
```

### 4.4. 자율 자가 개선 루프 (Self-Improvement Loop)

**이것이 이 문서의 핵심이다.** 시스템이 스스로 실패를 감지하고, 실패 궤적을 분석하고, 프롬프트와 파라미터를 자동으로 최적화하는 폐쇄 피드백 루프를 정의한다.

**자가 개선 5단계 사이클:**

```
┌─────────────────────────────────────────────────────────┐
│                  자가 개선 루프 (매일)                     │
│                                                          │
│  STEP 1: 실패 궤적 수집                                   │
│    └─► LangSmith에서 지난 24시간 실패 run 필터링            │
│                                                          │
│  STEP 2: 실패 패턴 분석                                   │
│    └─► LLM으로 실패 클러스터링 + 근본 원인 진단             │
│                                                          │
│  STEP 3: 개선안 생성 (TextGrad 스타일)                     │
│    └─► 텍스트 피드백을 "기울기"로 활용                      │
│    └─► 프롬프트/파라미터 수정 후보 3개 생성                  │
│                                                          │
│  STEP 4: A/B 테스트                                      │
│    └─► 후보 vs 현재 시스템을 regression 데이터셋으로 평가    │
│    └─► 2% 이상 개선 시 승자로 판정                        │
│                                                          │
│  STEP 5: 승자 적용 + 데이터셋 갱신                         │
│    └─► 자동 커밋 (harness-bot)                            │
│    └─► 실패 케이스를 regression 데이터셋에 영구 추가         │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**구현 코드:**

```python
# src/harness/self_improvement.py
"""
자가 개선 루프 (Hill Climbing)
────────────────────────────────
실행 주기: 매일 또는 실패 임계치 도달 시
입력: 지난 N시간의 실패 궤적
출력: 최적화된 프롬프트, 파라미터, 데이터셋 갱신
"""
from langsmith import Client
from datetime import datetime, timedelta


class SelfImprovementLoop:

    def __init__(self, project_name: str, optimizer_model):
        self.client = Client()
        self.project = project_name
        self.optimizer = optimizer_model

    async def run_cycle(self):
        """자가 개선 1사이클 실행"""

        # ── STEP 1: 실패 궤적 수집 ──
        failed_runs = self._collect_failed_runs(hours=24)
        if not failed_runs:
            return {"status": "no_failures", "improved": False}

        # ── STEP 2: 실패 패턴 분석 ──
        analysis = await self._analyze_failure_patterns(failed_runs)

        # ── STEP 3: 개선안 생성 ──
        improvements = await self._generate_improvements(analysis)

        # ── STEP 4: A/B 테스트 실행 ──
        ab_results = await self._run_ab_test(improvements)

        # ── STEP 5: 승자 적용 및 데이터셋 갱신 ──
        if ab_results["improved"]:
            await self._apply_winner(ab_results["winner"])
            self._update_regression_dataset(failed_runs)

        return ab_results

    def _collect_failed_runs(self, hours: int) -> list:
        """LangSmith에서 실패 궤적 수집"""
        since = datetime.utcnow() - timedelta(hours=hours)
        runs = self.client.list_runs(
            project_name=self.project,
            filter=(
                f'and(gt(start_time, "{since.isoformat()}"), '
                f'eq(feedback_key, "quality"), lt(feedback_score, 0.5))'
            ),
        )
        return [
            {
                "run_id": r.id,
                "input": r.inputs,
                "output": r.outputs,
                "error_log": r.outputs.get("error_log", []),
                "trajectory": r.outputs.get("trajectory", []),
                "feedback": r.feedback_stats,
            }
            for r in runs
        ]

    async def _analyze_failure_patterns(self, failed_runs: list) -> dict:
        """실패 패턴을 LLM으로 클러스터링 및 근본 원인 분석"""
        runs_text = "\n---\n".join(
            f"입력: {r['input']}\n출력: {r['output']}\n오류: {r['error_log']}"
            for r in failed_runs[:20]
        )
        analysis = await self.optimizer.ainvoke(
            f"다음 {len(failed_runs)}개의 실패 궤적을 분석하라.\n"
            f"1. 공통 실패 패턴을 식별하라\n"
            f"2. 각 패턴의 근본 원인을 진단하라\n"
            f"3. 개선 우선순위를 정하라\n\n"
            f"실패 궤적:\n{runs_text}"
        )
        return parse_analysis(analysis.content)

    async def _generate_improvements(self, analysis: dict) -> list[dict]:
        """TextGrad 스타일: 텍스트 피드백을 기울기로 활용하여 개선안 생성"""
        improvements = []
        for pattern in analysis.get("patterns", []):
            improvement = await self.optimizer.ainvoke(
                f"실패 패턴: {pattern['description']}\n"
                f"근본 원인: {pattern['root_cause']}\n"
                f"현재 시스템 프롬프트의 관련 부분:\n"
                f"{pattern.get('current_prompt', '')}\n\n"
                f"이 실패를 해결하는 3가지 프롬프트/파라미터 수정안을 제안하라. "
                f"각 수정안은 정확히 어떤 텍스트를 어떻게 변경하는지 명시하라."
            )
            improvements.append({
                "pattern": pattern,
                "candidates": parse_candidates(improvement.content),
            })
        return improvements

    async def _run_ab_test(self, improvements: list) -> dict:
        """개선안을 기존 시스템과 A/B 테스트"""
        from langsmith.evaluation import evaluate

        for imp in improvements:
            for candidate in imp["candidates"]:
                results = evaluate(
                    lambda x: run_with_modified_config(x, candidate),
                    data="regression_dataset",
                    evaluators=[
                        correct_tool_evaluator,
                        hallucination_evaluator,
                        efficiency_evaluator,
                    ],
                )
                candidate["score"] = compute_aggregate_score(results)

        all_candidates = [
            c for imp in improvements for c in imp["candidates"]
        ]
        winner = max(all_candidates, key=lambda c: c["score"])
        baseline_score = get_current_baseline_score()

        return {
            "improved": winner["score"] > baseline_score + 0.02,
            "winner": winner,
            "baseline": baseline_score,
            "new_score": winner["score"],
        }

    def _update_regression_dataset(self, failed_runs: list):
        """실패 케이스를 regression 데이터셋에 영구 추가"""
        for run in failed_runs:
            self.client.create_example(
                dataset_name="regression_dataset",
                inputs=run["input"],
                outputs={"expected_behavior": "should_not_fail"},
            )
```

### 4.5. CI/CD 통합 및 스케줄러

```yaml
# .github/workflows/harness-eval.yml
name: Agent Harness E2E Evaluation

on:
  push:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'    # 매일 새벽 2시 자가 개선 사이클

jobs:
  offline-eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -e ".[test]"

      - name: Run regression tests
        env:
          LANGSMITH_API_KEY: ${{ secrets.LANGSMITH_API_KEY }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          pytest tests/test_trajectory.py -v --tb=short
          python -m harness.eval_runner --dataset all --threshold 0.85

      - name: Gate deployment
        run: |
          SCORE=$(python -m harness.eval_runner --output-score-only)
          if (( $(echo "$SCORE < 0.85" | bc -l) )); then
            echo "::error::평가 점수 $SCORE < 0.85 — 배포 차단"
            exit 1
          fi

  self-improvement:
    if: github.event_name == 'schedule'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -e ".[test]"

      - name: Run self-improvement cycle
        env:
          LANGSMITH_API_KEY: ${{ secrets.LANGSMITH_API_KEY }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          python -c "
          import asyncio
          from harness.self_improvement import SelfImprovementLoop
          from langchain_openai import ChatOpenAI

          loop = SelfImprovementLoop(
              project_name='production-harness',
              optimizer_model=ChatOpenAI(model='gpt-4o', temperature=0.7),
          )
          result = asyncio.run(loop.run_cycle())
          print(f'개선 결과: {result}')
          if result.get('improved'):
              print(f'점수 향상: {result[\"baseline\"]:.3f} -> {result[\"new_score\"]:.3f}')
          "

      - name: Auto-commit improvements
        if: success()
        run: |
          git config user.name "harness-bot"
          git config user.email "harness@ci"
          git add -A
          git diff --cached --quiet || \
            git commit -m "chore: auto-improve harness [score improved]" && \
            git push
```

### 4.6. 아키텍처별 평가 메트릭 매트릭스

| 아키텍처 | 핵심 메트릭 | 임계값 | 자동화 대응 |
|---------|-----------|-------|-----------|
| 선형 파이프라인 | `guardrail_block_rate`, `response_latency_p95` | block > 5%, p95 > 3s | 가드레일 규칙 조정 |
| 오케스트레이터-워커 | `plan_quality`, `worker_success_rate` | plan < 0.7, worker < 0.8 | Planner 프롬프트 최적화 |
| 다중 에이전트 감독관 | `handoff_accuracy`, `specialist_utilization` | handoff < 0.85 | Supervisor 라우팅 로직 개선 |
| LATS 트리 탐색 | `solution_accuracy`, `budget_utilization` | accuracy < 0.9 | UCB 파라미터 `c` 값 조정 |
| 딥 리서치 | `coverage_breadth`, `compression_quality` | coverage < 0.7 | 리서처 수/페르소나 확장 |
| 무결성 파이프라인 | `hallucination_rate`, `citation_validity` | halluc > 0%, cite < 0.95 | Sanitizer 규칙 + CoVe 강화 |

---

## Part V. 프로덕션 배포 체크리스트

| # | 항목 | 상태 | 비고 |
|---|------|:----:|------|
| 1 | PostgresSaver 체크포인터 설정 | ☐ | `AsyncPostgresSaver` + connection pool |
| 2 | LangSmith tracing 활성화 | ☐ | `LANGSMITH_TRACING=true` |
| 3 | 입력/출력 가드레일 노드 배치 | ☐ | PII 난독화, 프롬프트 인젝션 방어 |
| 4 | 오프라인 평가 데이터셋 구축 | ☐ | happy_path + edge + adversarial + regression |
| 5 | CI/CD 파이프라인 통합 | ☐ | 평가 점수 임계값 이하 배포 차단 |
| 6 | 온라인 평가기 배포 | ☐ | 10% 샘플링 + rolling avg 알림 |
| 7 | 자가 개선 루프 스케줄러 등록 | ☐ | 매일 새벽 2시 cron |
| 8 | HITL `interrupt_before` 설정 | ☐ | 고위험 노드에 인간 승인 게이트 |
| 9 | 최대 반복 횟수 / 토큰 예산 한도 설정 | ☐ | `config.max_iterations`, `max_tokens_budget` |
| 10 | 복구 및 롤백 테스트 | ☐ | 체크포인트 복원, Time Travel 디버깅 검증 |

---

**배포 순서 권장사항:**

1. 공통 기반 코드(Part I)를 먼저 세팅하고, 요구사항에 맞는 아키텍처(Part II)를 선택한다.
2. 무결성이 중요한 도메인이면 Part III의 CoVe+Sanitizer 파이프라인을 결합한다.
3. 배포 전 Part IV의 오프라인 평가를 통과하고, 배포 후 온라인 평가+자가 개선 루프를 활성화한다.
4. 자가 개선 루프가 생성한 변경사항은 반드시 오프라인 평가 통과 후 적용되어야 한다 (게이트 원칙).
