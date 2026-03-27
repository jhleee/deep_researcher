# LangGraph Agent Harness - 개발 핸드오프 문서

> 작성일: 2026-03-28
> 기준 커밋: `4e1e806` (master)

---

## 1. 프로젝트 개요

`langgraph_harness_guide.md` 설계서를 기반으로 LangGraph 에이전트 하네스를 구현하는 프로젝트.
**외부 시스템(DB, LLM API, LangSmith) 없이 로컬/파일시스템만으로 동작**하는 것이 현재 단계의 핵심 제약이다.

---

## 2. 완료된 작업

### Phase 1: 공통 기반 (Part I) ✅

| 커밋 | 파일 | 내용 |
|------|------|------|
| `1063c79` | 23개 파일 | 프로젝트 스캐폴딩 전체 |

**구현된 모듈:**

| 모듈 | 역할 |
|------|------|
| `src/harness/base.py` | `HarnessState` TypedDict (messages, plan, artifacts, metadata, error_log, iteration_count), `HarnessConfig` dataclass, `create_initial_state()` |
| `src/harness/checkpointer.py` | `get_checkpointer()` 팩토리 (memory/json_file/sqlite/postgres), `JsonFileCheckpointer` 클래스 |
| `src/harness/guardrails.py` | `InputGuardrail` (PII 마스킹, 프롬프트 인젝션 차단, 길이 제한), `OutputGuardrail` (유해성 필터, 길이 잘라내기) |
| `src/harness/sanitizer.py` | `DeterministicSanitizer` (인용/URL/수치 검증), `LocalChunkDB` (JSON 파일 기반 청크 DB) |
| `src/harness/online_evaluator.py` | `OnlineEvaluator` (로컬 JSON 메트릭 기록, 품질 저하 알림) |
| `src/harness/self_improvement.py` | `LocalFailureStore`, `LocalRegressionDataset`, `SelfImprovementLoop` (실패 수집/분석/회귀 데이터셋) |

**테스트:** 62개 (test_base, test_checkpointer, test_guardrails, test_sanitizer, test_online_evaluator, test_self_improvement)

### Phase 2: 무결성 보장 파이프라인 (Part III) ✅

| 커밋 | 파일 | 내용 |
|------|------|------|
| `853720b` | models.py, test_models.py | Pydantic 모델 + FakeStructuredChatModel |
| `83f7f85` | zero_hallucination.py, conftest.py, test_zero_hallucination.py | 상태 스키마 + 9개 노드 함수 + 라우팅 |
| `1930c1a` | __init__.py, test_zero_hallucination.py | 그래프 조립 + E2E 테스트 |
| `4e1e806` | CLAUDE.md | 진행 상태 업데이트 |

**구현된 모듈:**

| 모듈 | 역할 |
|------|------|
| `src/harness/models.py` | `SubTask`, `ExecutionPlan`, `ResearchQuestions` Pydantic 모델, `FakeStructuredChatModel` (테스트용), `parse_questions()`, `assign_persona()` 헬퍼 |
| `src/architectures/zero_hallucination.py` | 4단계 무결성 파이프라인 전체 |

**파이프라인 4단계:**

```
1. Planner        → ExecutionPlan 생성 (structured output)
2. STORM Workers   → 3 페르소나 병렬 탐색 + 압축 (Send API)
3. CoVe 검증       → 질문 생성 → 격리 검증 → 교차 대조
4. Sanitizer       → 결정론적 검증 ⇄ 자기 교정 (max 3회) → 최종 출력
```

**노드 함수 (팩토리 패턴):**
- `make_planner_node()`, `make_storm_worker()`, `make_synthesis_node()`
- `make_cove_plan_node()`, `make_factored_verifier()`, `make_cross_check_node()`
- `make_sanitizer_node()`, `make_self_correction_node()`, `finalize_node()`

**라우팅:**
- `dispatch_storm_workers()` → Send API로 작업별 워커 스폰
- `dispatch_verifiers()` → Send API로 질문별 검증기 스폰
- `route_after_sanitizer()` → 오류 시 self_correct, 아니면 finalize

**테스트:** 54개 추가 (23 models + 31 pipeline, E2E 2개 포함)

---

## 3. 현재 상태

| 항목 | 값 |
|------|-----|
| 총 테스트 | **116개 전체 통과** (0.49초) |
| 총 소스 파일 | 8개 (harness 6 + architectures 2) |
| 총 테스트 파일 | 8개 |
| Python | >= 3.11 |
| 핵심 의존성 | langgraph>=0.4.0, langchain-core>=0.3.0, pydantic>=2.0 |
| 외부 시스템 | 없음 (전부 로컬/인메모리) |

**프로젝트 트리:**
```
src/
├── harness/
│   ├── __init__.py
│   ├── base.py              # 공통 상태/설정
│   ├── checkpointer.py      # 체크포인터 팩토리
│   ├── guardrails.py        # 입출력 가드레일
│   ├── models.py            # 공유 Pydantic 모델 + FakeModel
│   ├── online_evaluator.py  # 로컬 평가기
│   ├── sanitizer.py         # 결정론적 검증기
│   └── self_improvement.py  # 자가 개선 루프
└── architectures/
    ├── __init__.py
    └── zero_hallucination.py  # Part III 4단계 파이프라인
tests/
├── conftest.py              # 공통 픽스처 (19개)
├── test_base.py             # 7 tests
├── test_checkpointer.py     # 12 tests
├── test_guardrails.py       # 15 tests
├── test_models.py           # 23 tests
├── test_online_evaluator.py # 4 tests
├── test_sanitizer.py        # 16 tests
├── test_self_improvement.py # 8 tests
├── test_zero_hallucination.py # 31 tests
└── datasets/                # JSON 테스트 데이터
```

---

## 4. 미구현 항목 (설계서 기준)

### Part II: 5대 아키텍처 (미구현)

| 아키텍처 | 파일 | 우선순위 |
|----------|------|---------|
| 선형 파이프라인 + 가드레일 (2.1) | `linear_pipeline.py` | 높음 - 가장 단순, 기반 검증에 적합 |
| 오케스트레이터-워커 (2.2) | `orchestrator_worker.py` | 높음 - Send API 패턴이 Part III와 유사 |
| 다중 에이전트 감독관 (2.3) | `multi_agent_supervisor.py` | 중간 - `langgraph_supervisor` 의존성 필요 |
| LATS 트리 탐색 (2.4) | `lats_search.py` | 낮음 - 복잡도 높음, 별도 TreeNode 구현 필요 |
| 딥 리서치 (2.5) | `deep_research.py` | 중간 - Part III와 구조 유사 |

### Part IV: 자율 E2E 테스트 프레임워크 (일부 구현)

| 항목 | 상태 | 비고 |
|------|------|------|
| 오프라인 평가 (4.2) | 기반만 | conftest.py에 데이터셋 JSON 존재, 궤적 평가자 미구현 |
| 온라인 평가 (4.3) | ✅ 로컬 버전 | `online_evaluator.py` (JSON 메트릭, 알림) |
| 자가 개선 루프 (4.4) | ✅ 기반 | `self_improvement.py` (규칙 기반 분석, LLM 분석 미구현) |
| CI/CD 통합 (4.5) | 미구현 | `.github/workflows/harness-eval.yml` 비어 있음 |

### Part V: 프로덕션 배포 체크리스트 (미해당)

현재 로컬 전용 단계이므로 해당 없음. PostgresSaver, LangSmith, HITL 등은 외부 시스템 연동 단계에서 구현.

---

## 5. 핵심 설계 결정 (다음 개발자를 위한 참고)

1. **노드 함수 팩토리 패턴**: `make_xxx_node(model)` → 클로저 반환. 모델을 인자로 받으므로 테스트 시 FakeModel 주입 가능.

2. **FakeStructuredChatModel**: `FakeListChatModel` 확장. `with_structured_output(schema)` → `self | RunnableLambda(parse)` 체인. JSON 응답을 Pydantic 모델로 자동 파싱.

3. **Send API import**: `from langgraph.types import Send` (v1.0+ 방식). `langgraph.constants`에서 import하면 deprecation 경고.

4. **sanitizer_errors는 list[str]**: `SanitizeError` dataclass는 상태에 직접 저장하면 직렬화 문제 발생. `validate_as_strings()` 사용.

5. **correction_count는 reducer 없음**: `operator.add` 아닌 last-write-wins. 의도적으로 절대값 덮어쓰기.

6. **빈 계획/질문 시 폴백**: `dispatch_storm_workers`에서 계획이 비면 synthesis로 직행, `dispatch_verifiers`에서 질문이 비면 cross_check으로 직행.

---

## 6. 개발 환경

```bash
# 설치
pip install -e ".[test]"

# 전체 테스트
py -m pytest tests/ -v

# 특정 모듈
py -m pytest tests/test_zero_hallucination.py -v

# 커버리지
py -m pytest tests/ --cov=harness --cov=architectures --cov-report=term-missing
```

---

## 7. 권장 다음 작업

1. **선형 파이프라인 (Part II - 2.1)**: 가장 단순한 아키텍처. 기존 `InputGuardrail`/`OutputGuardrail`을 그래프 노드로 통합.
2. **오케스트레이터-워커 (Part II - 2.2)**: `ExecutionPlan`/`SubTask` 모델과 Send API를 재사용. Part III와 구조 유사.
3. **오프라인 평가 궤적 테스트 (Part IV - 4.2)**: `tests/datasets/` JSON을 활용한 궤적 기반 평가자 구현.
