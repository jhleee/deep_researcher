# LangGraph Agent Harness - 핸드오프 문서

> 작성일: 2026-03-29
> 기준 커밋: `b7d52d6` (master)

---

## 1. 프로젝트 현황

LangGraph 기반 Zero-Hallucination 리서치 파이프라인. 로컬 LLM(LM Studio) + 파일 기반 지식 베이스로 동작한다.

```
sessions/run_xxx/         ← 연구 세션별 전체 기록
  ├── session.json        ← 메타 (query, status, elapsed, completed_nodes)
  ├── config.json         ← 실행 시점 설정 스냅샷
  ├── traces.json         ← 노드 트레이스 + LLM 호출 (프롬프트/응답/역할)
  ├── checkpoint.json     ← 파이프라인 최종 상태
  ├── evaluation.json     ← 평가 점수
  ├── failure.json        ← 실패 시: traceback, phase, partial_state
  └── result.md           ← 최종 결과
```

| 항목 | 값 |
|------|-----|
| 테스트 | 175개 통과 |
| 소스 | harness 10개 + architectures 1개 |
| 의존성 | langgraph, langchain-core, pydantic, textual |
| LLM | LM Studio (qwen3.5-9b) — 로컬 |

---

## 2. 아키텍처

```
Query
  ↓
Term Resolver    ← 고유명사·전문용어 사전 확인 (KB + LLM)
  ↓
Planner          ← 구조화된 실행 계획 (ExecutionPlan)
  ↓
STORM Workers    ← 작업별 3 페르소나 병렬 탐색 + 압축 (Send API)
  ↓
Synthesis        ← 초안 합성
  ↓
CoVe Plan        ← 검증 질문 생성
  ↓
Factored Verifiers ← 질문별 독립 검증 (Send API)
  ↓
Cross-Check      ← 교차 대조
  ↓
Sanitizer        ← 결정론적 검증 (LLM 미사용)
  ↓ (errors? → Self-Correct ↻ max 3)
Finalize         ← 최종 출력
```

---

## 3. 자기 개선 루프 개발 방법론

이 프로젝트는 **"실행 → 실패 분석 → 프롬프트/아키텍처 수정 → 재실행"** 사이클을 반복하여 개발한다. 핵심은 **세션 디렉토리에 남은 기록만으로 문제의 근본 원인을 파악**할 수 있어야 한다는 것이다.

### 3.1. 사이클 구조

```
┌─────────────────────────────────────────────────┐
│  1. 실행                                         │
│     - TUI 또는 test_run.py로 파이프라인 실행       │
│     - sessions/{run_id}/ 에 전체 기록 저장         │
└──────────────────────┬──────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│  2. 분석                                         │
│     - session.json → 어디서 실패/성공했는지       │
│     - traces.json → 어떤 LLM 호출이 문제인지      │
│       (prompt_preview, response_preview, role)    │
│     - failure.json → traceback, phase, last_node │
│     - result.md → 최종 출력의 품질 확인            │
└──────────────────────┬──────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│  3. 원인 분류                                     │
│     아래 패턴 중 하나에 해당:                       │
│     (A) 주제 이탈 — LLM이 주제를 바꿈              │
│     (B) 컨텍스트 초과 — 프롬프트가 너무 김          │
│     (C) JSON 파싱 실패 — structured output 깨짐   │
│     (D) 인용 폭발 — Sanitizer가 대량 reject        │
│     (E) 검증 오류 — CoVe가 정확한 정보를 부정       │
│     (F) 외부 오류 — LLM 서버 연결, 타임아웃 등      │
└──────────────────────┬──────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│  4. 수정                                         │
│     원인별 처방:                                   │
│     (A) → 프롬프트에 TOPIC_PRESERVATION_RULE 강화  │
│           Term Resolver에 KB 자료 보강             │
│     (B) → max_chars 줄이기, findings 축약          │
│     (C) → structured output 재시도 로직 강화       │
│     (D) → Synthesis 프롬프트에서 인용태그 제거 지시 │
│     (E) → Factored Verifier에 term_context 전달   │
│     (F) → ThinkingModelWrapper에 에러 핸들링 추가  │
└──────────────────────┬──────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│  5. 검증                                         │
│     - py -m pytest tests/ 전체 통과 확인          │
│     - py -u scripts/test_run.py 라이브 재실행     │
│     - 오염 키워드 체크 (PROBLEM_KEYWORDS)          │
│     - Ground truth 키워드 체크                     │
│     - Sanitizer errors 수 확인                    │
│     - 만족스러우면 커밋, 아니면 1로 돌아감          │
└─────────────────────────────────────────────────┘
```

### 3.2. 실제 사이클 이력 (붉은사막 케이스)

| Cycle | 문제 | 원인 분류 | 수정 | 결과 |
|-------|------|-----------|------|------|
| 0 (초기) | "붉은 사막"을 RDR2로 대체 | (A) 주제 이탈 | TOPIC_PRESERVATION_RULE 추가 | 부분 개선, 여전히 RDR2 언급 |
| 1 | 여전히 RDR2 추측 | (A) LLM이 용어 자체를 모름 | Term Resolver 노드 신설, KB에 게임 정보 추가 | ✅ "Crimson Desert, Pearl Abyss" 정확 식별 |
| 2 | STORM Worker 컨텍스트 초과 | (B) 프롬프트 비대 | KB context 800자 제한, findings 마지막만 전달 | ✅ Worker 정상 실행 |
| 3 | Sanitizer가 106개 에러 발견 | (D) 인용 폭발 | Synthesis에서 인용태그 제거 지시, Self-Correct 개선 | ✅ Sanitizer 0 errors |
| 4 | CoVe Verifier가 정확한 정보 부정 | (E) 검증 오류 | Factored Verifier에 term_context 전달 | ✅ 검증 정상 |
| 검증 | KB 없이도 동작하는지? | - | Starfield 주제로 no-KB 테스트 | ✅ Ground truth 정확, 오염 없음 |

### 3.3. 핵심 원칙

1. **세션이 디버그 로그다** — `sessions/{run_id}/`만 보고 문제를 진단할 수 있어야 한다.
   - `traces.json`의 `prompt_preview`/`response_preview`가 핵심
   - `failure.json`의 `traceback` + `phase` + `last_node`로 에러 위치 특정
   - `session.json`의 `completed_nodes`로 어디까지 성공했는지 확인

2. **한 번에 하나만 고친다** — 여러 문제가 보여도 가장 먼저 발생하는 문제부터 수정. 앞쪽 노드 문제가 뒤쪽을 연쇄적으로 망가뜨리기 때문.

3. **빗나갈 때는 빠르게 끊는다** — 파이프라인이 10분 이상 걸리는데 중간 Worker 출력이 이미 주제에서 벗어났으면, 완주를 기다리지 않고 프로세스를 중단하고 수정.

4. **Ground truth + Contamination 이중 검증** — "맞는 키워드가 있는가" + "틀린 키워드가 없는가"를 둘 다 확인.

5. **로컬 모델의 한계를 인정한다** — 9B 모델은 모르는 것이 많다. "모른다"고 답하게 하는 것이 추측보다 낫다. KB와 Term Resolver가 이 한계를 보완하는 역할.

---

## 4. 모듈 구조

```
src/
├── harness/
│   ├── base.py              # HarnessState, HarnessConfig
│   ├── models.py            # Pydantic 모델, FakeStructuredChatModel
│   ├── guardrails.py        # InputGuardrail, OutputGuardrail
│   ├── sanitizer.py         # LocalChunkDB, DeterministicSanitizer
│   ├── checkpointer.py      # JsonFileCheckpointer
│   ├── online_evaluator.py  # OnlineEvaluator
│   ├── self_improvement.py  # LocalFailureStore, SelfImprovementLoop
│   ├── lmstudio.py          # ThinkingModelWrapper (로컬 LLM)
│   ├── knowledge_base.py    # LocalKnowledgeBase, RetrievalSource
│   ├── web_search.py        # WebSearchSource (agent-browser CLI)
│   ├── config_store.py      # AppConfig JSON 영속 저장
│   ├── session_store.py     # 세션 디렉토리 관리
│   ├── tui.py               # Textual TUI (8 패널)
│   └── __main__.py          # py -m harness
└── architectures/
    └── zero_hallucination.py  # 파이프라인 (10 노드)
```

### 핵심 설계 결정

| 결정 | 이유 |
|------|------|
| 노드 팩토리 패턴 `make_xxx_node(model)` | 테스트 시 FakeModel 주입 가능 |
| Term Resolver → Planner 순서 | LLM이 모르는 용어를 먼저 확인해야 주제 이탈 방지 |
| STORM Worker에 `messages` + `term_context` 전달 | Send API는 state 일부만 전달하므로 명시적 전달 필요 |
| Synthesis에서 인용 태그 사용 금지 | 로컬 모델이 [출처:...] 형식을 대량 생성 → Sanitizer 폭발 방지 |
| KB context 800자 제한 | 로컬 9B 모델의 컨텍스트 윈도우 보호 |
| Structured output 2회 재시도 | 첫 시도에서 JSON 대신 설명을 출력하는 경우 보완 |
| Self-Correct: 10개 이상 에러 시 인용 일괄 제거 | 개별 수정보다 인용 제거 + 본문 보존이 품질 유지에 유리 |

---

## 5. TUI

```bash
py -m harness          # 또는 harness-tui
```

8개 패널, 키보드 1~8 + q:

| 키 | 패널 | 기능 |
|----|------|------|
| 1 | Dashboard | 통계, 파이프라인 다이어그램 |
| 2 | Research | 쿼리 입력, 소스 선택 (Local/Web), 실행, 실시간 로그 |
| 3 | Trace | 노드별 LLM 호출 상세 (prompt/response/role) |
| 4 | Checkpoints | 체크포인트 조회/삭제 |
| 5 | Metrics | 평가 점수 + 알림 이력 |
| 6 | Failures | 실패 이력 + 패턴 분석 |
| 7 | Guardrails | Input/Output 가드레일 테스트 |
| 8 | Config | 전체 설정 편집 → .harness_config.json 저장 |

- 모든 패널은 앱 시작 시 마운트, display 토글로 전환 (상태 유실 없음)
- 사이드바에 SessionBox — 다른 탭에서도 실행 상태 확인 가능
- 아키텍처 단계별 로그 그룹핑 (🔎 Term Resolution, 📋 Plan-and-Execute, 🌪 STORM, 🔍 CoVe, 🛡 Sanitizer)

---

## 6. 소스 플러그인 확장

`RetrievalSource` ABC를 구현하면 어떤 소스든 연결 가능:

```python
# 현재 구현
kb = LocalKnowledgeBase(directory="./data/sources")   # 로컬 파일
kb.add_source(WebSearchSource(max_results=3))          # agent-browser 웹 검색

# 향후 확장 예시
kb.add_source(VectorStoreSource(collection="research"))
kb.add_source(APISource(endpoint="..."))
```

웹 검색은 `agent-browser` CLI 필요:
```bash
npm install -g agent-browser && agent-browser install
```

---

## 7. 테스트 가이드

```bash
py -m pytest tests/ -v                    # 전체 (175개)
py -m pytest tests/ -m "not integration"  # 외부 의존 없는 것만
py -u scripts/test_run.py                 # 라이브 파이프라인 (LM Studio 필요)
py -u scripts/test_run_no_kb.py           # KB 없이 라이브 테스트
```

라이브 테스트 스크립트는 실행 후 자동으로:
- 각 노드 출력 미리보기
- 오염 키워드 체크 (PROBLEM_KEYWORDS)
- Ground truth 키워드 체크
- 세션 디렉토리 저장

---

## 8. 다음 작업 권장

| 우선순위 | 작업 | 비고 |
|----------|------|------|
| 높음 | Term Resolver에 웹 검색 연동 | KB 없는 미지의 주제에 대한 용어 확인 자동화 |
| 높음 | 선형 파이프라인 (Part II - 2.1) | 가장 단순한 아키텍처, 기반 검증 |
| 중간 | 오케스트레이터-워커 (Part II - 2.2) | Send API 패턴 재사용 가능 |
| 중간 | 오프라인 궤적 평가 (Part IV - 4.2) | 세션 traces.json 기반 자동 평가 |
| 낮음 | LATS 트리 탐색 (Part II - 2.4) | 복잡도 높음 |

---

## 9. 알려진 제한사항

1. **로컬 9B 모델 한계**: 학습 데이터에 없는 주제는 정보 없음. KB 또는 웹 검색으로 보완 필수.
2. **컨텍스트 윈도우**: 프롬프트 합계가 ~6000자를 넘으면 400 에러. KB/term_context/findings에 길이 제한 필요.
3. **파이프라인 소요 시간**: 로컬 9B 모델 기준 전체 파이프라인 15~20분. 모델 크기·하드웨어에 따라 달라짐.
4. **Sanitizer 엄격성**: 인용 태그를 사용하면 대량 reject 발생. 현재는 인용 없이 자연어 작성으로 우회.
5. **Send API 상태 전달**: `Send()`는 부모 state의 일부만 전달. 새 필드 추가 시 `dispatch_*` 함수에서 명시적 전달 필요.
