# LangGraph Agent Harness

## 프로젝트 개요

LangGraph 기반 에이전트 하네스 프로덕션 구현. 5대 아키텍처 패턴(선형 파이프라인, 오케스트레이터-워커, 다중 에이전트 감독관, LATS 트리 탐색, 딥 리서치)과 자율 E2E 테스트 프레임워크를 포함한다.

**현재 단계:** 로컬/파일시스템 기반 초기 구현 (외부 시스템 없음)

## 빌드 & 테스트 명령어

```bash
# 설치
pip install -e ".[test]"

# 전체 테스트
py -m pytest tests/ -v

# 특정 모듈 테스트
py -m pytest tests/test_guardrails.py -v

# 커버리지 포함
py -m pytest tests/ --cov=harness --cov-report=term-missing

# 린트
ruff check src/ tests/
```

## 프로젝트 구조

```
src/
├── harness/           # 공통 기반 모듈
│   ├── base.py        # HarnessState, HarnessConfig, create_initial_state
│   ├── checkpointer.py    # 체크포인터 팩토리 (MemorySaver, JsonFileCheckpointer)
│   ├── guardrails.py      # 입출력 가드레일 (PII, 인젝션, 유해성)
│   ├── sanitizer.py       # 결정론적 검증기 (인용, URL, 수치 대조)
│   ├── online_evaluator.py    # 로컬 파일 기반 온라인 평가기
│   └── self_improvement.py    # 자가 개선 루프 (실패 수집/분석/회귀 데이터셋)
└── architectures/     # 아키텍처 구현 (향후)
    ├── linear_pipeline.py
    ├── orchestrator_worker.py
    ├── multi_agent_supervisor.py
    ├── lats_search.py
    └── deep_research.py
tests/
├── conftest.py        # 공통 픽스처
├── test_*.py          # 모듈별 유닛 테스트
└── datasets/          # 테스트 데이터셋 (JSON)
```

## 코딩 규칙

- **Python >= 3.11**, 타입 힌트 필수
- Pydantic v2 모델 사용 (스키마 정의)
- `ruff`로 린트 (line-length=100)
- 모든 public 함수/클래스에 docstring (한국어 가능)
- 외부 시스템 의존 금지 (현재 단계): PostgreSQL, LangSmith, 외부 API 직접 호출 없이 로컬 대체재 사용

## 테스트 규칙

- **모든 새 기능은 반드시 유닛 테스트와 함께 커밋한다**
- 테스트 파일명: `test_{모듈명}.py`
- 테스트 클래스명: `Test{클래스명}`
- 픽스처는 `conftest.py`에 정의
- 외부 의존 없는 순수 유닛 테스트가 기본
- `tmp_path` 픽스처로 파일 I/O 테스트 격리
- 테스트 실행 후 전체 통과 확인 후에만 커밋

## 커밋 규칙

- 기능 단위로 커밋 (하나의 기능 + 해당 테스트)
- 커밋 메시지 형식: `<type>: <description>`
  - feat: 새 기능
  - fix: 버그 수정
  - test: 테스트 추가/수정
  - refactor: 리팩토링
  - docs: 문서
  - chore: 빌드/설정

## 아키텍처 설계 원칙

| 원칙 | 구현 |
|------|------|
| 결정론적 제어 | conditional_edge + 가드레일 노드 |
| 상태 영속성 | MemorySaver / JsonFileCheckpointer (로컬) |
| 보안 격리 | 에이전트별 최소 권한 도구 세트 |
| 관측성 | 로컬 JSON 메트릭 파일 |
| 자기 교정 | 실패 궤적 수집 → 회귀 데이터셋 자동 축적 |

## 체크포인터 백엔드

| 백엔드 | 용도 | 의존성 |
|--------|------|--------|
| `memory` (기본) | 개발/테스트 | 없음 |
| `json_file` | 로컬 영속 | 없음 |
| `sqlite` | 중간 규모 | langgraph-checkpoint-sqlite |
| `postgres` | 프로덕션 | langgraph-checkpoint-postgres, asyncpg |

## 향후 구현 순서 (권장)

1. ~~공통 기반 (Part I)~~ ✅
2. 선형 파이프라인 아키텍처 (Part II - 2.1)
3. 오케스트레이터-워커 아키텍처 (Part II - 2.2)
4. 딥 리서치 아키텍처 (Part II - 2.5)
5. 무결성 파이프라인 CoVe+Sanitizer (Part III)
6. 오프라인 평가 하네스 (Part IV)
