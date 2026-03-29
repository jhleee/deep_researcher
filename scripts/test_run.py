"""라이브 파이프라인 테스트 스크립트.

LM Studio + 지식 베이스로 "붉은 사막" 주제 파이프라인을 실행하고
각 노드의 핵심 출력을 출력한다.
"""
import json
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.messages import HumanMessage

from architectures.zero_hallucination import build_zero_hallucination_pipeline
from harness.base import HarnessConfig, create_initial_state
from harness.knowledge_base import LocalKnowledgeBase
from harness.lmstudio import ThinkingModelWrapper, create_lmstudio_llm
from harness.session_store import SessionStore

QUERY = "rtx3080의 붉은사막 최적화 옵션 셋팅 가이드 조사."
KB_DIR = "data/crimson_desert"
RUN_ID = f"test_{int(time.time())}"

print(f"=== Run {RUN_ID} ===")
print(f"Query: {QUERY}")
print(f"KB: {KB_DIR}")
print()

# 1. KB 로딩
kb = LocalKnowledgeBase(directory=KB_DIR)
stats = kb.load()
print(f"KB loaded: {stats}")
print()

# 2. LLM 연결
llm = create_lmstudio_llm()
model = ThinkingModelWrapper(llm)

# 3. 파이프라인 빌드
config = HarnessConfig()
graph = build_zero_hallucination_pipeline(
    planner_model=model,
    worker_model=model,
    verifier_model=model,
    synthesizer_model=model,
    chunk_db=kb.chunk_db,
    config=config,
    knowledge_base=kb,
)
compiled = graph.compile()

# 4. 실행
initial = create_initial_state()
initial["messages"] = [HumanMessage(content=QUERY)]

store = SessionStore(RUN_ID)
start = time.time()
result = {}
step = 0

PROBLEM_KEYWORDS = ["Red Dead", "RDR2", "레드 데드", "레디뎀"]

for event in compiled.stream(initial, stream_mode="updates"):
    step += 1
    elapsed = time.time() - start
    for node_name, node_output in event.items():
        print(f"[{elapsed:.0f}s] ▸ {node_name}")

        # 핵심 출력 미리보기
        if node_name == "term_resolver":
            tc = node_output.get("term_context", "")
            print(f"  term_context ({len(tc)}자):")
            for line in tc[:500].split("\n"):
                print(f"    {line}")
            # RDR2 오염 체크
            for kw in PROBLEM_KEYWORDS:
                if kw in tc:
                    print(f"  ⚠ WARNING: '{kw}' found in term_context!")
        elif node_name == "planner":
            plan = node_output.get("plan", [])
            print(f"  tasks: {len(plan)}")
            for t in plan:
                print(f"    - {t.get('id')}: {t.get('description', '')[:80]}")
        elif node_name == "storm_worker":
            wr = node_output.get("worker_results", [])
            if wr:
                data = wr[0].get("data", "")
                print(f"  {wr[0].get('task_id')}: {data[:200]}")
                for kw in PROBLEM_KEYWORDS:
                    if kw in data:
                        print(f"  ⚠ WARNING: '{kw}' found in worker output!")
        elif node_name == "synthesis":
            draft = node_output.get("draft", "")
            print(f"  draft ({len(draft)}자): {draft[:200]}")
        elif node_name == "sanitizer":
            errors = node_output.get("sanitizer_errors", [])
            print(f"  errors: {len(errors)}")
        elif node_name == "finalize":
            msgs = node_output.get("messages", [])
            if msgs:
                c = msgs[-1].content if hasattr(msgs[-1], "content") else str(msgs[-1])
                print(f"  final ({len(c)}자)")

        result.update(node_output)
        print()

elapsed = time.time() - start
print(f"=== Done in {elapsed:.0f}s, {step} steps ===")
print()

# 최종 결과 출력
msgs = result.get("messages", [])
if msgs:
    final = msgs[-1].content if hasattr(msgs[-1], "content") else str(msgs[-1])
    print("=== FINAL RESULT ===")
    print(final)

    # RDR2 오염 체크
    print()
    clean = True
    for kw in PROBLEM_KEYWORDS:
        if kw in final:
            print(f"❌ FAIL: '{kw}' found in final result!")
            clean = False
    if clean:
        print("✅ PASS: No RDR2 contamination in final result")

    store.save_result(final)

store.save_session_meta(QUERY, "done", start, elapsed, step)
store.save_checkpoint(result)
print(f"\nSession saved: sessions/{RUN_ID}/")
