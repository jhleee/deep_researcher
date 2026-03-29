"""KB 없이 LLM 내부 지식만으로 파이프라인 테스트."""
import json
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.messages import HumanMessage

from architectures.zero_hallucination import build_zero_hallucination_pipeline
from harness.base import HarnessConfig, create_initial_state
from harness.lmstudio import ThinkingModelWrapper, create_lmstudio_llm
from harness.sanitizer import LocalChunkDB
from harness.session_store import SessionStore

QUERY = "RTX 4090으로 스타필드(Starfield) 울트라 세팅 가이드 조사."
RUN_ID = f"test_nokb_{int(time.time())}"

print(f"=== Run {RUN_ID} ===")
print(f"Query: {QUERY}")
print(f"KB: NONE (LLM internal knowledge only)")
print()

llm = create_lmstudio_llm()
model = ThinkingModelWrapper(llm)

config = HarnessConfig()
chunk_db = LocalChunkDB()
graph = build_zero_hallucination_pipeline(
    planner_model=model,
    worker_model=model,
    verifier_model=model,
    synthesizer_model=model,
    chunk_db=chunk_db,
    config=config,
)
compiled = graph.compile()

initial = create_initial_state()
initial["messages"] = [HumanMessage(content=QUERY)]

store = SessionStore(RUN_ID)
start = time.time()
result = {}
step = 0

GROUND_TRUTH_KEYWORDS = ["Starfield", "스타필드", "Bethesda", "베데스다"]
CONTAMINATION_KEYWORDS = ["Skyrim", "Fallout", "엘더스크롤"]

for event in compiled.stream(initial, stream_mode="updates"):
    step += 1
    elapsed = time.time() - start
    for node_name, node_output in event.items():
        print(f"[{elapsed:.0f}s] ▸ {node_name}")

        if node_name == "term_resolver":
            tc = node_output.get("term_context", "")
            print(f"  {tc[:300]}")
        elif node_name == "planner":
            plan = node_output.get("plan", [])
            print(f"  tasks: {len(plan)}")
            for t in plan:
                print(f"    - {t.get('id')}: {t.get('description', '')[:80]}")
        elif node_name == "storm_worker":
            wr = node_output.get("worker_results", [])
            if wr:
                print(f"  {wr[0].get('task_id')}: {wr[0].get('data', '')[:200]}")
        elif node_name == "synthesis":
            draft = node_output.get("draft", "")
            print(f"  draft ({len(draft)}자): {draft[:200]}")
        elif node_name == "sanitizer":
            print(f"  errors: {len(node_output.get('sanitizer_errors', []))}")
        elif node_name == "finalize":
            msgs = node_output.get("messages", [])
            if msgs:
                c = msgs[-1].content if hasattr(msgs[-1], "content") else ""
                print(f"  final ({len(c)}자)")

        result.update(node_output)
        print()

elapsed = time.time() - start
print(f"=== Done in {elapsed:.0f}s, {step} steps ===\n")

msgs = result.get("messages", [])
if msgs:
    final = msgs[-1].content if hasattr(msgs[-1], "content") else str(msgs[-1])
    print("=== FINAL RESULT ===")
    print(final[:1500])
    print()

    # Ground truth check
    gt_found = [kw for kw in GROUND_TRUTH_KEYWORDS if kw in final]
    contamination = [kw for kw in CONTAMINATION_KEYWORDS if kw in final]

    if gt_found:
        print(f"✅ Ground truth keywords found: {gt_found}")
    else:
        print("❌ No ground truth keywords in result!")

    if contamination:
        print(f"❌ Contamination found: {contamination}")
    else:
        print("✅ No contamination")

    store.save_result(final)

store.save_session_meta(QUERY, "done", start, elapsed, step)
store.save_checkpoint(result)
print(f"\nSession saved: sessions/{RUN_ID}/")
