"""자기 개선 루프 하네스 — 10개 연구 문제를 점진적 난이도로 실행.

각 문제에 대해:
1. 파이프라인 실행
2. Ground Truth / Contamination 검증
3. 결과 분석 및 세션 저장
4. 실패 시 원인 분류 기록

사용:
    python -u scripts/self_improvement_harness.py [--query-id N] [--all]
"""
import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.messages import HumanMessage

from architectures.zero_hallucination import build_zero_hallucination_pipeline
from harness.base import HarnessConfig, create_initial_state
from harness.knowledge_base import LocalKnowledgeBase
from harness.lmstudio import ThinkingModelWrapper, create_lmstudio_llm
from harness.sanitizer import LocalChunkDB
from harness.session_store import SessionStore


# ─── 10개 연구 문제 정의 (점진적 난이도) ───


@dataclass
class ResearchQuery:
    id: int
    difficulty: str  # easy / easy-medium / medium / medium-hard / hard
    query: str
    ground_truth: list[str]  # 결과에 반드시 포함되어야 할 키워드
    contamination: list[str]  # 결과에 포함되면 안 되는 키워드
    kb_dir: str | None = None  # KB 디렉토리 (None이면 KB 없이 실행)
    description: str = ""


QUERIES = [
    # ── Level 1: Easy (LLM이 잘 아는 주제) ──
    ResearchQuery(
        id=1,
        difficulty="easy",
        query="RTX 4090으로 사이버펑크 2077(Cyberpunk 2077) 울트라 세팅 최적화 가이드 조사.",
        ground_truth=["Cyberpunk 2077", "CD Projekt"],
        contamination=["GTA", "Watch Dogs", "와치독스"],
        description="잘 알려진 게임의 그래픽 설정 가이드. LLM 내부 지식으로 충분.",
    ),
    ResearchQuery(
        id=2,
        difficulty="easy",
        query="PlayStation 5와 Xbox Series X의 하드웨어 성능 비교 분석 조사.",
        ground_truth=["PlayStation 5", "Xbox Series X"],
        contamination=["Nintendo Switch", "PS3", "Xbox 360"],
        description="유명 콘솔 비교. LLM이 정확히 알고 있는 주제.",
    ),

    # ── Level 2: Easy-Medium (구체적 세부사항 필요) ──
    ResearchQuery(
        id=3,
        difficulty="easy-medium",
        query="Tesla Model 3 Highland 2024년형 주요 변경사항과 개선점 조사.",
        ground_truth=["Model 3", "테슬라"],
        contamination=["Model S", "Model X", "Rivian"],
        description="특정 모델의 연식 변경. 세부 사양 정확도가 중요.",
    ),
    ResearchQuery(
        id=4,
        difficulty="easy-medium",
        query="Python 3.12의 주요 신기능과 성능 개선사항 조사.",
        ground_truth=["Python 3.12"],
        contamination=["Java", "JavaScript", "Rust", "C++"],
        description="프로그래밍 언어 버전 업데이트. 정확한 기능 나열 필요.",
    ),

    # ── Level 3: Medium (주의 깊은 처리 필요) ──
    ResearchQuery(
        id=5,
        difficulty="medium",
        query="Apple Vision Pro의 공간 컴퓨팅 핵심 기술과 하드웨어 스펙 분석 조사.",
        ground_truth=["Vision Pro", "Apple"],
        contamination=["Meta Quest", "Oculus", "HoloLens"],
        description="비교적 최신 제품. 경쟁 제품과의 혼동 가능성.",
    ),
    ResearchQuery(
        id=6,
        difficulty="medium",
        query="SpaceX Starship 발사체의 재사용 기술 혁신과 발사 이력 조사.",
        ground_truth=["Starship", "SpaceX"],
        contamination=["Falcon 9", "Blue Origin", "SLS"],
        description="특정 발사체에 집중. 다른 로켓과 혼동 가능.",
    ),

    # ── Level 4: Medium-Hard (주제 이탈 가능성 높음) ──
    ResearchQuery(
        id=7,
        difficulty="medium-hard",
        query="RTX 3080으로 붉은사막(Crimson Desert) 최적화 옵션 셋팅 가이드 조사.",
        ground_truth=["Crimson Desert", "붉은 사막", "Pearl Abyss"],
        contamination=["Red Dead", "RDR2", "레드 데드", "레디뎀"],
        kb_dir="data/crimson_desert",
        description="LLM이 모르는 게임. KB 필수. 주제 이탈(→RDR2) 위험 높음.",
    ),
    ResearchQuery(
        id=8,
        difficulty="medium-hard",
        query="한국형 발사체 누리호(KSLV-II) 3차 발사 성과와 기술적 의의 조사.",
        ground_truth=["누리호", "KSLV-II"],
        contamination=[],  # 나로호 언급은 문맥상 정당 (역사적 비교)
        kb_dir="data/nuri_rocket",
        description="한국 우주 발사체. KB로 정확한 정보 제공. 나로호와 혼동 주의.",
    ),

    # ── Level 5: Hard (매우 구체적/생소한 주제) ──
    ResearchQuery(
        id=9,
        difficulty="hard",
        query="Project Cambria(현 Meta Quest Pro)의 XR 칩셋과 트래킹 기술 분석 조사.",
        ground_truth=["Quest Pro", "Meta", "Snapdragon XR2"],
        contamination=["Quest 2", "Quest 3", "Vision Pro"],
        kb_dir="data/quest_pro",
        description="코드명으로 질의. KB 없이는 정체 파악 불가. 다른 제품 혼동 위험.",
    ),
    ResearchQuery(
        id=10,
        difficulty="hard",
        query="중국 RISC-V 프로세서 Xuantie C910(玄铁)의 아키텍처와 기술적 의의 조사.",
        ground_truth=["RISC-V", "Xuantie", "C910", "T-Head"],
        contamination=["ARM", "x86", "Intel", "AMD"],
        kb_dir="data/xuantie_c910",
        description="매우 구체적 반도체 주제. KB 필수. ARM/x86 혼동 위험.",
    ),
]


# ─── 결과 분석 ───


@dataclass
class RunResult:
    query_id: int
    run_id: str
    cycle: int
    status: str  # "success" | "fail_gt" | "fail_contamination" | "fail_both" | "error"
    elapsed: float
    step_count: int
    gt_found: list[str]
    gt_missing: list[str]
    contamination_found: list[str]
    sanitizer_errors: int
    final_length: int
    error_msg: str = ""
    cause_category: str = ""  # A~F


def classify_failure(result: RunResult, node_outputs: dict) -> str:
    """실패 원인을 A~F로 분류한다."""
    if result.status == "error":
        return "(F) 외부 오류"
    if result.contamination_found:
        return "(A) 주제 이탈"
    if result.sanitizer_errors > 10:
        return "(D) 인용 폭발"
    if result.sanitizer_errors > 0:
        return "(D) Sanitizer 에러"
    if result.gt_missing:
        # GT 키워드가 없다면 → 주제를 아예 잘못 잡았을 수 있음
        draft = node_outputs.get("draft", "")
        if len(draft) < 50:
            return "(B) 컨텍스트 초과 또는 빈 응답"
        return "(A) 주제 이탈 또는 정보 부족"
    return ""


# ─── 파이프라인 실행 ───


def run_single_query(
    query: ResearchQuery,
    cycle: int,
    model: ThinkingModelWrapper,
) -> RunResult:
    """단일 쿼리를 파이프라인에 통과시키고 결과를 분석한다."""
    run_id = f"sil_q{query.id}_c{cycle}_{int(time.time())}"
    print(f"\n{'='*70}")
    print(f"[Query {query.id}] Cycle {cycle} | {query.difficulty.upper()}")
    print(f"  Q: {query.query}")
    print(f"  KB: {query.kb_dir or 'NONE'}")
    print(f"  Run: {run_id}")
    print(f"{'='*70}")

    # KB 로딩
    kb = None
    chunk_db = LocalChunkDB()
    if query.kb_dir:
        kb = LocalKnowledgeBase(directory=query.kb_dir)
        stats = kb.load()
        chunk_db = kb.chunk_db
        print(f"  KB loaded: {stats}")

    # 파이프라인 빌드
    config = HarnessConfig()
    graph = build_zero_hallucination_pipeline(
        planner_model=model,
        worker_model=model,
        verifier_model=model,
        synthesizer_model=model,
        chunk_db=chunk_db,
        config=config,
        knowledge_base=kb,
    )
    compiled = graph.compile()

    # 실행
    initial = create_initial_state()
    initial["messages"] = [HumanMessage(content=query.query)]

    store = SessionStore(run_id)
    start = time.time()
    result_state = {}
    step = 0
    completed_nodes = []

    try:
        for event in compiled.stream(initial, stream_mode="updates"):
            step += 1
            elapsed = time.time() - start
            for node_name, node_output in event.items():
                completed_nodes.append(node_name)
                tag = ""

                if node_name == "term_resolver":
                    tc = node_output.get("term_context", "")
                    tag = f"term_context={len(tc)}자"
                elif node_name == "planner":
                    plan = node_output.get("plan", [])
                    tag = f"tasks={len(plan)}"
                elif node_name == "storm_worker":
                    wr = node_output.get("worker_results", [])
                    if wr:
                        tag = f"{wr[0].get('task_id')}: {len(wr[0].get('data', ''))}자"
                elif node_name == "synthesis":
                    draft = node_output.get("draft", "")
                    tag = f"draft={len(draft)}자"
                elif node_name == "sanitizer":
                    errors = node_output.get("sanitizer_errors", [])
                    tag = f"errors={len(errors)}"
                elif node_name == "finalize":
                    msgs = node_output.get("messages", [])
                    if msgs:
                        c = msgs[-1].content if hasattr(msgs[-1], "content") else ""
                        tag = f"final={len(c)}자"

                print(f"  [{elapsed:5.0f}s] ▸ {node_name} {tag}")
                result_state.update(node_output)

    except Exception as e:
        elapsed = time.time() - start
        tb = traceback.format_exc()
        print(f"\n  ❌ ERROR: {e}")
        store.save_failure(
            [str(e)], {"query": query.query}, tb,
            phase="streaming", last_node=completed_nodes[-1] if completed_nodes else "",
        )
        store.save_session_meta(
            query.query, "error", start, elapsed, step,
            error_msg=str(e), last_node=completed_nodes[-1] if completed_nodes else "",
            completed_nodes=completed_nodes, phase="streaming",
        )
        return RunResult(
            query_id=query.id, run_id=run_id, cycle=cycle,
            status="error", elapsed=elapsed, step_count=step,
            gt_found=[], gt_missing=query.ground_truth,
            contamination_found=[], sanitizer_errors=0,
            final_length=0, error_msg=str(e),
            cause_category="(F) 외부 오류",
        )

    elapsed = time.time() - start

    # 최종 결과 추출
    msgs = result_state.get("messages", [])
    final = ""
    if msgs:
        final = msgs[-1].content if hasattr(msgs[-1], "content") else str(msgs[-1])

    # Ground Truth 검증
    gt_found = [kw for kw in query.ground_truth if kw in final]
    gt_missing = [kw for kw in query.ground_truth if kw not in final]

    # Contamination 검증
    contamination_found = [kw for kw in query.contamination if kw in final]

    # Sanitizer 에러 수
    sanitizer_errors = len(result_state.get("sanitizer_errors", []))

    # 종합 상태
    if gt_missing and contamination_found:
        status = "fail_both"
    elif gt_missing:
        status = "fail_gt"
    elif contamination_found:
        status = "fail_contamination"
    else:
        status = "success"

    run_result = RunResult(
        query_id=query.id, run_id=run_id, cycle=cycle,
        status=status, elapsed=elapsed, step_count=step,
        gt_found=gt_found, gt_missing=gt_missing,
        contamination_found=contamination_found,
        sanitizer_errors=sanitizer_errors,
        final_length=len(final),
    )
    run_result.cause_category = classify_failure(run_result, result_state)

    # 세션 저장
    store.save_session_meta(
        query.query, status, start, elapsed, step,
        last_node=completed_nodes[-1] if completed_nodes else "",
        completed_nodes=completed_nodes, phase="complete",
    )
    store.save_checkpoint(result_state)
    if final:
        store.save_result(final)
    store.save_evaluation({
        "error_free": 1.0 if not result_state.get("error_log") else 0.0,
        "has_response": 1.0 if final else 0.0,
        "ground_truth_score": len(gt_found) / max(len(query.ground_truth), 1),
        "contamination_free": 1.0 if not contamination_found else 0.0,
        "sanitizer_clean": 1.0 if sanitizer_errors == 0 else 0.0,
    })

    # 결과 출력
    print(f"\n  ── Result ──")
    print(f"  Status: {status}")
    print(f"  Elapsed: {elapsed:.0f}s ({step} steps)")
    print(f"  Final: {len(final)}자")
    if gt_found:
        print(f"  ✅ GT found: {gt_found}")
    if gt_missing:
        print(f"  ❌ GT missing: {gt_missing}")
    if contamination_found:
        print(f"  ❌ Contamination: {contamination_found}")
    else:
        print(f"  ✅ No contamination")
    if sanitizer_errors:
        print(f"  ⚠ Sanitizer errors: {sanitizer_errors}")
    if run_result.cause_category:
        print(f"  Cause: {run_result.cause_category}")
    print(f"  Session: sessions/{run_id}/")

    return run_result


# ─── 메인 루프 ───


def run_self_improvement_loop(
    query_ids: list[int] | None = None,
    max_cycles: int = 3,
) -> list[RunResult]:
    """자기 개선 루프를 실행한다.

    각 쿼리에 대해 max_cycles 회까지 반복하되, 성공하면 다음으로 넘어간다.
    """
    queries = QUERIES
    if query_ids:
        queries = [q for q in QUERIES if q.id in query_ids]

    print(f"╔{'═'*68}╗")
    print(f"║ Self-Improvement Loop Harness                                      ║")
    print(f"║ Queries: {len(queries)}, Max cycles per query: {max_cycles}                           ║")
    print(f"╚{'═'*68}╝")

    # LLM 연결
    print("\nConnecting to LM Studio...")
    llm = create_lmstudio_llm()
    model = ThinkingModelWrapper(llm)
    print("Connected.\n")

    all_results: list[RunResult] = []
    summary: dict[int, list[RunResult]] = {}

    for query in queries:
        summary[query.id] = []
        for cycle in range(max_cycles):
            result = run_single_query(query, cycle, model)
            all_results.append(result)
            summary[query.id].append(result)

            if result.status == "success":
                print(f"\n  ✅ Query {query.id} PASSED on cycle {cycle}")
                break
            else:
                print(f"\n  ↻ Query {query.id} needs improvement (cycle {cycle})")
                if cycle < max_cycles - 1:
                    print(f"  → Retrying... (cycle {cycle + 1})")

    # ── 최종 리포트 ──
    print(f"\n\n{'='*70}")
    print(f" SELF-IMPROVEMENT LOOP REPORT")
    print(f"{'='*70}")

    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_queries": len(queries),
        "total_runs": len(all_results),
        "results": [],
    }

    passed = 0
    for query in queries:
        results = summary.get(query.id, [])
        final_result = results[-1] if results else None
        cycles_used = len(results)
        final_status = final_result.status if final_result else "not_run"

        if final_status == "success":
            passed += 1
            icon = "✅"
        else:
            icon = "❌"

        print(f"\n  {icon} Q{query.id} [{query.difficulty}] — {final_status}")
        print(f"     Cycles: {cycles_used}, Final elapsed: {final_result.elapsed:.0f}s" if final_result else "")
        if final_result and final_result.gt_missing:
            print(f"     Missing GT: {final_result.gt_missing}")
        if final_result and final_result.contamination_found:
            print(f"     Contamination: {final_result.contamination_found}")
        if final_result and final_result.cause_category:
            print(f"     Cause: {final_result.cause_category}")

        report_data["results"].append({
            "query_id": query.id,
            "difficulty": query.difficulty,
            "query": query.query,
            "cycles_used": cycles_used,
            "final_status": final_status,
            "final_elapsed": final_result.elapsed if final_result else 0,
            "gt_found": final_result.gt_found if final_result else [],
            "gt_missing": final_result.gt_missing if final_result else [],
            "contamination": final_result.contamination_found if final_result else [],
            "cause": final_result.cause_category if final_result else "",
            "run_ids": [r.run_id for r in results],
        })

    print(f"\n  ── Summary ──")
    print(f"  Passed: {passed}/{len(queries)}")
    print(f"  Total runs: {len(all_results)}")
    total_time = sum(r.elapsed for r in all_results)
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f}min)")

    report_data["passed"] = passed
    report_data["total_time"] = total_time

    # 리포트 저장
    report_path = Path("metrics") / f"sil_report_{int(time.time())}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n  Report saved: {report_path}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--query-id", type=int, nargs="+", help="특정 쿼리만 실행")
    parser.add_argument("--max-cycles", type=int, default=3, help="쿼리당 최대 사이클 수")
    parser.add_argument("--all", action="store_true", help="전체 10개 실행")
    args = parser.parse_args()

    if args.all:
        run_self_improvement_loop(max_cycles=args.max_cycles)
    elif args.query_id:
        run_self_improvement_loop(query_ids=args.query_id, max_cycles=args.max_cycles)
    else:
        # 기본: 전체 실행
        run_self_improvement_loop(max_cycles=args.max_cycles)
