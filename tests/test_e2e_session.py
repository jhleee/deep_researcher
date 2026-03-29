"""세션 디렉토리 E2E 테스트 — FakeModel로 전체 파이프라인을 돌리고
세션 디렉토리에 모든 파일이 올바르게 저장되는지 검증한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from architectures.zero_hallucination import build_zero_hallucination_pipeline
from harness.base import HarnessConfig, create_initial_state
from harness.config_store import AppConfig
from harness.knowledge_base import LocalKnowledgeBase
from harness.models import FakeStructuredChatModel
from harness.online_evaluator import OnlineEvaluator
from harness.sanitizer import LocalChunkDB
from harness.session_store import SessionStore
from harness.tui import LLMCall, NodeTrace, RunSession


@pytest.fixture
def sample_data_dir(tmp_path):
    """샘플 자료 디렉토리 생성."""
    d = tmp_path / "sources"
    d.mkdir()
    (d / "ev_market.md").write_text(
        "# 전기차 시장\n\n"
        "2025년 글로벌 전기차 시장 규모는 약 5,000억 달러이다.\n\n"
        "BYD가 302만 대로 1위, Tesla가 179만 대로 2위이다.\n\n"
        "LFP 배터리 점유율은 40%이다.",
        encoding="utf-8",
    )
    (d / "ev_policy.md").write_text(
        "# 전기차 정책\n\n"
        "EU는 2035년부터 내연기관 신차 판매를 금지한다.\n\n"
        "미국 IRA에 따른 최대 $7,500 세액공제가 적용된다.\n\n"
        "한국의 EV 보조금은 최대 680만 원이다.",
        encoding="utf-8",
    )
    return d


def _make_fake_model(responses: list[str]) -> FakeStructuredChatModel:
    """FakeModel 생성 헬퍼."""
    return FakeStructuredChatModel(responses=responses)


class TestE2ESessionDirectory:
    """FakeModel 기반 전체 파이프라인 + 세션 저장 E2E 테스트."""

    def test_full_pipeline_creates_session_dir(self, tmp_path, sample_data_dir):
        run_id = "run_e2e_test"
        sessions_dir = tmp_path / "sessions"

        # 1. 지식 베이스 로딩
        kb = LocalKnowledgeBase(directory=str(sample_data_dir))
        stats = kb.load()
        assert stats["files"] == 2
        assert stats["chunks"] >= 2

        # 2. FakeModel 응답 준비
        plan_json = json.dumps({
            "goal": "전기차 시장 분석",
            "sub_tasks": [
                {"id": "t1", "description": "시장 규모 조사", "dependencies": []},
                {"id": "t2", "description": "정책 비교 분석", "dependencies": []},
            ],
        })
        responses = [
            plan_json,                    # planner
            "BYD 302만 대 판매 1위",      # storm worker t1 - persona 1
            "Tesla 179만 대 2위",          # storm worker t1 - persona 2
            "LFP 배터리 40% 점유",         # storm worker t1 - persona 3
            "시장 5000억달러 BYD 1위",     # storm worker t1 - compress
            "EU 2035 내연기관 금지",        # storm worker t2 - persona 1
            "미국 IRA 7500달러 공제",       # storm worker t2 - persona 2
            "한국 680만원 보조금",           # storm worker t2 - persona 3
            "EU 2035금지 미국IRA 한국보조금",  # storm worker t2 - compress
            "전기차 시장 종합 보고서 초안",    # synthesis
            "BYD 판매량 검증?",             # cove_plan
            "BYD 302만 대 확인",            # factored verifier
            "검증 완료된 초안",              # cross_check
        ]
        model = _make_fake_model(responses)

        # 3. 파이프라인 빌드 + 실행
        config = HarnessConfig(guardrail_strict=True)
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

        initial = create_initial_state()
        initial["messages"] = [HumanMessage(content="전기차 시장 동향을 분석해주세요")]

        result = {}
        for event in compiled.stream(initial, stream_mode="updates"):
            for node_name, node_output in event.items():
                result.update(node_output)

        # 4. 세션 저장
        store = SessionStore(run_id, base_dir=str(sessions_dir))
        store.save_config(AppConfig())
        store.save_input_guardrail(True, "", "전기차 시장 동향을 분석해주세요", "전기차 시장 동향을 분석해주세요")
        store.save_knowledge_base(stats, ["Local(2)"])

        # 간단한 트레이스 기록
        traces = [
            NodeTrace("planner", "1/7", "계획", 1.0, "2개 작업",
                       llm_calls=[LLMCall(1, "planner", "prompt", plan_json[:200], 0.5, "100→200")]),
            NodeTrace("storm_worker", "2/7", "탐색", 5.0, "t1 수집",
                       llm_calls=[LLMCall(2, "storm_worker", "p1", "r1", 1.0, "50→30")]),
            NodeTrace("synthesis", "3/7", "합성", 7.0, "초안 생성"),
        ]
        all_calls = [c for t in traces for c in t.llm_calls]
        store.save_traces(traces, all_calls)

        store.save_checkpoint(result)

        evaluator = OnlineEvaluator(metrics_dir=str(tmp_path / "metrics"))
        scores = evaluator.evaluate_run(result, run_id)
        store.save_evaluation(scores)

        final_text = result.get("messages", [{}])[-1]
        final_content = final_text.content if hasattr(final_text, "content") else str(final_text)
        store.save_output_guardrail(True, "", final_content)
        store.save_result(final_content)
        store.save_session_meta(
            "전기차 시장 동향을 분석해주세요", "done", 1000.0, 10.0, 7,
        )

        # 5. 검증 — 모든 파일 존재
        session_dir = sessions_dir / run_id
        expected = [
            "session.json",
            "config.json",
            "input_guardrail.json",
            "knowledge_base.json",
            "traces.json",
            "checkpoint.json",
            "evaluation.json",
            "output_guardrail.json",
            "result.md",
        ]
        for f in expected:
            assert (session_dir / f).exists(), f"Missing: {f}"

        # 6. 검증 — 내용 정합성
        session_meta = json.loads(
            (session_dir / "session.json").read_text(encoding="utf-8")
        )
        assert session_meta["status"] == "done"
        assert session_meta["step_count"] == 7

        traces_data = json.loads(
            (session_dir / "traces.json").read_text(encoding="utf-8")
        )
        assert traces_data["total_nodes"] == 3
        assert traces_data["total_llm_calls"] == 2

        kb_data = json.loads(
            (session_dir / "knowledge_base.json").read_text(encoding="utf-8")
        )
        assert kb_data["stats"]["files"] == 2

        eval_data = json.loads(
            (session_dir / "evaluation.json").read_text(encoding="utf-8")
        )
        assert "error_free" in eval_data["scores"]

        result_text = (session_dir / "result.md").read_text(encoding="utf-8")
        assert len(result_text) > 0

    def test_session_list_after_runs(self, tmp_path):
        """여러 세션 실행 후 list_sessions가 정상 동작하는지 확인."""
        base = str(tmp_path / "sessions")
        for i in range(3):
            s = SessionStore(f"run_{i}", base_dir=base)
            s.save_session_meta(f"query {i}", "done", 1000.0 + i, float(i), i)

        sessions = SessionStore.list_sessions(base)
        assert len(sessions) == 3
        assert all("run_id" in s for s in sessions)
