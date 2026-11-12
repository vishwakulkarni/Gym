# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recorded failures that have no recovery left settle as zeros by default."""

import orjson
import pytest

from nemo_gym.metrics.reward_profile import compute_aggregate_metrics
from nemo_gym.rollouts.rollout_collection import (
    RolloutAggregationConfig,
    RolloutAggregationHelper,
    RolloutCollectionConfig,
    RolloutCollectionHelper,
)
from nemo_gym.rollouts.rollout_recovery import RunManifest
from nemo_gym.rollouts.rollout_store import RolloutStore


SUCCESS = {"reward": 1.0, "response": {}}
FAIL = {"_ng_failure_class": "agent_run_error", "_ng_unrouted_run_error": True}
ZERO = {"_ng_failure_class": "agent_run_error", "_ng_failure_terminal": True, "_ng_failure_counts_as_zero": True}


def _run(tmp_path, outcomes):
    output = tmp_path / "rollouts.jsonl"
    rows = [
        {"_ng_task_index": i, "_ng_rollout_index": 0, "agent_ref": {"name": "agent"}} for i in range(len(outcomes))
    ]
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"".join(orjson.dumps(r) + b"\n" for r in rows))
    manifest = RunManifest.create(source, rows, {}, {"agent": {"responses_api_agents": {"impl": {}}}})
    with RolloutStore.start_or_resume(output, lambda: (rows, manifest), resume=False) as store:
        for row, outcome in zip(store.pending(), outcomes):
            if outcome == "missing":
                continue
            store.record_dispatch(row)
            if outcome != "live":
                store.record_outcome(row | outcome)
    return output


def _rejection(output):
    store = RolloutStore.read(output)
    c = store.coverage()
    return RolloutCollectionConfig(
        input_jsonl_fpath="unused",
        output_jsonl_fpath=str(output),
        require_complete=True,
    ).completion_rejection(
        expected=c["expected"],
        results=store.selected("success"),
        completed=c["successful"] - c["masked_retryable"] + c["final_zeros"],
        retryable=c["retryable"] > 0,
        made_progress=True,
    )


async def _aggregate(output, monkeypatch, **overrides):
    captured = []

    async def score(self, results, rows, destination, **kwargs):
        captured.extend(results)
        metrics = compute_aggregate_metrics(results)
        path = destination.with_name("rollouts_aggregate_metrics.json")
        path.write_bytes(orjson.dumps([{"agent_ref": {"name": "agent"}, **metrics.model_dump()}]))
        return path

    monkeypatch.setattr(RolloutCollectionHelper, "_call_aggregate_metrics", score)
    await RolloutAggregationHelper().run_from_config(
        RolloutAggregationConfig(
            input_glob=str(output),
            output_jsonl_fpath=str(output),
            merge_shards=False,
            disable_health_check=True,
            **overrides,
        )
    )
    return captured


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        FAIL,
        FAIL | {"failure_outcome_action": "fail"},
        {
            "_ng_failure_class": "rollout_allocation_ended",
            "_ng_failure_id": "unclassified",
            "_ng_failure_events": {},
        },
    ],
)
async def test_existing_non_retryable_failure_completes_and_enters_denominator(tmp_path, monkeypatch, failure):
    output = _run(tmp_path, [SUCCESS] * 338 + [failure])
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    assert _rejection(output) is None
    scored = await _aggregate(output, monkeypatch)
    assert len(scored) == 339
    assert sum(r["reward"] for r in scored) == 338
    assert compute_aggregate_metrics(scored).key_metrics["mean/reward"] == pytest.approx(338 / 339)
    c = RolloutStore.read(output).coverage()
    assert c["final_zeros"] == c["failed"] == 1 and c["complete"]
    assert c["stopped_not_retried"] == 0
    assert RolloutStore.read(output).pending() == []
    assert all(p.read_bytes() == contents for p, contents in before.items())
    assert "_ng_failure_counts_as_zero" not in RolloutStore.read(output).failures()[0]


@pytest.mark.parametrize("gap", ["missing", "live", {"_ng_failure_class": "provider_unavailable"}])
def test_retryable_missing_or_live_work_still_blocks_collection(tmp_path, gap):
    output = _run(tmp_path, [SUCCESS, FAIL, gap])
    assert _rejection(output) is not None
    assert RolloutStore.read(output).coverage()["final_zeros"] == 1


@pytest.mark.asyncio
async def test_all_settled_failures_produce_a_valid_zero_score(tmp_path, monkeypatch):
    output = _run(tmp_path, [FAIL, FAIL | {"reward": None}])
    assert _rejection(output) is None
    scored = await _aggregate(output, monkeypatch)
    assert [r["reward"] for r in scored] == [0.0, 0.0]
    assert compute_aggregate_metrics(scored).key_metrics["mean/reward"] == 0.0
    assert output.with_name("rollouts_aggregate_metrics.json").exists()


@pytest.mark.asyncio
async def test_zeros_are_not_duplicated_and_completed_grades_stay_intact(tmp_path, monkeypatch):
    output = _run(tmp_path, [FAIL | {"reward": 0.75, "evaluation_completed": True}, ZERO, FAIL])
    for _ in range(2):
        scored = await _aggregate(output, monkeypatch)
        assert sorted(r["reward"] for r in scored) == [0.0, 0.0, 0.75]
        assert RolloutStore.read(output).coverage()["final_zeros"] == 3
        assert _rejection(output) is None


@pytest.mark.asyncio
async def test_saved_retry_cap_settles_exhausted_failures(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "1")
    output = _run(tmp_path, [SUCCESS, {"_ng_failure_class": "provider_unavailable"}])
    monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "3")
    assert _rejection(output) is None
    scored = await _aggregate(output, monkeypatch)
    assert [r["reward"] for r in scored] == [1.0, 0.0]


@pytest.mark.parametrize("cap", [1, 3])
@pytest.mark.parametrize("answer_shape", ["typed", "legacy", "partial"])
def test_saved_judge_answer_is_not_zeroed_while_reverification_has_budget(tmp_path, monkeypatch, cap, answer_shape):
    monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", str(cap))
    judge = {
        "_ng_failure_class": "judge_failed",
        "response": {
            "id": "answer-1",
            "object": "response",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "answer"}]}],
        },
        "_ng_failure_id": "judge-1",
    }
    if answer_shape != "typed":
        judge["response"].pop("id")
        judge["response"].pop("object")
    if answer_shape == "partial":
        judge["_ng_failure_partial_response"] = judge.pop("response")
    output = _run(tmp_path, [SUCCESS, judge])
    store = RolloutStore.read(output)
    # Episode-budget extensions do not extend the separate saved-answer re-judgment cap.
    store._state.attempt_extension = 2
    assert store.coverage()["final_zeros"] == (1 if cap == 1 else 0)
    assert (_rejection(output) is None) == (cap == 1)


def test_unconfirmed_harness_termination_is_not_a_zero_even_at_the_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "1")
    output = _run(tmp_path, [SUCCESS, {"_ng_failure_class": "harness_termination_unconfirmed"}])
    assert RolloutStore.read(output).coverage()["final_zeros"] == 0
    assert _rejection(output) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["provider_credentials_rejected", "provider_quota_exhausted"])
async def test_provider_stops_are_not_scored_even_offline(tmp_path, monkeypatch, kind):
    output = _run(tmp_path, [SUCCESS, FAIL | {"failure_kind": kind}])
    assert _rejection(output) is not None
    assert RolloutStore.read(output).coverage()["final_zeros"] == 0
    assert [r["reward"] for r in await _aggregate(output, monkeypatch)] == [1.0]


@pytest.mark.asyncio
@pytest.mark.parametrize("cap", [1, 3])
async def test_infrastructure_timeout_cannot_keep_an_invalidated_grade(tmp_path, monkeypatch, cap):
    from tests.unit_tests.orchestration.test_recovery_infra_agent_timeout import LAST_503

    monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", str(cap))
    failure = {
        "reward": 1.0,
        "evaluation_completed": True,
        "failure_kind": "agent_timeout",
        "ng_model_call_capture": {"calls": LAST_503},
    }
    output = _run(tmp_path, [failure])
    assert [r["reward"] for r in await _aggregate(output, monkeypatch)] == ([0.0] if cap == 1 else [])
    assert bool(RolloutStore.read(output).pending()) == (cap == 3)
    assert RolloutStore.read(output).failures()[0]["reward"] == 1.0
