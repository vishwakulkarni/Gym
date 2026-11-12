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
        completed=c["successful"] - c["masked_retryable"] + c["final_zeros"] + c.get("failed_excluded", 0),
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
    output = _run(tmp_path, [SUCCESS | {"reward": 0.75}, ZERO, FAIL])
    for _ in range(2):
        scored = await _aggregate(output, monkeypatch)
        assert sorted(r["reward"] for r in scored) == [0.0, 0.0, 0.75]
        assert RolloutStore.read(output).coverage()["final_zeros"] == 2
        assert _rejection(output) is None


@pytest.mark.asyncio
async def test_saved_retry_cap_settles_exhausted_failures(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "1")
    output = _run(tmp_path, [SUCCESS, {"_ng_failure_class": "provider_unavailable"}])
    monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "3")
    assert _rejection(output) is None
    scored = await _aggregate(output, monkeypatch)
    assert [r["reward"] for r in scored] == [1.0, 0.0]


@pytest.mark.asyncio
async def test_explicit_fail_remains_excluded_not_zero(tmp_path, monkeypatch):
    output = _run(tmp_path, [SUCCESS, FAIL | {"failure_outcome_action": "fail"}])
    assert _rejection(output) is None
    scored = await _aggregate(output, monkeypatch)
    assert [r["reward"] for r in scored] == [1.0]
    c = RolloutStore.read(output).coverage()
    assert c["final_zeros"] == 0 and c["failed_excluded"] == 1


@pytest.mark.parametrize("prior_outcome", [SUCCESS, FAIL])
@pytest.mark.parametrize("new_attempt", [False, True])
def test_unresolved_dispatch_does_not_settle_from_its_older_outcome(tmp_path, prior_outcome, new_attempt):
    from nemo_gym.rollouts.omitted_rollouts import omitted_rollout_record

    output = _run(tmp_path, [SUCCESS, prior_outcome])
    saved = RolloutStore.read(output)
    rows = [orjson.loads(line) for line in (tmp_path / "source.jsonl").read_bytes().splitlines()]
    with RolloutStore.start_or_resume(output, lambda: (rows, saved.manifest), resume=True) as store:
        prior = next(r for r in store.selected("success") + store.failures() if r["_ng_task_index"] == 1)
        later = prior | {"_ng_attempt_index": int(new_attempt), "_ng_failure_class": "agent_run_error"}
        store.record_dispatch(later)
        store.record_unstarted(omitted_rollout_record(later) | {"_ng_unrouted_run_error": True})
    assert RolloutStore.read(output).coverage()["final_zeros"] == 0
    # A diagnostic attached to an already-saved success retains that score; a new reservation is missing.
    if prior_outcome == FAIL or new_attempt:
        assert _rejection(output) is not None


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


@pytest.mark.asyncio
async def test_a_recorded_completed_grade_is_not_replaced_by_the_default_zero(tmp_path, monkeypatch):
    output = _run(tmp_path, [FAIL | {"reward": 0.75, "evaluation_completed": True}])
    scored = await _aggregate(output, monkeypatch)
    assert [row["reward"] for row in scored] == [0.75]
    assert RolloutStore.read(output).failures()[0]["reward"] == 0.75


@pytest.mark.asyncio
async def test_explicit_fail_wins_over_the_existing_zero_class_override(tmp_path, monkeypatch):
    output = _run(tmp_path, [SUCCESS, FAIL | {"failure_outcome_action": "fail"}])
    scored = await _aggregate(output, monkeypatch, count_failure_classes_as_zero=["agent_run_error"])
    assert [row["reward"] for row in scored] == [1.0]


def test_an_explicit_fail_settles_a_saved_judge_answer_without_zeroing_it(tmp_path):
    failure = {
        "_ng_failure_class": "judge_failed",
        "failure_outcome_action": "fail",
        "response": {
            "object": "response",
            "id": "answer-1",
            "status": "completed",
            "output": [],
        },
    }
    output = _run(tmp_path, [SUCCESS, failure])
    assert _rejection(output) is None
    coverage = RolloutStore.read(output).coverage()
    assert coverage["failed_excluded"] == 1 and coverage["final_zeros"] == 0


def test_an_explicit_fail_does_not_settle_a_still_retryable_episode(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "3")
    output = _run(tmp_path, [SUCCESS, {"_ng_failure_class": "provider_unavailable", "failure_outcome_action": "fail"}])
    coverage = RolloutStore.read(output).coverage()
    assert coverage["retryable"] == 1 and not coverage["complete"]
    assert coverage.get("failed_excluded", 0) == coverage["final_zeros"] == 0
    assert _rejection(output) is not None


def test_unconfirmed_harness_termination_is_not_a_zero_even_at_the_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "1")
    output = _run(tmp_path, [SUCCESS, {"_ng_failure_class": "harness_termination_unconfirmed"}])
    assert RolloutStore.read(output).coverage()["final_zeros"] == 0
    assert _rejection(output) is not None
