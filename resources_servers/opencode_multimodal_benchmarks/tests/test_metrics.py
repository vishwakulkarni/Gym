# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from unittest.mock import MagicMock

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME, AggregateMetricsRequest
from nemo_gym.metrics import compute_aggregate_metrics
from resources_servers.opencode_multimodal_benchmarks.agentx import AgentXConfig, AgentXServer
from resources_servers.opencode_multimodal_benchmarks.app import VisualBenchmarksConfig, VisualBenchmarksServer
from resources_servers.opencode_multimodal_benchmarks.metrics import average_repeats


@pytest.fixture(params=["VStarBench", "HRBench8K", "VTCBench", "AgentX"])
def repeated_benchmark(request):
    benchmark = request.param
    config_type, server_type = (
        (AgentXConfig, AgentXServer) if benchmark == "AgentX" else (VisualBenchmarksConfig, VisualBenchmarksServer)
    )
    server = server_type(
        config=config_type(host="127.0.0.1", port=12345, name="benchmark", entrypoint="app.py"),
        server_client=MagicMock(spec=ServerClient),
    )
    rows = [
        {
            TASK_INDEX_KEY_NAME: task,
            ROLLOUT_INDEX_KEY_NAME: repeat,
            "reward": float(repeat == 0 and task < 2),
            "verifier_metadata": {
                "benchmark": benchmark,
                "index": str(task),
                "expected_samples": 3,
                "cycle_category": "1" if task < 2 else "2",
                "category": "spatial",
                "l2_category": "detail",
                "split": "test",
                "modality": "Image" if task < 2 else "Video",
            },
        }
        for task in range(3)
        for repeat in range(2)
    ]
    return benchmark, server, rows


def aggregate(server, rows):
    return compute_aggregate_metrics(
        rows,
        compute_metrics_fn=server.compute_metrics,
        get_key_metrics_fn=server.get_key_metrics,
        compute_repeat_metrics_fn=server.compute_repeat_metrics,
    )


def test_complete_repeats_have_headline_and_per_repeat_scores(repeated_benchmark):
    benchmark, server, rows = repeated_benchmark
    result = aggregate(server, rows)
    expected = 0.25 if benchmark == "HRBench8K" else 1 / 3
    assert result.agent_metrics["accuracy"] == pytest.approx(expected)
    assert result.agent_metrics[f"{benchmark}/completed"] == 3
    assert result.agent_metrics[f"{benchmark}/complete"] is True
    scores = {row[ROLLOUT_INDEX_KEY_NAME]: row["accuracy"] for row in result.repeat_level_metrics}
    assert scores == {0: pytest.approx(2 * expected), 1: 0}


def test_missing_repeat_is_partial_even_when_all_tasks_are_present(repeated_benchmark):
    benchmark, server, rows = repeated_benchmark
    result = aggregate(server, rows[:-1])
    assert result.agent_metrics[f"{benchmark}/complete"] is False
    assert "accuracy" not in result.agent_metrics
    assert f"{benchmark}/partial_accuracy" in result.agent_metrics
    assert f"{benchmark}/accuracy" not in result.agent_metrics


def test_duplicate_measurement_is_rejected(repeated_benchmark):
    _, server, rows = repeated_benchmark
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate(server, rows + rows[:1])


def test_duplicate_sample_identity_is_rejected(repeated_benchmark):
    _, server, rows = repeated_benchmark
    duplicate = deepcopy(rows[:2])
    for row in duplicate:
        row[TASK_INDEX_KEY_NAME] = 3
    with pytest.raises(ValueError, match="Duplicate benchmark sample ID"):
        aggregate(server, rows + duplicate)


def test_sample_repeat_guard_is_independent_of_framework_validation(repeated_benchmark):
    _, _, rows = repeated_benchmark
    with pytest.raises(ValueError, match="^Duplicate sample repeat$"):
        average_repeats([[rows[0], deepcopy(rows[0])]])


def test_repeat_metadata_must_be_identical(repeated_benchmark):
    _, _, rows = repeated_benchmark
    changed = deepcopy(rows[1])
    changed["verifier_metadata"]["category"] = "changed"
    with pytest.raises(ValueError, match="^Inconsistent metadata across sample repeats$"):
        average_repeats([[rows[0], changed]])


@pytest.mark.asyncio
async def test_uniformly_masked_repeat_is_partial_through_aggregation_handler(repeated_benchmark):
    benchmark, server, rows = repeated_benchmark
    for row in rows:
        row["mask_sample"] = row[ROLLOUT_INDEX_KEY_NAME] == 1
    result = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=rows))
    expected = 0.5 if benchmark == "HRBench8K" else 2 / 3
    assert result.agent_metrics[f"{benchmark}/complete"] is False
    assert result.agent_metrics[f"{benchmark}/partial_accuracy"] == pytest.approx(expected)
    assert result.agent_metrics["coverage/masked_rollouts"] == 3
    assert result.agent_metrics["num_repeats"] == 2
    assert "accuracy" not in result.agent_metrics
    assert f"{benchmark}/accuracy" not in result.agent_metrics
    assert f"{benchmark}/accuracy" not in result.key_metrics

    # Rejudging the failed attempts restores the full denominator and headline.
    for row in rows:
        row["mask_sample"] = False
    recovered = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=rows))
    assert recovered.agent_metrics[f"{benchmark}/complete"] is True
    assert recovered.agent_metrics["accuracy"] == pytest.approx(expected / 2)


def test_direct_metrics_exclude_masked_and_unfinished_measurements(repeated_benchmark):
    benchmark, server, rows = repeated_benchmark
    for row in rows:
        row["mask_sample"] = row[ROLLOUT_INDEX_KEY_NAME] == 1
        if row["mask_sample"]:
            row["reward"] = -100  # An invalid measurement must never enter a quality mean.
    tasks = [[row for row in rows if row[TASK_INDEX_KEY_NAME] == index] for index in range(3)]
    expected = 0.5 if benchmark == "HRBench8K" else 2 / 3
    for invalid_flag in ("mask_sample", "evaluation_completed"):
        if invalid_flag == "evaluation_completed":
            for row in rows:
                row["evaluation_completed"] = not row.pop("mask_sample")
        metrics = server.compute_metrics(tasks)
        assert metrics[f"{benchmark}/complete"] is False
        assert metrics[f"{benchmark}/partial_accuracy"] == pytest.approx(expected)
        assert "accuracy" not in metrics


@pytest.mark.asyncio
async def test_masks_do_not_make_an_unaffected_benchmark_partial(repeated_benchmark):
    benchmark, server, rows = repeated_benchmark
    other = "VTCBench" if benchmark != "VTCBench" else "VStarBench"
    additional = deepcopy(rows)
    for row in additional:
        row[TASK_INDEX_KEY_NAME] += 3
        row["verifier_metadata"]["benchmark"] = other
    for row in rows:
        row["mask_sample"] = row[ROLLOUT_INDEX_KEY_NAME] == 1
    result = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=rows + additional))
    assert result.agent_metrics[f"{benchmark}/complete"] is False
    assert result.agent_metrics[f"{other}/complete"] is True
    assert f"{benchmark}/accuracy" not in result.key_metrics
    assert f"{other}/accuracy" in result.key_metrics


@pytest.mark.asyncio
async def test_zerobench_extra_masked_attempt_does_not_certify_pass_at_five(repeated_benchmark):
    _, server, _ = repeated_benchmark
    rows = [
        {
            TASK_INDEX_KEY_NAME: task,
            ROLLOUT_INDEX_KEY_NAME: repeat,
            "reward": 1,
            "mask_sample": repeat == 5,
            "verifier_metadata": {"benchmark": "ZeroBench", "index": str(task), "expected_samples": 2},
        }
        for task in range(2)
        for repeat in range(6)
    ]
    result = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=rows))
    assert result.agent_metrics["ZeroBench/complete"] is False
    assert result.agent_metrics["ZeroBench/partial_pass@5"] == 1
    assert "accuracy" not in result.agent_metrics
    assert "ZeroBench/pass@5" not in result.key_metrics


def test_all_invalid_measurements_have_no_quality_score(repeated_benchmark):
    benchmark, server, rows = repeated_benchmark
    for row in rows:
        row["mask_sample"] = True
    tasks = [[row for row in rows if row[TASK_INDEX_KEY_NAME] == index] for index in range(3)]
    metrics = server.compute_metrics(tasks)
    assert metrics[f"{benchmark}/complete"] is False
    assert metrics[f"{benchmark}/completed"] == 0
    assert "accuracy" not in metrics and f"{benchmark}/partial_accuracy" not in metrics


def test_http_aggregation_keeps_masked_repeat_identities(repeated_benchmark):
    from fastapi.testclient import TestClient

    benchmark, server, rows = repeated_benchmark
    for row in rows:
        row["mask_sample"] = row[ROLLOUT_INDEX_KEY_NAME] == 1
    with TestClient(server.setup_webserver()) as client:
        response = client.post("/aggregate_metrics", json={"verify_responses": rows})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["agent_metrics"][f"{benchmark}/complete"] is False
    assert "accuracy" not in result["agent_metrics"]
    assert f"{benchmark}/accuracy" not in result["key_metrics"]


@pytest.mark.parametrize("benchmark", ["VStarBench", "HRBench8K", "VTCBench", "ZeroBench", "AgentX"])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("adjustment", [-0.05, 0.1])
def test_accuracy_uses_correctness_without_mutating_shaped_training_reward(benchmark, legacy, adjustment):
    config_type, server_type = (
        (AgentXConfig, AgentXServer) if benchmark == "AgentX" else (VisualBenchmarksConfig, VisualBenchmarksServer)
    )
    server = server_type(
        config=config_type(host="127.0.0.1", port=12345, name="benchmark", entrypoint="app.py"),
        server_client=MagicMock(spec=ServerClient),
    )
    score = 0.65 if benchmark == "AgentX" else 1.0
    rows = [
        {
            TASK_INDEX_KEY_NAME: 0,
            ROLLOUT_INDEX_KEY_NAME: repeat,
            "reward": score if repeat == 0 else 0.0,
            "verifier_metadata": {
                "benchmark": benchmark,
                "index": "0",
                "expected_samples": 1,
                "cycle_category": "1",
                "category": "spatial",
                "l2_category": "detail",
                "split": "test",
                "modality": "Image",
            },
        }
        for repeat in range(5)
    ]
    expected = aggregate(server, rows)
    assert expected.agent_metrics["accuracy"] == pytest.approx(1.0 if benchmark == "ZeroBench" else score / 5)
    for row in rows:
        base = row["reward"]
        auxiliary = adjustment if base > 0 else 0.0
        row["reward"] = base + auxiliary
        if legacy:
            row["opencode_tool_reward"] = {"base_reward": base}
        else:
            row["reward_components"] = {"verifier": base, "opencode_tool_reward": auxiliary}
    shaped = deepcopy(rows)
    result = aggregate(server, rows)
    assert result.agent_metrics["accuracy"] == expected.agent_metrics["accuracy"]
    assert result.agent_metrics[f"{benchmark}/accuracy"] == expected.agent_metrics[f"{benchmark}/accuracy"]
    if benchmark == "ZeroBench":
        assert result.agent_metrics["ZeroBench/pass@1"] == 0.2
        assert result.agent_metrics["ZeroBench/pass@5"] == 1.0
    if benchmark == "AgentX":
        assert result.agent_metrics["AgentX/Image/accuracy"] == pytest.approx(0.13)
    for actual, baseline in zip(result.repeat_level_metrics, expected.repeat_level_metrics, strict=True):
        for key in baseline:
            if key == "accuracy" or key.endswith("/accuracy") or key == "ZeroBench/pass@1":
                assert actual[key] == baseline[key]
        assert actual["mean/reward"] == rows[actual[ROLLOUT_INDEX_KEY_NAME]]["reward"]
    assert result.agent_metrics["mean/reward"] == pytest.approx((score + adjustment) / 5)
    assert rows == shaped
    assert rows[0]["reward"] == score + adjustment


@pytest.mark.asyncio
async def test_vtcbench_verification_with_tools_remains_correctness_only():
    from resources_servers.opencode_multimodal_benchmarks.tests.test_app import request
    from resources_servers.opencode_multimodal_benchmarks.tests.test_app import server as make_server
    from resources_servers.opencode_multimodal_benchmarks.tests.test_rewards import tool_response, trajectory

    server = make_server.__wrapped__()
    body = request("VTCBench", "A")
    body.response = tool_response("A")
    body.ng_trajectory = trajectory()
    result = await server.verify(body)
    assert result.reward == 1.0
    assert "opencode_tool_reward" not in result.model_dump()
    assert "reward_components" not in result.model_dump()
    metrics = server.compute_metrics([[result.model_dump()]])
    assert metrics["accuracy"] == metrics["VTCBench/accuracy"] == 1.0
    assert result.reward == 1.0
