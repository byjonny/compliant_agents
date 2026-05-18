from types import SimpleNamespace

from policy_tool_mapper import run_pipeline
from policy_tool_mapper.eval_mappings import evaluate_mappings
from policy_tool_mapper.state import (
    MappedStatement,
    PipelineState,
    PolicyStatement,
    ToolMapping,
    ToolProfile,
)
from policy_tool_mapper.utils.output_formatter import format_output


def mapping(tool_id: str, statements: list[dict], tool_name: str | None = None) -> dict:
    return {
        "tool_id": tool_id,
        "tool_name": tool_name or tool_id,
        "statements": statements,
    }


def statement(
    statement_id: str,
    text: str,
    *,
    confidence: str = "high",
) -> dict:
    return {"id": statement_id, "text": text, "confidence": confidence}


def test_evaluate_mappings_filters_medium_confidence_predictions():
    predicted = {
        "mappings": [
            mapping(
                "refund_order",
                [
                    statement("P1", "Refunds require proof of purchase."),
                    statement(
                        "P2",
                        "Customers may exchange unopened items.",
                        confidence="medium",
                    ),
                ],
            )
        ]
    }
    ground_truth = {
        "mappings": [
            mapping(
                "refund_order",
                [
                    statement("G1", "Refunds require proof of purchase."),
                    statement("G2", "Customers may exchange unopened items."),
                ],
            )
        ]
    }

    all_conf = evaluate_mappings(predicted, ground_truth, high_conf_only=False)
    high_only = evaluate_mappings(predicted, ground_truth, high_conf_only=True)

    assert all_conf["per_tool"]["refund_order"]["n_predicted"] == 2
    assert all_conf["overall"]["micro"] == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }
    assert high_only["per_tool"]["refund_order"]["n_predicted"] == 1
    assert high_only["overall"]["micro"] == {
        "precision": 1.0,
        "recall": 0.5,
        "f1": 0.6667,
    }


def test_evaluate_mappings_matches_compositional_statement_baskets():
    predicted = {
        "mappings": [
            mapping(
                "cancel_reservation",
                [
                    statement(
                        "P1",
                        "Cancel only when the customer is verified and the "
                        "reservation is refundable.",
                    )
                ],
            )
        ]
    }
    ground_truth = {
        "mappings": [
            mapping(
                "cancel_reservation",
                [
                    statement("G1", "customer is verified"),
                    statement("G2", "reservation is refundable"),
                ],
            )
        ]
    }

    result = evaluate_mappings(predicted, ground_truth, threshold=0.8)
    tool_result = result["per_tool"]["cancel_reservation"]

    assert tool_result["n_comp_matches"] == 2
    assert tool_result["n_matched_pred"] == 1
    assert tool_result["n_matched_gt"] == 2
    assert result["overall"]["micro"]["recall"] == 1.0


def test_format_output_uses_final_mappings_and_builds_statement_index():
    state: PipelineState = {
        "raw_policy_text": "",
        "raw_openapi_spec": {},
        "policy_statements": [
            PolicyStatement(id="PS-002", text="Second rule", section="B"),
            PolicyStatement(id="PS-001", text="First rule", section="A"),
        ],
        "tool_profiles": [
            ToolProfile(
                tool_id="refund_order",
                name="Refund order",
                description="Refund an order",
                semantic_profile="Writes refund state",
                parameters=["order_id"],
            )
        ],
        "mappings": [
            ToolMapping(
                tool_id="refund_order",
                statements=[MappedStatement(id="PS-001", confidence="medium")],
            )
        ],
        "final_mappings": [
            ToolMapping(
                tool_id="refund_order",
                statements=[
                    MappedStatement(id="PS-002", confidence="high"),
                    MappedStatement(id="PS-001", confidence="medium"),
                ],
            )
        ],
        "sweep_iterations": 1,
    }

    output = format_output(state, policy_file="policy.md", openapi_file="tools.json")

    assert output["metadata"]["total_mappings"] == 2
    assert output["mappings"][0]["tool_name"] == "Refund order"
    assert [s["id"] for s in output["mappings"][0]["statements"]] == [
        "PS-001",
        "PS-002",
    ]
    assert output["statement_index"]["PS-001"]["mapped_to_tools"] == [
        "refund_order"
    ]


def test_pipeline_paths_encode_domain_model_and_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(run_pipeline, "HERE", tmp_path / "policy_tool_mapper")
    paths = run_pipeline._paths("airline", "gpt-5.4", mode="retrieval")

    assert paths["policy"].name == "AirlinePolicy.md"
    assert paths["tools"].name == "AirlineTools.json"
    assert paths["ground_truth"].name == "airline-ground-truth.json"
    assert paths["mappings"].name == "airline-mappings-gpt-5.4-retrieval.json"
    assert paths["eval_high"].name == "airline-eval-gpt-5.4-retrieval-high.json"
    assert paths["eval_all"].name == "airline-eval-gpt-5.4-retrieval-all.json"


def test_run_mapping_builds_retrieval_command(monkeypatch, tmp_path):
    captured = {}

    def fake_run(cmd: list[str], label: str) -> int:
        captured["cmd"] = cmd
        captured["label"] = label
        return 0

    paths = {
        "policy": tmp_path / "Policy.md",
        "tools": tmp_path / "Tools.json",
        "mappings": tmp_path / "mappings.json",
    }
    args = SimpleNamespace(
        embed_model="text-embedding-test",
        ce_model="cross-encoder-test",
        ce_top_k=7,
    )
    monkeypatch.setattr(run_pipeline, "_run", fake_run)

    ok = run_pipeline.run_mapping(paths, "gpt-test", "retrieval", args)

    assert ok is True
    assert captured["label"] == "policy-map  [gpt-test]  mode=retrieval"
    assert captured["cmd"] == [
        "uv",
        "run",
        "policy-map",
        "--policy",
        str(paths["policy"]),
        "--openapi",
        str(paths["tools"]),
        "--output",
        str(paths["mappings"]),
        "--model",
        "gpt-test",
        "--mode",
        "retrieval",
        "--embed-model",
        "text-embedding-test",
        "--ce-model",
        "cross-encoder-test",
        "--ce-top-k",
        "7",
    ]
