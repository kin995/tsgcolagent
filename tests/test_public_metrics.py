from core.public_metrics import internal_metric_key, public_result


def test_internal_metric_keys_cover_aliases_without_hiding_paper_metrics():
    for key in (
        "sfc", "average_sfc", "delta_sfq", "estimated_tsq", "nrr", "pv",
        "csg", "dvs", "tic", "rcs", "state_flow_coverage",
        "weak_state_flow_coverage", "test_state_quality",
        "overall_quality_index", "weak_quality_flag_counts",
        "average_quality_gain", "quality_delta", "confidence", "reward",
        "success_rate", "success_count", "completion_rate", "defect_score",
        "estimated_mutation_score", "node_coverage", "path_coverage",
        "coverage_gap", "assertion_defect", "diagnosis_confidence",
        "repair_cost", "gaps", "metric_delta", "priority",
    ):
        assert internal_metric_key(key)
    for key in (
        "line_coverage", "branch_coverage", "ae", "mutation_score",
        "average_mutation_score", "pass_rate", "tir", "avg_exec",
        "duration_seconds", "preview", "classify_score",
    ):
        assert not internal_metric_key(key)


def test_public_result_removes_nested_process_metrics_and_text_without_mutating_source():
    source = {
        "source": "D:/tasks/SFC/demo.py",
        "final_report": {
            "line_coverage": 0.8,
            "sfc": 0.9,
            "nrr": 1.0,
            "tsq": 0.85,
            "quality_assessment": {"dimensions": {"coverage": {"sfc": 0.9}}},
            "details": [{"average_sfq": 0.7, "ae": 0.6}],
            "metric_names": ["SFC", "AE", "NRR", "mutation_score"],
            "target_methods": ["classify_score", "calculate_quality_gain"],
            "reason": "CSG=0.85; AE=0.60; SFC must improve; averageNRR 100%; defect_score=0.75; sfc_gap=0.2; repair_cost=5",
        },
        "paper_metrics": {"tir": 0.5, "avg_exec": 3.0},
        "outcome_summary": {"completion_rate": 0.8, "completed_execution": 8},
        "agent_decision": {"action": "repair", "gaps": {"coverage": 0.5}, "confidence": 0.8},
        "nodes": {"sfc": {"node_id": "sfc", "name": "calculate_quality_gain", "code": "def calculate_quality_gain(): pass", "defect_score": 0.9}},
        "metric_cards": [{"id": "sfc", "label": "SFC", "value": 0.9}, {"id": "ae", "label": "AE", "value": 0.6}],
    }

    result = public_result(source)

    assert result["final_report"]["line_coverage"] == 0.8
    assert result["final_report"]["details"] == [{"ae": 0.6}]
    assert result["final_report"]["metric_names"] == ["AE", "mutation_score"]
    assert result["final_report"]["target_methods"] == ["classify_score", "calculate_quality_gain"]
    assert result["paper_metrics"] == {"tir": 0.5, "avg_exec": 3.0}
    assert result["outcome_summary"] == {"completed_execution": 8}
    assert result["agent_decision"] == {"action": "repair"}
    assert result["nodes"] == {"sfc": {"node_id": "sfc", "name": "calculate_quality_gain", "code": "def calculate_quality_gain(): pass"}}
    assert result["metric_cards"] == [{"id": "ae", "label": "AE", "value": 0.6}]
    assert result["source"] == "D:/tasks/SFC/demo.py"
    assert "CSG" not in result["final_report"]["reason"]
    assert "0.85" not in result["final_report"]["reason"]
    assert "SFC" not in result["final_report"]["reason"]
    assert "100%" not in result["final_report"]["reason"]
    assert "0.75" not in result["final_report"]["reason"]
    assert "sfc_gap" not in result["final_report"]["reason"]
    assert "0.2" not in result["final_report"]["reason"]
    assert "repair_cost" not in result["final_report"]["reason"]
    assert "AE=0.60" in result["final_report"]["reason"]
    assert source["final_report"]["sfc"] == 0.9
