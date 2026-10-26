from __future__ import annotations

from pathlib import Path

from graph.state_graph import StateFlowGraph
from graph.structured_graph_builder import StructuredGraphBuilder
from llm_client import OpenAICompatibleLLM, extract_json_data
from memory.shared_graph import SharedGraphMemory
from agents.artifacts import AnalysisArtifact


class SourceAnalysisAgent:
    name = "测试状态智能体/源码分析模块"

    def __init__(self, memory: SharedGraphMemory, llm: OpenAICompatibleLLM | None = None, llm_required: bool = False):
        self.memory = memory
        self.llm = llm
        self.llm_required = llm_required

    def run(self, source_path: str | Path) -> StateFlowGraph:
        graph = StructuredGraphBuilder.from_source(source_path).build()
        graph.metadata["analysis_agent"] = self.name
        self._apply_llm_analysis(graph, Path(source_path))
        self.memory.set(graph)
        return graph

    def _apply_llm_analysis(self, graph: StateFlowGraph, source_path: Path) -> None:
        if not self.llm or not self.llm.enabled():
            graph.metadata.setdefault("analysis_strategy", "structured_context")
            return
        source_text = source_path.read_text(encoding="utf-8")
        language = str(graph.metadata.get("language", "python"))
        structured_context = graph.metadata.get("structured_context", {})
        node_digest = [
            {
                "id": node.node_id,
                "type": node.node_type,
                "name": node.name,
                "line_start": node.line_start,
                "line_end": node.line_end,
            }
            for node in graph.nodes.values()
        ][:120]
        valid_evidence = [item["id"] for item in node_digest]
        prompt = f"""
        You are the Source Analysis Agent module inside the Execution Agent of a three-agent unit-test generation system.
        Analyze the target {language} source, the normalized structured context, and the extracted graph nodes.

Return concise JSON only with this exact schema:
{{
  "behavior_summary": ["string"],
  "important_branches": [
    {{
      "evidence_id": "string",
      "expression": "string",
      "true_condition": "string",
      "false_condition": "string"
    }}
  ],
  "exception_contracts": [],
  "boundary_values": [],
  "test_generation_risks": []
}}

Use evidence_id only from this list:
{valid_evidence}

        Source:
        ```{language}
        {source_text}
        ```

        Structured context:
        {structured_context}

        Graph nodes:
        {node_digest}
""".strip()
        try:
            raw = self.llm.chat(
                [
                    {"role": "system", "content": "You analyze Java/Python code for test generation. Return compact JSON text."},
                    {"role": "user", "content": prompt},
                ]
            )
            graph.metadata["llm_source_analysis"] = raw
            graph.metadata["analysis_artifact"] = AnalysisArtifact.from_dict(extract_json_data(raw)).to_dict()
            graph.metadata["analysis_strategy"] = "ast_plus_llm"
            graph.metadata["llm_analysis_model"] = self.llm.config.model
        except Exception as exc:
            graph.metadata["llm_analysis_error"] = str(exc)
            if self.llm_required and _is_connection_error(exc):
                raise
            graph.metadata["analysis_strategy"] = "ast_only_after_llm_error"


def _is_connection_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "connection" in text or "not installed" in text or "http" in text
