from __future__ import annotations

from dataclasses import asdict, dataclass, field
from uuid import uuid4


@dataclass
class StateFlowEvent:
    event_type: str
    source_agent: str
    target_agent: str | None = None
    related_nodes: list[str] = field(default_factory=list)
    related_edges: list[str] = field(default_factory=list)
    payload: dict[str, object] = field(default_factory=dict)
    iteration: int = 0
    event_id: str = field(default_factory=lambda: f"sfe_{uuid4().hex[:12]}")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def append_state_flow_event(graph: object, event: StateFlowEvent) -> None:
    data = event.to_dict()
    flow_history = getattr(graph, "flow_history", None)
    if isinstance(flow_history, list):
        flow_history.append(data)
    metadata = getattr(graph, "metadata", None)
    if isinstance(metadata, dict):
        exchange = metadata.setdefault("agent_collaboration_exchange", [])
        if isinstance(exchange, list):
            exchange.append(
                {
                    "sender": event.source_agent,
                    "artifact_type": event.event_type,
                    "event_id": event.event_id,
                    "iteration": event.iteration,
                    "target_agent": event.target_agent,
                    "related_node_count": len(event.related_nodes),
                    "related_edge_count": len(event.related_edges),
                    "state_flow_event": True,
                }
            )
            del exchange[:-120]
