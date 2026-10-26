from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any

from core.state_flow_event import StateFlowEvent, append_state_flow_event
from graph.experience_memory import AgentExperienceMemory, resolve_knowledge_store_path
from memory.shared_graph import SharedGraphMemory


@dataclass
class KnowledgeContext:
    query: dict[str, object]
    hints: dict[str, object]
    selected_ids: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "hints": self.hints,
            "selected_ids": self.selected_ids,
        }


class TestKnowledgeAgent:
    name = "测试知识智能体"
    SYSTEM_IDENTITY = {
        "system_name": "ValiFixTest & TSGColAgent测试状态引导多智能体单元测试生成系统",
        "method": "ValiFixTest & TSGColAgent（测试状态引导机制TSG）",
        "shared_center": "共享TSG与测试状态中心",
        "top_level_agents": {
            "测试状态智能体": "分析和自修复测试状态、评价A-E质量、识别缺口并规划下一项具体任务",
            "测试生成智能体": "分析测试目标，生成、验证、执行，并使用状态定位修复策略和状态目标优化策略处理Python/Java测试",
            "测试知识智能体": "检索和学习经验，解释实验与报告，并与本地大模型共同完成单元测试领域对话",
        },
        "own_role": [
            "读取测试知识引导库和真实实验数据",
            "为每轮生成、失败恢复和C级优化提供可追溯知识",
            "回答系统自身、实验、报告、Python/Java单元测试和测试方法问题",
            "把符合条件的单元测试概念归纳为待真实任务验证的知识",
        ],
        "own_limits": [
            "不直接修改测试文件",
            "不替代测试状态智能体做最终质量分级",
            "不把未知、缺失或尚未执行的数据编造成结论",
            "不把单次对话内容未经真实任务验证直接标为已掌握",
        ],
        "round_policy": {
            "R1": "全部任务形成生成与正式评价基线",
            "R2": "只恢复D/E，使用失败原因、解决措施和历史知识重新生成或补齐证据",
            "R3": "统一优化全部C，保留通过测试并针对最大质量弱项做一次最小目标增强",
            "maximum": "最多三轮，不开启R4",
        },
        "languages": ["Python", "Java"],
        "scopes": ["文件级", "类级", "项目级", "数据集级"],
        "quality": ["LC", "BC", "AE", "MS", "PassRate", "AvgExec", "Time", "TIR"],
        "assistant_role": "作为系统助手统一解释当前系统能力、配置、实验、任务、报告、失败、知识和单元测试方法；回答以实时后端事实和系统目录为准。",
        "data_boundary": "前端只读取后端额外生成或提取的监控数据，不参与测试状态规划、测试生成、验证、评估、修复或优化。",
        "knowledge_lifecycle": "对话概念先进入正在学习或待复核状态；被后续真实任务多次使用且结果可行后标为已掌握，仍可继续累积新证据；负向证据进入应避免。",
    }

    def __init__(self, memory: SharedGraphMemory, experience: AgentExperienceMemory):
        self.memory = memory
        self.experience = experience

    def retrieve(self, intent: dict[str, Any] | None = None, model: str | None = None) -> KnowledgeContext:
        graph = self.memory.load()
        intent = intent or graph.metadata.get("last_test_intent_plan", {}) or {}
        query = {
            "language": graph.metadata.get("language"),
            "dominant_gap": self._dominant_gap(intent.get("gaps", {})),
            "action": intent.get("action"),
            "focus": intent.get("focus"),
            "target_states": list(intent.get("target_states", []) or [])[:20],
            "task_signature": self.experience.task_signature(graph),
        }
        hints = self.experience.hints(graph, model=model)
        failure_guidance = graph.metadata.get("failure_recovery_guidance")
        if isinstance(failure_guidance, dict) and failure_guidance:
            hints["followup_failure_guidance"] = failure_guidance
        selected_records = self.experience.retrieve_hybrid(graph, query=query, model=model, top_k=5)
        selected_ids = [self._stable_record_id(item) for item in selected_records]
        selected_ids.extend(
            str(item.get("experience_id"))
            for item in (hints.get("conversation_concepts", []) or [])
            if isinstance(item, dict) and item.get("experience_id")
        )
        if isinstance(failure_guidance, dict):
            selected_ids.extend(str(value) for value in failure_guidance.get("matched_experience_ids", []) or [] if value)
        selected_ids = list(dict.fromkeys(selected_ids))
        context = KnowledgeContext(query=query, hints=hints, selected_ids=selected_ids)
        graph.metadata["experience_hints"] = hints
        graph.metadata["last_knowledge_context"] = context.to_dict()
        history = graph.metadata.setdefault("knowledge_retrieval_history", [])
        if isinstance(history, list):
            history.append(context.to_dict())
        append_state_flow_event(
            graph,
            StateFlowEvent(
                event_type="KnowledgeRetrieved",
                source_agent=self.name,
                target_agent="测试生成智能体",
                related_nodes=query["target_states"],
                payload=context.to_dict(),
                iteration=graph.iteration,
            ),
        )
        graph.bump_version("test knowledge retrieved")
        self.memory.save(graph)
        return context

    def record_outcome(self, report: dict[str, Any], intent: dict[str, Any] | None = None) -> None:
        graph = self.memory.load()
        intent = intent or graph.metadata.get("last_test_intent_plan", {}) or {}
        outcome = {
            "task_source": graph.source_path,
            "experiment_round": graph.metadata.get("experiment_round", 1),
            "evaluation_profile": report.get("evaluation_profile", "full"),
            "quality_status": report.get("quality_status"),
            "pytest_passed": report.get("pytest_passed"),
            "line_coverage": report.get("line_coverage"),
            "branch_coverage": report.get("branch_coverage"),
            "mutation_score": report.get("mutation_score"),
            "sfc": report.get("sfc"),
            "ae": report.get("ae"),
            "tsq": report.get("tsq", report.get("sfq")),
            "failure_category": report.get("failure_category"),
            "action": intent.get("action"),
            "focus": intent.get("focus"),
        }
        graph.metadata["last_knowledge_outcome"] = outcome
        append_state_flow_event(
            graph,
            StateFlowEvent(
                event_type="KnowledgeUpdated",
                source_agent=self.name,
                target_agent="测试状态智能体",
                payload=outcome,
                iteration=graph.iteration,
            ),
        )
        context = graph.metadata.get("last_knowledge_context") if isinstance(graph.metadata.get("last_knowledge_context"), dict) else {}
        selected_ids = [str(value) for value in context.get("selected_ids", []) or [] if value]
        quality_status = str(report.get("quality_status") or "")
        successful = bool(report.get("pytest_passed")) and quality_status in {
            "quality_high", "quality_acceptable", "valid_usable", "quality_target_not_reached",
            "plateau_reached", "metric_unreliable", "budget_limited", "generated", "valid",
        }
        validated = (
            self.experience.validate_conversation_concepts(selected_ids, successful=successful, evidence=outcome)
            if outcome["evaluation_profile"] == "full" else []
        )
        if validated:
            graph.metadata["last_concept_validation"] = [
                {
                    "experience_id": item.get("experience_id"),
                    "topic": item.get("topic"),
                    "learning_stage": item.get("learning_stage"),
                    "validation_count": item.get("validation_count", 0),
                }
                for item in validated
            ]
        self.memory.save(graph)

    @classmethod
    def learn_from_conversation(
        cls,
        question: str,
        answer: str,
        *,
        model: str | None,
        conversation_id: str | None,
        knowledge_store: str | Path | None = None,
        force: bool = False,
    ) -> dict[str, Any] | None:
        experience = AgentExperienceMemory(Path(knowledge_store).expanduser().resolve() if knowledge_store else resolve_knowledge_store_path())
        record = experience.append_conversation_concept(
            question,
            answer,
            model=model,
            conversation_id=conversation_id,
            force=force,
        )
        if not record:
            return None
        if experience.last_write_errors or not any(
            item.get("experience_id") == record.get("experience_id")
            for item in experience.load()
        ):
            raise OSError("学习结果未成功写入知识库，不能标记为已学习")
        return {
            "record_id": record.get("experience_id"),
            "topic": record.get("topic"),
            "status": "已掌握" if record.get("learning_stage") in {"adopted", "mastered"} else "正在学习",
            "validation_count": record.get("validation_count", 0),
            "adoption_requirement": "被后续真实测试任务多次使用并获得可行结果",
        }

    @classmethod
    def answer_question(
        cls,
        question: str,
        *,
        snapshot: dict[str, Any],
        knowledge: dict[str, Any],
        llm: Any | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Answer dashboard questions; persistence is handled after a successful LLM reply."""
        current = snapshot.get("current_task") if isinstance(snapshot.get("current_task"), dict) else {}
        metrics = current.get("metrics") if isinstance(current.get("metrics"), dict) else {}
        overview = snapshot.get("overview") if isinstance(snapshot.get("overview"), dict) else {}
        failures = current.get("failure") if isinstance(current.get("failure"), dict) else {}
        catalog = snapshot.get("catalog") if isinstance(snapshot.get("catalog"), dict) else {}
        assistant_catalog = {
            "name": catalog.get("name"),
            "runtime": catalog.get("runtime"),
            "architecture": catalog.get("architecture", []),
            "entrypoints": catalog.get("entrypoints", []),
            "component_map": catalog.get("component_map", []),
            "dataset_families": catalog.get("dataset_families", []),
            "languages": catalog.get("languages", []),
            "scopes": catalog.get("scopes", []),
            "phases": catalog.get("phases", []),
            "agents": [
                {
                    "name": item.get("name"),
                    "mission": item.get("mission"),
                    "inputs": item.get("inputs", []),
                    "outputs": item.get("outputs", []),
                    "modules": item.get("modules", []),
                }
                for item in catalog.get("agents", []) if isinstance(item, dict)
            ],
            "quality_metrics": catalog.get("quality_metrics", []),
            "execution_policies": catalog.get("execution_policies", []),
            "artifacts": catalog.get("artifacts", []),
            "interfaces": catalog.get("interfaces", []),
        }
        sources = [value for value in [current.get("summary_path"), current.get("graph_path"), knowledge.get("path")] if value]
        relevant_knowledge = cls._relevant_knowledge(question, list(knowledge.get("records", []) or []), limit=12)
        facts = {
            "conversation_history": [
                {"role": item.get("role"), "content": str(item.get("content") or "")[:1500]}
                for item in (conversation_history or [])[-8:]
            ],
            "experiment": snapshot.get("experiment"),
            "overview": overview,
            "current_task": {
                "name": current.get("name"),
                "source": current.get("source"),
                "grade": current.get("grade"),
                "round": current.get("round"),
                "status": current.get("status"),
                "metrics": metrics,
                "failure": failures,
            },
            "parallel_execution": snapshot.get("parallel_execution"),
            "model_runtime": snapshot.get("assistant"),
            "system_identity": cls.SYSTEM_IDENTITY,
            "configured_system_catalog": assistant_catalog,
            "knowledge_counts": knowledge.get("counts", {}),
            "knowledge_retrieval": {
                "store_path": knowledge.get("path"),
                "stored_record_count": knowledge.get("stored_record_count", knowledge.get("total", 0)),
                "matched_record_count": len(relevant_knowledge),
                "status": "matched" if relevant_knowledge else "retrieved_without_match",
            },
            "relevant_knowledge": relevant_knowledge,
        }
        model_error = None
        if llm is not None and getattr(llm, "enabled", lambda: False)():
            try:
                answer = llm.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "你是本系统三个正式顶层智能体之一的测试知识智能体，正在与后端已配置的本地大模型共同回答问题。"
                                "你既是测试知识智能体，也是该系统面向用户的系统助手。你必须理解并准确说明自己的身份与边界、另外两个智能体、四层架构、共享TSG、测试状态中心、R1/R2/R3策略、入口命令、数据集、Python/Java能力、文件/类/项目范围、工具链、并行与续跑策略、质量指标、实验产物、前端功能和知识学习条件。"
                                "用户询问‘你、你们、系统、智能体、能力、流程、为什么这样设计’时，依据system_identity和configured_system_catalog回答，并明确区分自己的能力与整个系统的能力。"
                                "必须以给定事实为准；评价结果只解释公开的LC、BC、AE、MS、PassRate、AvgExec、Time、TIR，并区分单任务实际值与实验汇总值。缺少执行证据的指标要明确说明尚未记录，不得推测或以其他评分代替。"
                                "用户询问代码入口、某项功能在哪里、系统会产生什么文件或如何打包运行时，必须依据configured_system_catalog中的entrypoints、artifacts、interfaces和execution_policies回答。"
                                "对失败和优化问题应结合当前轮次、具体弱项、状态定位修复范围、状态目标优化措施、既往经验和可验证措施回答。"
                                "回答必须同时使用两类输入：一是实验报告和系统事实，二是knowledge_retrieval与relevant_knowledge中的知识库检索结果。"
                                "即使没有匹配记录，也必须说明本次检索未命中，而不能假装使用了不存在的经验。"
                                "对概念问题给出可复用、可验证的单元测试原则；不知道或当前尚无数据时明确说明，不得虚构。用清晰、自然且有针对性的中文回答。"
                            ),
                        },
                        {"role": "user", "content": f"问题：{question}\n可用事实与知识检索结果：{json.dumps(_compact_prompt_facts(facts), ensure_ascii=False, default=str)}"},
                    ]
                ).strip()
                if not answer:
                    raise RuntimeError("大模型返回了空回答")
                return cls._answer_payload(
                    answer,
                    sources=sources,
                    retrieval=facts["knowledge_retrieval"],
                    response_mode="model_and_knowledge",
                    model_used=True,
                    needs_learning=False,
                )
            except Exception as exc:
                model_error = f"{exc.__class__.__name__}: {exc}"
        else:
            model_error = "模型客户端当前不可用"

        knowledge_answer = cls._autonomous_answer(question, facts)
        can_answer = not knowledge_answer.startswith("我没有在当前实验报告或测试知识引导库中找到足够证据")
        if can_answer:
            return cls._answer_payload(
                knowledge_answer,
                sources=sources,
                retrieval=facts["knowledge_retrieval"],
                response_mode="knowledge_autonomous",
                model_used=False,
                needs_learning=False,
                model_error=model_error,
            )
        return cls._answer_payload(
            "这个问题超出了我当前实验事实和已掌握知识的范围。我还需要学习，已经记住这个问题；模型服务可用后我会自动补学并把结果写回知识库。",
            sources=sources,
            retrieval=facts["knowledge_retrieval"],
            response_mode="pending_learning",
            model_used=False,
            needs_learning=True,
            model_error=model_error,
        )

    @classmethod
    def _answer_payload(
        cls,
        answer: str,
        *,
        sources: list[object],
        retrieval: dict[str, object],
        response_mode: str,
        model_used: bool,
        needs_learning: bool,
        model_error: str | None = None,
    ) -> dict[str, Any]:
        return {
            "agent": cls.name,
            "answer": answer,
            "sources": sources,
            "model_used": model_used,
            "knowledge_used": True,
            "knowledge_retrieval": retrieval,
            "response_mode": response_mode,
            "needs_learning": needs_learning,
            "model_error": model_error,
            "read_only": False,
            "system_assistant": True,
            "knowledge_learning_enabled": True,
        }

    @staticmethod
    def _relevant_knowledge(question: str, records: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
        terms = _query_terms(question) - {"java", "python", "测试", "如何", "怎么", "什么", "原理", "请问"}
        ranked: list[tuple[int, int, dict[str, Any]]] = []
        for index, record in enumerate(records):
            text = " ".join(str(record.get(key) or "") for key in ("question", "topic", "solution", "usage", "action")).casefold()
            topic = " ".join(str(record.get(key) or "") for key in ("topic", "action", "usage", "solution")).casefold()
            score = sum(3 if term in topic else 1 for term in terms if term in text)
            if score:
                ranked.append((score, -index, record))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in ranked[:limit]]

    @classmethod
    def _autonomous_answer(cls, question: str, facts: dict[str, Any]) -> str:
        text = question.casefold()
        records = [
            record for record in facts.get("relevant_knowledge", [])
            if record.get("solution") and record.get("learning_stage") != "avoid"
            and record.get("status") != "应避免"
        ]
        concept = any(term in text for term in ["什么", "如何", "怎样", "怎么", "原理", "解释", "介绍", "提高", "区别"])
        system_question = any(term in text for term in ["你是谁", "系统", "智能体", "当前", "本次", "代码结构", "入口", "轮次", "r1", "r2", "r3"])
        if records and concept and not system_question:
            return "根据已保存知识（尚未验证的条目仅供参考）：\n" + "\n".join(
                f"- {str(record['solution'])[:1800]}" for record in records[:3]
            )
        basic_concept = any(term in text for term in ["单元测试的概念", "单元测试的原理", "什么是单元测试", "简单介绍单元测试", "介绍一下单元测试"])
        mutation_advice = "变异" in text and any(term in text for term in ["提高", "改进"])
        if concept and not system_question and not basic_concept and not mutation_advice:
            return "我没有在当前实验报告或测试知识引导库中找到足够证据。"
        if mutation_advice:
            facts = {**facts, "current_task": {}}
        return cls._factual_answer(question, facts)

    @staticmethod
    def _factual_answer(question: str, facts: dict[str, Any]) -> str:
        current = facts.get("current_task", {})
        metrics = current.get("metrics", {})
        overview = facts.get("overview", {})
        catalog = facts.get("configured_system_catalog", {})
        text = question.lower()
        if any(key in text for key in ["代码在哪", "哪个模块", "哪个文件实现", "代码结构", "实现位置"]):
            components = catalog.get("component_map", []) if isinstance(catalog, dict) else []
            return "代码职责索引为：" + "；".join(f"{item.get('path')}：{item.get('responsibility')}" for item in components if isinstance(item, dict))
        if any(key in text for key in ["你是谁", "你是什么", "你能做什么", "系统能力", "系统架构", "三个智能体", "智能体职责", "工作流程", "轮次逻辑", "r1", "r2", "r3"]):
            identity = TestKnowledgeAgent.SYSTEM_IDENTITY
            return (
                "我是测试知识智能体，是系统三个顶层智能体之一。我负责检索与学习测试经验、解释实验和报告，并与已配置的本地大模型共同回答Python、Java及单元测试方法问题；"
                "我不直接修改测试文件，也不负责最终质量分级。系统由测试状态智能体、测试生成智能体和测试知识智能体协作，通过共享TSG交换状态与证据。"
                f"轮次固定为：R1{identity['round_policy']['R1']}；R2{identity['round_policy']['R2']}；R3{identity['round_policy']['R3']}；{identity['round_policy']['maximum']}。"
            )
        if any(key in text for key in ["命令", "入口", "怎么运行", "如何运行", "启动", "打包"]):
            entries = catalog.get("entrypoints", []) if isinstance(catalog, dict) else []
            commands = [f"{item.get('name')}：{item.get('command')}" for item in entries if isinstance(item, dict)]
            return "系统支持后端命令和前端启动两种方式。" + "；".join(commands)
        if any(key in text for key in ["覆盖", "jacoco", "coverage", "变异", "pit", "断言", "ae", "tsq", "质量"]):
            if current.get("name"):
                return (
                    f"当前任务 {current.get('name')} 为 {current.get('grade')}·R{current.get('round')}。"
                    f"行覆盖率 {_pct(metrics.get('line_coverage'))}，分支覆盖率 {_pct(metrics.get('branch_coverage'))}，"
                    f"断言有效性 {_pct(metrics.get('ae'))}，变异得分 {_pct(metrics.get('mutation_score'))}，"
                    f"目标改善率 {_pct(metrics.get('tir'))}，平均执行次数 {metrics.get('avg_exec') if metrics.get('avg_exec') is not None else '—'}。"
                    "这些都是该单任务的实际值，不是平均值。"
                )
            guidance = _knowledge_guidance(facts.get("relevant_knowledge", []))
            if guidance:
                return "我从测试知识引导库检索到以下可复用措施：" + "；".join(guidance)
            if any(key in text for key in ["变异", "pit"]):
                return (
                    "提高Java变异检测率时，先保证JUnit稳定通过，再查看PIT存活变异所在行；"
                    "针对条件边界、算术运算和布尔取反加入能区分原实现与变异实现的精确断言。"
                    "输入应小而确定，并分别覆盖边界两侧，避免只断言非空或不抛异常。"
                )
            return "提高测试质量应同时检查可执行性、路径覆盖、断言有效性和变异杀伤能力，并优先补当前证据中最大的缺口。"
        if any(key in text for key in ["数据集", "testeval", "humaneval", "defects4j", "python", "java", "文件级", "类级", "项目级"]):
            datasets = catalog.get("dataset_families", []) if isinstance(catalog, dict) else []
            scopes = catalog.get("scopes", []) if isinstance(catalog, dict) else []
            dataset_text = "、".join(f"{item.get('name')}({item.get('language')})" for item in datasets if isinstance(item, dict))
            scope_text = "、".join(str(item.get("name")) for item in scopes if isinstance(item, dict))
            return f"系统面向Python与Java，已配置的数据集族包括{dataset_text}；处理范围包括{scope_text}。Python使用pytest、coverage.py和项目内变异执行器，Java使用JUnit 5、JaCoCo和PIT。"
        if any(key in text for key in ["页面", "界面", "前端", "可视化", "在哪看"]):
            interfaces = catalog.get("interfaces", []) if isinstance(catalog, dict) else []
            return "前端页面与职责为：" + "；".join(f"{item.get('name')}：{item.get('content')}" for item in interfaces if isinstance(item, dict))
        if any(key in text for key in ["文件", "产物", "摘要", "报告保存", "知识库路径", "状态图"]):
            artifacts = catalog.get("artifacts", []) if isinstance(catalog, dict) else []
            return "系统主要产物为：" + "；".join(f"{item.get('name')}({item.get('path')})：{item.get('content')}" for item in artifacts if isinstance(item, dict))
        if any(key in text for key in ["并行", "并发", "续跑", "耗时", "跳过"]):
            policies = catalog.get("execution_policies", []) if isinstance(catalog, dict) else []
            return "当前执行策略为：" + "；".join(f"{item.get('name')}={item.get('value')}，{item.get('description')}" for item in policies if isinstance(item, dict))
        if any(key in text for key in ["失败", "原因", "为什么", "修复"]):
            failure = current.get("failure", {})
            return (
                f"当前任务等级为 {current.get('grade') or '未知'}，失败类别为 {failure.get('category') or '无'}。"
                f"质量缺口：{json.dumps(failure.get('threshold_failures', {}), ensure_ascii=False)}；"
                f"弱项：{json.dumps(failure.get('weak_flags', []), ensure_ascii=False)}；"
                f"具体原因：{failure.get('reason') or '报告中没有记录额外错误'}。"
            )
        if any(key in text for key in ["知识", "经验", "学会", "学习"]):
            counts = facts.get("knowledge_counts", {})
            return f"测试知识引导库目前记录：已掌握 {counts.get('已掌握', 0)}，正在学习 {counts.get('正在学习', 0)}，待复核 {counts.get('待复核', 0)}，应避免 {counts.get('应避免', 0)}。"
        if any(key in text for key in ["多少", "任务", "结果", "进度"]):
            return (
                f"当前实验共有 {overview.get('total_tasks', 0)} 个任务，完整执行 {overview.get('completed_execution', 0)} 个，"
                f"未完整执行 {overview.get('not_completed_execution', 0)} 个，进入后续轮次 {overview.get('followup_task_count', 0)} 个。"
            )
        if any(key in text for key in ["单元测试", "测试概念", "测试原理", "怎么测试", "如何测试"]):
            guidance = _knowledge_guidance(facts.get("relevant_knowledge", []))
            base = (
                "单元测试是把一个函数、方法或类当作最小验证对象，用可重复的输入检查可观察结果。"
                "一个有效用例通常包含准备输入、执行目标和验证结果三部分；除正常路径外，还应覆盖边界、异常和状态变化。"
                "覆盖率说明代码是否走到，精确断言说明结果是否正确，变异检测则检验这些断言能否发现人为缺陷。"
            )
            return base + ((" 知识库还建议：" + "；".join(guidance)) if guidance else "")
        guidance = _knowledge_guidance(facts.get("relevant_knowledge", []))
        if guidance:
            return "根据测试知识引导库：" + "；".join(guidance)
        return "我没有在当前实验报告或测试知识引导库中找到足够证据。请提供任务名、失败信息或更具体的单元测试问题。"

    def _dominant_gap(self, gaps: object) -> str:
        if not isinstance(gaps, dict) or not gaps:
            return "initial_generation"
        numeric = []
        for key, value in gaps.items():
            try:
                numeric.append((str(key), float(value)))
            except (TypeError, ValueError):
                continue
        return max(numeric, key=lambda item: item[1])[0] if numeric else "initial_generation"

    def _stable_record_id(self, record: dict[str, Any]) -> str:
        direct = record.get("experience_id") or record.get("strategy_fingerprint") or record.get("candidate_hash")
        if direct:
            return str(direct)
        return "legacy:" + ":".join(
            [
                str(record.get("record_type") or "experience"),
                str(record.get("timestamp") or "unknown"),
                str(record.get("agent") or "unknown"),
            ]
        )


def _pct(value: object) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "暂无"


def _compact_prompt_facts(facts: dict[str, Any]) -> dict[str, Any]:
    """Bound individual fields without slicing away retrieval or invalidating JSON."""
    def compact(value: Any, depth: int = 0) -> Any:
        if isinstance(value, str):
            return value[:1000]
        if isinstance(value, dict):
            return {str(key): compact(item, depth + 1) for key, item in list(value.items())[:24]} if depth < 7 else {}
        if isinstance(value, list):
            return [compact(item, depth + 1) for item in value[:8]] if depth < 7 else []
        return value
    result = compact(facts)
    # These are mandatory inputs even when a catalog or failure log is huge.
    result["knowledge_retrieval"] = compact(facts.get("knowledge_retrieval", {}))
    result["relevant_knowledge"] = compact(facts.get("relevant_knowledge", []))
    return result


def _query_terms(text: str) -> set[str]:
    normalized = text.casefold()
    terms = {part for part in re.findall(r"[a-z0-9_]+", normalized) if len(part) >= 2}
    for run in re.findall(r"[\u4e00-\u9fff]+", normalized):
        if len(run) <= 4:
            terms.add(run)
        for size in (2, 3, 4):
            terms.update(run[index : index + size] for index in range(max(0, len(run) - size + 1)))
    return terms


def _knowledge_guidance(records: object, limit: int = 4) -> list[str]:
    if not isinstance(records, list):
        return []
    guidance: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        value = record.get("solution") or record.get("usage") or record.get("topic") or record.get("action")
        text = " ".join(str(value or "").split())
        if text and text not in guidance:
            guidance.append(text[:240])
        if len(guidance) >= limit:
            break
    return guidance
