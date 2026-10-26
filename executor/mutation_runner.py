from __future__ import annotations

import ast
import os
import shutil
import subprocess
from metrics.measurement import measured_python_run
import sys
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class MutantStatus(Enum):
    KILLED = "killed"
    SURVIVED = "survived"
    TIMEOUT = "timeout"
    INVALID = "invalid"
    INFRA_ERROR = "infra_error"
    EQUIVALENT = "equivalent"


@dataclass
class MutationResult:
    killed: int
    survived: int
    timeout: int
    invalid: int
    infra_error: int
    equivalent: int
    total: int
    score: float | None
    killed_lines: list[int]
    survived_lines: list[int]
    timeout_lines: list[int]
    invalid_lines: list[int]
    infra_error_lines: list[int]
    equivalent_lines: list[int]
    details: list[str]
    candidate_count: int = 0
    attempted_mutants: int = 0
    low_sample_reason: str = ""
    autonomy_action: str = ""
    timeout_retries: int = 0
    evaluation_profile: str = "full"
    profile_limit: int = 20
    selected_candidate_lines: list[int] | None = None
    candidate_results: list[dict[str, object]] | None = None

    def to_dict(self) -> dict[str, object]:
        effective_mutants = self.killed + self.survived
        low_sample_reason = self.low_sample_reason or _low_sample_reason(
            effective_mutants=effective_mutants,
            candidate_count=self.candidate_count,
            attempted_mutants=self.attempted_mutants,
            timeout_count=self.timeout,
            invalid=self.invalid,
            infra_error=self.infra_error,
            equivalent=self.equivalent,
            details=self.details,
        )
        return {
            "killed": self.killed,
            "survived": self.survived,
            "timeout": self.timeout,
            "invalid": self.invalid,
            "infra_error": self.infra_error,
            "equivalent": self.equivalent,
            "total": self.total,
            "effective_mutants": effective_mutants,
            "score_reliability": "unavailable" if effective_mutants == 0 else "low" if effective_mutants < 3 else "normal",
            "score": self.score,
            "killed_lines": self.killed_lines,
            "survived_lines": self.survived_lines,
            "timeout_lines": self.timeout_lines,
            "invalid_lines": self.invalid_lines,
            "infra_error_lines": self.infra_error_lines,
            "equivalent_lines": self.equivalent_lines,
            "details": self.details,
            "candidate_count": self.candidate_count,
            "attempted_mutants": self.attempted_mutants,
            "low_sample_reason": low_sample_reason,
            "autonomy_action": self.autonomy_action or _autonomy_action(low_sample_reason, self.timeout),
            "timeout_retries": self.timeout_retries,
            "evaluation_profile": self.evaluation_profile,
            "profile_limit": self.profile_limit,
            "selected_candidate_lines": self.selected_candidate_lines or [],
            "candidate_results": self.candidate_results or [],
            "final_evaluation": self.evaluation_profile == "full",
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "MutationResult":
        return cls(
            killed=int(data.get("killed", 0) or 0),
            survived=int(data.get("survived", 0) or 0),
            timeout=int(data.get("timeout", 0) or 0),
            invalid=int(data.get("invalid", 0) or 0),
            infra_error=int(data.get("infra_error", 0) or 0),
            equivalent=int(data.get("equivalent", 0) or 0),
            total=int(data.get("total", 0) or 0),
            score=None if data.get("score") is None else float(data["score"]),
            killed_lines=[int(value) for value in data.get("killed_lines", []) or []],
            survived_lines=[int(value) for value in data.get("survived_lines", []) or []],
            timeout_lines=[int(value) for value in data.get("timeout_lines", []) or []],
            invalid_lines=[int(value) for value in data.get("invalid_lines", []) or []],
            infra_error_lines=[int(value) for value in data.get("infra_error_lines", []) or []],
            equivalent_lines=[int(value) for value in data.get("equivalent_lines", []) or []],
            details=[str(value) for value in data.get("details", []) or []],
            candidate_count=int(data.get("candidate_count", 0) or 0),
            attempted_mutants=int(data.get("attempted_mutants", 0) or 0),
            low_sample_reason=str(data.get("low_sample_reason") or ""),
            autonomy_action=str(data.get("autonomy_action") or ""),
            timeout_retries=int(data.get("timeout_retries", 0) or 0),
            evaluation_profile=str(data.get("evaluation_profile") or "full"),
            profile_limit=int(data.get("profile_limit", 20) or 20),
            selected_candidate_lines=[int(value) for value in data.get("selected_candidate_lines", []) or []],
            candidate_results=[dict(value) for value in data.get("candidate_results", []) or [] if isinstance(value, dict)],
        )


class _SimpleMutator(ast.NodeTransformer):
    def __init__(self) -> None:
        self.mutations: list[tuple[int, ast.AST]] = []

    def collect(self, tree: ast.AST) -> list[tuple[int, ast.AST]]:
        self.mutations = []
        self.visit(tree)
        return self.mutations

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if node.ops:
            self.mutations.append((getattr(node, "lineno", 1), node))
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)):
            self.mutations.append((getattr(node, "lineno", 1), node))
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            self.mutations.append((getattr(node, "lineno", 1), node))
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, (ast.And, ast.Or)):
            self.mutations.append((getattr(node, "lineno", 1), node))
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, bool):
            self.mutations.append((getattr(node, "lineno", 1), node))
        elif isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            self.mutations.append((getattr(node, "lineno", 1), node))
        return node


class _ApplyOneMutation(ast.NodeTransformer):
    def __init__(self, target_index: int):
        self.target_index = target_index
        self.current_index = -1

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if node.ops:
            self.current_index += 1
            if self.current_index == self.target_index:
                node.ops[0] = self._flip_compare(node.ops[0])
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)):
            self.current_index += 1
            if self.current_index == self.target_index:
                node.op = ast.Sub() if isinstance(node.op, ast.Add) else ast.Add()
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            self.current_index += 1
            if self.current_index == self.target_index:
                return ast.copy_location(node.operand, node)
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, (ast.And, ast.Or)):
            self.current_index += 1
            if self.current_index == self.target_index:
                node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, bool):
            self.current_index += 1
            if self.current_index == self.target_index:
                return ast.copy_location(ast.Constant(value=not node.value), node)
        elif isinstance(node.value, int) and not isinstance(node.value, bool):
            self.current_index += 1
            if self.current_index == self.target_index:
                delta = 1 if node.value >= 0 else -1
                return ast.copy_location(ast.Constant(value=node.value + delta), node)
        elif isinstance(node.value, float):
            self.current_index += 1
            if self.current_index == self.target_index:
                delta = 1.0 if node.value >= 0 else -1.0
                return ast.copy_location(ast.Constant(value=node.value + delta), node)
        return node

    def _flip_compare(self, op: ast.cmpop) -> ast.cmpop:
        mapping = {
            ast.Gt: ast.LtE,
            ast.GtE: ast.Lt,
            ast.Lt: ast.GtE,
            ast.LtE: ast.Gt,
            ast.Eq: ast.NotEq,
            ast.NotEq: ast.Eq,
            ast.Is: ast.IsNot,
            ast.IsNot: ast.Is,
            ast.In: ast.NotIn,
            ast.NotIn: ast.In,
        }
        return mapping.get(type(op), ast.NotEq)()


class MutationRunner:
    def __init__(self, max_mutants: int = 20, timeout_retry_multiplier: int = 3):
        self.max_mutants = max_mutants
        self.timeout_retry_multiplier = timeout_retry_multiplier

    def run(
        self,
        source_path: str | Path,
        test_path: str | Path,
        cwd: str | Path | None = None,
        timeout: int = 5,
        *,
        evaluation_profile: str = "full",
        priority_lines: set[int] | None = None,
        prior_result: MutationResult | None = None,
    ) -> MutationResult:
        source_path = Path(source_path).resolve()
        test_path = Path(test_path).resolve()
        profile = "screening" if evaluation_profile == "screening" else "full"
        profile_limit = min(self.max_mutants, 8) if profile == "screening" else self.max_mutants
        source_text = source_path.read_text(encoding="utf-8")
        timeout_retries = 0
        baseline = self._run_pytest(test_path=test_path, target_source=source_path, cwd=Path(cwd or test_path.parent), timeout=timeout)
        if baseline == MutantStatus.TIMEOUT:
            timeout_retries += 1
            baseline = self._run_pytest(
                test_path=test_path,
                target_source=source_path,
                cwd=Path(cwd or test_path.parent),
                timeout=max(timeout + 1, timeout * self.timeout_retry_multiplier),
            )
        if baseline != MutantStatus.SURVIVED:
            reason = f"基线测试无法稳定通过，状态为 {baseline.value}，因此本轮不计算变异分数"
            return MutationResult(0, 0, 1 if baseline == MutantStatus.TIMEOUT else 0, 1, 0, 0, 0, None, [], [], [0] if baseline == MutantStatus.TIMEOUT else [], [0], [], [], [f"baseline test invalid: {baseline.value}", reason], 0, 0, reason, "已对基线超时进行一次更长时间重试" if timeout_retries else "等待执行智能体先修复基线测试", timeout_retries, profile, profile_limit, [])
        try:
            tree = ast.parse(source_text)
        except SyntaxError as exc:
            reason = f"源码语法错误导致无法生成变异体：{exc}"
            return MutationResult(0, 0, 0, 1, 0, 0, 0, None, [], [], [], [0], [], [], [reason], 0, 0, reason, "等待源码或任务输入修复后再运行变异", timeout_retries, profile, profile_limit, [])
        candidates = _SimpleMutator().collect(tree)
        if not candidates:
            reason = "源码中未发现可变异的比较、算术、布尔或常量节点"
            return MutationResult(0, 0, 0, 0, 0, 0, 0, None, [], [], [], [], [], [], ["no mutation candidates", reason], 0, 0, reason, "规划阶段应增加覆盖边界和分支区分断言；若源码本身过短则报告为低样本", timeout_retries, profile, profile_limit, [])

        killed = 0
        survived = 0
        timeout_count = 0
        invalid = 0
        infra_error = 0
        equivalent = 0
        killed_lines: list[int] = []
        survived_lines: list[int] = []
        timeout_lines: list[int] = []
        invalid_lines: list[int] = []
        infra_error_lines: list[int] = []
        equivalent_lines: list[int] = []
        details: list[str] = []
        candidate_results: list[dict[str, object]] = []

        selected = self._select_candidates(candidates, profile_limit, priority_lines or set())
        attempted_mutants = len(selected)
        prior_items = prior_result.candidate_results or [] if prior_result is not None else []
        prior_by_id = {
            int(item["id"]): str(item.get("status") or "")
            for item in prior_items
            if isinstance(item, dict) and str(item.get("id", "")).isdigit()
        }
        for index, line, _ in selected:
            prior_status = prior_by_id.get(index)
            if prior_status:
                status = MutantStatus(prior_status)
                details.append(f"reused {status.value} mutation at line {line}")
            else:
                status = None
            with tempfile.TemporaryDirectory(prefix="stateflow_mut_") as temp:
                if status is None:
                    temp_root = Path(temp)
                    mutant_source = temp_root / source_path.name
                    mutant_test = temp_root / test_path.name
                    mutated_tree = _ApplyOneMutation(index).visit(ast.parse(source_text))
                    ast.fix_missing_locations(mutated_tree)
                    mutant_source.write_text(ast.unparse(mutated_tree), encoding="utf-8")
                    shutil.copy2(test_path, mutant_test)
                    status = self._run_pytest(mutant_test, mutant_source, temp_root, timeout)
                    if status == MutantStatus.TIMEOUT:
                        timeout_retries += 1
                        retry_timeout = max(timeout + 1, timeout * self.timeout_retry_multiplier)
                        retry_status = self._run_pytest(mutant_test, mutant_source, temp_root, retry_timeout)
                        details.append(f"timeout retry mutation at line {line} with {retry_timeout}s -> {retry_status.value}")
                        status = retry_status
                candidate_results.append({"id": index, "line": line, "status": status.value})
                if status == MutantStatus.KILLED:
                    killed += 1
                    killed_lines.append(line)
                    details.append(f"killed mutation at line {line}")
                elif status == MutantStatus.SURVIVED:
                    survived += 1
                    survived_lines.append(line)
                    details.append(f"survived mutation at line {line}")
                elif status == MutantStatus.TIMEOUT:
                    timeout_count += 1
                    timeout_lines.append(line)
                    details.append(f"timeout mutation at line {line}")
                elif status == MutantStatus.INFRA_ERROR:
                    infra_error += 1
                    infra_error_lines.append(line)
                    details.append(f"infra_error mutation at line {line}")
                elif status == MutantStatus.INVALID:
                    invalid += 1
                    invalid_lines.append(line)
                    details.append(f"invalid mutation at line {line}")
                else:
                    equivalent += 1
                    equivalent_lines.append(line)
                    details.append(f"equivalent mutation at line {line}")
        total = killed + survived
        score = round(killed / total, 4) if total else None
        low_sample_reason = _low_sample_reason(
            effective_mutants=total,
            candidate_count=len(candidates),
            attempted_mutants=attempted_mutants,
            timeout_count=timeout_count,
            invalid=invalid,
            infra_error=infra_error,
            equivalent=equivalent,
            details=details,
        )
        return MutationResult(
            killed,
            survived,
            timeout_count,
            invalid,
            infra_error,
            equivalent,
            total,
            score,
            killed_lines,
            survived_lines,
            timeout_lines,
            invalid_lines,
            infra_error_lines,
            equivalent_lines,
            details,
            len(candidates),
            attempted_mutants,
            low_sample_reason,
            _autonomy_action(low_sample_reason, timeout_count),
            timeout_retries,
            profile,
            profile_limit,
            [line for _, line, _ in selected],
            candidate_results,
        )

    def _select_candidates(
        self,
        candidates: list[tuple[int, ast.AST]],
        limit: int,
        priority_lines: set[int],
    ) -> list[tuple[int, int, ast.AST]]:
        indexed = [(index, line, node) for index, (line, node) in enumerate(candidates)]
        if not priority_lines:
            return indexed[:limit]

        def score(item: tuple[int, int, ast.AST]) -> tuple[int, int, int]:
            index, line, node = item
            distance = min((abs(line - target) for target in priority_lines), default=10_000)
            proximity = 100 if distance == 0 else 60 if distance == 1 else 30 if distance <= 3 else 0
            semantic = 5 if isinstance(node, ast.Compare) else 4 if isinstance(node, ast.BoolOp) else 3 if isinstance(node, (ast.BinOp, ast.UnaryOp)) else 1
            return proximity, semantic, -index

        return sorted(indexed, key=score, reverse=True)[:limit]

    def _run_pytest(self, test_path: Path, target_source: Path, cwd: Path, timeout: int) -> MutantStatus:
        cwd = cwd.resolve()
        test_path = test_path.resolve()
        try:
            test_arg = str(test_path.relative_to(cwd))
        except ValueError:
            cwd = test_path.parent
            test_arg = test_path.name
        command = [sys.executable, "-m", "pytest", test_arg, "-q"]
        env = os.environ.copy()
        env["STATEFLOW_TARGET"] = str(target_source)
        try:
            completed = measured_python_run(
                command,
                kind="python_mutation",
                cwd=str(cwd),
                text=True,
                capture_output=True,
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return MutantStatus.TIMEOUT
        output = f"{completed.stdout}\n{completed.stderr}"
        if completed.returncode == 0:
            return MutantStatus.SURVIVED
        if completed.returncode in {2, 3, 4, 5} or self._looks_like_infra_error(output):
            return MutantStatus.INFRA_ERROR
        if completed.returncode == 1:
            return MutantStatus.KILLED
        return MutantStatus.INVALID

    def _looks_like_infra_error(self, output: str) -> bool:
        lowered = output.lower()
        markers = [
            "importerror",
            "modulenotfounderror",
            "fixture",
            "internalerror",
            "usageerror",
            "collection",
            "error collecting",
        ]
        return any(marker in lowered for marker in markers)


def _low_sample_reason(
    effective_mutants: int,
    candidate_count: int,
    attempted_mutants: int,
    timeout_count: int,
    invalid: int,
    infra_error: int,
    equivalent: int,
    details: list[str],
) -> str:
    if effective_mutants >= 3:
        return ""
    if candidate_count == 0:
        return "源码中缺少可变异结构，无法形成足够变异样本"
    if attempted_mutants < 3:
        return f"源码只发现 {candidate_count} 个可变异位置，最多只能尝试 {attempted_mutants} 个样本"
    non_effective = timeout_count + invalid + infra_error + equivalent
    if non_effective:
        parts = []
        if timeout_count:
            parts.append(f"超时 {timeout_count}")
        if invalid:
            parts.append(f"无效 {invalid}")
        if infra_error:
            parts.append(f"环境错误 {infra_error}")
        if equivalent:
            parts.append(f"疑似等价 {equivalent}")
        return "有效变异体偏少，主要因为" + "、".join(parts)
    if any("baseline test invalid" in item for item in details):
        return "基线测试未稳定通过，变异分数不可用"
    return "有效变异体少于 3 个，变异分数只作参考"


def _autonomy_action(low_sample_reason: str, timeout_count: int) -> str:
    if timeout_count:
        return "已对超时变异体自动进行一次更长时间重试；仍超时的样本会单独报告，后续规划应补更小输入或边界断言"
    if "源码只发现" in low_sample_reason or "缺少可变异结构" in low_sample_reason:
        return "系统会把该任务标记为低样本变异，不把高分当作强证据，并建议增加边界和分支区分断言"
    if low_sample_reason:
        return "系统会把低样本原因写入报告，并在下一轮规划中优先补正常行为、边界路径和杀变异断言"
    return ""
