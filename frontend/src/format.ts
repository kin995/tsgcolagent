import type { FlowEvent, Metrics, Task } from './types';

export const paperAverages = (tasks: Task[], total = tasks.length) => {
  const complete = total > 0 && tasks.length === total;
  const observedExec = tasks.map((task) => task.metrics.exec_count_status === 'complete' ? task.metrics.exec_count : undefined)
    .filter((value): value is number => typeof value === 'number' && Number.isFinite(value) && value >= 0);
  const exact = complete && observedExec.length === total;
  const comparable = tasks.reduce((sum, task) => sum + (task.metrics.tir_comparable || 0), 0);
  const improved = tasks.reduce((sum, task) => sum + (task.metrics.tir_improved || 0), 0);
  return {
    pass_rate: complete ? tasks.filter((task) => task.metrics.pass_rate === 1).length / total : '—',
    avg_exec: observedExec.length ? observedExec.reduce((sum, value) => sum + value, 0) / observedExec.length : '—',
    avg_exec_status: exact ? 'complete' : observedExec.length ? 'partial_evidence' : 'incomplete_evidence',
    average_time_seconds: complete && tasks.every((task) => typeof task.duration_seconds === 'number') ? tasks.reduce((sum, task) => sum + (task.duration_seconds || 0), 0) / total : '—',
    tir: comparable ? improved / comparable : '—', tir_comparable: comparable, tir_improved: improved,
    tir_excluded: tasks.reduce((sum, task) => sum + (task.metrics.tir_excluded || 0), 0),
    exec_complete_tasks: observedExec.length,
    selected_tasks: total,
    knowledge_occurrence: exact ? new Set(tasks.flatMap((task) => task.metrics.knowledge_record_ids || [])).size : '—',
    knowledge_use_frequency: exact ? tasks.reduce((sum, task) => sum + (task.metrics.knowledge_use_frequency || 0), 0) : '—',
  };
};

export const pct = (value: unknown, digits = 1) => typeof value === 'number' ? `${(value * 100).toFixed(digits)}%` : '—';
export const shortPath = (value?: string) => value ? value.replaceAll('\\', '/').split('/').slice(-3).join('/') : '—';
export const gradeName = (grade: string) => ({ A: '高质量', B: '良好', C: '可用', D: '无效', E: '未完成' } as Record<string, string>)[grade] || '运行中';
export const statusName = (status?: string) => ({ active: '正在执行', done: '已有证据', pending: '等待执行', standby: '等待调度', running: '运行中', completed: '已完成', queued: '排队中', failed: '运行失败', mixed: '融合视图', waiting: '等待启动', ready: '已就绪' } as Record<string, string>)[status || ''] || status || '未知';
export const eventLabel = (event?: string) => ({
  KnowledgeRetrieved: '检索测试知识', KnowledgeUpdated: '学习本轮经验', TestIntentPlanned: '规划下一测试任务',
  TestGenerated: '生成候选测试', GenerationCompleted: '完成测试生成', ValidationCompleted: '完成执行验证',
  EvaluationCompleted: '完成质量评估', TestStateEvaluated: '评估测试状态', StateRepaired: '修复测试状态',
  ExecutionPathRecorded: '记录执行路径', RepairScopeLocated: '定位状态修复范围', CoverageEvidenceUpdated: '更新覆盖证据',
  MutationEvidenceUpdated: '更新变异证据', AssertionEvidenceUpdated: '更新断言证据', StructuredContextBuilt: '构建源码结构上下文',
  ContextExtracted: '提取测试上下文', InitialGraphBuilt: '建立初始状态模型', ExecutionObserved: '记录执行证据',
  CoverageObserved: '记录覆盖证据', MutationObserved: '记录变异证据', ValidationPassed: '验证测试通过',
  EvaluationFeedback: '反馈质量评估', TestStateInitialized: '初始化测试状态', TestCandidateValidated: '验证候选测试',
  CompletenessReached: '达到完整执行条件',
} as Record<string, string>)[event || ''] || event || '等待事件';

export const metricEntries = (metrics: Metrics) => [
  { id: 'passed', label: '测试通过', value: metrics.tests_passed == null ? '—' : metrics.tests_passed ? '是' : '否', raw: metrics.tests_passed },
  { id: 'tests', label: '测试数量', value: metrics.collected_count ?? '—', raw: metrics.collected_count },
  { id: 'line', label: 'LC', value: pct(metrics.line_coverage), raw: metrics.line_coverage },
  { id: 'branch', label: 'BC', value: pct(metrics.branch_coverage), raw: metrics.branch_coverage },
  { id: 'ae', label: 'AE', value: pct(metrics.ae), raw: metrics.ae },
  { id: 'mutation', label: 'MS', value: pct(metrics.mutation_score), raw: metrics.mutation_score },
  { id: 'time', label: 'Time/s', value: typeof metrics.duration_seconds === 'number' && Number.isFinite(metrics.duration_seconds) ? metrics.duration_seconds.toFixed(2) : '—', raw: metrics.duration_seconds },
  { id: 'pass_rate', label: 'PassRate', value: pct(metrics.pass_rate), raw: metrics.pass_rate },
  { id: 'avg_exec', label: 'AvgExec', value: typeof metrics.avg_exec === 'number' ? metrics.avg_exec.toFixed(2) : '—', raw: metrics.avg_exec },
  { id: 'tir', label: 'TIR', value: pct(metrics.tir), raw: metrics.tir },
];

export const eventSummary = (event?: FlowEvent | null) => event ? `${eventLabel(event.event_type)} · ${event.source_agent || '系统'}${event.target_agent ? ` → ${event.target_agent}` : ''}` : '暂无执行记录';

export const localDateTime = (value?: string) => {
  if (!value) return '等待记录';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '时间无效';
  return new Intl.DateTimeFormat('zh-CN', {
    year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
  }).format(date).replaceAll('/', '-');
};

export const durationText = (seconds?: number | null) => {
  if (typeof seconds !== 'number' || !Number.isFinite(seconds) || seconds < 0) return '等待开始';
  const total = Math.floor(seconds);
  const days = Math.floor(total / 86400);
  const hours = Math.floor((total % 86400) / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const rest = total % 60;
  const clock = [hours, minutes, rest].map((part) => String(part).padStart(2, '0')).join(':');
  return days ? `${days}天 ${clock}` : clock;
};

export const elapsedSeconds = (startedAt?: string, finishedAt?: string, fallback?: number, now = Date.now()) => {
  const started = startedAt ? new Date(startedAt).getTime() : Number.NaN;
  const finished = finishedAt ? new Date(finishedAt).getTime() : Number.NaN;
  if (Number.isFinite(started)) return Math.max(0, ((Number.isFinite(finished) ? finished : now) - started) / 1000);
  return typeof fallback === 'number' ? Math.max(0, fallback) : undefined;
};
