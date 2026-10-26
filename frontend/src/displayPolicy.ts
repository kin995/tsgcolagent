/** Process-only measurements must never be rendered as experiment results. */
const processMetricToken = /(^|_)(sfc|nrr|tsq|sfq|csg|pv|dvs|tic|rcs)($|_)/i;
const processMetricAlias = /(state_?flow_?coverage|state_?flow_?quality|test_?state_?quality|complete_?state_?graph|state_?graph_?completeness|path_?validity|non?_?regression_?rate|state_?coverage|overall_?quality_?(index|score)|combined_?coverage|coverage_?percent|quality_?(gain|delta)|success_?(rate|count)|node_?coverage|edge_?coverage|path_?coverage|test_?intent_?coverage)/i;
const processMetricChinese = /(测试状态覆盖|状态流覆盖率?|状态流质量|测试状态质量分数|状态图完整性|状态图完整度|完整状态图得分|组合覆盖|综合覆盖|路径有效性|无回退率|无退化率|无回归率|非回归率|状态覆盖率|综合质量指数)/;
const processMetricKeys = new Set([
  'completion_rate', 'quality_assessment', 'thresholds', 'threshold_failures',
  'weak_quality_flags', 'weak_quality_flag_counts', 'confidence', 'reward',
  'expected_reward', 'predicted_reward', 'expected_gain', 'gaps', 'gap_values',
  'metric_delta', 'before_metrics', 'after_metrics', 'node_coverage',
  'edge_coverage', 'path_coverage', 'priority', 'cost', 'no_regression_after_repair',
]);
const processMetricSuffixes = ['_gap', '_defect', '_confidence', '_reward', '_cost', '_rate'];
const identityListKeys = new Set([
  'target_methods', 'target_states', 'target_branches', 'method_names',
  'function_names', 'called_functions', 'called_methods', 'test_names', 'sources',
]);
const processScoreSuffixes = [
  'defect_score', 'coverage_score', 'quality_score', 'validation_score',
  'logic_score', 'rule_score', 'feasibility_score', 'repair_scope_score',
  'path_score', 'node_score', 'edge_score', 'match_score', 'ranking_score',
  'knowledge_score', 'evidence_score', 'confidence_score', 'reward_score',
  'completeness_score', 'agent_score', 'candidate_score', 'total_score',
  'oracle_score',
];

function normalized(value: string) {
  return value.replace(/([a-z\d])([A-Z])/g, '$1_$2').replace(/[^A-Za-z\d]+/g, '_').toLowerCase();
}

export function isProcessMetricKey(key: string): boolean {
  const name = normalized(key);
  if (name === 'mutation_score' || name === 'average_mutation_score' || name === 'pass_rate' || name === 'average_pass_rate') return false;
  return processMetricKeys.has(name) || name === 'score' || name.endsWith('_mutation_score') || processScoreSuffixes.some((suffix) => name.endsWith(suffix)) || processMetricSuffixes.some((suffix) => name.endsWith(suffix)) || processMetricToken.test(name) || processMetricAlias.test(name) || processMetricChinese.test(key);
}

export function containsProcessMetric(text: string): boolean {
  return isProcessMetricKey(text) || processMetricChinese.test(text);
}

function publicText(value: string): string {
  if (!containsProcessMetric(value)) return value;
  const lines = value.split(/\r?\n/).filter((line) => !containsProcessMetric(line));
  return lines.join('\n') || '内部过程信息已隐藏';
}

function isProcessMetricRecord(value: unknown): boolean {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const record = value as Record<string, unknown>;
  return ['metric', 'metric_id', 'id', 'name', 'label', 'key'].some((key) => typeof record[key] === 'string' && containsProcessMetric(record[key]));
}

/** Also handles saved experiment snapshots and event payloads from older runs. */
export function publicDisplayData<T>(value: T, parentKey = ''): T {
  if (typeof value === 'string') return (identityListKeys.has(parentKey) ? value : publicText(value)) as T;
  if (Array.isArray(value)) return value
    .filter((item) => identityListKeys.has(parentKey) || (!(typeof item === 'string' && containsProcessMetric(item)) && !isProcessMetricRecord(item)))
    .map((item) => publicDisplayData(item, parentKey)) as T;
  if (value && typeof value === 'object') return Object.fromEntries(
    Object.entries(value).filter(([key]) => !isProcessMetricKey(key)).map(([key, item]) => [key, publicDisplayData(item, key)]),
  ) as T;
  return value;
}
