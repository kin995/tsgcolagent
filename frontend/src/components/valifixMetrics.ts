export const projects = [
  { id: 'Cli', name: 'Commons-Cli' },
  { id: 'Csv', name: 'Commons-Csv' },
  { id: 'Lang', name: 'Commons-Lang' },
  { id: 'Gson', name: 'Gson' },
] as const;

export const metrics = [
  { key: 'inc', label: 'Inc', fullName: '指令覆盖率', description: '已执行的字节码指令占全部可计数指令的比例。', unit: '%', group: '覆盖率', values: [74.40, 59.40, 80.12, 65.95] },
  { key: 'brc', label: 'Brc', fullName: '分支覆盖率', description: '已覆盖的条件分支占全部条件分支的比例。', unit: '%', group: '覆盖率', values: [60.90, 45.20, 62.37, 57.70] },
  { key: 'cxty', label: 'Cxty', fullName: '圈复杂度覆盖率', description: '已覆盖的复杂度单元占全部复杂度单元的比例。', unit: '%', group: '覆盖率', values: [66.00, 46.00, 64.38, 54.75] },
  { key: 'lc', label: 'LC', fullName: '行覆盖率', description: '生成测试执行到的有效代码行比例。', unit: '%', group: '覆盖率', values: [79.40, 66.50, 78.28, 67.15] },
  { key: 'mc', label: 'MC', fullName: '变异覆盖率', description: '测试杀死的变异体占有效变异体的比例。', unit: '%', group: '测试质量', values: [48.00, 45.75, 58.91, 53.22] },
  { key: 'ts', label: 'TS', fullName: '测试强度', description: '被测试杀死的变异体占测试覆盖到的变异体的比例。', unit: '%', group: '测试质量', values: [80.71, 74.80, 78.00, 79.00] },
  { key: 'ad', label: 'AD', fullName: '断言密度', description: '每个测试方法平均包含的断言数量。', unit: '条/方法', group: '测试质量', values: [2.32, 3.07, 6.16, 2.42] },
  { key: 'mar', label: 'MAR', fullName: '有意义断言比例', description: '有意义断言占全部断言的比例。', unit: '%', group: '测试质量', values: [96.96, 99.60, 99.33, 99.17] },
  { key: 'fdr', label: 'FDR', fullName: '缺陷检出率', description: '在故障版本失败且在修复版本通过的缺陷样本比例。', unit: '%', group: '测试质量', values: [79.27, 70.00, 69.74, 80.99] },
  { key: 'ftr', label: 'FTR', fullName: '不稳定测试率', description: '重复执行时出现通过与失败不一致的测试方法比例。', unit: '%', group: '测试质量', values: [3.90, 2.70, 8.30, 7.60] },
  { key: 'correct', label: 'Correct Rate', fullName: '语义正确率', description: '生成测试的语义正确程度。', unit: '%', group: '生成结果', values: [64.01, 37.33, 36.55, 48.56] },
  { key: 'pass', label: 'Pass Rate', fullName: '测试通过率', description: '生成测试实际执行通过的比例。', unit: '%', group: '生成结果', values: [70.01, 54.33, 52.55, 59.56] },
  { key: 'fail', label: 'Fail Rate', fullName: '测试失败率', description: '生成测试执行失败的比例。', unit: '%', group: '生成结果', values: [6.01, 8.33, 3.55, 7.56] },
  // The source figure swaps the labels of these two error series; the tabulated values
  // are used here because the accompanying text confirms Lang Build Error Rate = 6.55%.
  { key: 'build', label: 'Build Error Rate', fullName: '构建错误率', description: '生成结果中的构建错误比例。', unit: '%', group: '生成结果', values: [3.01, 12.33, 6.55, 11.56] },
  { key: 'syntax', label: 'Syntax Error Rate', fullName: '语法错误率', description: '生成结果中的语法错误比例。', unit: '%', group: '生成结果', values: [27.01, 28.33, 11.55, 20.56] },
] as const;

export type MetricKey = (typeof metrics)[number]['key'];
export const metricGroups = ['覆盖率', '测试质量', '生成结果'] as const;

export const stageMetrics = [
  { key: 'tm', label: 'TM', title: '测试方法数', unit: '个' },
  { key: 'tloc', label: 'TLOC', title: '测试代码行', unit: '行' },
  { key: 'assert', label: 'ASSERT', title: '断言数', unit: '条' },
  { key: 'brc', label: 'Brc', title: '分支覆盖', unit: '%' },
  { key: 'lc', label: 'Lc', title: '行覆盖', unit: '%' },
] as const;

type StageValues = { tm: number; tloc: number; assert: number; brc: number; lc: number };
type ProjectStages = { project: (typeof projects)[number]['id']; stages: [StageValues, StageValues, StageValues, StageValues] };

export const stageRecords: ProjectStages[] = [
  { project: 'Cli', stages: [
    { tm: 317, tloc: 4154, assert: 724, brc: 52.37, lc: 57.25 },
    { tm: 26, tloc: 478, assert: 52, brc: 62.33, lc: 79.72 },
    { tm: 15, tloc: 289, assert: 22, brc: -1.34, lc: -0.32 },
    { tm: 302, tloc: 3865, assert: 702, brc: 60.90, lc: 79.40 },
  ] },
  { project: 'Csv', stages: [
    { tm: 405, tloc: 6907, assert: 1490, brc: 40.00, lc: 46.21 },
    { tm: 7, tloc: 1223, assert: 26, brc: 45.38, lc: 66.94 },
    { tm: 2, tloc: 346, assert: 6, brc: -0.18, lc: -0.44 },
    { tm: 483, tloc: 6561, assert: 1484, brc: 45.20, lc: 66.50 },
  ] },
  { project: 'Lang', stages: [
    { tm: 2051, tloc: 33341, assert: 12618, brc: 49.73, lc: 52.55 },
    { tm: 24, tloc: 3735, assert: 142, brc: 66.90, lc: 78.68 },
    { tm: 17, tloc: 2456, assert: 85, brc: -4.53, lc: -0.40 },
    { tm: 2034, tloc: 30885, assert: 12533, brc: 62.37, lc: 78.28 },
  ] },
  { project: 'Gson', stages: [
    { tm: 1304, tloc: 20499, assert: 3146, brc: 44.25, lc: 49.56 },
    { tm: 35, tloc: 783, assert: 105, brc: 58.56, lc: 68.65 },
    { tm: 13, tloc: 400, assert: 26, brc: -0.86, lc: -1.50 },
    { tm: 1291, tloc: 20099, assert: 3120, brc: 57.70, lc: 67.15 },
  ] },
];

export const stageNames = [
  { code: 'F0', title: '修复前' },
  { code: 'F1', title: '局部修复' },
  { code: 'F2', title: '片段精简' },
  { code: 'F3', title: '最终输出' },
] as const;

export const overallMetrics = [
  { label: 'Oracle 行为一致率', value: '85.8%', note: '基于 200 个测试方法的人工核查' },
  { label: 'Cohen’s κ', value: '0.87', note: '两名评审者的一致性系数' },
  { label: 'Token Cost', value: '$2.515', note: '四个项目、六次运行的合计成本' },
] as const;
