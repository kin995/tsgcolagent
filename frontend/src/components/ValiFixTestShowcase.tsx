import { useEffect, useRef, useState } from 'react';
import { ArrowRight, BarChart3, CheckCircle2, ChevronRight, Code2, FileSearch2, FlaskConical, GitBranch, RotateCcw, ShieldCheck, Wrench, X } from 'lucide-react';
import promptTemplates from './valifix-prompts.json';
import { metricGroups, metrics, overallMetrics, projects, stageMetrics, stageNames, stageRecords, type MetricKey } from './valifixMetrics';
import './valifix-showcase.css';

type PromptGroup = 'context' | 'generation' | 'validation' | 'repair';

const promptGroups: Array<{ id: PromptGroup; label: string }> = [
  { id: 'context', label: 'Context' },
  { id: 'generation', label: 'Generation' },
  { id: 'validation', label: 'Validation' },
  { id: 'repair', label: 'Repair' },
];

const methodSteps = [
  { title: '整理上下文', icon: FileSearch2 },
  { title: '生成测试', icon: Code2 },
  { title: '动态验证', icon: ShieldCheck },
  { title: '最小修复', icon: Wrench },
  { title: '保留结果', icon: CheckCircle2 },
] as const;

const validationDimensions = [
  { title: '规则验证', detail: '语法、导入、类型匹配与变量名等基础约束。', points: ['核对包路径、导入和测试框架用法。', '定位语法、类型或变量名错误。'], example: '例如：把整数返回值赋给字符串变量会产生类型错误；修正变量类型即可保留其他测试代码。' },
  { title: '逻辑验证', detail: '正常流程、边界条件与异常处理的逻辑完整性。', points: ['比较断言预期与目标方法的实际行为。', '检查正常输入、边界输入和异常路径。'], example: '例如：对绝对值函数输入 −2 时，预期结果应为 2；断言写成 0 属于逻辑错误。' },
  { title: '可行性验证', detail: '测试能否编译、执行，以及环境是否匹配。', points: ['检查测试是否能在目标环境编译和运行。', '排查固定文件路径、缺失依赖等环境问题。'], example: '例如：测试依赖某台电脑上的固定临时目录；改用测试环境提供的临时目录后才能稳定执行。' },
] as const;

const repairLevels = [
  { title: '单行修复', tag: 'single', detail: '适用于孤立的断言使用、单行语法或边界条件错误；尽量不影响其他代码行。', points: ['只替换产生错误的代码行。', '修复后重新验证该错误及受影响的测试。'], example: '例如：绝对值函数的断言预期值写成 0，只将这一行的预期值改为 2。' },
  { title: '方法级修复', tag: 'method', detail: '当问题跨多行或涉及变量依赖时，只重构相关测试方法。', points: ['限定在一个测试方法内调整准备、调用和断言。', '保留同一测试类中已通过验证的其他方法。'], example: '例如：同一方法把正常输入与异常输入混用，需重新组织该方法里的变量、调用顺序和断言。' },
  { title: '类级修复', tag: 'class', detail: '当结构、继承、接口或成员变量相关问题跨越多个方法时，才扩大到测试类。', points: ['处理影响多个测试方法的共享成员和类结构。', '确认类级修改后原有通过项仍可执行。'], example: '例如：多个测试方法共用错误初始化的成员对象，需要修正测试类的共享初始化逻辑。' },
] as const;

export function ValiFixTestShowcase() {
  const [activeRepair, setActiveRepair] = useState(0);
  const [activeMetric, setActiveMetric] = useState<MetricKey>('lc');
  const [activePromptGroup, setActivePromptGroup] = useState<PromptGroup>('generation');
  const [activePromptId, setActivePromptId] = useState('ValiFixTestGenerationPrompt');
  const [activeStageProject, setActiveStageProject] = useState<(typeof projects)[number]['id']>('Cli');
  const [typeDialog, setTypeDialog] = useState<{ kind: 'validation' | 'repair'; index: number } | null>(null);
  const dialogClose = useRef<HTMLButtonElement>(null);
  const repair = repairLevels[activeRepair];
  const metric = metrics.find((item) => item.key === activeMetric)!;
  const prompt = promptTemplates.find((item) => item.id === activePromptId)!;
  const visiblePrompts = promptTemplates.filter((item) => item.group === activePromptGroup);
  const selectedStage = stageRecords.find((item) => item.project === activeStageProject)!;

  useEffect(() => {
    document.documentElement.scrollTop = 0;
    document.body.scrollTop = 0;
  }, []);

  useEffect(() => {
    if (!typeDialog) return;
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    dialogClose.current?.focus();
    const handleDialogKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setTypeDialog(null);
      if (event.key === 'Tab') { event.preventDefault(); dialogClose.current?.focus(); }
    };
    document.addEventListener('keydown', handleDialogKey);
    return () => { document.removeEventListener('keydown', handleDialogKey); previousFocus?.focus(); };
  }, [typeDialog]);

  const selectedType = typeDialog && (typeDialog.kind === 'validation' ? validationDimensions[typeDialog.index] : repairLevels[typeDialog.index]);

  return <div className="valifix-page">
    <section className="valifix-hero">
      <div>
        <span className="valifix-source-label"><GitBranch />METHOD VISUALIZATION</span>
        <h1>ValiFixTest <span>方法展示台</span></h1>
        <p>Generate unit tests from structured source context, validate each candidate, repair the smallest affected scope, and retain verified tests.</p>
        <div className="valifix-hero-tags"><span>动态验证</span><span>单行 → 方法 → 类级修复</span><span>保留已通过测试</span></div>
      </div>
    </section>

    <section className="valifix-panel valifix-prompt-panel" aria-labelledby="valifix-prompt-heading">
      <header className="valifix-section-heading"><span><Code2 /></span><div><h2 id="valifix-prompt-heading">提示工程</h2><p>Prompt Engineering</p></div><strong className="valifix-prompt-count">{promptTemplates.length} PROMPTS</strong></header>
      <div className="valifix-prompt-groups" role="group" aria-label="提示词类别">
        {promptGroups.map((group) => <button type="button" key={group.id} className={activePromptGroup === group.id ? 'selected' : ''} aria-pressed={activePromptGroup === group.id} onClick={() => { setActivePromptGroup(group.id); setActivePromptId(promptTemplates.find((item) => item.group === group.id)!.id); }}>{group.label}<small>{promptTemplates.filter((item) => item.group === group.id).length}</small></button>)}
      </div>
      <div className="valifix-prompt-layout">
        <div className="valifix-prompt-list" role="group" aria-label="选择提示词">
          {visiblePrompts.map((item, index) => <button type="button" key={item.id} className={activePromptId === item.id ? 'selected' : ''} aria-pressed={activePromptId === item.id} onClick={() => setActivePromptId(item.id)}><small>{String(index + 1).padStart(2, '0')}</small><span>{item.title}</span><ChevronRight aria-hidden="true" /></button>)}
        </div>
        <div className="valifix-prompt-display" aria-live="polite"><header><span>{activePromptGroup.toUpperCase()}</span><strong>{prompt.title}</strong></header><pre>{prompt.content}</pre></div>
      </div>
      <div className="valifix-compact-flow"><div className="valifix-compact-flow-heading"><GitBranch aria-hidden="true" /><strong>方法流程</strong><small>Context → Generation → Validation → Repair → Retain</small></div><div className="valifix-compact-steps">{methodSteps.map((step, index) => <div key={step.title}><span>{String(index + 1).padStart(2, '0')}</span><step.icon aria-hidden="true" /><b>{step.title}</b>{index < methodSteps.length - 1 && <ArrowRight className="valifix-compact-arrow" aria-hidden="true" />}</div>)}</div><p><RotateCcw aria-hidden="true" />修复后重新验证相关错误；多轮修复仍失败时保留已通过的测试。</p></div>
    </section>

    <section className="valifix-mechanisms" aria-label="动态验证与最小目标修复">
      <div className="valifix-panel"><header className="valifix-section-heading"><span><FlaskConical /></span><div><h2>动态验证看什么</h2><p>Locate a failure before selecting a repair scope.</p></div></header><div className="valifix-validation-list">{validationDimensions.map((item, index) => <button type="button" key={item.title} onClick={() => setTypeDialog({ kind: 'validation', index })} aria-haspopup="dialog"><i>{String(index + 1).padStart(2, '0')}</i><span><strong>{item.title}</strong><small>{item.detail}</small></span><ChevronRight aria-hidden="true" /></button>)}</div><p className="valifix-mechanism-note">验证报告记录错误位置、类型与历史。已通过且与错误无关的测试方法继续保留。</p></div>
      <div className="valifix-panel"><header className="valifix-section-heading"><span><Wrench /></span><div><h2>修复范围如何升级</h2><p>Start with the smallest change and expand only when needed.</p></div></header><div className="valifix-repair-list" role="group" aria-label="最小目标修复层级">{repairLevels.map((item, index) => <button type="button" key={item.tag} className={activeRepair === index ? 'selected' : ''} onClick={() => { setActiveRepair(index); setTypeDialog({ kind: 'repair', index }); }} aria-pressed={activeRepair === index} aria-haspopup="dialog"><span>{index + 1}</span><strong>{item.title}</strong><small>{item.tag}</small></button>)}</div><div className="valifix-repair-detail" aria-live="polite"><strong>{repair.title}</strong><p>{repair.detail}</p></div><p className="valifix-mechanism-note">若修复不能收敛，PreserveMaxCorrect 保留已验证通过的测试内容。</p></div>
    </section>

    <section className="valifix-panel" aria-labelledby="valifix-evidence-heading"><header className="valifix-section-heading"><span><BarChart3 /></span><div><h2 id="valifix-evidence-heading">项目指标对比</h2><p>Compare quality measures across four Defects4J projects.</p></div></header>
      <div className="valifix-metric-groups">{metricGroups.map((group) => <div className="valifix-metric-group" key={group}><h3>{group}</h3><div className="valifix-metric-tabs" role="group" aria-label={`${group}指标`}>{metrics.filter((item) => item.group === group).map((item) => <button type="button" key={item.key} className={activeMetric === item.key ? 'selected' : ''} onClick={() => setActiveMetric(item.key)} aria-pressed={activeMetric === item.key}><b>{item.label}</b><span>{item.fullName}</span></button>)}</div></div>)}</div>
      <div className="valifix-comparison"><div className="valifix-comparison-heading"><strong>{metric.fullName} · {metric.label}</strong><p>{metric.description}</p>{metric.unit !== '%' && <small>Bars are scaled to the highest value among the four projects.</small>}</div>{projects.map((project, index) => <div className="valifix-project-row" key={project.id}><span>{project.name}</span><div className="valifix-bar-track"><i style={{ width: `${Math.max(0, Math.min(100, metric.values[index] / (metric.unit === '%' ? 100 : Math.max(...metric.values)) * 100))}%` }} /></div><strong>{metric.values[index].toFixed(2)}{metric.unit === '%' ? '%' : ' 条/方法'}</strong></div>)}</div>
      <div className="valifix-stage-section"><header><div><h3>修复阶段指标</h3><p>Compare test volume and coverage at each method stage.</p></div><span>{stageMetrics.length} METRICS</span></header><div className="valifix-stage-projects" role="group" aria-label="选择阶段数据项目">{projects.map((project) => <button type="button" key={project.id} className={activeStageProject === project.id ? 'selected' : ''} aria-pressed={activeStageProject === project.id} onClick={() => setActiveStageProject(project.id)}>{project.name}</button>)}</div><div className="valifix-stage-table-wrap"><table><thead><tr><th scope="col">阶段</th>{stageMetrics.map((item) => <th scope="col" key={item.key}>{item.label}<small>{item.unit}</small></th>)}</tr></thead><tbody>{selectedStage.stages.map((row, index) => <tr key={stageNames[index].code}><th scope="row"><b>{stageNames[index].code}</b><span>{stageNames[index].title}</span></th>{stageMetrics.map((item) => <td key={item.key}>{item.unit === '%' ? `${row[item.key].toFixed(2)}%` : row[item.key].toLocaleString('en-US')}</td>)}</tr>)}</tbody></table></div><p className="valifix-stage-note">F2 的数量表示精简的测试片段；Brc 和 Lc 记录覆盖变化量。</p></div>
      <div className="valifix-overall-metrics"><header><h3>整体指标</h3><p>Method-wide quality and resource measures.</p></header><div>{overallMetrics.map((item) => <article key={item.label}><small>{item.label}</small><strong>{item.value}</strong><span>{item.note}</span></article>)}</div></div>
    </section>
    {selectedType && <div className="valifix-dialog-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) setTypeDialog(null); }}>
      <section className="valifix-type-dialog" role="dialog" aria-modal="true" aria-labelledby="valifix-type-title">
        <header><span>{typeDialog?.kind === 'validation' ? 'VALIDATION TYPE' : 'REPAIR SCOPE'}</span><button type="button" ref={dialogClose} onClick={() => setTypeDialog(null)} aria-label="关闭类型说明"><X /></button></header>
        <h2 id="valifix-type-title">{selectedType.title}</h2><p>{selectedType.detail}</p>
        <div className="valifix-dialog-section"><h3>{typeDialog?.kind === 'validation' ? '检查重点' : '处理方式'}</h3><ul>{selectedType.points.map((point) => <li key={point}>{point}</li>)}</ul></div>
        <div className="valifix-dialog-section example"><h3>示例</h3><p>{selectedType.example}</p></div>
      </section>
    </div>}
  </div>;
}
