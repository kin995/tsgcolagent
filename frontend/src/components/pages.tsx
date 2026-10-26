import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import {
  Activity, AlertTriangle, BarChart3, BookOpenCheck, Bot, Boxes, Braces, BrainCircuit, CheckCircle2, ChevronLeft, ChevronRight,
  CircleDot, Clock3, Code2, Cpu, Database, FileCode2, Flag, Gauge, GitBranch, Layers3, Network, Radar, Search, ShieldCheck,
  TestTube2, Timer, Workflow, XCircle,
} from 'lucide-react';
import type { DetailContent } from './DetailDrawer';
import { DetailList, StateModelVisual } from './DetailDrawer';
import { durationText, elapsedSeconds, eventLabel, eventSummary, gradeName, localDateTime, metricEntries, pct, shortPath, statusName } from '../format';
import { containsProcessMetric, isProcessMetricKey } from '../displayPolicy';
import type { AgentState, CatalogAgent, FlowEvent, KnowledgeSnapshot, RuntimeSnapshot, StateEdge, StateNode, Task } from '../types';
import './overview-evaluation.css';
import './task-summary.css';
import './run-model.css';
import './dataset-report-links.css';
import './knowledge-provenance.css';

type OpenDetail = (detail: DetailContent) => void;

export function OverviewPage({ runtime, open }: { runtime: RuntimeSnapshot; open: OpenDetail }) {
  const o = runtime.overview; const grades = o.grade_counts || {}; const current = runtime.current_task;
  const stats = [
    { label: '任务总数', value: o.total_tasks || 0, icon: FileCode2, detail: '本实验已经登记或发现的测试生成任务。', data: { total_tasks: o.total_tasks, completed_execution: o.completed_execution, not_completed_execution: o.not_completed_execution } },
    { label: '完整执行 A/B/C', value: o.completed_execution || 0, icon: CheckCircle2, detail: '测试真实运行并形成有效质量结论的任务。', data: { completed_execution: o.completed_execution, grade_counts: { A: grades.A || 0, B: grades.B || 0, C: grades.C || 0 }, tasks: runtime.tasks.filter((task) => ['A', 'B', 'C'].includes(task.grade)) } },
    { label: '未完整执行 D/E', value: o.not_completed_execution || 0, icon: XCircle, detail: '执行无效、被中断或缺少必要证据的任务。', data: { not_completed_execution: o.not_completed_execution, grade_counts: { D: grades.D || 0, E: grades.E || 0 }, failed_tasks: runtime.tasks.filter((task) => ['D', 'E'].includes(task.grade)), failure_categories: runtime.failures } },
    { label: '进入后续轮次', value: o.followup_task_count || 0, icon: Workflow, detail: 'R2只恢复D/E；R3统一优化全部C；最多三轮。', data: { followup_task_count: o.followup_task_count, followup_attempts: o.followup_attempts, tasks: runtime.tasks.filter((task) => task.round > 1) } },
  ];
  return <div className="page-stack">
    <PageTitle title={o.page_title || '实验总览'} />
    <ExperimentTiming runtime={runtime} open={open} />
    <ParallelExecution runtime={runtime} open={open} />
    {runtime.portfolio && <PortfolioBoard runtime={runtime} open={open} />}
    <section className="stat-grid">{stats.map(({ label, value, icon: Icon, detail, data }) => <button className="stat-tile" key={label} onClick={() => open({ title: label, eyebrow: '实验统计', body: <p>{detail}</p>, raw: data })}><span><Icon /></span><div><small>{label}</small><b>{value}</b></div><ChevronRight /></button>)}</section>
    <section className="grade-strip">{['A', 'B', 'C', 'D', 'E'].map((grade) => <button key={grade} className={`grade-cell grade-${grade.toLowerCase()}`} onClick={() => open({ title: `${grade} ${gradeName(grade)}`, eyebrow: '质量等级', body: <TaskMiniList tasks={runtime.tasks.filter((task) => task.grade === grade)} />, raw: { grade, count: grades[grade] || 0 } })}><i>{grade}</i><b>{grades[grade] || 0}</b><span>{gradeName(grade)}</span></button>)}</section>
    <section className="panel-section"><SectionHead icon={Gauge} title="评估指标" extra="8 项指标" /><EvaluationChart values={o.quality_averages || {}} open={open} /></section>
    <ExperimentDataDashboard runtime={runtime} open={open} />
    <DatasetComparison runtime={runtime} open={open} />
    <section className="panel-section">
      <SectionHead icon={Activity} title="当前任务指标" extra={current?.round_label || '等待任务'} />
      {current ? <><button className="task-banner" onClick={() => openTask(current, open)}><div><b>{current.name}</b><span>{current.source}</span></div><GradeMark task={current} /></button><MetricButtons task={current} open={open} /></> : <Empty text="实验形成测试状态后，当前任务会立即出现" />}
    </section>
  </div>;
}

export function AgentsPage({ runtime, open }: { runtime: RuntimeSnapshot; open: OpenDetail }) {
  const latest = runtime.flow_events.at(-1);
  const [puzzlePulse, setPuzzlePulse] = useState(0);
  useEffect(() => {
    const timer = window.setInterval(() => setPuzzlePulse((value) => value + 1), 2200);
    return () => window.clearInterval(timer);
  }, []);
  const puzzleOrder = [5, 9, 6, 2, 10, 1, 13, 7, 4, 14, 8, 3, 11, 0, 15, 12];
  const evidencedNodes = runtime.state_model.nodes.filter((node) => node.evidenced || node.visited).length;
  const puzzleSeed = (runtime.state_model.version || 0) * 7 + (runtime.state_model.iteration || 0) * 3 + runtime.state_model.nodes.length;
  const puzzleFillCount = Math.min(12, Math.max(6, 6 + Math.round((evidencedNodes / Math.max(1, runtime.state_model.nodes.length)) * 6)));
  const puzzleOffset = (puzzleSeed + puzzlePulse * 5) % puzzleOrder.length;
  const filledPuzzleCells = new Set(Array.from({ length: puzzleFillCount }, (_, index) => puzzleOrder[(puzzleOffset + index) % puzzleOrder.length]));
  return <div className="page-stack"><PageTitle title="多智能体协作台" meta="Shared test state · Agent roles · Live interactions" />
    <section className="collaboration-layout">
      <div className="agent-orbit">
        <div className="orbit-grid" aria-hidden="true" />
        <div className="orbit-ring ring-outer" aria-hidden="true" /><div className="orbit-ring ring-middle" aria-hidden="true" /><div className="orbit-ring ring-inner" aria-hidden="true" />
        <StellarNetwork />
        {runtime.catalog.agents.map((definition, index) => {
          const live = runtime.agents.find((agent) => agent.name === definition.name);
          const completed = live?.internal_tasks?.filter((task) => task.status === 'done' || task.status === 'completed').length || 0;
          const progress = definition.modules.length ? completed / definition.modules.length : 0;
          const AgentIcon = [Radar, Cpu, BrainCircuit][index] || Bot;
          return <button key={definition.id} style={{ '--agent-accent': agentColor(index) } as CSSProperties} className={`orbit-agent orbit-${index + 1} ${live?.status || 'standby'}`} onClick={() => openAgent(live, runtime, open, definition)}>
            <i className="agent-holo-scan" aria-hidden="true" /><i className="agent-channel" aria-hidden="true">{agentChannelLabel(definition.name)}</i>
            <span className="agent-emblem" style={{ background: `conic-gradient(${agentColor(index)} ${progress * 360}deg, #dfe6ef 0deg)` }}><i><AgentIcon /></i><u aria-hidden="true" /><u aria-hidden="true" /></span>
            <div className="agent-intel"><b>{definition.name}</b><em><i className={live?.status === 'active' || live?.status === 'running' ? 'live' : ''} />{agentWorkHeadline(live)}</em></div>
            <strong>{live?.event_count || 0}<small>状态事件</small></strong>
            <i className={`agent-signal-wave ${live?.status === 'active' || live?.status === 'running' ? 'live' : ''}`} aria-hidden="true">{Array.from({ length: 14 }, (_, waveIndex) => <u key={waveIndex} style={{ '--wave-height': `${3 + ((waveIndex * 7 + index * 5 + (live?.event_count || 0)) % 13)}px`, '--wave-delay': `${waveIndex * .08}s` } as CSSProperties} />)}</i>
          </button>;
        })}
        <button className="orbit-core" onClick={() => open({ title: '共享TSG', eyebrow: `状态版本 v${runtime.state_model.version || 0}`, body: <StateModelVisual model={runtime.state_model} /> })}>
          <i className="core-scan" aria-hidden="true" /><i className="core-crosshair" aria-hidden="true" />
          <span className="core-hex-layer core-hex-outer" aria-hidden="true" /><span className="core-hex-layer core-hex-mid" aria-hidden="true" /><span className="core-hex-layer core-hex-inner" aria-hidden="true" />
          <span className="puzzle-matrix" aria-hidden="true">{Array.from({ length: 16 }, (_, i) => <i key={i} className={filledPuzzleCells.has(i) ? 'filled' : ''} style={{ '--cell-delay': `${((i * 7 + puzzleSeed) % 16) * .09}s` } as CSSProperties} />)}</span>
          <span className="core-particles" aria-hidden="true">{Array.from({ length: 18 }, (_, i) => <i key={i} style={{ '--particle-angle': `${i * 20}deg`, '--particle-delay': `${-i * .31}s`, '--particle-duration': `${4.8 + (i % 5) * .8}s`, '--particle-radius': `${78 + (i % 4) * 13}px` } as CSSProperties} />)}</span>
          <span className="core-content"><Layers3 /><b>共享TSG</b><span>v{runtime.state_model.version || 0}</span><small>{runtime.current_task?.name || '等待任务状态'}</small><em>{runtime.parallel_execution?.active_task_count || 0}项活动 · {runtime.state_model.nodes.length}节点</em></span>
        </button>
      </div>
      <div className="live-exchange">
        <header><Network /><div><h2>当前交互</h2><p>状态事件驱动下一项工作</p></div><span>{runtime.flow_events.length}条</span></header>
        {latest ? <button className="current-exchange" onClick={() => open({ title: eventLabel(latest.event_type), eyebrow: '最新智能体交互', body: <EventDetail event={latest} />, raw: latest })}><span>{agentDisplayName(latest.source_agent)}</span><i>→</i><div><b>{eventLabel(latest.event_type)}</b><small>{latest.task_name ? `${latest.task_name} · ` : latest.experiment_label ? `${latest.experiment_label} · ` : ''}状态迭代{Number(latest.iteration || 0) + 1}</small></div><i>→</i><span>{agentDisplayName(latest.target_agent) || '共享TSG'}</span></button> : <Empty text="实验运行后显示智能体交互" />}
        <InteractionConsole events={runtime.flow_events} open={open} />
      </div>
    </section>
    <RealtimeStatusMonitor runtime={runtime} open={open} />
    <section className="panel-section"><SectionHead icon={Activity} title="协作时间线" extra="点击事件查看流向与证据" /><EventTable events={runtime.flow_events} open={open} /></section>
    <details className="data-detail-disclosure agent-capability-disclosure"><summary>查看智能体内部能力与事件证据</summary><section className="panel-section"><SectionHead icon={Workflow} title="内部能力图" extra="节点亮起表示已有真实事件证据" /><div className="capability-lanes">{runtime.catalog.agents.map((definition, index) => { const live = runtime.agents.find((agent) => agent.name === definition.name); return <div key={definition.id}><header><i style={{ background: agentColor(index) }} /><b>{definition.name}</b><span>{live?.internal_tasks?.filter((task) => task.event_count > 0).length || 0}/{definition.modules.length}</span></header><div>{definition.modules.map((module, moduleIndex) => { const state = live?.internal_tasks?.find((task) => task.name === module.name); return <button key={module.id} className={state?.event_count ? 'evidenced' : ''} onClick={() => open({ title: module.name, eyebrow: definition.name, body: <><div className="capability-route"><span>{moduleIndex + 1}</span><i /><b>{statusName(state?.status)}</b></div><p>{module.description}</p><EventDetail event={state?.last_event} /></>, raw: { status: state?.status, event_count: state?.event_count, last_event: state?.last_event } })}><i /><b>{module.name}</b><span>{state?.event_count || 0}</span></button>})}</div></div>})}</div></section></details>
  </div>;
}

function RealtimeStatusMonitor({ runtime, open }: { runtime: RuntimeSnapshot; open: OpenDetail }) {
  const running = runtime.overview.status === 'running';
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (!running) return undefined;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [running]);
  const current = runtime.current_task || runtime.tasks.at(-1);
  const monitoredMetrics = [
    { id: 'line_coverage', label: 'LC', value: current?.metrics.line_coverage, color: '#8fa8e8' },
    { id: 'ae', label: 'AE', value: current?.metrics.ae, color: '#b4a9dc' },
    { id: 'mutation_score', label: 'MS', value: current?.metrics.mutation_score, color: '#91c9be' },
    { id: 'branch', label: 'BC', value: current?.metrics.branch_coverage, color: '#56728c' },
  ];
  const chartTasks = runtime.tasks.filter((task) => [task.metrics.branch_coverage, task.metrics.ae, task.metrics.mutation_score].some((value) => typeof value === 'number')).slice(-24);
  const chartPoints = (metric: 'branch_coverage' | 'ae' | 'mutation_score') => chartTasks.map((task, index) => {
    const value = Number(task.metrics[metric] || 0);
    const x = chartTasks.length === 1 ? 50 : 4 + index * (92 / Math.max(1, chartTasks.length - 1));
    return `${x},${92 - Math.max(0, Math.min(1, value)) * 78}`;
  }).join(' ');
  const elapsed = running && runtime.overview.active_session_started_at
    ? Number(runtime.overview.accumulated_duration_before_session || 0) + Number(elapsedSeconds(runtime.overview.active_session_started_at, undefined, 0, now) || 0)
    : runtime.overview.duration_seconds;
  const systemSignals = [
    { label: '实验', value: statusName(runtime.overview.status), raw: runtime.overview },
    { label: '模型', value: runtime.overview.model_name || (runtime.assistant?.enabled ? runtime.assistant.model || runtime.assistant.provider : '未启用'), raw: { model: runtime.overview.model_name, assistant: runtime.assistant } },
    { label: '任务槽', value: String(runtime.parallel_execution?.task_workers || runtime.overview.task_workers || 1), raw: runtime.parallel_execution },
    { label: '活动', value: String(runtime.parallel_execution?.active_task_count || runtime.overview.active_task_count || 0), raw: runtime.parallel_execution?.active_tasks },
    { label: '队列', value: String(runtime.parallel_execution?.queued_task_count || runtime.overview.queued_task_count || 0), raw: runtime.parallel_execution },
    { label: '共享TSG', value: `v${runtime.state_model.version || 0}`, raw: runtime.state_model },
    { label: '节点/关系', value: `${runtime.state_model.nodes.length}/${runtime.state_model.edges.length}`, raw: { nodes: runtime.state_model.nodes.length, edges: runtime.state_model.edges.length } },
  ];
  return <section className="realtime-monitor">
    <header><span className="monitor-mark"><Activity /></span><div><h2>实时状态监控</h2><small>TSG LIVE TELEMETRY</small></div><em className={running ? 'running' : ''}><i />{running ? `LIVE · ${durationText(elapsed)}` : statusName(runtime.overview.status)}</em></header>
    <div className="monitor-main">
      <div className="monitor-quality">
        <div className="monitor-caption"><i />QUALITY SIGNAL <span>{current ? `${current.name} · ${current.round_label}` : '等待任务'}</span></div>
        <div className="monitor-metrics">{monitoredMetrics.map((metric) => <button key={metric.id} style={{ '--monitor-color': metric.color, '--monitor-value': `${Math.max(0, Math.min(100, Number(metric.value || 0) * 100))}%` } as CSSProperties} onClick={() => open({ title: metric.label, eyebrow: current?.name || '实时状态监控', body: <strong className="detail-value">{pct(metric.value)}</strong>, raw: { task: current?.name, metric: metric.id, value: metric.value } })}><small>{metric.label}</small><b>{pct(metric.value)}</b><span><i /></span></button>)}</div>
        <button className="monitor-chart" disabled={!chartTasks.length} onClick={() => open({ title: '任务质量信号', eyebrow: `最近${chartTasks.length}项任务`, raw: chartTasks })}>
          <span>任务质量信号</span><svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-label="最近任务质量趋势"><path d="M0 25H100M0 52H100M0 79H100" />{chartTasks.length > 0 && <><polyline className="branch" points={chartPoints('branch_coverage')} /><polyline className="ae" points={chartPoints('ae')} /><polyline className="mutation" points={chartPoints('mutation_score')} /></>}</svg><em>BC</em><em>AE</em><em>MS</em>
        </button>
      </div>
      <div className="monitor-agents"><div className="monitor-caption"><i />AGENT PULSE <span>{runtime.flow_events.length}条事件</span></div>{runtime.catalog.agents.map((definition, index) => { const live = runtime.agents.find((agent) => agent.name === definition.name); const active = live?.status === 'active' || live?.status === 'running'; return <button key={definition.id} style={{ '--monitor-color': agentColor(index) } as CSSProperties} onClick={() => openAgent(live, runtime, open, definition)}><span className="monitor-agent-node">{['S', 'G', 'K'][index] || 'A'}</span><div><b>{definition.name}</b><small>{agentWorkHeadline(live)}</small><i className={active ? 'active' : ''}>{Array.from({ length: 9 }, (_, item) => <u key={item} style={{ '--signal-height': `${4 + ((item * 5 + index * 4 + (live?.event_count || 0)) % 14)}px`, '--signal-delay': `${item * .09}s` } as CSSProperties} />)}</i></div><strong>{live?.event_count || 0}<small>{statusName(live?.status)}</small></strong></button>})}</div>
    </div>
    <footer>{systemSignals.map((signal, index) => <button key={signal.label} onClick={() => open({ title: signal.label, eyebrow: '实时系统状态', raw: signal.raw })}>{index === 0 && <i className={running ? 'online' : ''} />}<small>{signal.label}</small><b>{signal.value}</b></button>)}</footer>
  </section>;
}

function StellarNetwork() {
  const particles = [
    { path: 'M210 125 C240 200 330 135 350 235 S420 250 450 280', color: '#4f9f97', dur: '3.8s' },
    { path: 'M450 280 C405 300 415 220 360 235 S280 175 210 115', color: '#789aaa', dur: '5.1s' },
    { path: 'M690 160 C630 230 600 145 555 235 S490 255 450 280', color: '#5f91a0', dur: '4.4s' },
    { path: 'M450 280 C500 300 505 220 555 235 S625 205 690 150', color: '#8058e8', dur: '5.7s' },
    { path: 'M440 460 C360 420 520 390 430 345 S470 310 450 280', color: '#18a879', dur: '4.8s' },
    { path: 'M450 280 C425 315 485 335 435 365 S500 425 460 460', color: '#58a79c', dur: '6.2s' },
    { path: 'M210 115 C310 25 390 120 470 65 S610 75 690 145', color: '#6a8999', dur: '7.1s' },
    { path: 'M690 160 C790 235 620 295 720 360 S570 440 465 470', color: '#7651dc', dur: '6.6s' },
    { path: 'M445 455 C340 380 305 455 245 365 S305 205 215 135', color: '#4f9f97', dur: '7.5s' },
  ];
  return <svg className="stellar-network" viewBox="0 0 900 560" preserveAspectRatio="none" aria-hidden="true">
    <g className="stellar-lines">
      <path d="M210 125 C240 200 330 135 350 235 S420 250 450 280" /><path d="M210 115 C300 70 310 220 380 210 S410 280 450 280" />
      <path d="M690 160 C630 230 600 145 555 235 S490 255 450 280" /><path d="M690 150 C600 95 620 240 540 220 S500 285 450 280" />
      <path d="M440 460 C360 420 520 390 430 345 S470 310 450 280" /><path d="M460 460 C540 405 385 395 475 340 S425 305 450 280" />
      <path d="M210 115 C310 25 390 120 470 65 S610 75 690 145" /><path d="M210 130 C300 215 365 20 475 95 S600 40 690 155" />
      <path d="M200 125 C100 235 285 260 190 350 S330 440 435 470" /><path d="M215 135 C305 200 135 295 245 365 S340 380 445 455" />
      <path d="M690 160 C790 235 620 295 720 360 S570 440 465 470" /><path d="M680 170 C595 225 770 300 660 365 S555 385 455 455" />
    </g>
    <g className="stellar-particles">{particles.map((particle, index) => <circle key={index} r={index % 3 === 0 ? 3.2 : 2.2} fill={particle.color}><animateMotion dur={particle.dur} begin={`${index * -.47}s`} repeatCount="indefinite" path={particle.path} /></circle>)}</g>
  </svg>;
}

export function PhasesPage({ runtime, open }: { runtime: RuntimeSnapshot; open: OpenDetail }) {
  return <div className="page-stack"><PageTitle title="测试生成过程" meta="Test state events synchronized live" />
    <ParallelExecution runtime={runtime} open={open} />
    <ProcessAnalytics runtime={runtime} open={open} />
    <DatasetComparison runtime={runtime} open={open} />
    <section className="panel-section"><SectionHead icon={Timer} title="任务执行耗时" extra="每个任务的实际时长" /><TaskDurationChart tasks={runtime.tasks} open={open} /></section>
    <details className="data-detail-disclosure"><summary>查看阶段、工具链与状态事件明细</summary><section className="process-map">{runtime.phase_states.map((phase, index) => <button key={phase.id} className={phase.status || 'pending'} onClick={() => open({ title: phase.name, eyebrow: `第${index + 1}阶段 · ${phase.owner}`, body: <><p>{phase.description}</p><DetailList title="产生的信息" items={phase.artifacts} /><EventDetail event={phase.last_event} /></>, raw: phase })}><span>{String(index + 1).padStart(2, '0')}</span><div><h2>{phase.name}</h2><p>{phase.owner}</p></div><em>{statusName(phase.status)}</em><small>{phase.event_count || 0}条事件</small></button>)}</section><section className="panel-section"><SectionHead icon={Gauge} title="实际工具链" extra={languageName(runtime.overview.language)} /><div className="tool-grid">{runtime.toolchain.map((tool) => <button key={tool.name} className={tool.status} onClick={() => open({ title: tool.name, eyebrow: '执行工具', body: <><p>当前状态：{statusName(tool.status)}</p><p>当前结果：{formatToolValue(tool.value)}</p></>, raw: tool })}><ShieldCheck /><b>{tool.name}</b><span>{formatToolValue(tool.value)}</span><small>{statusName(tool.status)}</small></button>)}</div></section>{runtime.flow_events.length > 0 && <section className="panel-section"><SectionHead icon={Activity} title="状态事件时间线" extra="最近80条" /><EventTable events={runtime.flow_events.slice(-80)} open={open} /></section>}</details>
  </div>;
}

export function StatePage({ runtime, open }: { runtime: RuntimeSnapshot; open: OpenDetail }) {
  const groups = useMemo(() => groupNodes(runtime.state_model.nodes), [runtime.state_model.nodes]);
  const [selectedType, setSelectedType] = useState(groups[0]?.type || '');
  const selectedGroup = groups.find((group) => group.type === selectedType) || groups[0];
  const [selectedNodeId, setSelectedNodeId] = useState('');
  const selectedNode = runtime.state_model.nodes.find((node) => node.id === selectedNodeId);
  const relations = selectedNode ? runtime.state_model.edges.filter((edge) => edge.source === selectedNode.id || edge.target === selectedNode.id) : [];
  return <div className="page-stack"><PageTitle title="测试状态中心" meta="Test State Center" />
    <StateEvidenceDashboard runtime={runtime} groups={groups} open={open} />
    <ExecutionConsole runtime={runtime} open={open} />
    <DatasetComparison runtime={runtime} open={open} />
    <details className="data-detail-disclosure"><summary>查看单节点状态与智能体写入明细</summary><section className="state-browser">
      <div className="state-layers">{['程序状态', '测试状态', '运行证据', '决策状态'].map((layer) => <div key={layer}><header><Layers3 /><b>{layer}</b><span>{groups.filter((group) => group.layer === layer).reduce((sum, group) => sum + group.nodes.length, 0)}</span></header>{groups.filter((group) => group.layer === layer).map((group) => <button key={group.type} className={selectedGroup?.type === group.type ? 'selected' : ''} onClick={() => { setSelectedType(group.type); setSelectedNodeId(''); }}><span>{group.type}</span><b>{group.nodes.length}</b></button>)}</div>)}</div>
      <div className="node-inspector"><header><div><small>当前类型</small><h2>{selectedGroup?.type || '等待状态节点'}</h2></div><span>{selectedGroup?.nodes.length || 0}个节点</span></header><div className="node-table">{selectedGroup?.nodes.map((node) => <button key={node.id} className={selectedNodeId === node.id ? 'selected' : ''} onClick={() => setSelectedNodeId(node.id)}><i className={node.evidenced || node.visited ? 'visited' : ''} /><div><b>{node.name || node.id}</b><small>{node.id}</small></div><span>{node.line ? `L${node.line}` : '—'}</span><em>{node.evidenced || node.visited ? '已有证据' : '待验证'}</em></button>)}</div></div>
      <div className="relation-inspector">{selectedNode ? <><header><small>节点详情</small><h2>{selectedNode.name || selectedNode.id}</h2><button className="text-button" onClick={() => open({ title: selectedNode.name || selectedNode.id, eyebrow: selectedNode.type, body: <><p>综合证据：{selectedNode.evidenced || selectedNode.visited ? '已有真实证据' : '待验证'}</p><DetailList title="证据明细" items={[selectedNode.visited ? '程序执行已访问' : undefined, selectedNode.coverage_status ? `覆盖状态：${selectedNode.coverage_status}` : undefined, selectedNode.mutation_status ? `变异状态：${selectedNode.mutation_status}` : undefined, typeof selectedNode.assertion_effective === 'boolean' ? `断言有效：${selectedNode.assertion_effective ? '是' : '否'}` : undefined, selectedNode.repair_flag ? '已标记为修复目标' : undefined]} /><p>源码行：{selectedNode.line || '—'}</p><DetailList title="关联关系" items={relations.map((edge) => relationText(edge, selectedNode, runtime.state_model.nodes))} /></>, raw: { node: selectedNode, relations } })}>完整详情</button></header><div className="relation-list">{relations.slice(0, 30).map((edge, index) => <button key={edge.id || index} onClick={() => open({ title: edge.type || '状态关系', eyebrow: selectedNode.name, body: <p>{relationText(edge, selectedNode, runtime.state_model.nodes)}</p>, raw: edge })}><GitBranch /><div><b>{edge.type || 'relation'}</b><span>{relationText(edge, selectedNode, runtime.state_model.nodes)}</span></div><em>{edge.visited ? '已证实' : '待验证'}</em></button>)}</div></> : <Empty text="选择一个节点查看它的状态关系" />}</div>
    </section>
    <section className="metadata-grid">{[['当前测试意图', 'last_test_intent_plan'], ['质量评估反馈', 'last_evaluation_feedback'], ['知识检索上下文', 'last_knowledge_context'], ['最新执行报告', 'last_report']].map(([label, key]) => { const value = runtime.state_model.metadata?.[key]; return <button key={key} disabled={!value} onClick={() => value && open({ title: label, eyebrow: '测试状态中心', raw: value, body: <p>该信息由后台智能体真实写入共享TSG，并以图表方式展示。</p> })}><Braces /><b>{label}</b><span>{value ? '查看数据视图' : '当前实验尚未产生'}</span>{value ? <ChevronRight /> : <i>—</i>}</button>; })}</section>
    </details>
  </div>;
}

export function TasksPage({ runtime, open, openArtifact }: { runtime: RuntimeSnapshot; open: OpenDetail; openArtifact: (path: string) => void }) {
  const [selectedId, setSelectedId] = useState(taskKey(runtime.current_task || runtime.tasks[0]));
  const [grade, setGrade] = useState('ALL');
  const [dataset, setDataset] = useState('ALL');
  const [page, setPage] = useState(1);
  const pageSize = 10;
  const datasets = datasetSummaries(runtime);
  useEffect(() => { setDataset('ALL'); setGrade('ALL'); setPage(1); setSelectedId(taskKey(runtime.current_task || runtime.tasks[0])); }, [runtime.experiment?.id]);
  const datasetTasks = dataset === 'ALL' ? runtime.tasks : runtime.tasks.filter((task) => (task.dataset_name || task.experiment_label) === dataset);
  const tasks = grade === 'ALL' ? datasetTasks : datasetTasks.filter((task) => task.grade === grade);
  const pageCount = Math.max(1, Math.ceil(tasks.length / pageSize));
  const safePage = Math.min(page, pageCount);
  const pagedTasks = tasks.slice((safePage - 1) * pageSize, safePage * pageSize);
  const task = pagedTasks.find((item) => taskKey(item) === selectedId) || pagedTasks[0];
  return <div className="page-stack"><PageTitle title="任务与测试报告" meta="Experiment, dataset, and task views share the same results" />
    <DatasetComparison runtime={runtime} open={open} />
    <div className="dataset-task-filters"><button className={dataset === 'ALL' ? 'selected' : ''} onClick={() => { setDataset('ALL'); setPage(1); }}>全部数据集 <b>{runtime.tasks.length}</b></button>{datasets.map((item) => <button key={item.name} className={dataset === item.name ? 'selected' : ''} onClick={() => { setDataset(item.name); setPage(1); }}>{item.name} <b>{item.total}</b></button>)}</div>
    <div className="task-filters">{['ALL', 'A', 'B', 'C', 'D', 'E'].map((item) => <button key={item} className={grade === item ? 'selected' : ''} onClick={() => { setGrade(item); setPage(1); }}>{item === 'ALL' ? `全部 ${datasetTasks.length}` : `${item} ${datasetTasks.filter((task) => task.grade === item).length}`}</button>)}</div>
    <section className="task-workspace"><div className="task-list"><header><b>{dataset === 'ALL' ? '全部任务' : dataset}</b><span>{tasks.length}</span></header>{pagedTasks.map((item) => <button key={taskKey(item)} className={taskKey(task) === taskKey(item) ? 'selected' : ''} onClick={() => setSelectedId(taskKey(item))}><div><b>{item.name}</b><small>{item.dataset_name || item.experiment_label ? `${item.dataset_name || item.experiment_label} · ` : ''}{item.status || '等待结果'} · {followupStrategyName(item.round_strategy)} · {languageFromPath(item.source)}</small></div><GradeMark task={item} /></button>)}<Pagination page={safePage} totalPages={pageCount} total={tasks.length} onPage={setPage} /></div><div className="task-report">{task ? <><button className="task-banner" onClick={() => openTask(task, open)}><div><small>{task.dataset_name || task.experiment_label ? `${task.dataset_name || task.experiment_label} · ` : ''}{languageFromPath(task.source)} · {scopeFromTask(task)} · {followupStrategyName(task.round_strategy)}</small><b>{task.name}</b><span>{task.source}</span></div><GradeMark task={task} /></button><MetricButtons task={task} open={open} /><div className="report-blocks"><button onClick={() => open({ title: '质量与失败分析', eyebrow: task.round_label, body: <FailureDetail task={task} /> })}><AlertTriangle /><div><b>质量与失败分析</b><span>{failureSummary(task)}</span></div><ChevronRight /></button><button onClick={() => open({ title: '轮次与报告历史', eyebrow: task.round_label, body: <><p>内部执行报告：{task.reports_count || 0}份</p><p>当前实验轮次：R{task.round}</p><p>本轮策略：{followupStrategyName(task.round_strategy)}</p></>, raw: task.grade_history })}><Workflow /><div><b>轮次与报告历史</b><span>{followupStrategyName(task.round_strategy)} · {task.reports_count || 0}份执行报告</span></div><ChevronRight /></button></div><div className="artifact-list">{[['生成测试文件', task.test_path], ['任务摘要', task.summary_path], ['HTML测试报告', task.output_dir]].map(([label, path]) => path && <button key={label} onClick={() => label === 'HTML测试报告' ? openTaskReports(task, runtime.experiment?.id || 'all', open) : label === '任务摘要' ? openTaskSummary(task, open) : openArtifact(path)}><Code2 /><div><b>{label}</b><span>{label === 'HTML测试报告' ? '测试状态报告 · 源码覆盖报告' : label === '任务摘要' ? '状态 · 质量 · 运行证据' : shortPath(path)}</span></div><ChevronRight /></button>)}</div></> : <Empty text="没有符合筛选条件的任务" />}</div></section>
  </div>;
}

export function KnowledgePage({ knowledge, error = '', open }: { knowledge: KnowledgeSnapshot | null; error?: string; open: OpenDetail }) {
  const [status, setStatus] = useState('全部'); const [query, setQuery] = useState(''); const [page, setPage] = useState(1);
  if (!knowledge) return <div className="page-stack"><PageTitle title="测试知识引导库" meta="Test Knowledge Guidance Library" /><section className="panel-section knowledge-loading">{error || '正在读取所选实验的知识记录'}</section></div>;
  const pageSize = 18;
  const records = (knowledge?.records || []).filter((record) => (status === '全部' || record.status === status) && `${record.topic} ${record.action} ${record.usage} ${record.agent} ${record.failure_category} ${record.source} ${record.solution} ${record.dataset} ${knowledgeSourceLabel(record.source_type)}`.toLowerCase().includes(query.toLowerCase()));
  const pageCount = Math.max(1, Math.ceil(records.length / pageSize));
  const safePage = Math.min(page, pageCount);
  const visibleRecords = records.slice((safePage - 1) * pageSize, safePage * pageSize);
  const imported = knowledge?.scope === 'imported_run';
  const availability = knowledge?.availability;
  const lifecycle = [
    { name: '正在学习', note: '对话概念已归纳，等待真实任务采用与验证' },
    { name: '待复核', note: '候选经验正向证据不足，暂不提高检索优先级' },
    { name: '已掌握', note: '已被真实测试任务多次采用并取得可行结果' },
    { name: '应避免', note: '被候选验证拒绝或已有负向执行证据' },
    ...(imported ? [{ name: '经验摘要', note: '数据集执行经验的归纳，并非单条已验证知识' }] : []),
  ];
  return <div className="page-stack"><PageTitle title="测试知识引导库" meta="Test Knowledge Guidance Library" />
    {imported && availability && <section className="knowledge-provenance"><div><h2>本次实验可恢复的知识</h2><p>{availability.note}</p></div><div className="knowledge-provenance-counts"><span>本次实验记录 <b>{availability.run_record_count}</b></span><span>实验前已有记录 <b>{availability.baseline_record_count}</b></span><span>数据集经验摘要 <b>{availability.dataset_summary_count}</b></span>{Boolean(availability.post_import_learning_count) && <span>导入后学习 <b>{availability.post_import_learning_count}</b></span>}</div>{availability.missing_datasets.length > 0 && <small>未附原始库：{availability.missing_datasets.join('、')}</small>}</section>}
    <section className="knowledge-summary">{lifecycle.map((item) => <button key={item.name} className={status === item.name ? 'selected' : ''} onClick={() => { setStatus(status === item.name ? '全部' : item.name); setPage(1); }}><div><span>{item.name}</span><small>{item.note}</small></div><b>{knowledge?.counts?.[item.name] || 0}</b><ChevronRight /></button>)}</section>
    <section className="panel-section"><div className="table-toolbar"><SectionHead icon={BookOpenCheck} title="经验与学习记录" extra={`${records.length}/${knowledge?.total || 0}`} /><label><Search /><input value={query} onChange={(event) => { setQuery(event.target.value); setPage(1); }} placeholder="搜索用途、概念、策略、数据集或失败类型" /></label></div><div className="knowledge-table"><header><span>学习状态</span><span>概念、动作或策略</span><span>适用用途</span><span>负责智能体</span><span>验证证据</span><span>来源</span></header>{visibleRecords.map((record) => <button key={record.id} onClick={() => open({ title: record.topic || record.action || '经验记录', eyebrow: `${record.status} · ${knowledgeSourceLabel(record.source_type)}`, body: <><p className="knowledge-status-note">{knowledgeStatusDescription(record.status)}</p><section className="detail-section"><h3>适用用途</h3><p>{record.usage || '用于相似测试问题的分析与修复'}</p></section><DetailList title="归属与结果" items={[record.agent, record.failure_category, record.source_type === 'dataset_summary' ? '数据集经验归纳，不代表单条知识的验证状态' : record.accepted ? '已验证有效' : '尚未达到掌握条件', record.source_type === 'dataset_summary' ? `汇总证据 ${record.support_count || 0} 项` : record.support_count ? `历史应用 ${record.support_count} 次，涉及 ${record.independent_task_count || 0} 个任务` : `成功验证 ${record.validation_count || 0} 次`]} /><DetailList title="来源" items={[knowledgeSourceLabel(record.source_type), record.dataset, record.source]} />{record.solution && <section className="detail-section"><h3>归纳内容或解决措施</h3><p>{record.solution}</p></section>}</> })}><span><i className={`knowledge-dot ${knowledgeTone(record.status)}`} />{record.status}</span><b>{record.topic || record.action || '未命名经验'}</b><span>{record.usage || '相似测试问题分析'}</span><span>{record.agent || '系统归纳'}</span><span>{record.source_type === 'dataset_summary' ? `${record.support_count || 0}项汇总证据` : record.support_count ? `${record.support_count}次历史证据` : record.accepted ? '有效' : `${record.validation_count || 0}/2次`}</span><span className="knowledge-record-origin"><b>{knowledgeSourceLabel(record.source_type)}</b><small>{record.dataset || shortPath(record.source)}</small></span></button>)}</div><Pagination page={safePage} totalPages={pageCount} total={records.length} onPage={setPage} /></section>
  </div>;
}

export function SystemPage({ runtime, open }: { runtime: RuntimeSnapshot; open: OpenDetail }) {
  const catalog = runtime.catalog;
  const usage = runtime.overview?.model_usage;
  const runModelStats = usage ? [
    { label: '模型调用', value: usage.calls.toLocaleString('en-US'), note: `${usage.recorded_tasks} 项任务有调用记录` },
    { label: 'Token 总量', value: usage.total_tokens.toLocaleString('en-US'), note: `输入 ${usage.prompt_tokens.toLocaleString('en-US')} · 输出 ${usage.completion_tokens.toLocaleString('en-US')}` },
    { label: '模型耗时', value: durationText(usage.duration_seconds), note: '全部模型请求的累计耗时' },
  ] : [];
  return <div className="page-stack"><PageTitle title="系统在线" meta="Supports Python and Java test tasks" />
    {usage && <section className="panel-section run-model-panel"><SectionHead icon={BrainCircuit} title="本次模型运行" extra={runtime.overview.model_name || '已记录模型调用'} /><div className="run-model-grid">{runModelStats.map((item) => <button type="button" key={item.label} onClick={() => open({ title: item.label, eyebrow: runtime.overview.model_name || '本次实验', body: <><strong className="detail-value">{item.value}</strong><p>{item.note}</p></> })}><small>{item.label}</small><b>{item.value}</b><span>{item.note}</span></button>)}</div></section>}
    <section className="language-grid">{catalog.languages.map((language) => <button key={language.id} onClick={() => open({ title: `${language.name}测试能力`, eyebrow: language.test_framework, body: <><DetailList title="结构分析" items={language.analysis} /><DetailList title="执行证据" items={language.evidence} /><DetailList title="覆盖与变异工具" items={[...language.coverage_tools, ...language.mutation_tools]} /><p className="detail-note">生成文件：{language.generated_test}</p></>, raw: language })}><span>{language.name === 'Java' ? <Code2 /> : <FileCode2 />}</span><div><h2>{language.name}</h2><p>{language.test_framework} · {language.coverage_tools.join(' / ')} · {language.mutation_tools.join(' / ')}</p></div><ChevronRight /></button>)}</section>
    <section className="panel-section"><SectionHead icon={Timer} title="执行与缩时策略" extra="正式结果仍使用完整评估" /><div className="scope-grid">{(catalog.execution_policies || []).map((policy) => <button key={policy.id} onClick={() => open({ title: policy.name, eyebrow: policy.value, body: <p>{policy.description}</p>, raw: policy })}><Timer /><b>{policy.name}</b><span>{policy.description}</span><small>{policy.value}</small></button>)}</div></section>
    <section className="panel-section"><SectionHead icon={Boxes} title="测试范围" extra="文件、类、项目与数据集" /><div className="scope-grid">{catalog.scopes.map((scope) => <button key={scope.id} onClick={() => open({ title: scope.name, eyebrow: scope.entry, body: <p>{scope.description}</p>, raw: scope })}><Database /><b>{scope.name}</b><span>{scope.description}</span><small>{scope.entry}</small></button>)}</div></section>
    <section className="panel-section"><SectionHead icon={Gauge} title="评价指标" extra="来自真实执行证据" /><div className="metric-catalog">{catalog.quality_metrics.filter((metric) => ![metric.id, metric.name, catalogMetricDescription(metric)].some(containsProcessMetric)).map((metric) => <button key={metric.id} onClick={() => open({ title: metric.name, eyebrow: '质量指标', body: <p>{catalogMetricDescription(metric)}</p>, raw: { ...metric, description: catalogMetricDescription(metric) } })}><CircleDot /><div><b>{metric.name}</b><span>{catalogMetricDescription(metric)}</span></div><ChevronRight /></button>)}</div></section>
  </div>;
}

function PageTitle({ title, meta }: { title: string; meta?: string }) { return <div className="page-title"><h1>{title}</h1>{meta && <p>{meta}</p>}</div>; }
function SectionHead({ icon: Icon, title, extra }: { icon: typeof Activity; title: string; extra?: string }) { return <div className="section-head"><Icon /><h2>{title}</h2>{extra && <span>{extra}</span>}</div>; }
function Empty({ text }: { text: string }) { return <div className="empty"><CircleDot />{text}</div>; }
function Pagination({ page, totalPages, total, onPage }: { page: number; totalPages: number; total: number; onPage: (page: number) => void }) {
  return <nav className="pagination" aria-label="分页导航"><span>共 {total} 条</span><button disabled={page <= 1} onClick={() => onPage(page - 1)}><ChevronLeft />上一页</button><b>{page} / {totalPages}</b><button disabled={page >= totalPages} onClick={() => onPage(page + 1)}>下一页<ChevronRight /></button></nav>;
}
function GradeMark({ task }: { task: Task }) { return <span className={`grade-mark grade-${task.grade.toLowerCase()}`}>{task.round_label}</span>; }

function ExperimentTiming({ runtime, open }: { runtime: RuntimeSnapshot; open: OpenDetail }) {
  const overview = runtime.overview;
  const running = overview.status === 'running';
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    setNow(Date.now());
    if (!running || !overview.started_at) return undefined;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [running, overview.started_at]);
  const elapsed = overview.duration_source === 'historical_task_runtime'
    ? overview.duration_seconds
    : running
    ? overview.active_session_started_at
      ? Number(overview.accumulated_duration_before_session || 0) + Number(elapsedSeconds(overview.active_session_started_at, undefined, 0, now) || 0)
      : overview.duration_seconds
    : overview.finished_at
      ? elapsedSeconds(overview.started_at, overview.finished_at, overview.duration_seconds, now)
      : overview.duration_seconds;
  const endText = overview.finished_at ? localDateTime(overview.finished_at) : running ? '运行结束后记录' : overview.status === 'queued' ? '等待开始' : '尚未记录';
  const recoveredHistory = overview.duration_source === 'historical_task_runtime';
  const timing = [
    { id: 'started', label: recoveredHistory ? '最近恢复时间' : '开始时间', value: localDateTime(overview.started_at), icon: Clock3, note: recoveredHistory ? '旧实验没有保存首次启动时间；这里显示现有摘要中最近一次恢复实验的时间。' : '实验进程真正开始执行的时间，不使用排队登记时间。' },
    { id: 'elapsed', label: running ? '累计实验时长' : '累计总时长', value: durationText(elapsed), icon: Timer, note: overview.duration_source === 'historical_task_runtime' ? '旧实验缺少完整会话计时，当前由全部任务实际耗时按当时并发数折算，已包含终止前完成的实验，不再只显示恢复运行的几秒钟。' : running ? '保留此前已完成时长，并叠加本次续跑时间。' : '由各次真实运行区间累计，终止到续跑之间的空闲时间不计入。' },
    { id: 'finished', label: recoveredHistory ? '最近完成时间' : '结束时间', value: endText, icon: Flag, note: recoveredHistory ? '这是现有摘要最后一次完成记录，不代表完整累计时长只发生在该时间段内。' : '只有实验进程完成或失败后才记录结束时间。' },
  ];
  return <section className={`experiment-timing ${running ? 'is-running' : ''}`}>
    {timing.map(({ id, label, value, icon: Icon, note }) => <button key={id} onClick={() => open({ title: label, eyebrow: runtime.experiment?.label || '当前实验', body: <><strong className="detail-value timing-detail-value">{value}</strong><p>{note}</p></>, raw: { status: overview.status, started_at: overview.started_at, duration_seconds: elapsed, finished_at: overview.finished_at } })}><span><Icon /></span><div><small>{label}</small><b>{value}</b></div>{id === 'elapsed' && running && <i aria-label="正在计时" />}</button>)}
  </section>;
}

function PortfolioBoard({ runtime, open }: { runtime: RuntimeSnapshot; open: OpenDetail }) {
  const portfolio = runtime.portfolio!; const cards = portfolio.experiments;
  return <section className="portfolio-board"><header><div><Layers3 /><span><b>{portfolioLabel(portfolio.view)}</b><small>总实验与各数据集结果联动</small></span></div><div>{[['running', '运行中'], ['completed', '已完成'], ['queued', '排队中'], ['failed', '失败']].map(([key, label]) => <button key={key} onClick={() => open({ title: `${label}数据集`, eyebrow: '实验状态', body: <DetailList title="数据集" items={runtime.experiments.filter((item) => item.status === key).map((item) => item.label)} /> })}><i className={`status-${key}`} />{label}<b>{portfolio.counts[key] || 0}</b></button>)}</div></header>{runtime.experiment?.report_path && <a className="portfolio-report-link" href={rawArtifactUrl(runtime.experiment.id, runtime.experiment.report_path)} target="_blank" rel="noreferrer"><FileCode2 />完整实验报告<ChevronRight /></a>}<div className="portfolio-card-grid">{cards.map((item) => <button key={item.id} onClick={() => open({ title: item.label, eyebrow: `${languageName(item.language)} · ${scopeName(item.scope)}`, body: <><p>{item.status === 'queued' ? `当前位于队列第${item.queue_position || 1}位。` : `该数据集实验当前${datasetCardStatus(item)}。`}</p>{item.started_at && <p>开始时间：{localDateTime(item.started_at)}</p>}{typeof item.duration_seconds === 'number' && <p>累计耗时：{durationText(item.duration_seconds)}{item.duration_source === 'historical_task_runtime' ? '（由完整任务历史恢复）' : ''}</p>}{item.finished_at && <p>结束时间：{localDateTime(item.finished_at)}</p>}{item.batch_report_path && <a className="portfolio-detail-report" href={rawArtifactUrl(item.id, item.batch_report_path)} target="_blank" rel="noreferrer">查看数据集质量报告</a>}</> })}><span className={`experiment-state status-${item.status}`}><i /></span><div><header><b>{item.label}</b><em>{datasetCardStatus(item)}</em></header><p>{shortPath(item.output_dir)}</p><i><b style={{ width: `${Math.max(2, (item.progress || 0) * 100)}%` }} /></i><footer><span>{languageName(item.language)} · {scopeName(item.scope)}</span><strong>{item.completed_execution || 0}项有效 / {item.total_tasks || 0}项</strong></footer>{item.started_at && <small className="portfolio-duration" title={item.duration_source === 'historical_task_runtime' ? '由完整任务历史恢复' : '累计实验时长'}><Timer />{durationText(item.duration_seconds)}</small>}</div><ChevronRight /></button>)}</div>{!cards.length && <Empty text="当前范围内没有数据集实验" />}</section>;
}

function MetricButtons({ task, open }: { task: Task; open: OpenDetail }) { return <div className="metric-button-grid">{metricEntries(task.metrics).map((metric) => <button key={metric.id} onClick={() => open({ title: metric.label, eyebrow: `${task.name} · 实际值`, body: <><strong className="detail-value">{metric.value}</strong><p>{metricDescription(metric.id)}</p></>, raw: { task: task.name, metric: metric.id, value: metric.raw, ...(metric.id === 'tir' ? { improved: task.metrics.tir_improved, comparable: task.metrics.tir_comparable, excluded: task.metrics.tir_excluded, status: task.metrics.tir_status } : {}), ...(metric.id === 'avg_exec' ? { known_count: task.metrics.exec_count, status: task.metrics.exec_count_status } : {}) } })}><small>{metric.label}</small><b>{metric.value}</b><ChevronRight /></button>)}</div>; }
const evaluationRates = [
  { key: 'average_line_coverage', label: 'LC', title: '平均LC', name: '行覆盖率', color: '#5f7fd0', note: '有效任务的平均行覆盖率。' },
  { key: 'average_branch_coverage', label: 'BC', title: '平均BC', name: '分支覆盖率', color: '#42a19b', note: '有效任务的平均分支覆盖率。' },
  { key: 'average_ae', label: 'AE', title: '平均AE', name: '断言有效性', color: '#c99959', note: '有效任务的平均断言有效性。' },
  { key: 'average_mutation_score', label: 'MS', title: '平均MS', name: '变异得分', color: '#9079bf', note: '有效任务的平均变异得分。' },
  { key: 'pass_rate', label: 'PassRate', title: 'PassRate', name: '通过率', color: '#4caa87', note: '完成测试执行并通过的任务比例。' },
  { key: 'tir', label: 'TIR', title: 'TIR', name: '目标改善率', color: '#658fb9', note: '可比较评价目标中的改善比例。' },
] as const;

function EvaluationChart({ values, open }: { values: Record<string, number | string | unknown[]>; open: OpenDetail }) {
  const valueFor = (key: string) => { const value = values[key]; return typeof value === 'number' && Number.isFinite(value) ? value : null; };
  const countFor = (key: string) => valueFor(key) ?? '—';
  const tirUnrecorded = valueFor('tir') === null && valueFor('tir_comparable') === 0 && valueFor('tir_excluded') === 0;
  const execUnrecorded = valueFor('avg_exec') === null && valueFor('exec_complete_tasks') === 0 && valueFor('selected_tasks') !== null;
  const execComplete = valueFor('exec_complete_tasks');
  const execSelected = valueFor('selected_tasks');
  const execPartial = execComplete !== null && execSelected !== null && execComplete > 0 && execComplete < execSelected;
  const execMissingCount = execPartial ? execSelected - execComplete : 0;
  const showDetail = (key: string, title: string, display: string, note: string) => open({ title, eyebrow: '实验统计', body: <><strong className="detail-value">{display}</strong><p>{note}</p>{key === 'tir' && <p>{tirUnrecorded ? '本次运行未记录事前目标及行动前后对照，无法计算 TIR。' : `改善 ${countFor('tir_improved')} 项 / 可比较 ${countFor('tir_comparable')} 项；缺证或未完成 ${countFor('tir_excluded')} 项。`}</p>}{key === 'avg_exec' && <p>{execUnrecorded ? '本次运行未记录测试工具的完整执行会话计数；任务轮次不能代替。' : execPartial ? `按 ${execComplete} 项完整计数任务计算；${execMissingCount} 项缺证未纳入，不能代表全部 ${execSelected} 项的精确平均。` : `${countFor('exec_complete_tasks')} / ${countFor('selected_tasks')} 项任务计数完整。`}</p>}</> });
  const execution = [
    { key: 'avg_exec', title: 'AvgExec', name: '平均执行会话', unit: '次/任务', icon: Workflow, note: '按有完整计数的任务计算；缺证任务不纳入。' },
    { key: 'average_time_seconds', title: '平均Time/s', name: '平均任务耗时', unit: '秒/任务', icon: Timer, note: '按已记录任务累计耗时计算，单位为秒。' },
  ] as const;
  return <div className="evaluation-visual">
    <div className="evaluation-plot">
      <div className="evaluation-axis" aria-hidden="true"><span>比例指标</span><div><i>0</i><i>25</i><i>50</i><i>75</i><i>100%</i></div><span /></div>
      <div className="evaluation-rows">{evaluationRates.map((item) => {
        const raw = valueFor(item.key); const fill = raw === null ? 0 : Math.max(0, Math.min(100, raw * 100)); const display = pct(raw);
        return <button type="button" className={`evaluation-row ${raw === null ? 'missing' : ''}`} key={item.key} style={{ '--evaluation-color': item.color, '--evaluation-fill': `${fill}%` } as CSSProperties} onClick={() => showDetail(item.key, item.title, display, item.note)}>
          <span className="evaluation-label"><b>{item.label}</b><small>{item.name}</small></span>
          <span className="evaluation-track" aria-hidden="true"><i />{raw !== null && <u />}</span>
          <strong>{display}</strong>
        </button>;
      })}</div>
    </div>
    <div className="evaluation-execution">{execution.map(({ key, title, name, unit, icon: Icon, note }) => {
      const raw = valueFor(key); const display = raw === null ? '—' : raw.toFixed(2);
      return <button type="button" key={key} className={`evaluation-execution-card ${raw === null ? 'missing' : ''}`} onClick={() => showDetail(key, title, display, note)}>
        <span><Icon aria-hidden="true" /><small>{title}</small></span><strong>{display}</strong><em>{key === 'avg_exec' && execPartial ? '完整计数任务均值' : name}</em><i>{unit}</i>
      </button>;
    })}</div>
    <div className="evaluation-evidence"><span><b>TIR</b> {tirUnrecorded ? '未采集目标对照记录' : `${countFor('tir_improved')} 项改善 / ${countFor('tir_comparable')} 项可比较，${countFor('tir_excluded')} 项缺证或未完成`}</span><span><b>AvgExec</b> {execUnrecorded ? '未采集工具级执行计数' : execPartial ? `按 ${execComplete} / ${execSelected} 项完整计数任务计算；${execMissingCount} 项未纳入` : `${countFor('exec_complete_tasks')} / ${countFor('selected_tasks')} 项任务计数完整`}</span><small>缺证显示“—”</small></div>
  </div>;
}
function TaskMiniList({ tasks }: { tasks: Task[] }) { return tasks.length ? <div className="detail-task-list">{tasks.map((task) => <div key={task.id}><b>{task.name}</b><span>{task.round_label}</span></div>)}</div> : <p>当前没有该等级任务。</p>; }

function ParallelExecution({ runtime, open }: { runtime: RuntimeSnapshot; open: OpenDetail }) {
  const parallel = runtime.parallel_execution;
  if (!parallel) return null;
  const active = parallel.active_tasks || [];
  const mode = parallel.python_parallel && parallel.java_parallel ? 'Python / Java 任务级并行' : parallel.python_parallel ? 'Python 任务级并行' : parallel.java_parallel ? 'Java 任务级并行' : '当前语言串行';
  const items = [
    { label: '任务工作槽', value: parallel.task_workers, note: '同一数据集目录内可同时推进的源码任务数。', icon: Cpu },
    { label: '正在执行', value: parallel.active_task_count, note: '当前已经写入实时共享TSG的活动任务。', icon: Activity },
    { label: '等待任务', value: parallel.queued_task_count, note: '尚未进入工作槽的当前数据集任务。', icon: Clock3 },
    { label: '模型通道', value: parallel.llm_concurrency, note: '同一时刻允许占用本地Ollama的请求数，其他任务继续执行非模型阶段或等待通道。', icon: BrainCircuit },
  ];
  return <section className="parallel-board"><header><div><Network /><span><b>并行执行状态</b><small>{mode}</small></span></div><em>{parallel.active_task_count}/{parallel.task_workers}槽活动</em></header><div>{items.map(({ label, value, note, icon: Icon }) => <button key={label} onClick={() => open({ title: label, eyebrow: mode, body: <><strong className="detail-value">{value}</strong><p>{note}</p>{label === '正在执行' && (active.length ? <div className="detail-task-list">{active.map((task) => <div key={task.id || task.name}><b>{task.name}</b><span>{task.round_label || '运行中'}</span></div>)}</div> : <p>当前没有已形成TSG的活动任务。</p>)}</>, raw: label === '正在执行' ? active : { value, mode, knowledge_store: parallel.knowledge_store } })}><Icon /><span><small>{label}</small><b>{value}</b></span></button>)}</div></section>;
}
function ExecutionConsole({ runtime, open }: { runtime: RuntimeSnapshot; open: OpenDetail }) {
  const events = runtime.flow_events.slice(-18);
  const consoleRef = useRef<HTMLDivElement>(null);
  useEffect(() => { if (consoleRef.current) consoleRef.current.scrollTop = consoleRef.current.scrollHeight; }, [events.length]);
  const fallback = runtime.phase_states.filter((phase) => phase.status !== 'pending').map((phase) => phase.last_event).filter(Boolean) as FlowEvent[];
  const route = events.length ? events : fallback;
  return <section className="panel-section execution-live"><SectionHead icon={Workflow} title="实时执行过程" extra={`${runtime.flow_events.length}条真实状态事件`} /><div className="execution-console-grid">
    <div className="execution-route">{route.length ? route.map((event, index) => <button key={`${event.event_id || event.timestamp || index}-${index}`} className={index === route.length - 1 ? 'current' : ''} onClick={() => open({ title: eventLabel(event.event_type), eyebrow: `${event.task_name || '当前任务'} · 状态迭代${Number(event.iteration || 0) + 1}`, body: <EventDetail event={event} />, raw: event })}><span>{String(index + 1).padStart(2, '0')}</span><i /><div><b>{executionStage(event)}</b><small>{agentDisplayName(event.source_agent) || '智能体系统'} · {event.task_name || runtime.current_task?.name || '等待任务'}</small></div><em>R{eventRound(event, runtime)}</em></button>) : <Empty text="实验启动后按真实路由显示执行过程" />}</div>
    <div className="execution-terminal" ref={consoleRef}><header><i /><i /><i /><b>TSG LIVE EVENT CONSOLE</b><span>{statusName(runtime.overview.status)}</span></header><div>{route.length ? route.map((event, index) => <button key={`${event.event_id || index}-log`} onClick={() => open({ title: eventLabel(event.event_type), eyebrow: '实时执行信息', body: <EventDetail event={event} />, raw: event })}><time>{eventTime(event.timestamp)}</time><em>{agentChannelLabel(agentDisplayName(event.source_agent))}</em><span>{eventLabel(event.event_type)}</span><small>{event.task_name || event.experiment_label || '当前实验'} · iteration {Number(event.iteration || 0) + 1}</small></button>) : <p>WAITING FOR STATE EVENTS...</p>}</div></div>
  </div></section>;
}

type DatasetSummary = { name: string; tasks: Task[]; total: number; valid: number; tests: number; grades: Record<string, number>; coverage: number | null; mutation: number | null; ae: number | null; branch: number | null };

function summarizeDataset(name: string, tasks: Task[]): DatasetSummary {
  const validTasks = tasks.filter((task) => ['A', 'B', 'C'].includes(task.grade));
  const average = (key: keyof Task['metrics']) => { const values = validTasks.map((task) => task.metrics[key]).filter((value): value is number => typeof value === 'number'); return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null; };
  return { name, tasks, total: tasks.length, valid: validTasks.length, tests: tasks.filter((task) => task.test_artifact_available || task.test_path).length, grades: Object.fromEntries(['A', 'B', 'C', 'D', 'E'].map((grade) => [grade, tasks.filter((task) => task.grade === grade).length])), coverage: average('line_coverage'), mutation: average('mutation_score'), ae: average('ae'), branch: average('branch_coverage') };
}

function datasetSummaries(runtime: RuntimeSnapshot): DatasetSummary[] {
  const grouped = new Map<string, Task[]>();
  runtime.tasks.forEach((task) => { const name = task.dataset_name || task.experiment_label || runtime.experiment?.label || '当前实验'; grouped.set(name, [...(grouped.get(name) || []), task]); });
  const order = runtime.experiments.map((item) => item.label);
  return [...grouped.entries()].map(([name, tasks]) => summarizeDataset(name, tasks)).sort((left, right) => { const a = order.indexOf(left.name); const b = order.indexOf(right.name); return (a < 0 ? 999 : a) - (b < 0 ? 999 : b) || left.name.localeCompare(right.name); });
}

function DatasetComparison({ runtime, open }: { runtime: RuntimeSnapshot; open: OpenDetail }) {
  const groups = datasetSummaries(runtime);
  if (!groups.length) return null;
  const rows = groups.length > 1 ? [summarizeDataset('全部数据集', runtime.tasks), ...groups] : groups;
  const metric = (value: number | null) => value === null ? '待复验' : pct(value);
  return <section className="dataset-comparison panel-section"><SectionHead icon={Database} title="数据集结果矩阵" extra={`${groups.length}个数据集 · 总览与分项`} /><div className="dataset-matrix"><header><span>实验范围</span><span>有效结果</span><span>LC</span><span>MS</span><span>AE</span><span>BC</span><span>A/B/C/D/E</span></header>{rows.map((row) => <button key={row.name} className={`${row.name === '全部数据集' ? 'total' : ''} ${row.valid === 0 && row.tests > 0 ? 'awaiting-revalidation' : ''}`} onClick={() => open({ title: row.name, eyebrow: `${row.valid}/${row.total}项有效`, body: <>{row.valid === 0 && row.tests > 0 && <p>已有{row.tests}个测试文件，但当前历史运行没有形成可用执行证据。请使用修复后的执行链路重新实验，LC、BC、AE与MS才会成为有效结果。</p>}<TaskMiniList tasks={row.tasks} /></>, raw: row })}><span><b>{row.name}</b><small>{row.total}项任务 · {row.tests}个测试文件</small></span><i><u style={{ width: `${row.total ? row.valid / row.total * 100 : 0}%` }} /><em>{row.valid}/{row.total}</em></i><strong style={{ '--heat': row.coverage || 0 } as CSSProperties}>{metric(row.coverage)}</strong><strong style={{ '--heat': row.mutation || 0 } as CSSProperties}>{metric(row.mutation)}</strong><strong style={{ '--heat': row.ae || 0 } as CSSProperties}>{metric(row.ae)}</strong><strong style={{ '--heat': row.branch || 0 } as CSSProperties}>{metric(row.branch)}</strong><span className="dataset-grades">{['A', 'B', 'C', 'D', 'E'].map((grade) => <u key={grade} className={`grade-${grade.toLowerCase()}`}>{grade}<b>{row.grades[grade] || 0}</b></u>)}</span></button>)}</div></section>;
}

function ExperimentDataDashboard({ runtime, open }: { runtime: RuntimeSnapshot; open: OpenDetail }) {
  const averages = runtime.overview.quality_averages || {};
  const metricDefs = [
    ['LC', 'average_line_coverage', '#8fa8e8'], ['BC', 'average_branch_coverage', '#b4a9dc'],
    ['AE', 'average_ae', '#d6a961'], ['MS', 'average_mutation_score', '#ce929b'],
    ['PassRate', 'pass_rate', '#70ac9d'], ['TIR', 'tir', '#738fbf'],
  ] as const;
  const metricValues = metricDefs.map(([, key]) => {
    const value = averages[key];
    return typeof value === 'number' && Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : null;
  });
  const radarPoint = (index: number, value: number) => {
    const angle = -Math.PI / 2 + index * Math.PI * 2 / metricValues.length;
    return { x: 60 + Math.cos(angle) * value * 38, y: 56 + Math.sin(angle) * value * 38 };
  };
  const radarRing = (value: number) => metricDefs.map((_, index) => {
    const point = radarPoint(index, value);
    return `${point.x},${point.y}`;
  }).join(' ');
  const completeRadar = metricValues.every((value) => value !== null);
  const measuredCount = metricValues.filter((value) => value !== null).length;
  const grades = runtime.overview.grade_counts || {};
  const gradeDefs = [['A', '#78b8ac'], ['B', '#8fa8e8'], ['C', '#e6b96f'], ['D', '#e3a082'], ['E', '#d29aa5']] as const;
  const gradeTotal = gradeDefs.reduce((sum, [grade]) => sum + Number(grades[grade] || 0), 0);
  let cursor = 0;
  const gradeStops = gradeDefs.map(([grade, color]) => { const start = cursor; cursor += gradeTotal ? Number(grades[grade] || 0) / gradeTotal * 360 : 0; return `${color} ${start}deg ${cursor}deg`; }).join(',');
  const validTasks = runtime.tasks.filter((task) => ['A', 'B', 'C'].includes(task.grade));
  const scatter = validTasks.filter((task) => typeof task.metrics.line_coverage === 'number' && typeof task.metrics.mutation_score === 'number');
  const qualityGrades = [
    { id: 'a', label: 'A · 高质量', note: '完整执行且综合质量优异', tasks: scatter.filter((task) => task.grade === 'A') },
    { id: 'b', label: 'B · 良好', note: '完整执行，关键指标表现良好', tasks: scatter.filter((task) => task.grade === 'B') },
    { id: 'c', label: 'C · 可用', note: '完整执行，仍有质量提升空间', tasks: scatter.filter((task) => task.grade === 'C') },
  ];
  const mutationBuckets = [0, .25, .5, .75].map((floor, index) => ({ floor, ceiling: index === 3 ? 1.001 : floor + .25, count: scatter.filter((task) => Number(task.metrics.mutation_score) >= floor && Number(task.metrics.mutation_score) < (index === 3 ? 1.001 : floor + .25)).length }));
  const maxBucket = Math.max(1, ...mutationBuckets.map((item) => item.count));
  const slowest = runtime.tasks.filter((task) => typeof task.duration_seconds === 'number').sort((a, b) => Number(b.duration_seconds) - Number(a.duration_seconds)).slice(0, 8);
  const maxDuration = Math.max(1, ...slowest.map((task) => Number(task.duration_seconds || 0)));
  return <section className="experiment-data-dashboard">
    <div className="data-chart metric-radar">
      <header><BarChart3 /><span><b>六维质量指标</b><small>有效任务均值及实验整体比例 · 缺证显示“—”</small></span></header>
      <div className="radar-content">
        <figure className="radar-plot">
          <svg viewBox="0 0 120 112" role="img" aria-label={`六维质量指标雷达图，${measuredCount}项有数据，${metricDefs.length - measuredCount}项缺证`}>
            <polygon className="radar-grid outer" points={radarRing(1)} />
            <polygon className="radar-grid inner" points={radarRing(.5)} />
            {metricDefs.map(([label], index) => {
              const end = radarPoint(index, 1);
              const textPoint = radarPoint(index, 1.2);
              return <g key={label}><line className="radar-axis" x1="60" y1="56" x2={end.x} y2={end.y} /><text className="radar-axis-label" x={textPoint.x} y={textPoint.y} textAnchor={index === 0 || index === 3 ? 'middle' : index < 3 ? 'start' : 'end'} dominantBaseline="middle">{label}</text></g>;
            })}
            {completeRadar
              ? <polygon className="radar-value" points={metricValues.map((value, index) => { const point = radarPoint(index, value!); return `${point.x},${point.y}`; }).join(' ')} />
              : metricValues.map((value, index) => {
                const next = metricValues[(index + 1) % metricValues.length];
                if (value === null || next === null) return null;
                const start = radarPoint(index, value);
                const end = radarPoint((index + 1) % metricValues.length, next);
                return <line className="radar-partial" key={metricDefs[index][0]} x1={start.x} y1={start.y} x2={end.x} y2={end.y} />;
              })}
            {metricValues.map((value, index) => {
              if (value === null) return null;
              const point = radarPoint(index, value);
              return <circle className="radar-point" key={metricDefs[index][0]} cx={point.x} cy={point.y} r="1.8" style={{ fill: metricDefs[index][2] }} />;
            })}
          </svg>
          {measuredCount === 0 && <figcaption>暂无可比较的质量数据</figcaption>}
        </figure>
        <div className="radar-legend">{metricDefs.map(([label, key, color]) => <button key={key} onClick={() => open({ title: label, eyebrow: key === 'pass_rate' || key === 'tir' ? '实验整体比例' : '有效任务平均值', body: <strong className="detail-value">{pct(averages[key])}</strong>, raw: averages })}><i style={{ background: color }} /><span>{label}</span><b>{pct(averages[key])}</b></button>)}</div>
      </div>
    </div>
    <div className="data-chart grade-donut-card"><header><Gauge /><span><b>等级结构</b><small>{gradeTotal}项任务</small></span></header><div><button className="grade-donut" style={{ background: gradeTotal ? `conic-gradient(${gradeStops})` : '#edf1f5' }} onClick={() => open({ title: 'A-E等级结构', eyebrow: '全部任务', raw: grades })}><i><b>{runtime.overview.completed_execution || 0}</b><small>有效</small></i></button><div className="grade-legend">{gradeDefs.map(([grade, color]) => <button key={grade} onClick={() => open({ title: `${grade}级任务`, body: <TaskMiniList tasks={runtime.tasks.filter((task) => task.grade === grade)} /> })}><i style={{ background: color }} /><span>{grade}</span><b>{grades[grade] || 0}</b></button>)}</div></div></div>
    <div className="data-chart scatter-card"><header><CircleDot /><span><b>质量分布（LC × MS × AE）</b><small>{scatter.length}项有效结果 · 右上区域代表覆盖与变异表现更高</small></span></header><div className="quality-scatter"><svg viewBox="0 0 116 110" aria-label="LC和MS质量分布图"><rect className="zone zone-both" x="18" y="28" width="63" height="63" /><rect className="zone zone-mutation" x="81" y="28" width="21" height="63" /><rect className="zone zone-coverage" x="18" y="7" width="63" height="21" /><rect className="zone zone-strong" x="81" y="7" width="21" height="21" /><path d="M18 7V91H102M18 70H102M18 49H102M18 28H102M39 7V91M60 7V91M81 7V91" />{[0,25,50,75,100].map((value, index) => <text key={`x-${value}`} x={18 + index * 21} y="100" textAnchor="middle">{value}</text>)}{[0,25,50,75,100].map((value, index) => <text key={`y-${value}`} x="14" y={92 - index * 21} textAnchor="end">{value}</text>)}<text className="axis-title" x="60" y="108" textAnchor="middle">LC（%）</text><text className="axis-title" transform="translate(4 50) rotate(-90)" textAnchor="middle">MS（%）</text><text className="zone-label" x="91.5" y="14" textAnchor="middle">高LC / 高MS</text><text className="zone-label" x="91.5" y="36" textAnchor="middle">高LC / 低MS</text>{scatter.map((task) => <circle key={`${task.experiment_id || ''}-${task.id}`} cx={18 + Number(task.metrics.line_coverage) * 84} cy={91 - Number(task.metrics.mutation_score) * 84} r={1.1 + Math.max(0, Math.min(1, Number(task.metrics.ae || 0))) * 2.2} className={`point-${task.grade.toLowerCase()}`}><title>{task.name} · LC {pct(task.metrics.line_coverage)} · MS {pct(task.metrics.mutation_score)} · AE {pct(task.metrics.ae)}</title></circle>)}</svg><div className="quality-grade-legend">{qualityGrades.map((group) => <button key={group.id} className={group.id} onClick={() => open({ title: group.label, eyebrow: `${group.tasks.length}项任务`, body: <><p>{group.note}</p><TaskMiniList tasks={group.tasks} /></>, raw: group.tasks })}><i /><span>{group.label}</span><b>{group.tasks.length}</b><small>{group.note}</small></button>)}</div></div><footer><span>LC：行覆盖率</span><i className="ae-size-legend"><u /><u /><u />圆点大小：AE</i><span>MS：变异得分</span></footer></div>
    <div className="data-chart histogram-card"><header><Activity /><span><b>MS 分布</b><small>按25%区间</small></span></header><div className="histogram">{mutationBuckets.map((bucket) => <button key={bucket.floor} style={{ '--bar-height': `${bucket.count / maxBucket * 100}%` } as CSSProperties} onClick={() => open({ title: `${bucket.floor * 100}%–${Math.min(100, bucket.ceiling * 100)}%`, eyebrow: 'MS区间', raw: scatter.filter((task) => Number(task.metrics.mutation_score) >= bucket.floor && Number(task.metrics.mutation_score) < bucket.ceiling) })}><b>{bucket.count}</b><i><span /></i><small>{bucket.floor * 100}–{Math.min(100, Math.round(bucket.ceiling * 100))}%</small></button>)}</div></div>
    <div className="data-chart slow-task-card"><header><Timer /><span><b>任务耗时排行</b><small>跨续跑累计</small></span></header><div>{slowest.length ? slowest.map((task) => <button key={`${task.experiment_id || ''}-${task.id}`} onClick={() => openTask(task, open)}><span><b>{task.name}</b><small>{task.attempt_count || 1}次运行</small></span><i><u style={{ width: `${Number(task.duration_seconds || 0) / maxDuration * 100}%` }} /></i><em>{durationText(task.duration_seconds)}</em></button>) : <Empty text="任务完成后显示耗时" />}</div></div>
  </section>;
}

function InteractionConsole({ events, open }: { events: FlowEvent[]; open: OpenDetail }) {
  const ref = useRef<HTMLDivElement>(null); const visible = events.slice(-80);
  useEffect(() => { if (ref.current) ref.current.scrollTop = ref.current.scrollHeight; }, [visible.length]);
  if (!visible.length) return <Empty text="实验运行后显示智能体交互" />;
  return <div className="interaction-console" ref={ref}>{visible.map((event, index) => <button key={`${event.event_id || event.timestamp || index}-${index}`} onClick={() => open({ title: eventLabel(event.event_type), eyebrow: `${agentDisplayName(event.source_agent)} → ${agentDisplayName(event.target_agent) || '共享TSG'}`, body: <EventDetail event={event} />, raw: event })}><time>{eventTime(event.timestamp)}</time><span>{agentDisplayName(event.source_agent) || '系统'}</span><i>›</i><span>{agentDisplayName(event.target_agent) || '共享TSG'}</span><b>{eventLabel(event.event_type)}</b><em>R{Number(event.payload?.experiment_round || 1)}</em></button>)}</div>;
}

function ProcessAnalytics({ runtime, open }: { runtime: RuntimeSnapshot; open: OpenDetail }) {
  const total = Math.max(1, runtime.tasks.length);
  const rounds = [1, 2, 3].map((round) => ({ round, count: runtime.tasks.filter((task) => Number(task.round || 1) >= round).length }));
  const strategies = [{ label: '首轮完成', tasks: runtime.tasks.filter((task) => Number(task.round || 1) === 1), color: '#8fa8e8' }, { label: '失败恢复 R2', tasks: runtime.tasks.filter((task) => Number(task.round || 1) === 2), color: '#e6b96f' }, { label: '质量增强 R3', tasks: runtime.tasks.filter((task) => Number(task.round || 1) >= 3), color: '#91c9be' }];
  const grades = runtime.overview.grade_counts || {};
  return <section className="process-analytics"><div className="analytics-card round-funnel"><header><Workflow /><span><b>正式轮次流量</b><small>由全部历史任务直接还原</small></span></header><div>{rounds.map((item) => <button key={item.round} onClick={() => open({ title: `R${item.round}参与任务`, eyebrow: `${item.count}项`, body: <TaskMiniList tasks={runtime.tasks.filter((task) => Number(task.round || 1) >= item.round)} /> })}><span>R{item.round}</span><i style={{ width: `${item.count / total * 100}%` }} /><b>{item.count}</b></button>)}</div></div><div className="analytics-card strategy-donut"><header><Radar /><span><b>任务处理路径</b><small>首轮、恢复与增强</small></span></header><div>{strategies.map((item) => <button key={item.label} onClick={() => open({ title: item.label, body: <TaskMiniList tasks={item.tasks} /> })}><i style={{ background: item.color, '--share': item.tasks.length / total } as CSSProperties} /><span><b>{item.tasks.length}</b><small>{item.label}</small></span></button>)}</div></div><div className="analytics-card grade-flow"><header><BarChart3 /><span><b>最终等级产出</b><small>A-E真实结果</small></span></header><div>{['A', 'B', 'C', 'D', 'E'].map((grade) => <button key={grade} onClick={() => open({ title: `${grade}级任务`, body: <TaskMiniList tasks={runtime.tasks.filter((task) => task.grade === grade)} /> })}><span>{grade}</span><i><u className={`grade-${grade.toLowerCase()}`} style={{ width: `${Number(grades[grade] || 0) / total * 100}%` }} /></i><b>{grades[grade] || 0}</b></button>)}</div></div></section>;
}

function StateEvidenceDashboard({ runtime, groups, open }: { runtime: RuntimeSnapshot; groups: ReturnType<typeof groupNodes>; open: OpenDetail }) {
  const failures = new Map<string, number>(); runtime.tasks.forEach((task) => { if (['D', 'E'].includes(task.grade)) { const key = task.failure?.category || '未完成/无明确类别'; failures.set(key, (failures.get(key) || 0) + 1); } });
  const topFailures = [...failures.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6); const maxFailure = Math.max(1, ...topFailures.map((item) => item[1])); const maxNodes = Math.max(1, ...groups.map((group) => group.nodes.length));
  const valid = runtime.tasks.filter((task) => ['A', 'B', 'C'].includes(task.grade));
  const stateMetric = (key: 'branch_coverage' | 'mutation_score') => { const values = valid.map((task) => task.metrics[key]).filter((value): value is number => typeof value === 'number'); return { values, average: values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null }; };
  const bc = stateMetric('branch_coverage'); const ms = stateMetric('mutation_score');
  return <section className="state-evidence-dashboard"><div className="analytics-card node-distribution"><header><Network /><span><b>状态节点类型分布</b><small>当前共享TSG结构</small></span></header><div>{groups.slice(0, 8).map((group) => <button key={group.type} onClick={() => open({ title: group.type, eyebrow: group.layer, raw: group.nodes })}><span>{group.type}</span><i><u style={{ width: `${group.nodes.length / maxNodes * 100}%` }} /></i><b>{group.nodes.length}</b></button>)}</div></div><div className="analytics-card failure-distribution"><header><AlertTriangle /><span><b>无效状态原因</b><small>D/E任务分类</small></span></header><div>{topFailures.length ? topFailures.map(([label, value]) => <button key={label} onClick={() => open({ title: label, body: <TaskMiniList tasks={runtime.tasks.filter((task) => (task.failure?.category || '未完成/无明确类别') === label)} /> })}><span>{label}</span><i><u style={{ width: `${value / maxFailure * 100}%` }} /></i><b>{value}</b></button>) : <Empty text="当前没有无效任务" />}</div></div><div className="analytics-card state-quality-metrics"><header><Radar /><span><b>分支与变异证据</b><small>BC分支覆盖与MS变异得分</small></span></header><div>{[{ key: 'BC', detail: '分支覆盖率', data: bc, color: '#188f88' }, { key: 'MS', detail: '变异得分', data: ms, color: '#3e7897' }].map((item) => <button key={item.key} onClick={() => open({ title: item.key, eyebrow: item.detail, body: <><strong className="detail-value">{pct(item.data.average)}</strong><p>{item.data.values.length}项有效任务已形成该指标。</p></>, raw: item.data.values })}><span><b>{item.key}</b><small>{item.detail}</small></span><strong>{pct(item.data.average)}</strong><i><u style={{ width: `${(item.data.average ?? 0) * 100}%`, background: item.color }} /></i><em>{item.data.values.length}/{valid.length}项有证据</em></button>)}</div></div></section>;
}

function TaskDurationChart({ tasks, open }: { tasks: Task[]; open: OpenDetail }) {
  const data = tasks.filter((task) => Number(task.duration_seconds) > 0).sort((left, right) => Number(left.duration_seconds) - Number(right.duration_seconds));
  if (!data.length) return <Empty text="任务完成并写入时长后显示耗时分布" />;
  const values = data.map((task) => Number(task.duration_seconds || 0));
  const average = values.reduce((sum, value) => sum + value, 0) / values.length;
  const summary = [
    { label: '平均Time', value: average, note: `${data.length}项有计时` },
  ];
  const ranked = [...data].reverse().slice(0, 18); const max = Math.max(1, Number(ranked[0]?.duration_seconds || 0));
  return <div className="duration-dashboard"><div className="duration-kpis">{summary.map((item) => <button key={item.label} onClick={() => open({ title: item.label, eyebrow: '累计任务计时', body: <><strong className="detail-value timing-detail-value">{durationText(item.value)}</strong><p>{item.note}</p></>, raw: item })}><small>{item.label}</small><b>{durationText(item.value)}</b><span>{item.note}</span></button>)}</div><div className="duration-ranking"><header><span>耗时较长任务</span><small>按累计时长从高到低 · 点击查看任务详情</small></header><div>{ranked.map((task, index) => <button key={taskKey(task)} onClick={() => openTask(task, open)}><em>{String(index + 1).padStart(2, '0')}</em><span><b>{task.name}</b><small>{task.dataset_name || task.experiment_label || '当前实验'} · {task.attempt_count || 1}次运行</small></span><i><u style={{ width: `${Number(task.duration_seconds || 0) / max * 100}%` }} /></i><strong>{durationText(task.duration_seconds)}</strong></button>)}</div></div></div>;
}

function executionStage(event: FlowEvent) { const name = String(event.event_type || '').toLowerCase(); if (/source|structure|analy/.test(name)) return '分析待测意图与源码结构'; if (/stateinitial|statebuild/.test(name)) return '构建测试状态'; if (/knowledge.*retriev/.test(name)) return '检索测试知识引导库'; if (/intent|plan|route|decision/.test(name)) return '制定测试计划'; if (/generat|candidatecreated/.test(name)) return '测试代码生成'; if (/valid|execut|coverage|assert|mutation/.test(name)) return '测试代码验证'; if (/repair|patch/.test(name)) return '测试代码修复'; if (/optimiz|enhance/.test(name)) return '测试代码优化'; if (/evaluat|quality|graded|feedback/.test(name)) return '评估并同步测试状态'; if (/knowledge.*updated|learn|conversation/.test(name)) return '经验学习'; return eventLabel(event.event_type); }
function eventRound(event: FlowEvent, runtime: RuntimeSnapshot) { return Number(event.payload?.experiment_round || runtime.current_task?.round || 1); }
function eventTime(timestamp?: string) { if (!timestamp) return '--:--:--'; const value = new Date(timestamp); return Number.isNaN(value.getTime()) ? '--:--:--' : value.toLocaleTimeString('zh-CN', { hour12: false }); }

function EventTable({ events, open }: { events: FlowEvent[]; open: OpenDetail }) { if (!events.length) return <Empty text="等待后台写入状态事件" />; return <div className="event-table">{[...events].reverse().map((event, index) => <button key={`${event.task_id || event.experiment_id || ''}-${event.event_id || index}`} onClick={() => open({ title: eventLabel(event.event_type), eyebrow: `${event.task_name || event.experiment_label || '当前实验'} · 状态迭代${Number(event.iteration || 0) + 1}`, body: <EventDetail event={event} />, raw: event })}><span>{String(events.length - index).padStart(3, '0')}</span><div><b>{eventLabel(event.event_type)}</b><small>{event.task_name ? `${event.task_name} · ` : event.experiment_label ? `${event.experiment_label} · ` : ''}{agentDisplayName(event.source_agent) || '系统'} {event.target_agent ? `→ ${agentDisplayName(event.target_agent)}` : ''}</small></div><em>状态迭代{Number(event.iteration || 0) + 1}</em><ChevronRight /></button>)}</div>; }
function EventDetail({ event }: { event?: FlowEvent | null }) { return event ? <><p>{eventSummary(event)}</p><DetailList title="关联状态节点" items={event.related_nodes || []} /><DetailList title="关联状态关系" items={event.related_edges || []} /></> : <p>尚无相关事件。</p>; }
function FailureDetail({ task }: { task: Task }) {
  const failure = task.failure || {}; const valid = ['A', 'B', 'C'].includes(task.grade);
  return <>
    <p>{failure.reason || failure.category || (valid ? '测试结果有效，仍可继续优化。' : '该历史任务没有保存可解析的失败原因。')}</p>
    <DetailList title="失败类别" items={[failure.category]} />
    <DetailList title="失败发生阶段" items={[failure.stage]} />
    {failure.error && <section className="detail-section"><h3>原始执行错误</h3><pre className="failure-error">{failure.error}</pre></section>}
    <DetailList title="失败测试" items={failure.failed_tests || []} />
    <DetailList title="各轮已经采用的措施" items={failure.attempted_measures || []} />
    <DetailList title="历史经验中的解决措施" items={failure.previous_solutions || []} />
    <DetailList title="建议的下一步措施" items={failure.suggested_solutions || []} />
    <DetailList title="已写入测试知识引导库" items={[failure.knowledge_record_id]} />
    <DetailList title="质量弱项" items={failure.weak_flags || []} />
    <ThresholdBars values={failure.threshold_failures || {}} valid={valid} />
  </>;
}
function ThresholdBars({ values, valid }: { values: Record<string, number>; valid: boolean }) { const items = Object.entries(values).filter(([key]) => !isProcessMetricKey(key)); if (!items.length) return <div className={`quality-signal ${valid ? 'good' : 'warn'}`}><i /><div><b>{valid ? '未发现阈值失败' : '未形成可用质量指标'}</b><span>{valid ? '当前有效指标满足结果分级要求' : '任务在完成测试执行与质量评估前失败，请依据上方原因处理'}</span></div></div>; return <section className="threshold-bars"><h3>未达标指标</h3>{items.map(([key, value]) => <div key={key}><header><span>{key.replaceAll('_', ' ')}</span><b>{pct(value)}</b></header><i><span style={{ width: `${Math.max(4, Math.min(100, Number(value) * 100))}%` }} /></i></div>)}</section>; }

function openTask(task: Task, open: OpenDetail) { open({ title: task.name, eyebrow: `${task.round_label} · ${languageFromPath(task.source)}`, body: <><DetailList title="文件" items={[task.source, task.test_path]} /><DetailList title="状态" items={[task.status, task.stop_reason, followupStrategyName(task.round_strategy), `${task.iterations || 0}次内部迭代`]} /><DetailList title="累计计时" items={[`累计耗时：${durationText(task.duration_seconds)}`, `运行区间：${task.attempt_count || 1}次`, task.started_at ? `首次开始：${localDateTime(task.started_at)}` : undefined, task.finished_at ? `最近完成：${localDateTime(task.finished_at)}` : undefined]} /><FailureDetail task={task} /></> }); }

function openTaskSummary(task: Task, open: OpenDetail) {
  const metrics = metricEntries(task.metrics);
  open({
    title: `${task.name} · 任务摘要`,
    eyebrow: `${task.dataset_name || task.experiment_label || '当前实验'} · ${task.round_label}`,
    body: <div className="task-summary-visual">
      <div className="task-summary-facts">
        <div><small>结果等级</small><b>{task.grade} · {gradeName(task.grade)}</b></div>
        <div><small>实验轮次</small><b>R{task.round}</b></div>
        <div><small>测试数量</small><b>{task.metrics.collected_count ?? '—'}</b></div>
        <div><small>累计耗时</small><b>{durationText(task.duration_seconds)}</b></div>
      </div>
      <section className="detail-section"><h3>公开评估指标</h3><div className="task-summary-metrics">{metrics.map((metric) => <div key={metric.id}><span>{metric.label}</span><b>{metric.value}</b></div>)}</div></section>
      <DetailList title="输入与产物" items={[task.source, task.test_path]} />
      <DetailList title="执行记录" items={[task.round_strategy ? followupStrategyName(task.round_strategy) : undefined, task.reports_count ? `${task.reports_count}份执行报告` : undefined, task.started_at ? `开始：${localDateTime(task.started_at)}` : undefined, task.finished_at ? `结束：${localDateTime(task.finished_at)}` : undefined]} />
      {['D', 'E'].includes(task.grade) && <FailureDetail task={task} />}
    </div>,
  });
}

function taskReportPath(task: Task, fileName: string) {
  const base = String(task.output_dir || '').replace(/[\\/]+$/, '');
  return base ? `${base}${base.includes('\\') ? '\\' : '/'}${fileName}` : '';
}

function rawArtifactUrl(experimentId: string, path: string) { return `/api/artifact/raw?experiment=${encodeURIComponent(experimentId)}&path=${encodeURIComponent(path)}`; }

function openTaskReports(task: Task, experimentId: string, open: OpenDetail) {
  const reports = [
    { label: '测试状态报告', file: 'state_flow_graph.html', description: '查看测试状态、节点关系与执行证据' },
    { label: '源码覆盖报告', file: 'source_coverage.html', description: '查看本任务的源码行级覆盖结果' },
  ].map((item) => ({ ...item, path: taskReportPath(task, item.file) }));
  open({ title: `${task.name} · HTML报告`, eyebrow: '选择要查看的实验报告', body: <div className="task-report-links">{reports.map((report) => <a key={report.file} href={rawArtifactUrl(experimentId, report.path)} target="_blank" rel="noreferrer"><FileCode2 /><span><b>{report.label}</b><small>{report.description}</small><em>{report.file}</em></span><ChevronRight /></a>)}</div> });
}
function openAgent(agent: AgentState | undefined, runtime: RuntimeSnapshot, open: OpenDetail, definition?: CatalogAgent) { const def = definition || runtime.catalog.agents.find((item) => item.name === agent?.name); open({ title: def?.name || agent?.name || '智能体', eyebrow: statusName(agent?.status), body: <>{def && <><p>{def.mission}</p><DetailList title="输入" items={def.inputs} /><DetailList title="输出" items={def.outputs} /></>}<EventDetail event={agent?.last_event} /></>, raw: { definition: def, runtime: agent } }); }

function groupNodes(nodes: StateNode[]) { const map = new Map<string, StateNode[]>(); nodes.forEach((node) => { const type = node.type || 'State'; map.set(type, [...(map.get(type) || []), node]); }); return [...map.entries()].map(([type, list]) => ({ type, nodes: list, layer: nodeLayer(type) })).sort((a, b) => layerOrder(a.layer) - layerOrder(b.layer) || b.nodes.length - a.nodes.length); }
function nodeLayer(type: string) { const value = type.toLowerCase(); if (/(test|assert|oracle)/.test(value)) return '测试状态'; if (/(runtime|mutation|coverage|execution|type)/.test(value)) return '运行证据'; if (/(intent|decision|feedback|gap|repair|quality)/.test(value)) return '决策状态'; return '程序状态'; }
function layerOrder(layer: string) { return ['程序状态', '测试状态', '运行证据', '决策状态'].indexOf(layer); }
function relationText(edge: StateEdge, node: StateNode, nodes: StateNode[]) { const otherId = edge.source === node.id ? edge.target : edge.source; const other = nodes.find((item) => item.id === otherId); return `${edge.source === node.id ? '指向' : '来自'} ${other?.name || otherId}`; }
function languageName(language?: string) { return ({ java: 'Java', python: 'Python', mixed: 'Python + Java', auto: '自动识别' } as Record<string, string>)[language || ''] || '自动识别'; }
function scopeName(scope?: string) { return ({ file: '文件级', class: '类级', project: '项目级', file_function: '文件 / 函数级', file_function_class: '文件 / 函数 / 类级', file_class: '文件 / 类级', class_project: '类 / 项目级', project_file: '项目 / 文件级', mixed_scope: '项目 / 文件级', datasets: '全数据集级', portfolio: '实验组合' } as Record<string, string>)[scope || ''] || '待识别范围'; }
function portfolioLabel(view: string) { return ({ all: '各数据集实验', running: '正在运行的数据集', completed: '已完成的数据集', queued: '排队中的数据集' } as Record<string, string>)[view] || '数据集实验'; }
function experimentStatus(status?: string) { return ({ running: '正在运行', completed: '已经完成', queued: '等待执行', failed: '运行失败', waiting: '等待数据' } as Record<string, string>)[status || ''] || '状态待确认'; }
function datasetCardStatus(item: RuntimeSnapshot['experiments'][number]) { return item.status === 'completed' && Number(item.total_tasks || 0) > 0 && Number(item.completed_execution || 0) === 0 ? '待重新验证' : experimentStatus(item.status); }
function languageFromPath(path?: string) { return path?.toLowerCase().endsWith('.java') ? 'Java' : path?.toLowerCase().endsWith('.py') ? 'Python' : '未知语言'; }
function scopeFromTask(task: Task) { return scopeName(task.scope || (languageFromPath(task.source) === 'Java' ? 'class' : 'file')); }
function formatToolValue(value: unknown) { if (typeof value === 'boolean') return value ? '通过' : '未通过'; if (typeof value === 'number') return pct(value); return '等待证据'; }
function failureSummary(task: Task) { const failure = task.failure || {}; return failure.reason || failure.category || failure.weak_flags?.join('、') || (['A', 'B', 'C'].includes(task.grade) ? '结果有效' : '等待失败报告'); }
function followupStrategyName(strategy?: string) { return ({ initial_generation: '首轮生成与验证', optimize_existing: '保留通过测试并定向优化', regenerate_after_failure: '检索失败知识后重新生成', restart_incomplete: '补齐证据后重新执行' } as Record<string, string>)[strategy || ''] || '首轮生成与验证'; }
function taskKey(task?: Task | null) { return task ? `${task.dataset_id || task.experiment_id || ''}::${task.id}` : ''; }
function knowledgeTone(status: string) { return ({ 已掌握: 'mastered', 正在学习: 'learning', 待复核: 'pending', 应避免: 'avoid' } as Record<string, string>)[status] || 'pending'; }
function agentColor(index: number) { return ['#3f63e8', '#7852d6', '#15966f'][index] || '#657087'; }
function agentDisplayName(name?: string) { return ({ TestStateAgent: '测试状态智能体', PlanningAgent: '测试状态智能体', EvaluationAgent: '测试状态智能体', TestGenerationAgent: '测试生成智能体', ExecutionAgent: '测试生成智能体', TestKnowledgeAgent: '测试知识智能体' } as Record<string, string>)[name || ''] || name || ''; }
function agentActionLabel(agent?: AgentState) { if (!agent?.current_action) return '等待状态引导'; const action = eventLabel(agent.current_action); return agent.interaction_role === 'received' ? `接收：${action}` : action; }
function agentChannelLabel(name: string) { return ({ 测试状态智能体: 'TSG.STATE', 测试生成智能体: 'TSG.GENERATE', 测试知识智能体: 'TSG.KNOWLEDGE' } as Record<string, string>)[name] || 'TSG.AGENT'; }
function agentWorkHeadline(agent?: AgentState) {
  if (!agent?.current_action) return '等待共享TSG分配下一项任务';
  const event = agent.current_action.toLowerCase();
  if (agent.name === '测试状态智能体') {
    if (/repair|scope|invalid/.test(event)) return '复核失败证据，重建状态并界定下一步处理范围';
    if (/evaluat|quality|graded/.test(event)) return '汇总覆盖、断言与变异证据，计算质量等级';
    if (/intent|plan|decision|route/.test(event)) return '比较剩余质量缺口，规划下一智能体和具体任务';
    if (/knowledge/.test(event)) return '接收知识上下文，校验其与当前测试意图的匹配度';
    return '解析状态变化，决定继续生成、修复、优化或结束';
  }
  if (agent.name === '测试生成智能体') {
    if (/repair|scope|patch/.test(event)) return '根据失败栈和质量缺口修复最小必要测试片段';
    if (/execut|valid|coverage|mutation|assert/.test(event)) return '运行候选测试并收集通过、覆盖、断言与变异证据';
    if (/optimiz|enhance/.test(event)) return '保留已通过行为，定向增强当前最弱质量维度';
    if (/generat|candidate/.test(event)) return '围绕目标路径、边界和可判别预言构造候选测试';
    return '按测试意图生成、验证并迭代测试代码';
  }
  if (/conversation|question/.test(event)) return '联合大模型回答，并归纳可验证的单元测试概念';
  if (/updated|learn/.test(event)) return '归纳本轮收益、失败原因和无效动作，更新知识状态';
  if (/retriev|knowledge/.test(event)) return '按语言、结构签名和质量缺口检索可复用经验';
  return '维护测试知识引导库并向协作智能体提供证据';
}
function agentWorkContext(agent: AgentState | undefined, runtime: RuntimeSnapshot) {
  if (!agent?.last_event) return runtime.current_task ? `等待处理：${runtime.current_task.name}` : '等待实验产生真实状态事件';
  const payload = agent.last_event.payload || {};
  const detail = [payload.focus, payload.action, payload.failure_category, payload.quality_status].find((item) => typeof item === 'string' && item);
  return [runtime.current_task?.name, detail].filter(Boolean).join(' · ') || `状态迭代${Number(agent.last_event.iteration || 0) + 1}`;
}
function knowledgeStatusDescription(status: string) { return ({ 已掌握: '该知识已在真实测试任务中多次采用并取得可行结果，可提高检索优先级，但仍会继续接受新证据。', 正在学习: '该概念由测试知识智能体与大模型归纳，正在等待真实任务采用。', 待复核: '该候选策略已有记录但正向证据不足，系统会谨慎使用并继续观察。', 应避免: '该策略已被候选验证拒绝或产生负向证据，不应重复采用。', 经验摘要: '这是数据集运行经验的汇总，不等同于单条知识的验证结论。' } as Record<string, string>)[status] || '等待更多真实执行证据。'; }
function knowledgeSourceLabel(source?: string) { return ({ run_record: '本次实验', historical_baseline: '实验前知识', dataset_summary: '数据集摘要', post_import_learning: '导入后学习' } as Record<string, string>)[source || ''] || '知识库'; }
function metricDescription(id: string) { return ({ passed: '来自pytest或JUnit真实执行结果。', tests: '实际收集或执行的测试数量。', line: 'LC：Python由coverage.py、Java由JaCoCo提供。', branch: 'BC：已执行分支占可识别分支的比例。', ae: 'AE：断言的可达性、具体性和行为判别能力。', mutation: 'MS：有效变异体被测试检测并杀死的比例。', time: 'Time/s：当前任务跨续跑累计耗时，单位为秒；缺失时不推测。', pass_rate: 'PassRate：最终收集到测试且执行通过的任务比例；单任务为是或否对应的100%或0%。', avg_exec: 'AvgExec：按工具级测试执行会话计数；单任务展示该任务累计次数。联合测试与覆盖计一次，PIT一次评估计一次，缺证不推测。', tir: 'TIR：预先登记且可比较的评价目标改善比例；不用于智能体决策。详见原始记录中的分子、分母和排除原因。' } as Record<string, string>)[id] || ''; }
function catalogMetricDescription(metric: { id: string; description: string }) { return metric.id === 'coverage' ? 'LC行覆盖率与BC分支覆盖率，由Python coverage.py或Java JaCoCo提供。' : metric.description; }
