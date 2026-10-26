import { useEffect, useState } from 'react';
import {
  BarChart3, BookOpenCheck, Bot, GitBranch, Layers3, Menu, MessageSquareText,
  Play, RefreshCw, ScanSearch, TestTube2, Wifi, WifiOff, Workflow, X,
} from 'lucide-react';
import { connectRuntime, fetchArtifact, fetchKnowledge, fetchRuntime, selectedExperiment } from './api';
import type { Artifact, KnowledgeSnapshot, RuntimeSnapshot } from './types';
import { paperAverages } from './format';
import DetailDrawer, { DataVisual, StateModelVisual, type DetailContent } from './components/DetailDrawer';
import { ChatWorkspace } from './components/ChatWorkspace';
import { ExperimentLauncher } from './components/ExperimentLauncher';
import { ExperimentSwitcher } from './components/ExperimentSwitcher';
import { ValiFixTestShowcase } from './components/ValiFixTestShowcase';
import { AgentsPage, KnowledgePage, OverviewPage, PhasesPage, StatePage, SystemPage, TasksPage } from './components/pages';
import './overlay.css';

type View = 'overview' | 'run' | 'valifix' | 'agents' | 'phases' | 'state' | 'tasks' | 'knowledge' | 'system';

function initialView(): View {
  return new URLSearchParams(window.location.search).get('view') === 'valifix' ? 'valifix' : 'overview';
}

const nav = [
  { id: 'overview' as const, label: '实验总览', icon: BarChart3 },
  { id: 'run' as const, label: '启动实验', icon: Play },
  { id: 'valifix' as const, label: 'ValiFixTest方法展示台', icon: ScanSearch },
  { id: 'state' as const, label: '测试状态中心', icon: GitBranch },
  { id: 'agents' as const, label: '多智能体协作台', icon: Bot },
  { id: 'phases' as const, label: '测试生成过程', icon: Workflow },
  { id: 'tasks' as const, label: '任务与报告', icon: TestTube2 },
  { id: 'knowledge' as const, label: '测试知识引导库', icon: BookOpenCheck },
];

const runtimeCache = new Map<string, RuntimeSnapshot>();

function rememberRuntime(id: string, snapshot: RuntimeSnapshot) {
  runtimeCache.delete(id);
  runtimeCache.set(id, snapshot);
  while (runtimeCache.size > 4) {
    const oldest = [...runtimeCache.keys()].find((key) => key !== 'all');
    if (!oldest) break;
    runtimeCache.delete(oldest);
  }
}

function datasetPreview(snapshot: RuntimeSnapshot, id: string): RuntimeSnapshot | null {
  const card = snapshot.experiments.find((item) => item.id === id);
  if (!card) return null;
  const tasks = snapshot.tasks.filter((task) => task.experiment_id === id || task.dataset_id === id || task.dataset_name === card.label || task.experiment_label === card.label);
  if (!tasks.length) return null;
  const gradeCounts = Object.fromEntries(['A', 'B', 'C', 'D', 'E'].map((grade) => [grade, tasks.filter((task) => task.grade === grade).length]));
  const valid = tasks.filter((task) => ['A', 'B', 'C'].includes(task.grade) && task.metrics.tests_passed);
  const average = (key: keyof (typeof valid)[number]['metrics']) => {
    const values = valid.map((task) => task.metrics[key]).filter((value): value is number => typeof value === 'number');
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : undefined;
  };
  return {
    ...snapshot,
    experiment: { id: card.id, output_dir: card.output_dir, label: card.label, registered_at: '', status: card.status || 'completed', started_at: card.started_at, finished_at: card.finished_at, duration_seconds: card.duration_seconds },
    portfolio: undefined,
    tasks,
    current_task: tasks[0],
    phase_states: [],
    agents: [],
    toolchain: [],
    flow_events: [],
    state_model: { nodes: [], edges: [] },
    failures: {},
    parallel_execution: snapshot.parallel_execution ? { ...snapshot.parallel_execution, active_task_count: 0, queued_task_count: 0, active_tasks: [] } : undefined,
    overview: {
      ...snapshot.overview,
      language: card.language,
      scope: card.scope,
      status: card.status || 'completed',
      total_tasks: card.total_tasks || tasks.length,
      completed_execution: Number(gradeCounts.A) + Number(gradeCounts.B) + Number(gradeCounts.C),
      not_completed_execution: Number(gradeCounts.D) + Number(gradeCounts.E),
      grade_counts: gradeCounts,
      followup_task_count: tasks.filter((task) => Number(task.round || 1) > 1).length,
      followup_attempts: tasks.reduce((sum, task) => sum + Math.max(0, Number(task.round || 1) - 1), 0),
      dataset_count: 1,
      completed_datasets: card.status === 'completed' ? 1 : 0,
      active_task_count: 0,
      queued_task_count: 0,
      quality_averages: { average_line_coverage: average('line_coverage') ?? '—', average_branch_coverage: average('branch_coverage') ?? '—', average_ae: average('ae') ?? '—', average_mutation_score: average('mutation_score') ?? '—', ...paperAverages(tasks, card.total_tasks || tasks.length) },
      model_usage: null,
      started_at: card.started_at,
      finished_at: card.finished_at,
      duration_seconds: card.duration_seconds,
    },
  };
}

export default function App() {
  const [view, setView] = useState<View>(initialView);
  const [experimentId, setExperimentId] = useState(selectedExperiment());
  const [runtime, setRuntime] = useState<RuntimeSnapshot | null>(null);
  const [knowledge, setKnowledge] = useState<KnowledgeSnapshot | null>(null);
  const [knowledgeError, setKnowledgeError] = useState('');
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState('');
  const [detail, setDetail] = useState<DetailContent | null>(null);
  const [chatOpen, setChatOpen] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    if (!detail && !chatOpen && !menuOpen) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      if (detail) setDetail(null);
      else if (chatOpen) setChatOpen(false);
      else setMenuOpen(false);
    };
    document.addEventListener('keydown', closeOnEscape);
    return () => document.removeEventListener('keydown', closeOnEscape);
  }, [detail, chatOpen, menuOpen]);

  useEffect(() => {
    let alive = true;
    let stream: EventSource | null = null;
    const request = new AbortController();
    const cached = runtimeCache.get(experimentId);
    if (cached) setRuntime(cached);
    setConnected(false);
    fetchRuntime(experimentId, request.signal).then((data) => {
      if (!alive) return;
      rememberRuntime(experimentId, data);
      setRuntime(data);
      setError('');
      const status = String(data.overview.status || '').toLowerCase();
      if (!['running', 'queued', 'waiting'].includes(status)) {
        setConnected(true);
        return;
      }
      stream = connectRuntime(experimentId, (next) => {
        if (!alive) return;
        rememberRuntime(experimentId, next);
        setRuntime(next);
        setConnected(true);
        setError('');
      }, () => alive && setConnected(false));
    }).catch((cause) => alive && cause instanceof DOMException && cause.name === 'AbortError' ? undefined : alive && setError(String(cause)));
    return () => { alive = false; request.abort(); stream?.close(); };
  }, [experimentId]);

  useEffect(() => {
    if (view !== 'knowledge') return undefined;
    let alive = true;
    const request = new AbortController();
    setKnowledge(null);
    setKnowledgeError('');
    const refresh = () => fetchKnowledge(experimentId, request.signal).then((data) => {
      if (!alive) return;
      setKnowledge(data);
      setKnowledgeError('');
    }).catch((cause) => {
      if (alive && !(cause instanceof DOMException && cause.name === 'AbortError')) setKnowledgeError('知识记录暂时无法读取，请稍后重试。');
    });
    refresh(); const timer = window.setInterval(refresh, 15000);
    return () => { alive = false; request.abort(); window.clearInterval(timer); };
  }, [experimentId, view]);

  function selectExperiment(id: string) {
    if (id !== experimentId) { setKnowledge(null); setKnowledgeError(''); }
    const cached = runtimeCache.get(id);
    const source = runtimeCache.get('all') || runtime;
    const preview = !cached && source && id !== 'all' ? datasetPreview(source, id) : null;
    if (cached) setRuntime(cached);
    else if (preview) { rememberRuntime(id, preview); setRuntime(preview); }
    setExperimentId(id);
    const url = new URL(window.location.href);
    if (id) url.searchParams.set('experiment', id); else url.searchParams.delete('experiment');
    window.history.replaceState({}, '', url);
  }

  async function openArtifact(path: string) {
    if (!runtime?.experiment?.id) return;
    setDetail({ title: '正在读取文件', body: <p>{path}</p> });
    try {
      const artifact: Artifact = await fetchArtifact(runtime.experiment.id, path);
      if (artifact.kind === 'state_model' && artifact.state_model) {
        setDetail({ title: '测试状态中心', eyebrow: `${artifact.size.toLocaleString()} 字节 · 专用可视化`, body: <StateModelVisual model={artifact.state_model} path={artifact.path} /> });
        return;
      }
      if (artifact.kind === 'data') {
        setDetail({ title: artifact.name, eyebrow: `${artifact.size.toLocaleString()} 字节 · 数据视图`, body: <DataVisual data={artifact.data} title="报告内容" /> });
        return;
      }
      if (artifact.kind === 'html' && artifact.content) {
        setDetail({ title: artifact.name, eyebrow: `${artifact.size.toLocaleString()} 字节 · 报告页面`, body: <iframe className="artifact-report-frame" title={artifact.name} sandbox="" srcDoc={artifact.content} /> });
        return;
      }
      if (artifact.hidden_internal_metrics) {
        setDetail({ title: artifact.name, body: <p>这份历史报告含内部过程指标，已暂停页面预览。请重新生成报告查看评价指标。</p> });
        return;
      }
      setDetail({ title: artifact.name, eyebrow: `${artifact.size.toLocaleString()} 字节`, body: <>{artifact.truncated && <p className="detail-note">文件较大，当前展示前200,000个字符；原始文件保持完整。</p>}<pre className="artifact-content">{artifact.content || '文件内容为空'}</pre></>, raw: { path: artifact.path, size: artifact.size } });
    } catch (cause) { setDetail({ title: '无法读取文件', body: <p>{String(cause)}</p>, raw: { path } }); }
  }

  function go(next: View) {
    setView(next);
    setMenuOpen(false);
    const url = new URL(window.location.href);
    if (next === 'valifix') url.searchParams.set('view', 'valifix');
    else url.searchParams.delete('view');
    window.history.replaceState({}, '', url);
  }
  const content = view === 'valifix' ? <ValiFixTestShowcase /> : runtime ? <>
    {view === 'overview' && <OverviewPage runtime={runtime} open={setDetail} />}
    {view === 'run' && <ExperimentLauncher runtime={runtime} open={setDetail} onStarted={() => { selectExperiment('all'); go('overview'); }} />}
    {view === 'agents' && <AgentsPage runtime={runtime} open={setDetail} />}
    {view === 'phases' && <PhasesPage runtime={runtime} open={setDetail} />}
    {view === 'state' && <StatePage runtime={runtime} open={setDetail} />}
    {view === 'tasks' && <TasksPage runtime={runtime} open={setDetail} openArtifact={openArtifact} />}
    {view === 'knowledge' && <KnowledgePage knowledge={knowledge} error={knowledgeError} open={setDetail} />}
    {view === 'system' && <SystemPage runtime={runtime} open={setDetail} />}
  </> : <div className="loading-state"><RefreshCw className="spin" /><h2>正在连接测试系统</h2><p>{error || '读取实验注册表与实时状态'}</p></div>;

  return <div className={`app-shell ${chatOpen ? 'chat-visible' : ''}`}>
    <aside id="app-navigation" className={`sidebar ${menuOpen ? 'open' : ''}`}>
      <div className="brand"><span><Layers3 /></span><div><b>单元测试代码智能生成系统</b><small>ValiFixTest · TSGColAgent</small></div><button className="sidebar-close" aria-label="关闭导航" onClick={() => setMenuOpen(false)}><X /></button></div>
      <nav>{nav.map(({ id, label, icon: Icon }) => <button key={id} className={`${view === id ? 'selected' : ''} ${id === 'valifix' ? 'valifix-nav-item' : ''}`.trim()} onClick={() => go(id)}><Icon /><span>{label}</span></button>)}</nav>
      <button className="sidebar-chat" onClick={() => { setMenuOpen(false); setChatOpen(true); }}><MessageSquareText /><div><b>测试知识智能体</b><small>报告查询与专业问答</small></div></button>
      <button className={`system-state ${view === 'system' ? 'selected' : ''}`} onClick={() => go('system')} title="查看系统运行能力"><i className={connected ? 'online' : ''} /><div><b>{connected ? '系统在线' : '正在重连'}</b><small>语言、范围、质量与执行策略</small></div></button>
    </aside>
    {menuOpen && <button type="button" className="sidebar-backdrop" aria-label="点击页面关闭导航" onClick={() => setMenuOpen(false)} />}
    <div className="workspace">
      <header className="topbar">
        <button className="menu-button" title="打开导航" aria-controls="app-navigation" aria-expanded={menuOpen} onClick={() => setMenuOpen(true)}><Menu /></button>
        {view === 'valifix' ? <div className="method-context"><i><ScanSearch /></i><span><small>当前页面</small><b>ValiFixTest 方法展示</b></span></div> : <ExperimentSwitcher runtime={runtime} onSelect={selectExperiment} />}
        {view === 'valifix' ? <div className="connection"><ScanSearch /><span>方法概览</span></div> : <div className={`connection ${connected ? 'online' : ''}`}>{connected ? <Wifi /> : <WifiOff />}<span>{connected ? '实时同步' : '等待更新'}</span></div>}
        <button className="top-chat" title="打开测试知识智能体" aria-label="打开测试知识智能体" onClick={() => setChatOpen(true)}><MessageSquareText /><span>对话</span></button>
      </header>
      <main>{content}</main>
    </div>
    <ChatWorkspace open={chatOpen} runtime={runtime} onClose={() => setChatOpen(false)} />
    {detail && <DetailDrawer detail={detail} onClose={() => setDetail(null)} />}
  </div>;
}
