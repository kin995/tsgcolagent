import type { CSSProperties, ReactNode } from 'react';
import { BarChart3, CircleDot, GitBranch, Layers3, Network, ShieldCheck, X } from 'lucide-react';
import type { RuntimeSnapshot } from '../types';
import { containsProcessMetric, isProcessMetricKey, publicDisplayData } from '../displayPolicy';

export interface DetailContent { title: string; eyebrow?: string; body?: ReactNode; raw?: unknown }

export default function DetailDrawer({ detail, onClose }: { detail: DetailContent | null; onClose: () => void }) {
  if (!detail) return null;
  return <div className="drawer-layer" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
    <aside className="detail-drawer" role="dialog" aria-modal="true" aria-label={detail.title}>
      <header><div><small>{detail.eyebrow || '详细信息'}</small><h2>{detail.title}</h2></div><button className="icon-button" title="关闭详情" onClick={onClose}><X /></button></header>
      <div className="drawer-content">{detail.body}{detail.raw !== undefined && <DataVisual data={detail.raw} />}</div>
    </aside>
  </div>;
}

export function DetailList({ title, items }: { title: string; items: Array<string | undefined> }) {
  const values = items.filter((item): item is string => typeof item === 'string' && Boolean(item.trim()) && !containsProcessMetric(item) && !item.includes('内部过程指标已隐藏'));
  if (!values.length) return null;
  return <section className="detail-section"><h3>{title}</h3><div className="visual-tags">{values.map((item) => <span key={item}>{item}</span>)}</div></section>;
}

export function StateModelVisual({ model, path }: { model: RuntimeSnapshot['state_model'] & { source_path?: string }; path?: string }) {
  const nodes = model.nodes || []; const edges = model.edges || [];
  const evidenced = nodes.filter((node) => node.evidenced || node.visited).length;
  const layers = ['程序状态', '测试状态', '运行证据', '决策状态'].map((name) => {
    const values = nodes.filter((node) => stateLayer(node.type) === name);
    return { name, count: values.length, evidenced: values.filter((node) => node.evidenced || node.visited).length };
  });
  const nodeTypes = countBy(nodes, (node) => node.type || 'State');
  const edgeTypes = countBy(edges, (edge) => edge.type || 'relation');
  return <div className="tsm-detail">
    <section className="tsm-detail-head"><span><Network /></span><div><small>测试状态中心实时解析</small><b>共享TSG v{model.version || 0}</b><p>状态迭代{model.iteration || 0} · {nodes.length}节点 · {edges.length}关系</p></div><i className={nodes.length ? 'online' : ''} /></section>
    {nodes.length ? <>
      <div className="tsm-kpis"><div><Layers3 /><small>状态节点</small><b>{nodes.length}</b></div><div><GitBranch /><small>状态关系</small><b>{edges.length}</b></div><div><ShieldCheck /><small>证据节点</small><b>{evidenced}</b></div><div><CircleDot /><small>待验证节点</small><b>{nodes.length - evidenced}</b></div></div>
      <section className="tsm-layer-map"><header><b>四层状态空间</b><span>各层节点与证据数量</span></header><div>{layers.map((layer, index) => <article key={layer.name} style={{ '--layer-index': index } as CSSProperties}><i><span style={{ height: layer.count ? '100%' : '0%' }} /></i><small>0{index + 1}</small><b>{layer.name}</b><em>{layer.evidenced}个有证据 / {layer.count}个节点</em></article>)}</div></section>
      <div className="tsm-distributions"><Distribution title="节点类型分布" items={nodeTypes} total={nodes.length} /><Distribution title="关系类型分布" items={edgeTypes} total={edges.length} /></div>
      <section className="tsm-evidence-stream"><header><b>状态证据样本</b><span>展示最近可读取的结构状态</span></header>{nodes.slice(0, 8).map((node) => <article key={node.id}><i className={node.evidenced || node.visited ? 'evidenced' : ''} /><div><b>{node.name || node.id}</b><span>{node.type || 'State'} · {node.line ? `源码行 ${node.line}` : '结构节点'}</span></div><em>{node.evidenced || node.visited ? '已有证据' : '待验证'}</em></article>)}</section>
    </> : <p className="detail-note">测试状态文件已经读取，但当前快照尚未形成可展示节点。</p>}
    {(model.source_path || path) && <p className="tsm-source">数据来源：{model.source_path || path}</p>}
  </div>;
}

function Distribution({ title, items, total }: { title: string; items: Array<[string, number]>; total: number }) {
  return <section><header><b>{title}</b><span>{items.length}类</span></header>{items.slice(0, 8).map(([label, count]) => <div key={label}><span>{label}</span><i><b style={{ width: `${count / Math.max(1, total) * 100}%` }} /></i><em>{count}</em></div>)}</section>;
}

function countBy<T>(values: T[], keyFor: (value: T) => string): Array<[string, number]> {
  const counts = new Map<string, number>(); values.forEach((value) => { const key = keyFor(value); counts.set(key, (counts.get(key) || 0) + 1); });
  return [...counts.entries()].sort((a, b) => b[1] - a[1]);
}

function stateLayer(type?: string) {
  const value = String(type || '').toLowerCase();
  if (/(test|assert|oracle)/.test(value)) return '测试状态';
  if (/(runtime|mutation|coverage|execution|type)/.test(value)) return '运行证据';
  if (/(intent|decision|feedback|gap|repair|quality)/.test(value)) return '决策状态';
  return '程序状态';
}

export function DataVisual({ data, title = '数据视图' }: { data: unknown; title?: string }) {
  if (data == null) return null;
  const visible = publicDisplayData(data);
  if (Array.isArray(visible)) return <ArrayVisual values={visible} title={title} />;
  if (typeof visible === 'object') return <ObjectVisual value={visible as Record<string, unknown>} title={title} />;
  return <section className="data-visual"><header><CircleDot /><h3>{title}</h3></header><strong className="visual-single">{displayValue(visible)}</strong></section>;
}

const visibleMetricKey = (key: string) => !isProcessMetricKey(key);

function ObjectVisual({ value, title }: { value: Record<string, unknown>; title: string }) {
  const entries = Object.entries(value).filter(([key, item]) => visibleMetricKey(key) && item !== undefined && item !== null);
  const scalars = entries.filter(([, item]) => !Array.isArray(item) && typeof item !== 'object').slice(0, 18);
  const numerics = scalars.filter(([, item]) => typeof item === 'number') as Array<[string, number]>;
  const nested = entries.filter(([, item]) => Array.isArray(item) || (item && typeof item === 'object')).slice(0, 8);
  const max = Math.max(1, ...numerics.map(([, item]) => Math.abs(item)));
  return <section className="data-visual">
    <header><BarChart3 /><h3>{title}</h3><span>{entries.length}项信息</span></header>
    {scalars.length > 0 && <div className="visual-metric-grid">{scalars.map(([key, item]) => <div key={key}><small>{labelFor(key)}</small><b>{displayValue(item, key)}</b>{typeof item === 'number' && <i><span style={{ width: `${Math.min(100, ratioKey(key) ? Math.abs(item) * 100 : Math.abs(item) / max * 100)}%` }} /></i>}</div>)}</div>}
    {nested.map(([key, item]) => Array.isArray(item) ? <ArrayVisual key={key} values={item} title={labelFor(key)} compact /> : <NestedObject key={key} title={labelFor(key)} value={item as Record<string, unknown>} />)}
  </section>;
}

function ArrayVisual({ values, title, compact = false }: { values: unknown[]; title: string; compact?: boolean }) {
  const shown = values.slice(0, compact ? 20 : 40);
  const scalar = shown.every((item) => item == null || typeof item !== 'object');
  const distribution = scalar ? null : distributionFor(values);
  return <section className="data-visual visual-array"><header><Layers3 /><h3>{title}</h3><span>{values.length}项</span></header>
    {distribution && <div className="visual-distribution"><h4>按{labelFor(distribution.key)}分布</h4>{distribution.items.map(([label, count]) => <div key={label}><span>{displayValue(label, distribution.key)}</span><i><b style={{ width: `${count / distribution.max * 100}%` }} /></i><em>{count}</em></div>)}</div>}
    {scalar ? <div className="visual-tags">{shown.map((item, index) => <span key={`${index}-${String(item)}`}>{displayValue(item)}</span>)}</div> : <div className="visual-records">{shown.map((item, index) => <RecordCard key={index} value={item} index={index} />)}</div>}
    {values.length > shown.length && <p className="visual-more">另有{values.length - shown.length}项，可在对应业务页面筛选查看</p>}
  </section>;
}

function NestedObject({ title, value }: { title: string; value: Record<string, unknown> }) {
  const entries = Object.entries(value).filter(([key, item]) => visibleMetricKey(key) && item != null).slice(0, 16);
  if (!entries.length) return null;
  return <div className="visual-nested"><h4>{title}</h4><div>{entries.map(([key, item]) => <span key={key}><small>{labelFor(key)}</small><b>{typeof item === 'object' ? countValue(item) : displayValue(item, key)}</b></span>)}</div></div>;
}

function RecordCard({ value, index }: { value: unknown; index: number }) {
  if (!value || typeof value !== 'object') return <span>{displayValue(value)}</span>;
  const entries = Object.entries(value as Record<string, unknown>).filter(([key, item]) => visibleMetricKey(key) && item != null && typeof item !== 'object').slice(0, 5);
  const heading = entries.find(([key]) => ['name', 'title', 'label', 'action', 'event_type', 'id'].includes(key));
  return <article><i>{String(index + 1).padStart(2, '0')}</i><div><b>{heading ? displayValue(heading[1], heading[0]) : `记录 ${index + 1}`}</b><p>{entries.filter(([key]) => key !== heading?.[0]).map(([key, item]) => `${labelFor(key)}：${displayValue(item, key)}`).join(' · ') || '包含关联数据'}</p></div></article>;
}

function displayValue(value: unknown, key = ''): string {
  if (typeof value === 'boolean') return value ? '是' : '否';
  if (typeof value === 'number') return ratioKey(key) ? `${(value * 100).toFixed(1)}%` : value.toLocaleString();
  if (value == null || value === '') return '暂无';
  if (key === 'language') return ({ java: 'Java', python: 'Python', mixed: 'Python + Java', auto: '自动识别' } as Record<string, string>)[String(value)] || String(value);
  if (key === 'status' || key === 'execution_status') return ({ completed: '已完成', running: '运行中', queued: '排队中', failed: '运行失败', mixed: '融合视图', waiting: '等待中', done: '已有证据', pending: '等待执行' } as Record<string, string>)[String(value)] || String(value);
  if (key === 'mode') return ({ execute: '真实执行', generate: '仅生成', portfolio: '多实验融合' } as Record<string, string>)[String(value)] || String(value);
  if (key === 'scope') return ({ file: '文件级', class: '类级', project: '项目级', datasets: '数据集级', portfolio: '多实验融合' } as Record<string, string>)[String(value)] || String(value);
  if (key.endsWith('_at')) { const date = new Date(String(value)); return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString('zh-CN', { hour12: false }); }
  return String(value);
}
function countValue(value: unknown): string { return Array.isArray(value) ? `${value.length}项` : value && typeof value === 'object' ? `${Object.keys(value).length}项` : displayValue(value); }
function ratioKey(key: string) { return ['line_coverage', 'branch_coverage', 'mutation_score', 'ae'].includes(key); }
function distributionFor(values: unknown[]) {
  const objects = values.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === 'object' && !Array.isArray(item));
  const key = ['grade', 'type', 'status', 'event_type', 'source_agent'].find((candidate) => objects.filter((item) => item[candidate] != null).length >= Math.max(1, objects.length / 2));
  if (!key) return null;
  const counts = new Map<string, number>(); objects.forEach((item) => { const label = String(item[key] ?? '其他'); counts.set(label, (counts.get(label) || 0) + 1); });
  const items = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 12);
  return { key, items, max: Math.max(1, ...items.map(([, count]) => count)) };
}
function labelFor(key: string): string { return ({ language: '语言', mode: '模式', status: '状态', execution_status: '执行状态', queue_position: '队列位置', progress: '目录进度', total_tasks: '任务总数', completed_execution: '完整执行', not_completed_execution: '未完整执行', followup_task_count: '后续轮次任务', followup_attempts: '后续轮次次数', grade_counts: '质量等级', quality_averages: '质量平均值', scope: '测试范围', dataset_count: '实验目录数', completed_datasets: '已完成实验目录', started_at: '开始时间', duration_seconds: '耗时秒数', finished_at: '结束时间', tasks: '任务', failed_tasks: '失败任务', failure_categories: '失败分类', name: '名称', type: '类型', visited: '程序访问', evidenced: '综合证据', line: '源码行', coverage_status: '覆盖状态', mutation_status: '变异状态', assertion_effective: '断言有效', repair_flag: '修复标记', source: '来源', target: '目标', event_type: '事件', source_agent: '发起智能体', target_agent: '接收智能体', iteration: '迭代', event_count: '事件数量', tests_passed: '测试通过', line_coverage: '行覆盖率', branch_coverage: '分支覆盖率', mutation_score: '变异得分', ae: '断言有效性' } as Record<string, string>)[key] || key.replaceAll('_', ' '); }
