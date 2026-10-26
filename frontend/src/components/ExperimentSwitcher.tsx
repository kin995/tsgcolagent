import { useEffect, useRef, useState } from 'react';
import { Check, CheckCircle2, ChevronDown, Clock3, FolderOpen, GitCompareArrows, Layers3, PlayCircle } from 'lucide-react';
import type { RuntimeSnapshot } from '../types';

export function ExperimentSwitcher({ runtime, onSelect }: { runtime: RuntimeSnapshot | null; onSelect: (id: string) => void }) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const close = (event: MouseEvent) => { if (!root.current?.contains(event.target as Node)) setOpen(false); };
    document.addEventListener('mousedown', close); return () => document.removeEventListener('mousedown', close);
  }, []);
  const current = runtime?.experiment;
  return <div className="experiment-switcher" ref={root}>
    <button type="button" className="experiment-trigger" onClick={() => setOpen(!open)} aria-expanded={open}>
      <span><GitCompareArrows /></span><div><small>当前实验范围</small><b>{current?.label || '实验总览'}</b></div><em>{statusLabel(current?.status)}</em><ChevronDown />
    </button>
    {open && <div className="experiment-menu portfolio-menu">
      <header><div><h2>选择实验数据范围</h2><p>点击实验总览或一个数据集，页面图表与任务同步切换</p></div><span>{runtime?.experiments.length || 0}个数据集</span></header>
      <div className="portfolio-overview-choice"><button type="button" className={current?.id === 'all' ? 'selected' : ''} onClick={() => { onSelect('all'); setOpen(false); }}><Layers3 /><span><b>实验总览</b><small>全部数据集的综合结果</small></span><em>{runtime?.experiments.length || 0}个数据集</em></button></div>
      <section className="experiment-menu-list"><header><span>各数据集实验</span><em>点击切换该数据集的图表、任务与报告</em></header>{runtime?.experiments.map((item) => {
        const selected = item.id === current?.id;
        return <button type="button" key={item.id} className={selected ? 'selected' : ''} onClick={() => { onSelect(item.id); setOpen(false); }}>
          <i>{selected ? <Check /> : item.status === 'running' ? <PlayCircle /> : item.status === 'queued' ? <Clock3 /> : <CheckCircle2 />}</i>
          <div><b>{item.label}</b><span><FolderOpen />{item.output_dir}</span><small><i><b style={{ width: `${Math.max(2, (item.progress || 0) * 100)}%` }} /></i>{item.completed_execution || 0}/{item.total_tasks || 0}项</small></div>
          <em className={`status-${item.status}`}>{item.status === 'queued' && item.queue_position ? `队列第${item.queue_position}位` : `${languageName(item.language)} · ${scopeName(item.scope)}`}</em>
        </button>;
      })}</section>
      <footer>总览与各数据集共用同一批真实实验产物</footer>
    </div>}
  </div>;
}

function statusLabel(status?: string) { return ({ running: '运行中', completed: '已完成', queued: '排队中', failed: '运行失败', waiting: '等待中', mixed: '融合视图' } as Record<string, string>)[status || ''] || '已登记'; }
function languageName(language?: string) { return ({ java: 'Java', python: 'Python', mixed: 'Python + Java', auto: '自动识别' } as Record<string, string>)[language || ''] || '待识别'; }
function scopeName(scope?: string) { return ({ file: '文件级', class: '类级', project: '项目级', file_function: '文件 / 函数级', file_function_class: '文件 / 函数 / 类级', file_class: '文件 / 类级', class_project: '类 / 项目级', project_file: '项目 / 文件级', mixed_scope: '项目 / 文件级', datasets: '全数据集级', portfolio: '实验组合' } as Record<string, string>)[scope || ''] || '待识别'; }
