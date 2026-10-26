import { useState } from 'react';
import { Braces, Database, FileCode2, FolderTree, KeyRound, LoaderCircle, Play, Sparkles } from 'lucide-react';
import { startExperiment } from '../api';
import type { ExperimentRequest, RuntimeSnapshot } from '../types';
import type { DetailContent } from './DetailDrawer';

type OpenDetail = (detail: DetailContent) => void;

export function ExperimentLauncher({ runtime, open, onStarted }: { runtime: RuntimeSnapshot; open: OpenDetail; onStarted: (id: string) => void }) {
  const [request, setRequest] = useState<ExperimentRequest>({ mode: 'single', source: '', output: '', language: 'auto', label: '', llm: false, llm_provider: 'ollama', llm_model: 'deepseek-r1:14b', llm_base_url: 'http://127.0.0.1:11434', llm_api_key: '' });
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState('');
  const canStart = Boolean(request.source.trim() && request.output.trim() && !busy);
  const modes = [
    { id: 'single' as const, label: '单文件/类', icon: FileCode2, description: '直接运行一个Python文件或Java类文件。' },
    { id: 'directory' as const, label: '项目/目录', icon: FolderTree, description: '递归发现目录内任务并执行批量测试生成。' },
    { id: 'datasets' as const, label: '数据集根目录', icon: Database, description: '按数据集清单执行Python与Java数据集。' },
  ];

  async function start() {
    if (!request.source.trim() || !request.output.trim() || busy) return;
    setBusy(true); setNotice('');
    try {
      const result = await startExperiment(request);
      setNotice(result.queue_position <= 1 ? '实验已进入执行队列，即将开始运行' : `实验已排队，当前位于第${result.queue_position}位`);
      onStarted(result.experiment.id);
    } catch (cause) {
      setNotice(String(cause));
    } finally {
      setBusy(false);
    }
  }

  return <div className="page-stack">
    <div className="page-title"><h1>启动实验</h1><p>Configure source code and output paths to start test generation</p></div>
    <section className="launcher-layout">
      <div className="launch-form">
        <header><span><Play /></span><div><h2>新实验</h2><p>选择真实源代码、项目或数据集目录</p></div></header>
        <div className="mode-switch">{modes.map(({ id, label, icon: Icon, description }) => <button key={id} className={request.mode === id ? 'selected' : ''} onClick={() => setRequest({ ...request, mode: id })} title={description}><Icon /><span>{label}</span></button>)}</div>
        <label className="field"><span>{request.mode === 'single' ? '源代码文件' : request.mode === 'directory' ? '项目或源码目录' : '数据集根目录'}</span><input value={request.source} onChange={(event) => setRequest({ ...request, source: event.target.value })} placeholder={request.mode === 'single' ? 'D:\\project\\src\\Calculator.java' : 'D:\\project\\datasets'} /></label>
        <label className="field"><span>实验输出目录</span><input value={request.output} onChange={(event) => setRequest({ ...request, output: event.target.value })} placeholder="D:\\project\\runs\\experiment-name" /></label>
        <div className="field-row">
          <label className="field"><span>编程语言</span><select value={request.language} disabled={request.mode === 'single' || request.mode === 'datasets'} onChange={(event) => setRequest({ ...request, language: event.target.value as ExperimentRequest['language'] })}><option value="auto">自动识别</option><option value="python">Python</option><option value="java">Java</option><option value="all">Python + Java</option></select></label>
          <label className="field"><span>实验名称</span><input value={request.label} onChange={(event) => setRequest({ ...request, label: event.target.value })} placeholder="留空则使用源目录名" /></label>
        </div>
        <label className="toggle-row"><input type="checkbox" checked={request.llm} onChange={(event) => setRequest({ ...request, llm: event.target.checked })} /><span><Sparkles />启用大模型与测试知识智能体</span><small>当前配置只随本次启动提交到本机后端，不写入浏览器持久化数据</small></label>
        {request.llm && <section className="model-config">
          <header><KeyRound /><div><b>本次实验模型</b><span>直接沿用智能体原有LLM调用链</span></div></header>
          <div className="field-row">
            <label className="field"><span>模型服务</span><select value={request.llm_provider} onChange={(event) => setRequest({ ...request, llm_provider: event.target.value })}><option value="ollama">Ollama 本地模型</option><option value="openai">OpenAI兼容服务</option></select></label>
            <label className="field"><span>模型名称</span><input list="economical-models" value={request.llm_model} onChange={(event) => setRequest({ ...request, llm_model: event.target.value })} /><datalist id="economical-models"><option value="deepseek-r1:14b" /><option value="qwen2.5-coder:14b" /><option value="qwen2.5-coder:7b" /><option value="gpt-4.1-mini" /><option value="gpt-4o-mini" /></datalist></label>
          </div>
          <label className="field"><span>服务地址</span><input value={request.llm_base_url} onChange={(event) => setRequest({ ...request, llm_base_url: event.target.value })} placeholder="http://127.0.0.1:11434" /></label>
          <label className="field"><span>API密钥</span><input type="password" autoComplete="off" value={request.llm_api_key} onChange={(event) => setRequest({ ...request, llm_api_key: event.target.value })} placeholder="本地Ollama可留空" /></label>
        </section>}
        {notice && <p className={notice.startsWith('实验已') ? 'form-notice success' : notice.startsWith('ValiFixTest') ? 'form-notice info' : 'form-notice error'}>{notice}</p>}
        <section className="generation-methods"><header><div><b>选择生成方法</b><span>填写源代码与输出目录后即可启动</span></div>{!canStart && !busy && <small>尚需填写两个必填目录</small>}</header><div className="launch-actions"><button className="primary-command" disabled={!canStart} onClick={start}>{busy ? <LoaderCircle className="spin" /> : <Play />}<span>{busy ? '正在启动' : 'TSGColAgent 生成'}</span></button><button type="button" className="secondary-command" disabled><Sparkles /><span>ValiFixTest方法生成</span></button></div></section>
      </div>
      <div className="launch-side">
        <section><header><Braces /><div><h2>执行入口</h2><p>实际调用项目已有运行器</p></div></header>{modes.map((mode) => <button key={mode.id} onClick={() => open({ title: mode.label, eyebrow: '实验入口', body: <p>{mode.description}</p>, raw: mode })}><mode.icon /><div><b>{mode.label}</b><span>{mode.description}</span></div></button>)}</section>
        <section><header><Database /><div><h2>已登记实验</h2><p>可切换查看，不重新遍历数据集</p></div></header>{runtime.experiments.slice(0, 8).map((item) => <button key={item.id} onClick={() => onStarted(item.id)}><i className={item.status === 'running' ? 'running' : ''} /><div><b>{item.label}</b><span>{item.output_dir}</span></div></button>)}</section>
      </div>
    </section>
  </div>;
}
