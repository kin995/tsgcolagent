import { useEffect, useRef, useState } from 'react';
import { BookOpenCheck, Check, ChevronLeft, LoaderCircle, MessageSquarePlus, PanelRightClose, Pencil, Send, Sparkles, Trash2, X } from 'lucide-react';
import { createConversation, deleteConversation, fetchConversation, fetchConversations, renameConversation, sendConversationMessage } from '../api';
import type { Conversation, RuntimeSnapshot } from '../types';

export function ChatWorkspace({ open, runtime, onClose }: { open: boolean; runtime: RuntimeSnapshot | null; onClose: () => void }) {
  const [chats, setChats] = useState<Conversation[]>([]);
  const [active, setActive] = useState<Conversation | null>(null);
  const [question, setQuestion] = useState('');
  const [busy, setBusy] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [title, setTitle] = useState('');
  const endRef = useRef<HTMLDivElement>(null);

  async function reloadChats() { const result = await fetchConversations(); setChats(result.chats); return result.chats; }
  async function load(id: string) { const chat = await fetchConversation(id); setActive(chat); setTitle(chat.title); }
  async function addChat() { const chat = await createConversation(); setChats((items) => [chat, ...items]); setActive(chat); setTitle(chat.title); }

  useEffect(() => {
    if (!open) return;
    reloadChats().then((items) => { if (!active && items[0]) load(items[0].id); }).catch(() => undefined);
  }, [open]);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [active?.messages, busy]);
  useEffect(() => {
    const chatId = active?.id;
    if (!open || !chatId || busy) return;
    let cancelled = false;
    const timer = window.setInterval(() => {
      fetchConversation(chatId).then((chat) => {
        if (!cancelled) setActive((current) => current?.id === chatId ? chat : current);
      }).catch(() => undefined);
    }, 5000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [open, active?.id, busy]);

  async function send() {
    const value = question.trim(); if (!value || busy) return;
    let chat = active;
    if (!chat) chat = await createConversation();
    const optimistic = { id: `local-${Date.now()}`, role: 'user' as const, content: value, created_at: new Date().toISOString() };
    setActive({ ...chat, messages: [...(chat.messages || []), optimistic] }); setQuestion(''); setBusy(true);
    try {
      const reply = await sendConversationMessage(chat.id, value);
      setActive(reply.conversation); setTitle(reply.conversation.title); await reloadChats();
    } catch (cause) {
      setActive((current) => current ? { ...current, messages: [...(current.messages || []), { id: `error-${Date.now()}`, role: 'assistant', agent: '测试知识智能体', content: `暂时无法完成查询：${String(cause)}`, created_at: new Date().toISOString() }] } : current);
    } finally { setBusy(false); }
  }

  async function saveTitle() { if (!active) return; const renamed = await renameConversation(active.id, title); setActive({ ...active, ...renamed }); setRenaming(false); await reloadChats(); }
  async function remove() { if (!active) return; await deleteConversation(active.id); const next = (await reloadChats())[0]; setActive(null); if (next) await load(next.id); }

  if (!open) return null;
  return <><button type="button" className="chat-backdrop" aria-label="点击页面关闭测试知识智能体对话" onClick={onClose} /><aside className="chat-workspace" role="dialog" aria-modal="true" aria-label="测试知识智能体对话">
    <div className="chat-list">
      <header><div><Sparkles /><b>测试知识智能体</b></div><button title="新建对话" onClick={addChat}><MessageSquarePlus /></button></header>
      <p>对话名称与实验任务相互独立</p>
      <div>{chats.map((chat) => <button key={chat.id} className={active?.id === chat.id ? 'selected' : ''} onClick={() => load(chat.id)}><b>{chat.title}</b><span>{chat.last_message || '尚未开始对话'}</span><small>{chat.message_count || 0}条</small></button>)}</div>
    </div>
    <div className="chat-main">
      <header>
        <button className="mobile-back" title="对话列表"><ChevronLeft /></button>
        <div>{renaming && active ? <label className="rename-box"><input value={title} autoFocus onChange={(event) => setTitle(event.target.value)} onKeyDown={(event) => event.key === 'Enter' && saveTitle()} /><button title="确认名称" onClick={saveTitle}><Check /></button><button title="取消" onClick={() => setRenaming(false)}><X /></button></label> : <><b>{active?.title || '新的单元测试对话'}</b><small>{active?.context_label ? `已识别上下文：${active.context_label}` : modelStatus(runtime)}</small></>}</div>
        {active && !renaming && <button title="重命名对话" onClick={() => setRenaming(true)}><Pencil /></button>}
        {active && <button title="删除对话" onClick={remove}><Trash2 /></button>}
        <button title="关闭对话" onClick={onClose}><PanelRightClose /></button>
      </header>
      <div className="chat-messages">
        {!active?.messages?.length && <div className="chat-welcome"><span><Sparkles /></span><h2>你想了解哪部分测试？</h2><p>可以查询某个实验或任务的覆盖率、失败原因、报告与知识经验，也可以讨论Python、Java单元测试设计。问题中提到任务名时，我会自动读取对应上下文。</p><div>{['概括当前实验结果', '分析某个失败任务的原因', '怎样提高Java变异检测率'].map((prompt) => <button key={prompt} onClick={() => setQuestion(prompt)}>{prompt}</button>)}</div></div>}
        {active?.messages?.map((message) => <article key={message.id} className={`chat-message ${message.role}`}><header><b>{message.role === 'assistant' ? responseLabel(message.response_mode, message.model_used) : '你'}</b>{message.context?.label && <span>{message.context.label}</span>}</header><p>{message.content}</p>{message.pending_learning && <div className="chat-learning"><BookOpenCheck /><span>问题已进入自主学习队列</span><b>等待模型恢复 · 已尝试{message.pending_learning.attempts || 0}次</b></div>}{message.learning && <div className="chat-learning"><BookOpenCheck /><span>已归纳到测试知识引导库</span><b>{message.learning.topic} · {message.learning.status}</b></div>}{message.sources?.length ? <details><summary>查看数据来源</summary>{message.sources.map((source) => <code key={source}>{source}</code>)}</details> : null}</article>)}
        {busy && <article className="chat-message assistant pending"><LoaderCircle className="spin" /><span>测试知识智能体正在检索知识；模型可用时联合回答，否则自主判断并进入学习闭环</span></article>}
        <div ref={endRef} />
      </div>
      <div className="chat-composer"><textarea value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); send(); } }} placeholder="询问实验数据、测试报告、失败原因或单元测试知识" /><button title="发送" disabled={!question.trim() || busy} onClick={send}><Send /></button></div>
    </div>
  </aside></>;
}

function modelStatus(runtime: RuntimeSnapshot | null) { const model = runtime?.assistant; return model?.enabled ? `${model.provider} / ${model.model} · 后端统一模型与密钥` : '报告查询可用 · 开放式问答随实验后端模型配置'; }

function responseLabel(mode?: string, modelUsed?: boolean) {
  if (mode === 'knowledge_autonomous') return '测试知识智能体 · 知识自主回答';
  if (mode === 'pending_learning') return '测试知识智能体 · 待学习';
  if (mode === 'autonomous_learning_completed') return '测试知识智能体 · 自动补学完成';
  return modelUsed ? '测试知识智能体 + LLM' : '测试知识智能体';
}
