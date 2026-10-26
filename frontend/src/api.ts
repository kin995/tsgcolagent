import type { Artifact, ChatReply, Conversation, ExperimentRequest, KnowledgeSnapshot, RuntimeSnapshot } from './types';
import { containsProcessMetric, publicDisplayData } from './displayPolicy';

export function selectedExperiment(): string { return new URLSearchParams(window.location.search).get('experiment') || 'all'; }

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, { cache: 'no-store', ...init });
  const payload = await response.json();
  if (!response.ok) throw new Error(publicDisplayData(String(payload.error || `请求失败：${response.status}`)));
  return publicDisplayData(payload) as T;
}

export const fetchRuntime = (experiment = selectedExperiment(), signal?: AbortSignal) => json<RuntimeSnapshot>(`/api/runtime?experiment=${encodeURIComponent(experiment)}`, { signal });
export const fetchKnowledge = (experiment = selectedExperiment(), signal?: AbortSignal) => json<KnowledgeSnapshot>(`/api/knowledge?experiment=${encodeURIComponent(experiment)}`, { signal });
export const fetchConversations = () => json<{ chats: Conversation[] }>('/api/chats');
export const fetchConversation = (id: string) => json<Conversation>(`/api/chats/${encodeURIComponent(id)}`);
export const createConversation = (title?: string) => json<Conversation>('/api/chats', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ title }) });
export const renameConversation = (id: string, title: string) => json<Conversation>(`/api/chats/${encodeURIComponent(id)}/rename`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ title }) });
export const deleteConversation = (id: string) => json<{ deleted: boolean }>(`/api/chats/${encodeURIComponent(id)}`, { method: 'DELETE' });
export const sendConversationMessage = (id: string, question: string, experiment?: string) => json<ChatReply>(`/api/chats/${encodeURIComponent(id)}/messages`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question, experiment: experiment || undefined }) });
export const startExperiment = (request: ExperimentRequest) => json<{ experiment: { id: string; label: string; output_dir: string; status: string }; process_id?: number; queue_position: number; log_path: string }>('/api/experiments/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(request) });
export const fetchArtifact = async (experiment: string, path: string): Promise<Artifact> => {
  const response = await fetch(`/api/artifact?experiment=${encodeURIComponent(experiment)}&path=${encodeURIComponent(path)}`, { cache: 'no-store' });
  const artifact = await response.json() as Artifact & { error?: string };
  if (!response.ok) throw new Error(publicDisplayData(String(artifact.error || `请求失败：${response.status}`)));
  if (artifact.kind === 'html' && artifact.content && containsProcessMetric(artifact.content)) {
    return { ...artifact, content: undefined, hidden_internal_metrics: true };
  }
  return publicDisplayData(artifact);
};

export function connectRuntime(experiment: string, onSnapshot: (data: RuntimeSnapshot) => void, onError: () => void): EventSource {
  const source = new EventSource(`/api/events?experiment=${encodeURIComponent(experiment)}`);
  source.addEventListener('snapshot', (event) => onSnapshot(publicDisplayData(JSON.parse((event as MessageEvent).data))));
  source.onerror = onError;
  return source;
}
