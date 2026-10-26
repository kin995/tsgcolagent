import { afterEach, describe, expect, it, vi } from 'vitest';
import { connectRuntime, fetchArtifact, fetchRuntime, selectedExperiment, sendConversationMessage, startExperiment } from './api';

describe('dashboard API client', () => {
  afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

  it('reads the experiment id from the URL', () => {
    window.history.replaceState({}, '', '/?experiment=java-run');
    expect(selectedExperiment()).toBe('java-run');
  });

  it('requests the selected runtime snapshot', async () => {
    const mock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({ schema_version: 1 }), { status: 200 }));
    await fetchRuntime('run-1');
    expect(mock).toHaveBeenCalledWith('/api/runtime?experiment=run-1', { cache: 'no-store' });
  });

  it('filters saved process scores before runtime data reaches any page', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({
      schema_version: 1,
      overview: { quality_averages: { average_sfc: .8, average_mutation_score: .7 }, overall_quality_index: .9 },
      tasks: [{ metrics: { nrr: 1, mutation_score: .7 } }],
    }), { status: 200 }));
    const result = await fetchRuntime('old-run');
    expect(result.overview.quality_averages).toEqual({ average_mutation_score: .7 });
    expect(result.overview).not.toHaveProperty('overall_quality_index');
    expect(result.tasks[0].metrics).toEqual({ mutation_score: .7 });
  });

  it('blocks historical HTML previews that still contain process metrics', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({
      path: 'old.html', name: 'old.html', size: 99, kind: 'html', content: '<p>SFC: 80%</p>',
    }), { status: 200 }));
    const artifact = await fetchArtifact('old-run', 'old.html');
    expect(artifact.hidden_internal_metrics).toBe(true);
    expect(artifact.content).toBeUndefined();
  });

  it('filters process scores from live event snapshots', () => {
    let receive: ((event: MessageEvent) => void) | undefined;
    vi.stubGlobal('EventSource', class {
      addEventListener(_type: string, listener: (event: MessageEvent) => void) { receive = listener; }
    });
    const onSnapshot = vi.fn();
    connectRuntime('running', onSnapshot, vi.fn());
    receive?.({ data: JSON.stringify({ overview: { quality_averages: { average_sfc: .8, average_mutation_score: .6 } } }) } as MessageEvent);
    expect(onSnapshot).toHaveBeenCalledWith({ overview: { quality_averages: { average_mutation_score: .6 } } });
  });

  it('sends a persistent conversation to the test knowledge agent', async () => {
    const mock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({ conversation: {}, message: {}, answer: {} }), { status: 200 }));
    await sendConversationMessage('chat-1', 'Calculator任务的覆盖率？');
    expect(mock).toHaveBeenCalledWith('/api/chats/chat-1/messages', expect.objectContaining({ method: 'POST' }));
  });

  it('starts the existing terminal runner through the backend', async () => {
    const mock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({ experiment: { id: 'run-1' }, process_id: 8 }), { status: 201 }));
    await startExperiment({ mode: 'single', source: 'D:\\Calculator.java', output: 'D:\\runs\\calculator', language: 'java', llm: true });
    expect(mock).toHaveBeenCalledWith('/api/experiments/start', expect.objectContaining({ method: 'POST' }));
  });
});
