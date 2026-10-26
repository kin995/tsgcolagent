import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import App from './App';
import type { KnowledgeSnapshot, RuntimeSnapshot } from './types';

const api = vi.hoisted(() => ({ fetchRuntime: vi.fn(), fetchKnowledge: vi.fn() }));

vi.mock('./api', () => ({
  selectedExperiment: () => 'knowledge-master-case',
  fetchRuntime: api.fetchRuntime,
  fetchKnowledge: api.fetchKnowledge,
}));

afterEach(() => { cleanup(); api.fetchRuntime.mockReset(); api.fetchKnowledge.mockReset(); });

describe('knowledge scope switch', () => {
  it('does not display the previous experiment library while the new one loads', async () => {
    const master = {
      experiment: { id: 'knowledge-master-case', label: 'DeepSeek-R1', output_dir: 'D:/runs', status: 'completed' },
      experiments: [{ id: 'knowledge-child-case', label: 'TestEval', output_dir: 'D:/runs/TestEval', status: 'completed', total_tasks: 1 }],
      tasks: [], overview: { status: 'completed', total_tasks: 1 },
      catalog: { agents: [], languages: [], scopes: [], phases: [], quality_metrics: [] },
      phase_states: [], toolchain: [], agents: [], state_model: { nodes: [], edges: [] }, flow_events: [],
    } as unknown as RuntimeSnapshot;
    const child = { ...master, experiment: { ...master.experiment, id: 'knowledge-child-case', label: 'TestEval' } } as RuntimeSnapshot;
    const oldKnowledge = {
      path: '', total: 1, counts: { 已掌握: 1 }, records: [{ id: 'old', topic: '旧实验知识', status: '已掌握' }],
    } as KnowledgeSnapshot;
    const newKnowledge = {
      path: '', total: 1, counts: { 待复核: 1 }, records: [{ id: 'new', topic: '新数据集知识', status: '待复核' }],
    } as KnowledgeSnapshot;
    let resolveNew!: (value: KnowledgeSnapshot) => void;
    const pendingNew = new Promise<KnowledgeSnapshot>((resolve) => { resolveNew = resolve; });
    api.fetchRuntime.mockImplementation((id: string) => Promise.resolve(id === 'knowledge-master-case' ? master : child));
    api.fetchKnowledge.mockImplementation((id: string) => id === 'knowledge-master-case' ? Promise.resolve(oldKnowledge) : pendingNew);

    const { container } = render(<App />);
    fireEvent.click(screen.getByRole('button', { name: '测试知识引导库' }));
    await waitFor(() => expect(screen.getByText('旧实验知识')).toBeTruthy());

    fireEvent.click(container.querySelector('.experiment-trigger')!);
    fireEvent.click(container.querySelector('.experiment-menu-list button')!);
    expect(screen.queryByText('旧实验知识')).toBeNull();
    expect(screen.getByText('正在读取所选实验的知识记录')).toBeTruthy();

    resolveNew(newKnowledge);
    await waitFor(() => expect(screen.getByText('新数据集知识')).toBeTruthy());
  });
});
