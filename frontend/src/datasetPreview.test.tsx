import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import App from './App';
import type { RuntimeSnapshot } from './types';

const api = vi.hoisted(() => ({ fetchRuntime: vi.fn() }));

vi.mock('./api', () => ({
  selectedExperiment: () => 'all',
  fetchRuntime: api.fetchRuntime,
}));

afterEach(() => { cleanup(); api.fetchRuntime.mockReset(); });

describe('dataset preview', () => {
  it('does not show master model usage while a dataset snapshot is loading', async () => {
    const master = {
      experiment: { id: 'all', label: 'DeepSeek-R1', output_dir: 'D:/runs', status: 'completed' },
      experiments: [{ id: 'preview-model-isolation', label: 'TestEval', output_dir: 'D:/runs/TestEval', status: 'completed', total_tasks: 1 }],
      tasks: [{ id: 'task-1', name: 'task-1', dataset_id: 'preview-model-isolation', dataset_name: 'TestEval', grade: 'A', round: 2, round_label: 'R2', metrics: { tests_passed: true, line_coverage: .5 } }],
      overview: { status: 'completed', total_tasks: 1, followup_task_count: 287, followup_attempts: 311, dataset_count: 7, model_name: 'deepseek-r1:14b', model_usage: {
        recorded_tasks: 711, calls: 3305, prompt_tokens: 100, completion_tokens: 50, total_tokens: 150, duration_seconds: 30,
      } },
      catalog: { agents: [], languages: [], scopes: [], phases: [], quality_metrics: [] },
      phase_states: [], toolchain: [], agents: [], state_model: { nodes: [], edges: [] }, flow_events: [],
    } as unknown as RuntimeSnapshot;
    const dataset = {
      ...master,
      experiment: { id: 'preview-model-isolation', label: 'TestEval', output_dir: 'D:/runs/TestEval', status: 'completed' },
      overview: { ...master.overview, model_usage: { ...master.overview.model_usage, recorded_tasks: 1, calls: 17 } },
    } as unknown as RuntimeSnapshot;
    let resolveDataset!: (snapshot: RuntimeSnapshot) => void;
    const pendingDataset = new Promise<RuntimeSnapshot>((resolve) => { resolveDataset = resolve; });
    api.fetchRuntime.mockImplementation((id: string) => id === 'all' ? Promise.resolve(master) : pendingDataset);

    const { container } = render(<App />);
    await waitFor(() => expect(container.querySelector('.experiment-trigger')).toBeTruthy());
    fireEvent.click(container.querySelector('.experiment-trigger')!);
    fireEvent.click(container.querySelector('.experiment-menu-list button')!);
    expect(container.querySelector('.stat-grid')?.textContent).toContain('进入后续轮次1');
    expect(container.querySelector('.stat-grid')?.textContent).not.toContain('287');
    fireEvent.click(container.querySelector('.system-state')!);

    expect(screen.queryByRole('heading', { name: '本次模型运行' })).toBeNull();
    expect(screen.queryByText('3,305')).toBeNull();

    resolveDataset(dataset);
    await waitFor(() => expect(screen.getByText('17')).toBeTruthy());
  });
});
