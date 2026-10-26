import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { OverviewPage } from './pages';
import type { RuntimeSnapshot } from '../types';

afterEach(cleanup);

describe('completed experiment report links', () => {
  it('exposes the public master and dataset reports from the selected run', () => {
    const dataset = {
      id: 'child-1', parent_id: 'master-1', label: 'TestEval', output_dir: 'D:\\runs\\TestEval',
      batch_report_path: 'D:\\runs\\TestEval\\batch_quality_report.html',
      status: 'completed', total_tasks: 2, completed_execution: 1,
    };
    const runtime = {
      experiment: { id: 'master-1', label: 'DeepSeek-R1', report_path: 'D:\\runs\\all_datasets_report.html' },
      experiments: [dataset], tasks: [], current_task: null,
      portfolio: { view: 'all', experiments: [dataset], counts: { completed: 1, running: 0, queued: 0, failed: 0 } },
      overview: { status: 'completed', total_tasks: 2, grade_counts: {} },
    } as unknown as RuntimeSnapshot;
    const open = vi.fn();
    render(<OverviewPage runtime={runtime} open={open} />);

    const masterLink = screen.getByRole('link', { name: /完整实验报告/ });
    expect(masterLink.getAttribute('href')).toContain('experiment=master-1');
    fireEvent.click(screen.getByRole('button', { name: /已完成 1/ }));
    expect(open.mock.calls.at(-1)?.[0].raw).toBeUndefined();
    fireEvent.click(screen.getByRole('button', { name: /TestEval.*1项有效/ }));
    expect(open.mock.calls.at(-1)?.[0].raw).toBeUndefined();
    const { container } = render(<>{open.mock.calls.at(-1)?.[0].body}</>);
    const childLink = container.querySelector('a[href*="child-1"]');
    expect(childLink?.textContent).toContain('数据集质量报告');
  });
});
