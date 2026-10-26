import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { TasksPage } from './pages';
import { DetailList } from './DetailDrawer';
import type { RuntimeSnapshot, Task } from '../types';

afterEach(cleanup);

describe('task summary in the results interface', () => {
  it('opens a compact public task view instead of the original summary JSON', () => {
    const task = {
      id: 'task-1', name: 'Example.java', source: 'HumanEvalJava/Example.java',
      output_dir: 'D:\\runs\\HumanEvalJava\\Example',
      summary_path: 'D:\\runs\\HumanEvalJava\\Example\\stateflow_summary.json',
      test_path: 'D:\\runs\\HumanEvalJava\\Example\\ExampleTest.java',
      failure_report: 'D:\\runs\\HumanEvalJava\\Example\\task_failure_report.html',
      status: 'quality_high', grade: 'A', round: 1, round_label: 'A·R1',
      duration_seconds: 12.5, reports_count: 2,
      metrics: { tests_passed: true, collected_count: 3, line_coverage: .9, ae: .85, mutation_score: .7 },
      failure: {},
    } as Task;
    const runtime = {
      experiment: { id: 'child', label: 'HumanEvalJava' }, experiments: [],
      tasks: [task], current_task: task, overview: { status: 'completed', total_tasks: 1, grade_counts: { A: 1 } },
    } as unknown as RuntimeSnapshot;
    const open = vi.fn(); const openArtifact = vi.fn();
    render(<TasksPage runtime={runtime} open={open} openArtifact={openArtifact} />);

    expect(screen.queryByRole('button', { name: /失败报告 HTML/ })).toBeNull();

    fireEvent.click(screen.getByRole('button', { name: /任务摘要/ }));

    expect(openArtifact).not.toHaveBeenCalled();
    const detail = open.mock.calls.at(-1)?.[0];
    expect(detail.title).toContain('Example.java');
    expect(detail.raw).toBeUndefined();
    const { container } = render(<>{detail.body}</>);
    expect(container.textContent).toContain('90.0%');
    expect(container.textContent).toContain('ExampleTest.java');
    expect(container.textContent).not.toContain('sfc');
  });
});

describe('failure detail rendering', () => {
  it('does not crash when an imported failure list contains a structured record', () => {
    const { container } = render(<DetailList title="失败测试" items={[
      { test_name: 'java_junit_execution' }, 'test_boundary_case', 'weak_内部过程指标已隐藏',
    ] as unknown as string[]} />);
    expect(container.textContent).toContain('test_boundary_case');
    expect(container.textContent).not.toContain('内部过程指标已隐藏');
  });
});
