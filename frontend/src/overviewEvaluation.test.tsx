import { cleanup, fireEvent, render } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { OverviewPage } from './components/pages';
import type { RuntimeSnapshot } from './types';

afterEach(cleanup);

const runtimeWith = (qualityAverages: Record<string, number | string | null>) => ({
  experiment: { id: 'all' }, experiments: [], tasks: [], current_task: null, portfolio: null,
  overview: { status: 'completed', total_tasks: 1, quality_averages: qualityAverages },
}) as unknown as RuntimeSnapshot;

describe('overview evaluation chart', () => {
  it('uses the concise overview title for a directly selected experiment', () => {
    const runtime = {
      ...runtimeWith({}),
      experiment: { id: '1dee7992e202c48f', label: 'DeepSeek-R1（2026-09-15）', status: 'completed' },
      overview: { status: 'completed', total_tasks: 711, quality_averages: {} },
    } as RuntimeSnapshot;
    const { container } = render(<OverviewPage runtime={runtime} open={vi.fn()} />);
    expect(container.querySelector('.page-title h1')?.textContent).toBe('实验总览');
    expect(container.querySelector('.page-title p')).toBeNull();
  });

  it('reads the overview title from the saved page data when provided', () => {
    const runtime = {
      ...runtimeWith({}),
      overview: { status: 'completed', total_tasks: 711, page_title: '页面数据中的总览标题', quality_averages: {} },
    } as RuntimeSnapshot;
    const { container } = render(<OverviewPage runtime={runtime} open={vi.fn()} />);
    expect(container.querySelector('.page-title h1')?.textContent).toBe('页面数据中的总览标题');
  });

  it('plots six percentages and keeps execution measures on separate numeric cards', () => {
    const open = vi.fn();
    const { container } = render(<OverviewPage runtime={runtimeWith({
      average_line_coverage: 1, average_branch_coverage: .8, average_ae: .9,
      average_mutation_score: .5, pass_rate: 1, tir: .4,
      avg_exec: 2.5, average_time_seconds: 75.84,
    })} open={open} />);
    const chart = container.querySelector('.evaluation-visual')!;
    expect(chart.querySelectorAll('.evaluation-row')).toHaveLength(6);
    expect(chart.querySelectorAll('.evaluation-execution-card')).toHaveLength(2);
    expect(chart.textContent).toContain('50.0%');
    expect(chart.textContent).toContain('75.84');
    expect(chart.textContent).toContain('2.50');
    fireEvent.click(chart.querySelector('.evaluation-row')!);
    expect(open).toHaveBeenCalledWith(expect.objectContaining({ title: '平均LC', eyebrow: '实验统计' }));
    expect(open.mock.calls[0][0].raw).toBeUndefined();
  });

  it('distinguishes missing evidence from a real zero', () => {
    const { container } = render(<OverviewPage runtime={runtimeWith({
      average_line_coverage: 0, average_branch_coverage: '—', average_ae: .6,
      average_mutation_score: .4, pass_rate: null, tir: .25,
      avg_exec: null, average_time_seconds: 0,
    })} open={vi.fn()} />);
    const rows = Array.from(container.querySelectorAll<HTMLElement>('.evaluation-row'));
    expect(rows[0].textContent).toContain('0.0%');
    expect(rows[0].classList.contains('missing')).toBe(false);
    expect(rows[0].style.getPropertyValue('--evaluation-fill')).toBe('0%');
    expect(rows[1].textContent).toContain('—');
    expect(rows[1].classList.contains('missing')).toBe(true);
    expect(rows[4].querySelector('.evaluation-track u')).toBeNull();
    const execution = Array.from(container.querySelectorAll<HTMLElement>('.evaluation-execution-card'));
    expect(execution[0].textContent).toContain('—');
    expect(execution[1].textContent).toContain('0.00');
  });

  it('explains when TIR and AvgExec were not recorded by the run', () => {
    const open = vi.fn();
    const { container } = render(<OverviewPage runtime={runtimeWith({
      tir: null, tir_improved: 0, tir_comparable: 0, tir_excluded: 0,
      avg_exec: null, exec_complete_tasks: 0, selected_tasks: 711,
    })} open={open} />);
    const chart = container.querySelector('.evaluation-visual')!;
    expect(chart.textContent).toContain('未采集目标对照记录');
    expect(chart.textContent).toContain('未采集工具级执行计数');

    fireEvent.click(chart.querySelectorAll('.evaluation-row')[5]);
    const tirDetail = render(open.mock.calls[0][0].body);
    expect(tirDetail.container.textContent).toContain('无法计算 TIR');
    fireEvent.click(chart.querySelector('.evaluation-execution-card')!);
    const execDetail = render(open.mock.calls[1][0].body);
    expect(execDetail.container.textContent).toContain('任务轮次不能代替');
  });

  it('shows observed AvgExec and labels its partial evidence coverage', () => {
    const open = vi.fn();
    const { container } = render(<OverviewPage runtime={runtimeWith({
      tir: .2453, tir_improved: 662, tir_comparable: 2699, tir_excluded: 113,
      avg_exec: 11658 / 689, avg_exec_status: 'partial_evidence', exec_complete_tasks: 689, selected_tasks: 711,
    })} open={open} />);
    const chart = container.querySelector('.evaluation-visual')!;
    expect(chart.textContent).toContain('16.92');
    expect(chart.textContent).toContain('689 / 711');
    expect(chart.textContent).toContain('22 项未纳入');
    fireEvent.click(chart.querySelector('.evaluation-execution-card')!);
    const detail = render(open.mock.calls[0][0].body);
    expect(detail.container.textContent).toContain('按 689 项完整计数任务计算');
    expect(detail.container.textContent).toContain('22 项缺证未纳入');
  });
});
