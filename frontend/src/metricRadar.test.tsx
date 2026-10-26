import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render } from '@testing-library/react';
import { OverviewPage } from './components/pages';
import type { RuntimeSnapshot } from './types';

afterEach(cleanup);

const runtimeWith = (qualityAverages: Record<string, number | string | null>) => ({
  experiment: { id: 'all' }, experiments: [], tasks: [], current_task: null, portfolio: null,
  overview: { status: 'completed', total_tasks: 1, quality_averages: qualityAverages },
}) as unknown as RuntimeSnapshot;

describe('experiment quality radar', () => {
  it('plots the six comparable percentage metrics on six axes', () => {
    const open = vi.fn();
    const { container } = render(<OverviewPage runtime={runtimeWith({
      average_line_coverage: 1, average_branch_coverage: .8, average_ae: .6,
      average_mutation_score: .4, pass_rate: 1, tir: .25,
    })} open={open} />);
    const radar = container.querySelector('.metric-radar')!;
    expect(radar.textContent).toContain('六维质量指标');
    expect(Array.from(radar.querySelectorAll('.radar-legend button'), (item) => item.querySelector('span')?.textContent)).toEqual(['LC', 'BC', 'AE', 'MS', 'PassRate', 'TIR']);
    expect(radar.querySelectorAll('.radar-axis')).toHaveLength(6);
    expect(radar.querySelectorAll('.radar-point')).toHaveLength(6);
    expect(radar.querySelector('.radar-value')?.getAttribute('points')?.trim().split(/\s+/)).toHaveLength(6);
    fireEvent.click(radar.querySelectorAll('.radar-legend button')[4]);
    expect(open).toHaveBeenCalledWith(expect.objectContaining({ title: 'PassRate', eyebrow: '实验整体比例' }));
  });

  it('does not turn missing evidence into a zero or close an incomplete polygon', () => {
    const { container } = render(<OverviewPage runtime={runtimeWith({
      average_line_coverage: .7, average_branch_coverage: .6, average_ae: .8,
      average_mutation_score: 0, pass_rate: '—', tir: null,
    })} open={vi.fn()} />);
    const radar = container.querySelector('.metric-radar')!;
    expect(radar.querySelector('svg[role="img"]')?.getAttribute('aria-label')).toContain('4项有数据，2项缺证');
    expect(radar.querySelector('.radar-value')).toBeNull();
    expect(radar.querySelectorAll('.radar-point')).toHaveLength(4);
    expect(radar.querySelectorAll('.radar-partial')).toHaveLength(3);
    expect(Array.from(radar.querySelectorAll('.radar-legend button'), (item) => item.querySelector('b')?.textContent)).toEqual(['70.0%', '60.0%', '80.0%', '0.0%', '—', '—']);
  });
});
