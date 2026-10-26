import { describe, expect, it } from 'vitest';
import { durationText, elapsedSeconds, metricEntries, paperAverages } from './format';
import type { Task } from './types';

describe('experiment time formatting', () => {
  it('shows current metrics and seconds without exposing legacy quality scores', () => {
    const entries = metricEntries({ branch_coverage: .5, sfc: .8, tsq: .9, duration_seconds: 12.345 });
    expect(entries.map((entry) => entry.label)).toEqual(['测试通过', '测试数量', 'LC', 'BC', 'AE', 'MS', 'Time/s', 'PassRate', 'AvgExec', 'TIR']);
    expect(entries.find((entry) => entry.id === 'branch')?.value).toBe('50.0%');
    expect(entries.find((entry) => entry.id === 'time')?.value).toBe('12.35');
    expect(metricEntries({}).find((entry) => entry.id === 'time')?.value).toBe('—');
    expect(metricEntries({ duration_seconds: 0 }).find((entry) => entry.id === 'time')?.value).toBe('0.00');
  });

  it('formats a duration without losing days', () => {
    expect(durationText(90061)).toBe('1天 01:01:01');
  });

  it('keeps completed elapsed time fixed and running time increasing', () => {
    const started = '2026-01-01T00:00:00.000Z';
    expect(elapsedSeconds(started, '2026-01-01T00:00:08.000Z', undefined, Date.parse('2026-01-01T00:01:00.000Z'))).toBe(8);
    expect(elapsedSeconds(started, undefined, undefined, Date.parse('2026-01-01T00:00:09.000Z'))).toBe(9);
  });

  it('computes observed AvgExec from complete counts without using partial counts', () => {
    const tasks = [
      { metrics: { exec_count: 0, exec_count_status: 'complete' } },
      { metrics: { exec_count: 4, exec_count_status: 'complete' } },
      { metrics: { exec_count: 9, exec_count_status: 'incomplete_evidence' } },
    ] as Task[];
    const values = paperAverages(tasks, 3);
    expect(values.avg_exec).toBe(2);
    expect(values.exec_complete_tasks).toBe(2);
    expect(paperAverages(tasks.slice(2), 1).avg_exec).toBe('—');
  });
});
