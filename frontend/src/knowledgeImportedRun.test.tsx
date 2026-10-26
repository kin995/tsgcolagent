import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { KnowledgePage } from './components/pages';
import type { KnowledgeSnapshot } from './types';

afterEach(cleanup);

describe('imported run knowledge library', () => {
  it('shows recovered run records and source limits without presenting summaries as mastered knowledge', () => {
    const knowledge = {
      path: '', scope: 'imported_run', total: 3,
      counts: { 已掌握: 1, 正在学习: 1, 待复核: 0, 应避免: 0, 经验摘要: 1 },
      availability: {
        complete: false,
        note: '原始外置知识库未随运行包提供；以下为可恢复记录与数据集经验摘要。',
        recovered_snapshot_records: 3, recovered_journal_records: 2,
        run_record_count: 1, baseline_record_count: 1, dataset_summary_count: 1,
        missing_datasets: ['TestEval'],
      },
      records: [
        { id: 'run', status: '正在学习', topic: '边界输入', source_type: 'run_record', dataset: 'HumanEvalJava' },
        { id: 'baseline', status: '已掌握', topic: '历史断言', source_type: 'historical_baseline' },
        { id: 'summary', status: '经验摘要', topic: '项目经验', source_type: 'dataset_summary', dataset: 'TestEval', support_count: 63 },
      ],
    } as unknown as KnowledgeSnapshot;
    const open = vi.fn();
    const { container } = render(<KnowledgePage knowledge={knowledge} open={open} />);

    expect(screen.getByText('本次实验可恢复的知识')).toBeTruthy();
    expect(screen.getByText(/原始外置知识库未随运行包提供/)).toBeTruthy();
    expect(container.querySelector('.knowledge-provenance')?.textContent).toContain('本次实验记录 1');
    expect(container.querySelector('.knowledge-provenance')?.textContent).toContain('数据集经验摘要 1');
    expect(screen.getByText(/未附原始库：TestEval/)).toBeTruthy();
    expect(screen.getAllByText('经验摘要').length).toBeGreaterThan(0);
    expect(screen.getByText('63项汇总证据')).toBeTruthy();

    fireEvent.click(screen.getByText('项目经验'));
    expect(open).toHaveBeenCalledWith(expect.objectContaining({ title: '项目经验' }));
    expect(open.mock.calls[0][0].raw).toBeUndefined();
    const detail = render(open.mock.calls[0][0].body);
    expect(detail.container.textContent).toContain('汇总证据 63 项');
    expect(detail.container.textContent).not.toContain('涉及 0 个任务');
  });
});
