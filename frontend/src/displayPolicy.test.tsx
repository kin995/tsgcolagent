import { describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { fireEvent, render, screen } from '@testing-library/react';
import { publicDisplayData } from './displayPolicy';
import { DataVisual, StateModelVisual } from './components/DetailDrawer';
import { OverviewPage, SystemPage } from './components/pages';
import type { RuntimeSnapshot } from './types';

describe('public experiment result display', () => {
  it('removes process scores and their aliases from historical nested data', () => {
    const visible = publicDisplayData({
      quality_averages: { average_sfc: .8, average_tsq: .9, average_line_coverage: .7, average_mutation_score: .6 },
      report: { nrr: 1, pv: 1, csg: .8, mutation_score: .6, combined_coverage: .8 },
      knowledge: { confidence: .9, reward: .8, quality_delta: { sfc: .2 }, success_count: 3, defect_score: .2, estimated_mutation_score: .4, mutation_score: .6 },
      candidates: [{ id: 'SFC', value: .8 }, { id: 'LC', value: .7 }],
      weak_flags: ['SFC below target', 'AE below target'],
      stop_reason: 'SFC below target; AE below target',
      overall_quality_index: .9,
    });
    expect(visible.quality_averages).toEqual({ average_line_coverage: .7, average_mutation_score: .6 });
    expect(visible.report).toEqual({ mutation_score: .6 });
    expect(visible.knowledge).toEqual({ mutation_score: .6 });
    expect(visible.candidates).toEqual([{ id: 'LC', value: .7 }]);
    expect(visible.weak_flags).toEqual(['AE below target']);
    expect(visible.stop_reason).not.toMatch(/SFC/i);
    expect(visible).not.toHaveProperty('overall_quality_index');
  });

  it('keeps process scores out of generic report details', () => {
    const html = renderToStaticMarkup(<DataVisual data={{
      line_coverage: .7,
      average_sfc: .8,
      detail: { NRR: 1, TSQ: .9, SFQ: .9, CSG: .9, PV: 1, DVS: .4, TIC: .4, RCS: .4, branch_coverage: .5 },
      notes: ['状态图完整度 CSG=0.9', 'LC 70%'],
      catalog_description: '行、分支和组合覆盖',
    }} />);
    expect(html).toContain('行覆盖率');
    expect(html).toContain('分支覆盖率');
    expect(html).not.toMatch(/\b(SFC|NRR|TSQ|SFQ|CSG|PV|DVS|TIC|RCS)\b/i);
    expect(html).not.toContain('状态图完整度');
    expect(html).not.toContain('组合覆盖');
  });

  it('keeps source function names ending in score while filtering process scores', () => {
    const visible = publicDisplayData({
      target_methods: ['classify_score', 'calculate_total'],
      source_identifiers: { classify_score: 1 },
      metrics: { defect_score: .5, coverage_score: .6, quality_score: .7, validation_score: .8 },
    });
    expect(visible.target_methods).toEqual(['classify_score', 'calculate_total']);
    expect(visible.source_identifiers).toEqual({ classify_score: 1 });
    expect(visible.metrics).toEqual({});
    const html = renderToStaticMarkup(<DataVisual data={{ target_methods: ['classify_score'], defect_score: .5 }} />);
    expect(html).toContain('classify_score');
    expect(html).not.toContain('defect score');
  });

  it('removes the non-paper completion rate while retaining task counts and Pass Rate', () => {
    const visible = publicDisplayData({ completion_rate: .5, completed_execution: 1, total_tasks: 2, pass_rate: .5 });
    expect(visible).toEqual({ completed_execution: 1, total_tasks: 2, pass_rate: .5 });
    const runtime = {
      experiment: { id: 'all' }, experiments: [], tasks: [], current_task: null, portfolio: null,
      overview: {
        status: 'completed', total_tasks: 2, completed_execution: 1, not_completed_execution: 1,
        completion_rate: .5, grade_counts: { A: 1, B: 0, C: 0, D: 0, E: 1 },
        quality_averages: {
          average_line_coverage: .7, average_branch_coverage: .6, average_ae: .8,
          average_mutation_score: .5, pass_rate: .5, avg_exec: 2,
          average_time_seconds: 5, tir: .4,
        },
      },
    } as unknown as RuntimeSnapshot;
    const open = vi.fn();
    render(<OverviewPage runtime={runtime} open={open} />);
    expect(screen.getAllByText('PassRate').length).toBeGreaterThan(0);
    expect(screen.getAllByText('50.0%').length).toBeGreaterThan(0);
    expect(document.body.textContent).not.toContain('完成率');
    expect(document.body.textContent).not.toContain('undefined');
    fireEvent.click(screen.getByRole('button', { name: /任务总数/ }));
    expect(open.mock.calls[0][0].raw).toEqual({ total_tasks: 2, completed_execution: 1, not_completed_execution: 1 });
  });

  it('filters internal gaps and rates while preserving paper metrics and source method names', () => {
    const visible = publicDisplayData({
      gaps: { coverage_gap: .3 }, gap_values: [.3],
      metric_delta: { ae: .1 }, before_metrics: { ae: .7 }, after_metrics: { ae: .8 },
      graph: {
        node_coverage: .7, average_edge_coverage: .8, path_coverage: .6,
        test_intent_coverage: .5, priority: .9, coverage_gap: .3,
        assertion_defect: .4, knowledge_confidence: .7, agent_reward: .6,
        repair_cost: .5, failure_rate: .2,
        line_coverage: .8, branch_coverage: .7, ae: .9,
        mutation_score: .6, pass_rate: .5, average_pass_rate: .5,
      },
      target_methods: ['classify_score', 'estimate_cost', 'calculate_gap'],
    });
    expect(visible).toEqual({
      graph: { line_coverage: .8, branch_coverage: .7, ae: .9, mutation_score: .6, pass_rate: .5, average_pass_rate: .5 },
      target_methods: ['classify_score', 'estimate_cost', 'calculate_gap'],
    });
    const html = renderToStaticMarkup(<DataVisual data={visible} />);
    expect(html).toContain('estimate_cost');
    expect(html).not.toContain('coverage gap');
  });

  it('shows state nodes without a synthetic state coverage percentage', () => {
    const html = renderToStaticMarkup(<StateModelVisual model={{ version: 1, nodes: [{ id: 'node-1', evidenced: true }, { id: 'node-2' }], edges: [] }} />);
    expect(html).toContain('证据节点');
    expect(html).not.toContain('证据覆盖');
    expect(html).not.toContain('50%');
  });

  it('keeps the paper coverage entry while replacing an old catalog description', () => {
    const runtime = { catalog: {
      runtime: 'Python', languages: [], scopes: [], execution_policies: [],
      quality_metrics: [
        { id: 'coverage', name: '覆盖证据', description: '行、分支和组合覆盖' },
        { id: 'sfc', name: 'SFC', description: '状态流覆盖率' },
      ],
    } } as unknown as RuntimeSnapshot;
    const html = renderToStaticMarkup(<SystemPage runtime={runtime} open={() => undefined} />);
    expect(html).toContain('LC行覆盖率');
    expect(html).toContain('BC分支覆盖率');
    expect(html).not.toContain('组合覆盖');
    expect(html).not.toMatch(/\bSFC\b/);
  });
});
