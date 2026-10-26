import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { SystemPage } from './components/pages';
import type { RuntimeSnapshot } from './types';

afterEach(cleanup);

describe('recorded model use', () => {
  it('shows run-level DeepSeek usage as system execution data', () => {
    const runtime = {
      overview: { status: 'completed', model_name: 'deepseek-r1:14b', model_usage: {
        recorded_tasks: 711, calls: 3305, prompt_tokens: 22696799,
        completion_tokens: 4035036, total_tokens: 26731835, duration_seconds: 56781.0529,
      } },
      catalog: { languages: [], scopes: [], quality_metrics: [], execution_policies: [] },
    } as unknown as RuntimeSnapshot;
    const open = vi.fn();
    render(<SystemPage runtime={runtime} open={open} />);

    expect(screen.getByRole('heading', { name: '本次模型运行' })).toBeTruthy();
    expect(screen.getByText('deepseek-r1:14b')).toBeTruthy();
    expect(screen.getByText('3,305')).toBeTruthy();
    expect(screen.getByText('26,731,835')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: /模型调用.*3,305/ }));
    expect(open).toHaveBeenCalledWith(expect.objectContaining({ title: '模型调用' }));
  });
});
