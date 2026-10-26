import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { renderToStaticMarkup } from 'react-dom/server';
import { afterEach, describe, expect, it, vi } from 'vitest';
import App from './App';
import { ExperimentLauncher } from './components/ExperimentLauncher';
import type { RuntimeSnapshot } from './types';

vi.mock('./api', () => ({
  selectedExperiment: () => 'all',
  fetchRuntime: () => new Promise(() => undefined),
  fetchConversations: () => Promise.resolve({ chats: [] }),
}));

afterEach(cleanup);

describe('launcher and dismissible panels', () => {
  it('labels the active generation method TSGColAgent', () => {
    const html = renderToStaticMarkup(<ExperimentLauncher runtime={{ experiments: [] } as unknown as RuntimeSnapshot} open={() => undefined} onStarted={() => undefined} />);
    expect(html).toContain('TSGColAgent 生成');
    expect(html).not.toContain('ValiFixTest &amp; TSGColAgent方法生成');
  });

  it('closes the navigation sidebar when the page backdrop is clicked', () => {
    render(<App />);
    fireEvent.click(screen.getByRole('button', { name: '打开导航' }));
    expect(document.querySelector('.sidebar')?.classList.contains('open')).toBe(true);
    fireEvent.click(screen.getByRole('button', { name: '点击页面关闭导航' }));
    expect(document.querySelector('.sidebar')?.classList.contains('open')).toBe(false);
  });

  it('keeps chat open for clicks inside and closes it for clicks outside or Escape', () => {
    render(<App />);
    fireEvent.click(screen.getByRole('button', { name: '打开测试知识智能体' }));
    const dialog = screen.getByRole('dialog', { name: '测试知识智能体对话' });
    fireEvent.click(dialog);
    expect(screen.getByRole('dialog', { name: '测试知识智能体对话' })).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: '点击页面关闭测试知识智能体对话' }));
    expect(screen.queryByRole('dialog', { name: '测试知识智能体对话' })).toBeNull();

    fireEvent.click(screen.getByRole('button', { name: '打开测试知识智能体' }));
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(screen.queryByRole('dialog', { name: '测试知识智能体对话' })).toBeNull();
  });
});
