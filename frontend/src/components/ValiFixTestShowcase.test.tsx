import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { ValiFixTestShowcase } from './ValiFixTestShowcase';

afterEach(cleanup);

describe('ValiFixTest method showcase', () => {
  it('shows all prompt categories and switches the complete English template', () => {
    const { container } = render(<ValiFixTestShowcase />);
    expect(container.textContent).toContain('12 PROMPTS');
    expect(container.textContent).toContain('Strictly create objects according to constructors and methods that actually exist in the source code.');

    fireEvent.click(screen.getByRole('button', { name: 'Validation 2' }));
    fireEvent.click(screen.getByRole('button', { name: /Dynamic validation/ }));
    expect(container.textContent).toContain('Priority 0: package path checking.');
    expect(container.textContent).toContain('[Test Quality Suggestions]: Optional');
    expect(container.textContent).toContain('整理上下文');
    expect(container.textContent).not.toMatch(/论文|文献来源|历史研究结果|非当前|并非当前|第 \d+ 页/);
  });

  it('opens concrete validation and repair explanations in dismissible dialogs', () => {
    render(<ValiFixTestShowcase />);
    fireEvent.click(screen.getByRole('button', { name: /规则验证 语法/ }));
    expect(within(screen.getByRole('dialog')).getByText(/把整数返回值赋给字符串变量/)).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: '关闭类型说明' }));
    expect(screen.queryByRole('dialog')).toBeNull();

    fireEvent.click(screen.getByRole('button', { name: /方法级修复/ }));
    expect(within(screen.getByRole('dialog')).getByText(/重新组织该方法里的变量、调用顺序和断言/)).toBeTruthy();
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('covers all fifteen project measures with correct units and stage data', () => {
    const { container } = render(<ValiFixTestShowcase />);
    const measureGroups = ['覆盖率指标', '测试质量指标', '生成结果指标'];
    expect(measureGroups.flatMap((name) => within(screen.getByRole('group', { name })).getAllByRole('button'))).toHaveLength(15);
    expect(container.textContent).toContain('79.40%');

    fireEvent.click(screen.getByRole('button', { name: /AD 断言密度/ }));
    expect(container.textContent).toContain('6.16 条/方法');
    fireEvent.click(screen.getByRole('button', { name: /FTR 不稳定测试率/ }));
    expect(container.textContent).toContain('8.30%');
    fireEvent.click(screen.getByRole('button', { name: /Syntax Error Rate 语法错误率/ }));
    expect(container.textContent).toContain('27.01%');

    fireEvent.click(within(screen.getByRole('group', { name: '选择阶段数据项目' })).getByRole('button', { name: 'Commons-Lang' }));
    expect(container.textContent).toContain('-4.53%');
    expect(container.textContent).toContain('85.8%');
    expect(container.textContent).toContain('$2.515');
  });
});
