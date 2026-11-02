import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

const api = vi.hoisted(() => ({
  getEmbeddingLifecycle: vi.fn(),
  controlEmbeddingBuild: vi.fn(),
  currentPlatform: vi.fn(() => 'macos'),
}));
vi.mock('@/lib/api', () => api);
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

import { EmbeddingLifecycle } from './EmbeddingLifecycle';

describe('embedding lifecycle', () => {
  afterEach(() => {
    vi.clearAllMocks();
    document.body.innerHTML = '';
  });
  it('shows active and interrupted builds with corpus progress and recovery actions', async () => {
    api.getEmbeddingLifecycle.mockResolvedValue({
      revision: 7,
      active: {
        status: { id: 1, state: 'active', dimensions: 768 },
        profile: { model: 'old', backend: 'local', dimension_policy: { Explicit: 768 } },
        corpora: [],
      },
      pending: {
        status: { id: 2, state: 'interrupted', dimensions: 4096 },
        profile: { model: 'new', backend: 'server', dimension_policy: 'Native' },
        corpora: [{ corpus: 'Emails', requestedEnabled: true, completed: 3, pending: 2, failed: 1 }],
      },
      retained: null,
    });
    api.controlEmbeddingBuild.mockResolvedValue(undefined);
    const element = document.createElement('div');
    document.body.appendChild(element);
    const root = createRoot(element);
    await act(async () => {
      root.render(<EmbeddingLifecycle policy={{ Explicit: 768 }} onPolicy={vi.fn()} onBuild={vi.fn()} busy={false} />);
    });
    expect(element.textContent).toContain('old');
    expect(element.textContent).toContain('new');
    expect(element.textContent).toContain('3 / 5');
    expect(element.textContent).toContain('embeddingLifecycle.resume');
    const resume = [...element.querySelectorAll('button')].find((b) => b.textContent === 'embeddingLifecycle.resume');
    await act(async () => {
      resume?.click();
    });
    expect(api.controlEmbeddingBuild).toHaveBeenCalledWith('resume', 2, 7);
    await act(async () => root.unmount());
  });
  it('shows actionable rollback refusal without changing the retained index', async () => {
    api.getEmbeddingLifecycle.mockResolvedValue({
      revision: 9,
      active: null,
      pending: null,
      retained: {
        status: { id: 3, state: 'retained', dimensions: 768 },
        profile: { model: 'old', backend: 'server', dimension_policy: { Explicit: 768 } },
        corpora: [],
      },
    });
    api.controlEmbeddingBuild.mockRejectedValueOnce(new Error('Select and authorize the retained destination'));
    const element = document.createElement('div');
    document.body.appendChild(element);
    const root = createRoot(element);
    await act(async () => {
      root.render(<EmbeddingLifecycle policy="Native" onPolicy={vi.fn()} onBuild={vi.fn()} busy={false} />);
    });
    const rollback = [...element.querySelectorAll('button')].find(
      (b) => b.textContent === 'embeddingLifecycle.rollback',
    );
    await act(async () => {
      rollback?.click();
    });
    expect(api.controlEmbeddingBuild).toHaveBeenCalledWith('rollback', null, 9);
    expect(element.textContent).toContain('Select and authorize the retained destination');
    await act(async () => root.unmount());
  });
  it('previews retained rollback with the current settings before any action commits', async () => {
    api.getEmbeddingLifecycle.mockResolvedValue({
      revision: 9,
      active: null,
      pending: null,
      retained: {
        status: { id: 3, state: 'retained', dimensions: 768 },
        profile: { model: 'old', backend: 'server', dimension_policy: { Explicit: 768 } },
        corpora: [],
      },
    });
    const onRollback = vi.fn();
    const element = document.createElement('div');
    document.body.appendChild(element);
    const root = createRoot(element);
    await act(async () => {
      root.render(
        <EmbeddingLifecycle
          policy="Native"
          onPolicy={vi.fn()}
          onBuild={vi.fn()}
          onRollback={onRollback}
          busy={false}
        />,
      );
    });
    await act(async () => {
      [...element.querySelectorAll('button')].find((b) => b.textContent === 'embeddingLifecycle.rollback')?.click();
    });
    expect(onRollback).toHaveBeenCalledOnce();
    expect(api.controlEmbeddingBuild).not.toHaveBeenCalled();
    await act(async () => root.unmount());
  });
  it('does not let an older poll overwrite the refreshed status after stop', async () => {
    vi.useFakeTimers();
    const running = {
      revision: 1,
      active: null,
      retained: null,
      pending: { status: { id: 5, state: 'building', dimensions: 768 }, profile: null, corpora: [] },
    };
    const paused = {
      ...running,
      revision: 2,
      pending: { ...running.pending, status: { ...running.pending.status, state: 'paused' } },
    };
    let finishOldPoll: (value: typeof running) => void = () => {};
    api.getEmbeddingLifecycle
      .mockReset()
      .mockResolvedValueOnce(running)
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            finishOldPoll = resolve;
          }),
      )
      .mockResolvedValue(paused);
    api.controlEmbeddingBuild.mockResolvedValue(undefined);
    const element = document.createElement('div');
    document.body.appendChild(element);
    const root = createRoot(element);
    await act(async () => {
      root.render(<EmbeddingLifecycle policy="Native" onPolicy={vi.fn()} onBuild={vi.fn()} busy={false} />);
    });
    await act(async () => {
      vi.advanceTimersByTime(3000);
    });
    const stop = [...element.querySelectorAll('button')].find((b) => b.textContent === 'embeddingLifecycle.stop');
    await act(async () => {
      stop?.click();
    });
    expect(element.textContent).toContain('embeddingLifecycle.resume');
    await act(async () => {
      finishOldPoll(running);
    });
    expect(element.textContent).toContain('embeddingLifecycle.resume');
    await act(async () => root.unmount());
    vi.useRealTimers();
  });
});
