import { describe, expect, it } from 'vitest';
import { applySseEvent } from './sse';
import type { Message } from '@/types/chat';

describe('applySseEvent', () => {
  it('appends content and clears thinking', () => {
    const msg = { text: '', isUser: false, isThinking: true } as Message;
    const out = applySseEvent(msg, { type: 'content', content: '你好' });
    expect(out.text).toBe('你好');
    expect(out.isThinking).toBe(false);
  });

  it('stores hitl_request', () => {
    const msg = { text: '', isUser: false, isThinking: true } as Message;
    const out = applySseEvent(msg, { type: 'hitl_request', hitl: { route: 'scope_select', prompt: '选一个', options: ['A', 'B'] } });
    expect(out.isThinking).toBe(false);
    expect(out.hitl?.prompt).toBe('选一个');
    expect(out.hitl?.options).toEqual(['A', 'B']);
  });

  it('stores trace', () => {
    const msg = { text: '', isUser: false } as Message;
    const out = applySseEvent(msg, { type: 'trace', rag_trace: { route: 'scope_select' } });
    expect(out.ragTrace?.route).toBe('scope_select');
  });

  it('appends rag_step to ragSteps and groups ungrouped steps', () => {
    const msg = { text: '', isUser: false, ragSteps: [], _groupedSteps: [] } as Message;
    const out = applySseEvent(msg, { type: 'rag_step', step: { label: '检索中', group: null } });
    expect(out.ragSteps?.[0]).toMatchObject({ label: '检索中' });
    expect(out._groupedSteps?.[0]?.steps?.[0]).toMatchObject({ label: '检索中' });
  });

  it('appends error text and clears thinking', () => {
    const msg = { text: '前半句', isUser: false, isThinking: true } as Message;
    const out = applySseEvent(msg, { type: 'error', content: 'boom' });
    expect(out.isThinking).toBe(false);
    expect(out.text).toContain('[Error: boom]');
  });
});
