import { describe, expect, it } from 'vitest';
import { applySseEvent, consumeThink, splitThinking } from './sse';
import type { Message } from '@/types/chat';

describe('applySseEvent', () => {
  it('appends content and clears thinking', () => {
    const msg = { text: '', isUser: false, isThinking: true } as Message;
    const out = applySseEvent(msg, { type: 'content', content: '你好' });
    expect(out.text).toBe('你好');
    expect(out.isThinking).toBe(false);
  });

  it('binds run events to the assistant message', () => {
    const msg = { text: '', isUser: false } as Message;
    const out = applySseEvent(msg, { type: 'run', run_id: 'run_123', status: 'running' });
    expect(out.runId).toBe('run_123');
  });

  it('stores hitl_request', () => {
    const msg = { text: '', isUser: false, isThinking: true } as Message;
    const out = applySseEvent(msg, {
      type: 'hitl_request',
      hitl: { route: 'scope_select', prompt: '选一个', options: ['A', 'B'] },
    });
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

  it('routes <think>...</think> block to thinkingText and hides from text', () => {
    const msg = { text: '', isUser: false, isThinking: true } as Message;
    const out = applySseEvent(msg, { type: 'content', content: '<think>reasoning</think>真实回答' });
    expect(out.text).toBe('真实回答');
    expect(out.thinkingText).toBe('reasoning');
    expect(out._hidingThink).toBe(false);
  });

  it('suppresses <think> content across multiple chunks via state machine', () => {
    let msg = { text: '', isUser: false, isThinking: true } as Message;
    msg = applySseEvent(msg, { type: 'content', content: '<think>reas' });
    expect(msg.text).toBe('');
    expect(msg.thinkingText).toBe('reas');
    expect(msg._hidingThink).toBe(true);
    msg = applySseEvent(msg, { type: 'content', content: 'oning跨 chunk</think>\n真实' });
    expect(msg.text).toBe('\n真实');
    expect(msg.thinkingText).toBe('reasoning跨 chunk');
    expect(msg._hidingThink).toBe(false);
    msg = applySseEvent(msg, { type: 'content', content: '回答' });
    expect(msg.text).toBe('\n真实回答');
  });
});

describe('consumeThink', () => {
  it('returns empty visible when input is fully inside think', () => {
    const { visible, hidden, hiding } = consumeThink('<think>reasoning</think>', { hiding: false });
    expect(visible).toBe('');
    expect(hidden).toBe('reasoning');
    expect(hiding).toBe(false);
  });

  it('resumes visible after </think>', () => {
    const { visible, hidden, hiding } = consumeThink('<think>x</think>Hi', { hiding: false });
    expect(visible).toBe('Hi');
    expect(hidden).toBe('x');
    expect(hiding).toBe(false);
  });

  it('keeps hiding=true when </think> not yet arrived', () => {
    const { visible, hidden, hiding } = consumeThink('more thinking', { hiding: true });
    expect(visible).toBe('');
    expect(hidden).toBe('more thinking');
    expect(hiding).toBe(true);
  });
});

describe('splitThinking', () => {
  it('extracts inline think block from historical message text', () => {
    const { text, thinkingText } = splitThinking('<think>reasoning</think>真实回答');
    expect(text).toBe('真实回答');
    expect(thinkingText).toBe('<think>reasoning</think>');
  });

  it('returns empty thinkingText when no think block present', () => {
    const { text, thinkingText } = splitThinking('纯回答，无 think 块');
    expect(text).toBe('纯回答，无 think 块');
    expect(thinkingText).toBe('');
  });
});
