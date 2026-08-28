import type { GroupedRagStep, Message, RagStep, RagTrace } from '@/types/chat';

// 推理模型（DeepSeek-R1 / Qwen3-Thinking）的 `<think>...</think>` 块。
// 流式状态机：进入 think 后立即把内容路由到 thinkingText（不写入 text），
// 等 `</think>` 才恢复写 text——避免思考内容闪现给用户。
// 假设：标签作为原子 token 整体到达 chunk，不跨 chunk 边界（DeepSeek-R1 API 实测如此）。
const _THINK_OPEN = '<think>';
const _THINK_CLOSE = '</think>';
const _THINK_RE = /<think>[\s\S]*?<\/think>/g;

interface _ThinkState { hiding: boolean }

/**
 * 消费一段入站 content，把 think 块内容路由到 hidden，text 内容路由到 visible。
 * 单次扫描：找到 <think> → 切到隐藏态；找到 </think> → 切回可见态。
 */
export function consumeThink(
  input: string,
  state: _ThinkState,
): { visible: string; hidden: string; hiding: boolean } {
  let remaining = input;
  let visible = '';
  let hidden = '';
  let hiding = state.hiding;
  while (remaining.length > 0) {
    const marker = hiding ? _THINK_CLOSE : _THINK_OPEN;
    const idx = remaining.indexOf(marker);
    if (idx === -1) {
      // 标签原子：剩余整段要么全可见要么全隐藏。
      if (hiding) hidden += remaining;
      else visible += remaining;
      remaining = '';
      continue;
    }
    if (hiding) {
      hidden += remaining.slice(0, idx);
      remaining = remaining.slice(idx + marker.length);
      hiding = false;
    } else {
      visible += remaining.slice(0, idx);
      remaining = remaining.slice(idx + marker.length);
      hiding = true;
    }
  }
  return { visible, hidden, hiding };
}

/** 单次切分：把历史消息里 inline 的 `<think>...</think>` 抽到 thinkingText。 */
export function splitThinking(text: string): { text: string; thinkingText: string } {
  const thinkingText = (text.match(_THINK_RE) || []).join('').trim();
  const cleaned = text.replace(_THINK_RE, '').trim();
  return { text: cleaned, thinkingText };
}

export function appendRagStepToGroups(prev: GroupedRagStep[], step: RagStep): GroupedRagStep[] {
  const groups = prev ? [...prev] : [];
  const g = step.group || null;
  const groupLabel = step.group_label || g;

  if (g) {
    const idx = groups.findIndex((grp) => grp.group === g);
    if (idx >= 0) {
      const existing = groups[idx];
      const updated: GroupedRagStep = {
        group: existing.group,
        label: existing.label || groupLabel,
        steps: [...existing.steps, step],
        collapsed: existing.collapsed,
      };
      groups[idx] = updated;
      return groups;
    }
    return [...groups, { group: g, label: groupLabel, steps: [step], collapsed: true }];
  }

  const last = groups.length > 0 ? groups[groups.length - 1] : null;
  if (last && last.group === null) {
    const updated = { ...last, steps: [...last.steps, step] };
    groups[groups.length - 1] = updated;
    return groups;
  }
  return [...groups, { group: null, label: null, steps: [step], collapsed: false }];
}

/**
 * 把单个 SSE 事件应用到一条消息上（纯函数）。
 * 覆盖 content / rag_step / trace / error / hitl_request；session_title 由 store 单独处理。
 */
export function applySseEvent(msg: Message, data: any): Message {
  switch (data?.type) {
    case 'content': {
      const incoming = data.content || '';
      const { visible, hidden, hiding } = consumeThink(incoming, { hiding: msg._hidingThink || false });
      return {
        ...msg,
        isThinking: false,
        text: (msg.text || '') + visible,
        thinkingText: (msg.thinkingText || '') + hidden,
        _hidingThink: hiding,
      };
    }
    case 'rag_step':
      return {
        ...msg,
        ragSteps: [...(msg.ragSteps || []), data.step],
        _groupedSteps: appendRagStepToGroups(msg._groupedSteps || [], data.step),
      };
    case 'trace':
      return { ...msg, ragTrace: (data.rag_trace as RagTrace) || null };
    case 'error':
      return { ...msg, isThinking: false, text: (msg.text || '') + `\n[Error: ${data.content}]` };
    case 'hitl_request':
      return {
        ...msg,
        isThinking: false,
        hitl: data.hitl,
        text: msg.text || '',
      };
    default:
      return msg;
  }
}
