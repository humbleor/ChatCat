import type { GroupedRagStep, Message, RagStep, RagTrace } from '@/types/chat';

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
    case 'content':
      return { ...msg, isThinking: false, text: (msg.text || '') + (data.content || '') };
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
