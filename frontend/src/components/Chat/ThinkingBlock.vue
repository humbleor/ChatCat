<template>
  <details v-if="msg.thinkingText || msg._hidingThink" class="reasoning-details thinking-block">
    <summary>
      <span><i class="fa-solid fa-brain" /> 思考</span>
      <small v-if="msg._hidingThink">生成中…</small>
    </summary>
    <div class="reasoning-content thinking-body">{{ displayText }}</div>
  </details>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import type { Message } from '@/types/chat';

const props = defineProps<{ msg: Message }>();

// 流式时显示已累积的 thinkingText（不含光标），关闭后内容已固定
const displayText = computed(() => props.msg.thinkingText || '');
</script>

<style scoped>
.thinking-block {
  margin-bottom: 14px;
}
.thinking-block > summary {
  font-size: 13px;
  padding: 8px 12px;
}
.thinking-block > summary > span {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--text-soft);
}
.thinking-block > summary > span i {
  color: var(--mint, #5ad6a6);
}
.thinking-block > summary > small {
  color: var(--muted);
  font-size: 11px;
  margin-left: auto;
}
.thinking-body {
  white-space: pre-wrap;
  font-size: 12.5px;
  line-height: 1.55;
  color: var(--text-soft);
  max-height: 320px;
  overflow-y: auto;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}
</style>