<template>
  <div class="hitl-message">
    <div class="hitl-prompt">{{ msg.hitl?.prompt }}</div>

    <div v-if="msg.hitl?.options?.length" class="hitl-options">
      <button
        v-for="opt in msg.hitl.options"
        :key="opt"
        type="button"
        class="hitl-option"
        :disabled="chatStore.isLoading"
        @click="handleReply(opt)"
      >
        {{ opt }}
      </button>
    </div>

    <div v-else class="hitl-reply-input">
      <input
        v-model="replyText"
        type="text"
        placeholder="输入补充信息后回车..."
        :disabled="chatStore.isLoading"
        @keydown.enter="handleReply(replyText)"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { useChatStore } from '@/stores/chat';
import type { Message } from '@/types/chat';

defineProps<{
  msg: Message;
}>();

const chatStore = useChatStore();
const replyText = ref('');

const handleReply = async (text: string) => {
  const value = (text || '').trim();
  if (!value) return;
  await chatStore.handleHitlReply(value);
  replyText.value = '';
};
</script>

<style scoped>
.hitl-message {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 10px 12px;
  border: 1px solid var(--border-color, rgba(127, 127, 127, 0.2));
  border-radius: 10px;
  background: var(--surface-color, rgba(127, 127, 127, 0.08));
}

.hitl-prompt {
  font-size: 14px;
  line-height: 1.5;
  white-space: pre-wrap;
}

.hitl-options {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.hitl-option {
  padding: 6px 12px;
  border: 1px solid rgba(127, 127, 127, 0.3);
  border-radius: 16px;
  background: transparent;
  cursor: pointer;
  font-size: 13px;
  color: inherit;
}

.hitl-option:hover {
  background: rgba(127, 127, 127, 0.12);
}

.hitl-option:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.hitl-reply-input input {
  width: 100%;
  padding: 6px 10px;
  border: 1px solid rgba(127, 127, 127, 0.3);
  border-radius: 8px;
  background: transparent;
  color: inherit;
  font-size: 13px;
}
</style>
