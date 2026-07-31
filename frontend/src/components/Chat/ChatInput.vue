<template>
  <div class="input-area-wrapper">
    <div class="input-area">
      <button
        class="attach-btn"
        type="button"
        title="当前版本暂不支持聊天附件"
        aria-label="聊天附件暂不可用"
        disabled
      >
        <i class="fa-solid fa-paperclip"></i>
      </button>

      <textarea
        ref="textareaRef"
        v-model="chatStore.userInput"
        class="chat-input-textarea"
        :placeholder="chatStore.inputPlaceholder"
        :disabled="chatStore.isInputLocked"
        rows="1"
        @keydown="handleKeyDown"
        @compositionstart="handleCompositionStart"
        @compositionend="handleCompositionEnd"
        @input="autoResize"
      ></textarea>

      <button
        v-if="chatStore.isViewingStreamingSession"
        type="button"
        class="send-btn stop-btn"
        title="终止回答"
        aria-label="终止回答"
        @click="chatStore.handleStop"
      >
        <i class="fa-solid fa-stop"></i>
      </button>

      <button
        v-else
        type="button"
        class="send-btn"
        :disabled="chatStore.isLoading"
        :title="chatStore.isLoading ? '当前已有回答正在生成' : '发送'"
        aria-label="发送消息"
        @click="onSend"
      >
        <i class="fa-regular fa-paper-plane"></i>
      </button>
    </div>

    <div class="input-footer">
      <span>AI 生成内容可能有误，重要结论请结合引用复核。</span>
      <span><kbd>Enter</kbd> 发送 · <kbd>Shift</kbd> + <kbd>Enter</kbd> 换行</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { nextTick, ref } from 'vue';
import { useChatStore } from '@/stores/chat';

const chatStore = useChatStore();
const textareaRef = ref<HTMLTextAreaElement | null>(null);
const isComposing = ref(false);

const handleCompositionStart = () => {
  isComposing.value = true;
};

const handleCompositionEnd = () => {
  isComposing.value = false;
};

const handleKeyDown = (event: KeyboardEvent) => {
  if (event.key === 'Enter' && !event.shiftKey && !isComposing.value) {
    event.preventDefault();
    onSend();
  }
};

const autoResize = () => {
  if (!textareaRef.value) return;
  textareaRef.value.style.height = 'auto';
  textareaRef.value.style.height = Math.min(textareaRef.value.scrollHeight, 140) + 'px';
};

const resetTextareaHeight = () => {
  if (textareaRef.value) textareaRef.value.style.height = 'auto';
};

const onSend = async () => {
  const text = chatStore.userInput.trim();
  if (!text || chatStore.isLoading || isComposing.value) return;
  await chatStore.handleSend();
  await nextTick();
  resetTextareaHeight();
};
</script>
