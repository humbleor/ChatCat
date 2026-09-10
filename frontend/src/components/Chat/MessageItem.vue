<template>
  <div :class="['message', msg.isUser ? 'user-message' : 'bot-message']">
    <div v-if="!msg.isUser" class="message-avatar" aria-hidden="true">
      <svg viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M32 16 C22 16 16 23 16 31 C16 40 22 46 32 46 C42 46 48 40 48 31 C48 23 42 16 32 16 Z" />
        <path d="M19 19 L13 7 L24 16 Z" />
        <path d="M45 19 L51 7 L40 16 Z" />
        <path d="M25 30 Q27.5 32.5 30 30" />
        <path d="M34 30 Q36.5 32.5 39 30" />
        <path d="M30 36 L34 36 L32 38.5 Z" fill="currentColor" stroke="none" />
        <path d="M32 38.5 Q30.5 41 28 40.5" />
        <path d="M32 38.5 Q33.5 41 36 40.5" />
        <path d="M20 50 Q20 56 28 56 L40 56 Q48 56 48 50" />
        <path d="M48 52 Q56 50 58 42 Q59 38 56 36" />
      </svg>
    </div>

    <div class="message-column">
      <div v-if="!msg.isUser" class="message-author">
        <span>ChatCat</span>
        <small v-if="msg.ragTrace?.retrieved_chunks?.length">
          已引用 {{ msg.ragTrace.retrieved_chunks.length }} 个来源
        </small>
      </div>

      <template v-if="msg.isUser">
        <MessageContent :text="msg.text" :is-user="true" :msg-index="msgIndex" />
      </template>

      <template v-else>
        <ThinkingTrace v-if="msg.isThinking && !msg.text" :msg="msg" :msg-index="msgIndex" />

        <HitlMessage v-else-if="msg.hitl" :msg="msg" />

        <template v-else>
          <ThinkingBlock v-if="msg.thinkingText || msg._hidingThink" :msg="msg" />
          <MessageContent :text="msg.text" :is-user="false" :msg-index="msgIndex" @cite-click="onCiteClick" />
          <References ref="referencesRef" :msg="msg" :msg-index="msgIndex" @cite-click="onCiteClick" />
          <RetrievalTraceDetails :msg="msg" />
        </template>
      </template>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import MessageContent from './MessageContent.vue';
import ThinkingTrace from './ThinkingTrace.vue';
import ThinkingBlock from './ThinkingBlock.vue';
import References from './References.vue';
import RetrievalTraceDetails from './RetrievalTraceDetails.vue';
import HitlMessage from './HitlMessage.vue';
import type { Message } from '@/types/chat';

defineProps<{
  msg: Message;
  msgIndex: number;
}>();

const emit = defineEmits<{
  (e: 'cite-click', msgIndex: number, chunkIndex: number): void;
}>();

const referencesRef = ref<InstanceType<typeof References> | null>(null);

const openReferences = () => {
  referencesRef.value?.openDetails();
};

defineExpose({ openReferences });

const onCiteClick = (msgIndex: number, chunkIndex: number) => {
  emit('cite-click', msgIndex, chunkIndex);
};
</script>
