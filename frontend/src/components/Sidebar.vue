<template>
  <aside class="sidebar">
    <div class="sidebar-header">
      <div class="brand-mark" aria-hidden="true">
        <svg viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
          <path d="M32 16 C22 16 16 23 16 31 C16 40 22 46 32 46 C42 46 48 40 48 31 C48 23 42 16 32 16 Z" />
          <path d="M19 19 L13 7 L24 16 Z" />
          <path d="M17.5 17 L15 11 L21 16 Z" fill="currentColor" fill-opacity="0.18" stroke="none" />
          <path d="M45 19 L51 7 L40 16 Z" />
          <path d="M46.5 17 L49 11 L43 16 Z" fill="currentColor" fill-opacity="0.18" stroke="none" />
          <path d="M25 30 Q27.5 32.5 30 30" />
          <path d="M34 30 Q36.5 32.5 39 30" />
          <path d="M30 36 L34 36 L32 38.5 Z" fill="currentColor" stroke="none" />
          <path d="M32 38.5 Q30.5 41 28 40.5" />
          <path d="M32 38.5 Q33.5 41 36 40.5" />
          <path d="M20 50 Q20 56 28 56 L40 56 Q48 56 48 50" />
          <path d="M48 52 Q56 50 58 42 Q59 38 56 36" />
        </svg>
      </div>
      <div class="brand-copy">
        <h1>ChatCat</h1>
        <span>Knowledge Copilot</span>
      </div>
    </div>

    <nav class="sidebar-nav" aria-label="主导航">
      <button
        type="button"
        :class="['nav-btn', { active: chatStore.activeNav === 'newChat' }]"
        aria-label="智能对话"
        @click="onNewChat"
      >
        <i class="fa-regular fa-message"></i>
        <span>智能对话</span>
      </button>
      <button
        type="button"
        :class="['nav-btn', { active: chatStore.activeNav === 'history' }]"
        aria-label="历史会话"
        @click="onHistory"
      >
        <i class="fa-solid fa-clock-rotate-left"></i>
        <span>历史会话</span>
        <small v-if="sessionStore.sessions.length" class="nav-count">
          {{ sessionStore.sessions.length }}
        </small>
      </button>
      <button
        v-if="authStore.isAdmin"
        type="button"
        :class="['nav-btn', { active: chatStore.activeNav === 'settings' }]"
        aria-label="知识库"
        @click="onSettings"
      >
        <i class="fa-regular fa-bookmark"></i>
        <span>知识库</span>
      </button>
    </nav>

    <template v-if="authStore.isAuthenticated">
      <div class="sidebar-section-label">最近会话</div>
      <div class="sidebar-recents">
        <button
          v-for="session in recentSessions"
          :key="session.session_id"
          type="button"
          :class="['recent-session', { active: session.session_id === chatStore.sessionId }]"
          @click="onLoadSession(session.session_id)"
        >
          <span class="recent-dot" aria-hidden="true"></span>
          <span class="recent-copy">
            <strong>{{ session.title || '未命名会话' }}</strong>
            <small>
              {{ session.isStreaming ? '生成中' : session.message_count + ' 条消息' }}
              · {{ formatRelativeTime(session.updated_at) }}
            </small>
          </span>
        </button>

        <div v-if="!recentSessions.length" class="recent-empty">还没有历史会话，问喵喵一个问题吧。</div>
      </div>
    </template>

    <div class="sidebar-bottom">
      <div class="theme-control">
        <span class="theme-control-label">
          <i :class="theme === 'light' ? 'fa-regular fa-sun' : 'fa-regular fa-moon'"></i>
          <span>{{ theme === 'light' ? '浅色模式' : '深色模式' }}</span>
        </span>
        <ThemeToggle :theme="theme" @toggle="$emit('toggle-theme')" />
      </div>

      <div v-if="authStore.isAuthenticated" class="user-panel">
        <span class="user-avatar">{{ userInitials }}</span>
        <span class="user-copy">
          <strong>{{ authStore.currentUser?.username }}</strong>
          <small>{{ roleLabel }}</small>
        </span>
        <span class="user-actions">
          <button type="button" title="退出登录" aria-label="退出登录" @click="onLogout">
            <i class="fa-solid fa-arrow-right-from-bracket"></i>
          </button>
        </span>
      </div>
    </div>
  </aside>
</template>

<script setup lang="ts">
import { computed, watch } from 'vue';
import ThemeToggle from '@/components/ThemeToggle.vue';
import { useAuthStore } from '@/stores/auth';
import { useChatStore } from '@/stores/chat';
import { useSessionStore } from '@/stores/sessions';

defineProps<{
  theme: 'dark' | 'light';
}>();

defineEmits<{
  (e: 'toggle-theme'): void;
}>();

const authStore = useAuthStore();
const chatStore = useChatStore();
const sessionStore = useSessionStore();

const recentSessions = computed(() => sessionStore.sessions.slice(0, 4));

const roleLabel = computed(() => (authStore.currentUser?.role === 'admin' ? '管理员' : '普通用户'));

const userInitials = computed(() => {
  const name = authStore.currentUser?.username || 'ME';
  return name.slice(0, 2).toUpperCase();
});

const refreshSessions = async () => {
  if (!authStore.isAuthenticated) return;
  try {
    await sessionStore.fetchSessions();
    chatStore.mergeCachedSessionsIntoHistory();
  } catch (error) {
    console.warn('加载历史会话失败', error);
  }
};

watch(
  () => authStore.isAuthenticated,
  (isAuthenticated) => {
    if (isAuthenticated) refreshSessions();
  },
  { immediate: true }
);

const onNewChat = () => {
  chatStore.handleNewChat();
};

const onHistory = async () => {
  chatStore.activeNav = 'history';
  sessionStore.showHistorySidebar = !sessionStore.showHistorySidebar;
  if (sessionStore.showHistorySidebar) {
    try {
      await sessionStore.fetchSessions();
      chatStore.mergeCachedSessionsIntoHistory();
    } catch (error: any) {
      alert(error.message);
    }
  }
};

const onSettings = () => {
  if (!authStore.isAdmin) {
    alert('仅管理员可访问文档管理');
    return;
  }
  chatStore.activeNav = 'settings';
  sessionStore.showHistorySidebar = false;
};

const onLoadSession = async (sessionId: string) => {
  try {
    await chatStore.loadSession(sessionId);
  } catch (error: any) {
    alert('加载会话失败：' + error.message);
  }
};

const onLogout = () => {
  sessionStore.showHistorySidebar = false;
  authStore.handleLogout();
};

const formatRelativeTime = (value: string) => {
  const timestamp = new Date(value).getTime();
  if (!Number.isFinite(timestamp)) return '刚刚';
  const diffMinutes = Math.max(0, Math.floor((Date.now() - timestamp) / 60000));
  if (diffMinutes < 1) return '刚刚';
  if (diffMinutes < 60) return diffMinutes + ' 分钟前';
  const diffHours = Math.floor(diffMinutes / 60);
  if (diffHours < 24) return diffHours + ' 小时前';
  const diffDays = Math.floor(diffHours / 24);
  if (diffDays < 7) return diffDays + ' 天前';
  return new Date(value).toLocaleDateString();
};
</script>
