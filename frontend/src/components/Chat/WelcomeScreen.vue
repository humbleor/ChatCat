<template>
  <section class="welcome-screen">
    <div class="welcome-avatar" aria-hidden="true">
      <svg viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
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
        <path d="M22 38 L12 36" />
        <path d="M22 40 L11 42" />
        <path d="M42 38 L52 36" />
        <path d="M42 40 L53 42" />
        <path d="M20 50 Q20 56 28 56 L40 56 Q48 56 48 50" />
        <path d="M26 56 L26 60" />
        <path d="M38 56 L38 60" />
        <path d="M48 52 Q56 50 58 42 Q59 38 56 36" />
      </svg>
    </div>
    <span class="welcome-eyebrow"><i class="fa-solid fa-paw"></i> ChatCat is ready</span>
    <h2>你好，我是<em>喵喵</em>。</h2>
    <p>我会在回答时检索你的知识库、展示处理过程，并把每个关键结论链接回原始证据。</p>

    <div class="welcome-capabilities">
      <span><i class="fa-solid fa-magnifying-glass"></i> 混合检索</span>
      <span><i class="fa-solid fa-diagram-project"></i> 并行 Agent</span>
      <span><i class="fa-solid fa-link"></i> 可追溯引用</span>
    </div>
  </section>
</template>
