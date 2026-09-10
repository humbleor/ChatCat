<template>
  <section class="upload-section">
    <div class="upload-section-head">
      <span class="upload-title-icon"><i class="fa-solid fa-cloud-arrow-up"></i></span>
      <div>
        <h2>快速入库</h2>
        <p>解析结构、三级分块并写入混合索引。</p>
      </div>
    </div>

    <input
      ref="fileInputRef"
      type="file"
      accept=".pdf,.doc,.docx,.xls,.xlsx,.html,.htm"
      multiple
      hidden
      @change="onFileSelect"
    />

    <button
      type="button"
      class="upload-dropzone"
      :class="{ 'has-files': documentStore.selectedFiles.length }"
      @click="triggerFileSelect"
      @dragover.prevent
      @drop.prevent="onFileDrop"
    >
      <span class="dropzone-icon"><i class="fa-solid fa-arrow-up-from-bracket"></i></span>
      <strong v-if="!documentStore.selectedFiles.length">拖放文件到这里</strong>
      <strong v-else>已选择 {{ documentStore.selectedFiles.length }} 个文件，继续添加</strong>
      <span v-if="!documentStore.selectedFiles.length">或点击选择 PDF、Word、Excel、HTML 文件（支持多选）</span>
    </button>

    <div v-if="documentStore.selectedFiles.length" class="selected-file-list">
      <div class="selected-file-list-head">
        <span>已选 {{ documentStore.selectedFiles.length }} 个文件</span>
        <button
          type="button"
          class="link-button"
          :disabled="documentStore.isUploading"
          @click="onClearList"
        >
          <i class="fa-regular fa-circle-xmark"></i> 清空列表
        </button>
      </div>
      <ul>
        <li v-for="file in documentStore.selectedFiles" :key="fileKey(file)">
          <span class="selected-file-icon"><i class="fa-regular fa-file-lines"></i></span>
          <span class="selected-file-copy">
            <strong>{{ file.name }}</strong>
            <small>
              {{ formatFileSize(file.size) }}
              <template v-if="documentStore.getFileJob(file.name)?.message">
                · {{ documentStore.getFileJob(file.name)?.message }}
              </template>
              <template v-else>· 等待上传</template>
            </small>
          </span>
          <span class="selected-file-status" :class="'status-' + (documentStore.getFileJob(file.name)?.status || 'pending')">
            <i :class="statusIcon(documentStore.getFileJob(file.name)?.status || 'pending')"></i>
            {{ statusLabel(documentStore.getFileJob(file.name)?.status || 'pending') }}
          </span>
          <button
            type="button"
            class="icon-button"
            :disabled="documentStore.isUploading && documentStore.getFileJob(file.name)?.status === 'running'"
            @click="onRemoveFile(file.name)"
            title="移除"
          >
            <i class="fa-solid fa-xmark"></i>
          </button>
        </li>
      </ul>

      <div class="selected-file-actions">
        <button
          type="button"
          class="btn-primary"
          :disabled="documentStore.isUploading"
          @click="onUpload"
        >
          <i :class="documentStore.isUploading ? 'fa-solid fa-spinner fa-spin' : 'fa-solid fa-arrow-up'"></i>
          {{ documentStore.isUploading ? '处理中' : '上传全部' }}
        </button>
      </div>
    </div>

    <div
      v-if="hasAnyJob"
      :class="['upload-progress', { collapsed: documentStore.uploadProgressCollapsed }]"
    >
      <button type="button" class="upload-progress-header" @click="onToggleCollapse">
        <span>
          <strong>上传进度</strong>
          <small>{{ completedFilesCount }} / {{ totalJobCount }} 个文件完成</small>
        </span>
        <span class="upload-toggle">
          {{ documentStore.uploadProgressCollapsed ? '展开' : '收起' }}
          <i :class="documentStore.uploadProgressCollapsed ? 'fa-solid fa-chevron-down' : 'fa-solid fa-chevron-up'"></i>
        </span>
      </button>

      <div v-show="!documentStore.uploadProgressCollapsed" class="upload-progress-list">
        <div
          v-for="file in jobFiles"
          :key="file.name"
          :class="['upload-file', 'status-' + (file.job?.status || 'pending')]"
        >
          <button type="button" class="upload-file-header" @click="documentStore.toggleFileJobCollapsed(file.name)">
            <span class="upload-file-name">
              <i :class="statusIcon(file.job?.status || 'pending')"></i>
              {{ file.name }}
            </span>
            <span class="upload-file-meta">
              {{ file.job?.message || statusLabel(file.job?.status || 'pending') }}
              <i :class="file.job?.collapsed ? 'fa-solid fa-chevron-down' : 'fa-solid fa-chevron-up'"></i>
            </span>
          </button>
          <div v-show="file.job && !file.job.collapsed" class="upload-step-list">
            <div
              v-for="step in file.job.steps"
              :key="step.key"
              :class="['upload-step', 'upload-step-' + step.status]"
            >
              <div class="upload-step-header">
                <span class="upload-step-label">
                  <i :class="stepIcon(step.status)"></i>
                  {{ step.label }}
                </span>
                <span class="upload-step-percent">{{ step.percent }}%</span>
              </div>
              <div class="upload-step-bar">
                <div class="upload-step-fill" :style="{ width: step.percent + '%' }"></div>
              </div>
              <div v-if="step.message" class="upload-step-message">{{ step.message }}</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="upload-pipeline-note">
      <div>
        <span>01</span>
        <p><strong>结构解析</strong><small>识别章节、表格与页面</small></p>
      </div>
      <div>
        <span>02</span>
        <p><strong>三级分块</strong><small>保留父子上下文关系</small></p>
      </div>
      <div>
        <span>03</span>
        <p><strong>混合索引</strong><small>Dense + BM25 同步写入</small></p>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import { useDocumentStore } from '@/stores/documents';
import type { UploadStep } from '@/types/document';

const documentStore = useDocumentStore();
const fileInputRef = ref<HTMLInputElement | null>(null);

const jobFiles = computed(() =>
  documentStore.selectedFiles.map((file) => ({
    name: file.name,
    job: documentStore.fileJobs[file.name],
  }))
);

const hasAnyJob = computed(() => jobFiles.value.some((item) => !!item.job));
const totalJobCount = computed(() => jobFiles.value.length);
const completedFilesCount = computed(
  () => jobFiles.value.filter((item) => item.job?.status === 'completed').length
);

const triggerFileSelect = () => {
  fileInputRef.value?.click();
};

const fileKey = (file: File) => `${file.name}_${file.size}`;

const onFileSelect = (event: Event) => {
  const files = (event.target as HTMLInputElement).files;
  if (!files?.length) return;
  documentStore.addSelectedFiles(Array.from(files));
  if (fileInputRef.value) fileInputRef.value.value = '';
};

const onFileDrop = (event: DragEvent) => {
  const files = event.dataTransfer?.files;
  if (!files?.length) return;
  documentStore.addSelectedFiles(Array.from(files));
};

const onRemoveFile = (filename: string) => {
  documentStore.removeSelectedFile(filename);
};

const onClearList = () => {
  documentStore.clearSelectedFiles();
};

const onUpload = async () => {
  try {
    await documentStore.uploadDocuments();
  } catch (error: any) {
    alert('上传文档失败：' + error.message);
  }
};

const onToggleCollapse = () => {
  documentStore.uploadProgressCollapsed = !documentStore.uploadProgressCollapsed;
};

const formatFileSize = (bytes: number) => {
  if (bytes < 1024 * 1024) return Math.max(1, Math.round(bytes / 1024)) + ' KB';
  return (bytes / 1024 / 1024).toFixed(1) + ' MB';
};

const statusLabel = (status: string) => {
  if (status === 'completed') return '完成';
  if (status === 'running') return '上传中';
  if (status === 'failed') return '失败';
  return '等待';
};

const statusIcon = (status: string) => {
  if (status === 'completed') return 'fa-solid fa-check';
  if (status === 'running') return 'fa-solid fa-spinner fa-spin';
  if (status === 'failed') return 'fa-solid fa-xmark';
  return 'fa-regular fa-clock';
};

const stepIcon = (status: UploadStep['status']) => {
  if (status === 'completed') return 'fa-solid fa-check';
  if (status === 'running') return 'fa-solid fa-spinner fa-spin';
  if (status === 'failed') return 'fa-solid fa-xmark';
  return 'fa-solid fa-circle';
};
</script>