import { defineStore } from 'pinia';
import api from '@/utils/api';
import type { DocumentItem, UploadStep, ActiveDeleteJob, DeleteStep } from '@/types/document';

export interface FileUploadJob {
  jobId: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  message: string;
  steps: UploadStep[];
  collapsed: boolean;
  error?: string;
}

export const useDocumentStore = defineStore('documents', {
  state: () => ({
    documents: [] as DocumentItem[],
    documentsLoading: false,
    selectedFiles: [] as File[],
    isUploading: false,
    fileJobs: {} as Record<string, FileUploadJob>,
    uploadProgressCollapsed: false,
    uploadPollTimers: {} as Record<string, any>,
    deleteJobs: {} as Record<string, ActiveDeleteJob>,
    deletePollTimers: {} as Record<string, any>,
    deleteRemoveTimers: {} as Record<string, any>,
  }),

  actions: {
    createUploadSteps(): UploadStep[] {
      return [
        { key: 'upload', label: '文档上传', percent: 0, status: 'pending', message: '' },
        { key: 'cleanup', label: '清理旧版本', percent: 0, status: 'pending', message: '' },
        { key: 'parse', label: '解析与分块', percent: 0, status: 'pending', message: '' },
        { key: 'parent_store', label: '父级分块入库', percent: 0, status: 'pending', message: '' },
        { key: 'vector_store', label: '向量化入库', percent: 0, status: 'pending', message: '' },
      ];
    },

    createDeleteSteps(): DeleteStep[] {
      return [
        { key: 'prepare', label: '准备删除', percent: 0, status: 'pending', message: '' },
        { key: 'milvus', label: '删除向量数据', percent: 0, status: 'pending', message: '' },
        { key: 'parent_store', label: '删除父级分块', percent: 0, status: 'pending', message: '' },
      ];
    },

    getFileJob(filename: string): FileUploadJob | undefined {
      return this.fileJobs[filename];
    },

    initFileJob(filename: string): FileUploadJob {
      const job: FileUploadJob = {
        jobId: '',
        status: 'pending',
        message: '',
        steps: this.createUploadSteps(),
        collapsed: false,
      };
      this.fileJobs = { ...this.fileJobs, [filename]: job };
      return job;
    },

    updateFileJobStep(filename: string, key: string, percent: number, status: UploadStep['status'], message: string) {
      const job = this.fileJobs[filename];
      if (!job) return;
      const idx = job.steps.findIndex((step) => step.key === key);
      if (idx === -1) return;
      const next = [...job.steps];
      next[idx] = {
        ...next[idx],
        percent: Math.max(0, Math.min(100, Math.round(percent || 0))),
        status,
        message,
      };
      this.fileJobs = { ...this.fileJobs, [filename]: { ...job, steps: next } };
    },

    syncFileJob(filename: string, serverJob: any) {
      const current = this.fileJobs[filename] || this.initFileJob(filename);
      this.fileJobs = {
        ...this.fileJobs,
        [filename]: {
          ...current,
          jobId: serverJob.job_id || current.jobId,
          status: serverJob.status || current.status,
          message: serverJob.message || current.message,
          collapsed: serverJob.status === 'completed' ? true : current.collapsed,
          steps: Array.isArray(serverJob.steps)
            ? serverJob.steps.map((step: any) => ({
                key: step.key,
                label: step.label,
                percent: step.percent,
                status: step.status,
                message: step.message || '',
              }))
            : current.steps,
        },
      };
    },

    finalizeCompletedFile(filename: string) {
      this.stopUploadJobPolling(filename);
      this.selectedFiles = this.selectedFiles.filter((file) => file.name !== filename);
      const { [filename]: _removed, ...rest } = this.fileJobs;
      this.fileJobs = rest;
    },

    failFileJob(filename: string, error: string) {
      const current = this.fileJobs[filename] || this.initFileJob(filename);
      this.fileJobs = {
        ...this.fileJobs,
        [filename]: {
          ...current,
          status: 'failed',
          message: error,
          collapsed: false,
        },
      };
    },

    toggleFileJobCollapsed(filename: string) {
      const job = this.fileJobs[filename];
      if (!job) return;
      this.fileJobs = {
        ...this.fileJobs,
        [filename]: { ...job, collapsed: !job.collapsed },
      };
    },

    addSelectedFiles(files: File[]) {
      const additions = files.filter(
        (file) => !this.selectedFiles.some((existing) => existing.name === file.name && existing.size === file.size)
      );
      if (!additions.length) return;
      this.selectedFiles = [...this.selectedFiles, ...additions];
      additions.forEach((file) => this.initFileJob(file.name));
    },

    removeSelectedFile(filename: string) {
      this.selectedFiles = this.selectedFiles.filter((file) => file.name !== filename);
      this.stopUploadJobPolling(filename);
      const { [filename]: _, ...rest } = this.fileJobs;
      this.fileJobs = rest;
    },

    clearSelectedFiles() {
      this.selectedFiles.forEach((file) => this.stopUploadJobPolling(file.name));
      this.fileJobs = {};
      this.selectedFiles = [];
      this.isUploading = false;
    },

    mergeDocumentsWithActiveDeletes(nextDocuments: DocumentItem[]): DocumentItem[] {
      const merged = Array.isArray(nextDocuments) ? [...nextDocuments] : [];
      Object.keys(this.deleteJobs).forEach((filename) => {
        const job = this.deleteJobs[filename];
        if (!job || job.status === 'failed') return;
        const exists = merged.some((doc) => doc.filename === filename);
        if (!exists) {
          const currentDoc = this.documents.find((doc) => doc.filename === filename);
          if (currentDoc) {
            merged.push(currentDoc);
          }
        }
      });
      return merged;
    },

    async loadDocuments() {
      this.documentsLoading = true;
      try {
        const response = await api.get('/documents');
        this.documents = this.mergeDocumentsWithActiveDeletes(response.data.documents || []);
      } catch (error: any) {
        const errMsg = error.response?.data?.detail || error.message || '加载文档列表失败';
        throw new Error(errMsg);
      } finally {
        this.documentsLoading = false;
      }
    },

    async uploadDocuments() {
      if (!this.selectedFiles.length) {
        throw new Error('请先选择文件');
      }
      if (this.isUploading) {
        throw new Error('已有上传任务进行中');
      }

      this.isUploading = true;
      this.uploadProgressCollapsed = false;

      const queue = [...this.selectedFiles];
      queue.forEach((file) => {
        this.initFileJob(file.name);
        const job = this.fileJobs[file.name];
        this.fileJobs = {
          ...this.fileJobs,
          [file.name]: { ...job, status: 'running', message: '准备上传' },
        };
        this.updateFileJobStep(file.name, 'upload', 0, 'running', '准备上传');
      });

      await Promise.all(
        queue.map((file) => this.uploadSingleFile(file).catch(() => undefined))
      );

      this.isUploading = false;
    },

    async uploadSingleFile(file: File) {
      const formData = new FormData();
      formData.append('file', file);

      try {
        const response = await api.post('/documents/upload/async', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          onUploadProgress: (progressEvent) => {
            if (!progressEvent.total) return;
            const percent = Math.round((progressEvent.loaded / progressEvent.total) * 100);
            this.updateFileJobStep(file.name, 'upload', percent, 'running', `已上传 ${percent}%`);
          },
        });

        const data = response.data;
        this.updateFileJobStep(file.name, 'upload', 100, 'completed', '文档上传完成');
        this.startUploadJobPolling(file.name, data.job_id, data.message);
      } catch (error: any) {
        const errMsg = error.response?.data?.detail || error.message || '上传失败';
        this.updateFileJobStep(file.name, 'upload', 100, 'failed', errMsg);
        this.failFileJob(file.name, '上传失败：' + errMsg);
      }
    },

    startUploadJobPolling(filename: string, jobId: string, initialMessage = '') {
      this.stopUploadJobPolling(filename);

      const current = this.fileJobs[filename];
      if (current) {
        this.fileJobs = {
          ...this.fileJobs,
          [filename]: { ...current, jobId, status: 'running', message: initialMessage || current.message },
        };
      }

      let consecutiveFailures = 0;

      const poll = async () => {
        try {
          const response = await api.get(`/documents/upload/jobs/${encodeURIComponent(jobId)}`, {
            timeout: 0,
          });
          consecutiveFailures = 0;
          this.syncFileJob(filename, response.data);

          if (response.data.status === 'completed') {
            this.finalizeCompletedFile(filename);
            await this.loadDocuments();
          } else if (response.data.status === 'failed') {
            this.stopUploadJobPolling(filename);
          }
        } catch (error: any) {
          consecutiveFailures += 1;
          if (consecutiveFailures >= 3) {
            const errMsg = error.response?.data?.detail || error.message || '查询失败';
            this.failFileJob(filename, '进度查询失败：' + errMsg);
            this.stopUploadJobPolling(filename);
          }
        }
      };

      poll();
      this.uploadPollTimers = {
        ...this.uploadPollTimers,
        [filename]: setInterval(poll, 1500),
      };
    },

    stopUploadJobPolling(filename: string) {
      const timer = this.uploadPollTimers[filename];
      if (!timer) return;
      clearInterval(timer);
      const { [filename]: _, ...rest } = this.uploadPollTimers;
      this.uploadPollTimers = rest;
    },

    stopAllUploadJobPolling() {
      Object.keys(this.uploadPollTimers).forEach((filename) => this.stopUploadJobPolling(filename));
    },

    isDeletingDocument(filename: string): boolean {
      const job = this.deleteJobs[filename];
      return !!(job && job.status === 'running');
    },

    isDeleteActionLocked(filename: string): boolean {
      const job = this.deleteJobs[filename];
      return !!(job && (job.status === 'running' || job.status === 'completed'));
    },

    getDeleteButtonIcon(filename: string): string {
      const job = this.deleteJobs[filename];
      if (job?.status === 'running') return 'fas fa-spinner fa-spin';
      if (job?.status === 'completed') return 'fas fa-check';
      return 'fas fa-trash';
    },

    setDeleteJob(filename: string, nextJob: Partial<ActiveDeleteJob>) {
      this.deleteJobs = {
        ...this.deleteJobs,
        [filename]: {
          ...(this.deleteJobs[filename] || {
            status: 'running',
            message: '',
            collapsed: false,
            steps: this.createDeleteSteps(),
          }),
          ...nextJob,
        },
      };
    },

    syncDeleteJob(filename: string, job: any) {
      const current = this.deleteJobs[filename] || {};
      this.setDeleteJob(filename, {
        jobId: job.job_id,
        status: job.status,
        message: job.message || '',
        collapsed: job.status === 'completed' ? true : Boolean(current.collapsed),
        steps: Array.isArray(job.steps)
          ? job.steps.map((step: any) => ({
              key: step.key,
              label: step.label,
              percent: step.percent,
              status: step.status,
              message: step.message || '',
            }))
          : this.createDeleteSteps(),
      });
    },

    async deleteDocument(filename: string) {
      if (this.isDeletingDocument(filename)) {
        return;
      }
      if (!confirm(`确定要删除文档 "${filename}" 吗？这将同时删除 Milvus 中的所有相关向量。`)) {
        return;
      }

      this.clearDeleteRemovalTimer(filename);
      this.setDeleteJob(filename, {
        status: 'running',
        message: '正在提交删除任务...',
        collapsed: false,
        steps: this.createDeleteSteps().map((step) =>
          step.key === 'prepare'
            ? { ...step, percent: 1, status: 'running' as const, message: '正在提交删除任务' }
            : step
        ),
      });

      try {
        const response = await api.delete(`/documents/delete/async/${encodeURIComponent(filename)}`);
        const data = response.data;
        this.setDeleteJob(filename, {
          jobId: data.job_id,
          status: 'running',
          message: data.message || `正在删除 ${filename}`,
          collapsed: false,
        });
        this.startDeleteJobPolling(filename, data.job_id);
      } catch (error: any) {
        const errMsg = error.response?.data?.detail || error.message || '删除请求失败';
        this.setDeleteJob(filename, {
          status: 'failed',
          message: '删除文档失败：' + errMsg,
          collapsed: false,
          steps: this.deleteJobs[filename]?.steps || this.createDeleteSteps(),
        });
      }
    },

    startDeleteJobPolling(filename: string, jobId: string) {
      this.stopDeleteJobPolling(filename);

      let consecutiveFailures = 0;

      const poll = async () => {
        try {
          const response = await api.get(`/documents/delete/jobs/${encodeURIComponent(jobId)}`, {
            timeout: 0,
          });
          consecutiveFailures = 0;
          const job = response.data;
          this.syncDeleteJob(filename, job);

          if (job.status === 'completed') {
            this.stopDeleteJobPolling(filename);
            this.scheduleDeletedDocumentRemoval(filename);
          } else if (job.status === 'failed') {
            this.stopDeleteJobPolling(filename);
          }
        } catch (error: any) {
          consecutiveFailures += 1;
          if (consecutiveFailures >= 3) {
            const errMsg = error.response?.data?.detail || error.message || '查询失败';
            this.setDeleteJob(filename, {
              status: 'failed',
              message: '删除进度查询失败：' + errMsg,
              collapsed: false,
              steps: this.deleteJobs[filename]?.steps || this.createDeleteSteps(),
            });
            this.stopDeleteJobPolling(filename);
          }
        }
      };

      poll();
      this.deletePollTimers = {
        ...this.deletePollTimers,
        [filename]: setInterval(poll, 1500),
      };
    },

    stopDeleteJobPolling(filename: string) {
      const timer = this.deletePollTimers[filename];
      if (!timer) return;
      clearInterval(timer);
      const { [filename]: _, ...rest } = this.deletePollTimers;
      this.deletePollTimers = rest;
    },

    stopAllDeleteJobPolling() {
      Object.keys(this.deletePollTimers).forEach((filename) => this.stopDeleteJobPolling(filename));
    },

    clearDeleteRemovalTimer(filename: string) {
      const timer = this.deleteRemoveTimers[filename];
      if (!timer) return;
      clearTimeout(timer);
      const { [filename]: _, ...rest } = this.deleteRemoveTimers;
      this.deleteRemoveTimers = rest;
    },

    scheduleDeletedDocumentRemoval(filename: string) {
      this.clearDeleteRemovalTimer(filename);
      const timer = setTimeout(async () => {
        this.documents = this.documents.filter((doc) => doc.filename !== filename);
        const { [filename]: _job, ...jobs } = this.deleteJobs;
        const { [filename]: _timer, ...timers } = this.deleteRemoveTimers;
        this.deleteJobs = jobs;
        this.deleteRemoveTimers = timers;
        await this.loadDocuments();
      }, 3000);
      this.deleteRemoveTimers = {
        ...this.deleteRemoveTimers,
        [filename]: timer,
      };
    },

    toggleDeleteJobCollapsed(filename: string) {
      const job = this.deleteJobs[filename];
      if (!job) return;
      this.setDeleteJob(filename, { collapsed: !job.collapsed });
    },
  },
});
export type { DocumentItem };