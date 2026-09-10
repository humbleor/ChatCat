import { createPinia, setActivePinia } from 'pinia';
import { readFileSync } from 'node:fs';
import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';
import { useDocumentStore } from './documents';
import api from '@/utils/api';

vi.mock('@/utils/api', () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
    delete: vi.fn(),
  },
}));

const flushPromises = async () => {
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
};

const createUploadJob = (overrides: Record<string, any> = {}) => ({
  job_id: 'job_upload_1',
  status: 'running',
  message: '正在向量化入库：450 / 770',
  steps: [
    { key: 'upload', label: '文档上传', percent: 100, status: 'completed', message: '文档上传完成' },
    { key: 'cleanup', label: '清理旧版本', percent: 100, status: 'completed', message: '清理完成' },
    { key: 'parse', label: '解析与分块', percent: 100, status: 'completed', message: '解析完成' },
    { key: 'parent_store', label: '父级分块入库', percent: 100, status: 'completed', message: '父级分块入库完成' },
    { key: 'vector_store', label: '向量化入库', percent: 58, status: 'running', message: '450 / 770' },
  ],
  ...overrides,
});

const mockFile = (name: string, size = 1024) =>
  ({ name, size } as File);

describe('document upload polling', () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    vi.useFakeTimers();
    vi.clearAllMocks();
  });

  afterEach(() => {
    const store = useDocumentStore();
    store.stopAllUploadJobPolling();
    store.stopAllDeleteJobPolling();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('does not stop upload polling when the settings view unmounts', () => {
    const source = readFileSync(new URL('../components/Documents/DocumentSettings.vue', import.meta.url), 'utf8');
    const unmountedBlock = source.match(/onUnmounted\(\(\) => \{([\s\S]*?)\}\);/);

    expect(unmountedBlock?.[1]).not.toContain('stopUploadJobPolling');
    expect(unmountedBlock?.[1]).not.toContain('stopAllUploadJobPolling');
    expect(unmountedBlock?.[1]).toContain('stopAllDeleteJobPolling');
  });

  it('continues polling upload progress until the active job completes', async () => {
    const store = useDocumentStore();
    const runningJob = createUploadJob();
    const completedJob = createUploadJob({
      status: 'completed',
      message: '文档处理完成',
      steps: [
        ...runningJob.steps.slice(0, 4),
        { key: 'vector_store', label: '向量化入库', percent: 100, status: 'completed', message: '770 / 770' },
      ],
    });
    const jobResponses = [runningJob, completedJob];

    vi.mocked(api.get).mockImplementation((url: string) => {
      if (url === '/documents') {
        return Promise.resolve({
          data: {
            documents: [{ filename: 'wuthering-waves.pdf', file_type: 'PDF', chunk_count: 770 }],
          },
        });
      }
      if (url === '/documents/upload/jobs/job_upload_1') {
        return Promise.resolve({ data: jobResponses.shift() || completedJob });
      }
      return Promise.reject(new Error(`Unexpected GET ${url}`));
    });

    store.selectedFiles = [mockFile('wuthering-waves.pdf')];
    store.initFileJob('wuthering-waves.pdf');
    store.startUploadJobPolling('wuthering-waves.pdf', 'job_upload_1');
    await flushPromises();

    const job = store.fileJobs['wuthering-waves.pdf'];
    expect(job.jobId).toBe('job_upload_1');
    expect(job.message).toBe('正在向量化入库：450 / 770');
    expect(job.steps.find((step) => step.key === 'vector_store')).toMatchObject({
      percent: 58,
      status: 'running',
    });
    expect(store.uploadPollTimers['wuthering-waves.pdf']).toBeDefined();

    await vi.advanceTimersByTimeAsync(1500);
    await flushPromises();

    expect(store.fileJobs['wuthering-waves.pdf']).toBeUndefined();
    expect(store.selectedFiles).toEqual([]);
    expect(store.uploadPollTimers['wuthering-waves.pdf']).toBeUndefined();
    expect(store.documents).toEqual([{ filename: 'wuthering-waves.pdf', file_type: 'PDF', chunk_count: 770 }]);
  });

  it('uploads multiple files in parallel and tracks per-file jobs independently', async () => {
    const store = useDocumentStore();

    const completedA = createUploadJob({
      job_id: 'job_a',
      status: 'completed',
      message: '文档处理完成',
      steps: createUploadJob().steps.map((step) => ({ ...step, status: 'completed', percent: 100 })),
    });
    const completedB = createUploadJob({
      job_id: 'job_b',
      status: 'completed',
      message: '文档处理完成',
      steps: createUploadJob().steps.map((step) => ({ ...step, status: 'completed', percent: 100 })),
    });

    let postCallIndex = 0;
    vi.mocked(api.post).mockImplementation((url: string) => {
      if (url === '/documents/upload/async') {
        const jobId = postCallIndex === 0 ? 'job_a' : 'job_b';
        postCallIndex += 1;
        return Promise.resolve({ data: { job_id: jobId, message: '文件已上传' } });
      }
      return Promise.reject(new Error(`Unexpected POST ${url}`));
    });

    vi.mocked(api.get).mockImplementation((url: string) => {
      if (url === '/documents') {
        return Promise.resolve({
          data: {
            documents: [
              { filename: '需求文档.pdf', file_type: 'PDF', chunk_count: 100 },
              { filename: '接口设计.docx', file_type: 'DOCX', chunk_count: 50 },
            ],
          },
        });
      }
      if (url === '/documents/upload/jobs/job_a') return Promise.resolve({ data: completedA });
      if (url === '/documents/upload/jobs/job_b') return Promise.resolve({ data: completedB });
      return Promise.reject(new Error(`Unexpected GET ${url}`));
    });

    store.selectedFiles = [mockFile('需求文档.pdf', 2048), mockFile('接口设计.docx', 1024)];
    await store.uploadDocuments();
    await flushPromises();
    await vi.advanceTimersByTimeAsync(1500);
    await flushPromises();

    expect(postCallIndex).toBe(2);
    expect(store.selectedFiles).toEqual([]);
    expect(store.fileJobs['需求文档.pdf']).toBeUndefined();
    expect(store.fileJobs['接口设计.docx']).toBeUndefined();
    expect(store.uploadPollTimers['需求文档.pdf']).toBeUndefined();
    expect(store.uploadPollTimers['接口设计.docx']).toBeUndefined();
  });

  it('keeps polling on transient errors and only fails after consecutive failures', async () => {
    const store = useDocumentStore();
    const completedJob = createUploadJob({
      status: 'completed',
      message: '文档处理完成',
      steps: createUploadJob().steps.map((step) => ({ ...step, status: 'completed', percent: 100 })),
    });

    const responses = [
      Promise.reject({ message: 'timeout of 60000ms exceeded' }),
      Promise.reject({ message: 'timeout of 60000ms exceeded' }),
      Promise.resolve({ data: completedJob }),
    ];

    vi.mocked(api.get).mockImplementation((url: string) => {
      if (url === '/documents') {
        return Promise.resolve({
          data: { documents: [{ filename: 'retry.pdf', file_type: 'PDF', chunk_count: 10 }] },
        });
      }
      if (url === '/documents/upload/jobs/job_retry') {
        const next = responses.shift();
        return next || Promise.resolve({ data: completedJob });
      }
      return Promise.reject(new Error(`Unexpected GET ${url}`));
    });

    store.selectedFiles = [mockFile('retry.pdf')];
    store.initFileJob('retry.pdf');
    store.startUploadJobPolling('retry.pdf', 'job_retry');
    await flushPromises();

    expect(store.fileJobs['retry.pdf']?.status).toBe('running');
    expect(store.uploadPollTimers['retry.pdf']).toBeDefined();

    await vi.advanceTimersByTimeAsync(1500);
    await flushPromises();

    expect(store.fileJobs['retry.pdf']?.status).toBe('running');
    expect(store.uploadPollTimers['retry.pdf']).toBeDefined();

    await vi.advanceTimersByTimeAsync(1500);
    await flushPromises();

    expect(store.fileJobs['retry.pdf']).toBeUndefined();
    expect(store.selectedFiles).toEqual([]);
    expect(store.uploadPollTimers['retry.pdf']).toBeUndefined();
  });

  it('keeps a failed file in selectedFiles so the user can retry', async () => {
    const store = useDocumentStore();
    const failedJob = createUploadJob({
      status: 'failed',
      message: '处理失败',
      steps: createUploadJob().steps.map((step, i) =>
        i === 4 ? { ...step, status: 'failed', percent: 0 } : step
      ),
    });

    vi.mocked(api.get).mockImplementation((url: string) => {
      if (url === '/documents') return Promise.resolve({ data: { documents: [] } });
      if (url === '/documents/upload/jobs/job_fail') return Promise.resolve({ data: failedJob });
      return Promise.reject(new Error(`Unexpected GET ${url}`));
    });

    store.selectedFiles = [mockFile('broken.pdf')];
    store.initFileJob('broken.pdf');
    store.startUploadJobPolling('broken.pdf', 'job_fail');
    await flushPromises();

    await vi.advanceTimersByTimeAsync(1500);
    await flushPromises();

    expect(store.fileJobs['broken.pdf']?.status).toBe('failed');
    expect(store.selectedFiles.map((f) => f.name)).toEqual(['broken.pdf']);
    expect(store.uploadPollTimers['broken.pdf']).toBeUndefined();
  });

  it('removes a selected file and stops its polling', async () => {
    const store = useDocumentStore();
    vi.mocked(api.get).mockResolvedValue({ data: createUploadJob() });

    store.selectedFiles = [mockFile('a.pdf'), mockFile('b.pdf')];
    store.initFileJob('a.pdf');
    store.initFileJob('b.pdf');
    store.startUploadJobPolling('a.pdf', 'job_a');
    store.startUploadJobPolling('b.pdf', 'job_b');

    store.removeSelectedFile('a.pdf');

    expect(store.selectedFiles.map((f) => f.name)).toEqual(['b.pdf']);
    expect(store.fileJobs['a.pdf']).toBeUndefined();
    expect(store.fileJobs['b.pdf']).toBeDefined();
    expect(store.uploadPollTimers['a.pdf']).toBeUndefined();
    expect(store.uploadPollTimers['b.pdf']).toBeDefined();
  });
});