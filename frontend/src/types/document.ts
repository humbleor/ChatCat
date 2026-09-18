export interface DocumentItem {
  document_id?: string;
  filename: string;
  file_type: string;
  chunk_count: number;
  status?: string;
  uploaded_at?: string;
}

export interface UploadStep {
  key: string;
  label: string;
  percent: number;
  status: 'pending' | 'running' | 'completed' | 'failed';
  message: string;
}

export interface UploadJob {
  job_id: string;
  status: 'running' | 'completed' | 'failed';
  message: string;
  steps: UploadStep[];
}

export interface DeleteStep {
  key: string;
  label: string;
  percent: number;
  status: 'pending' | 'running' | 'completed' | 'failed';
  message: string;
}

export interface DeleteJob {
  job_id: string;
  status: 'running' | 'completed' | 'failed';
  message: string;
  steps: DeleteStep[];
}

export interface ActiveDeleteJob {
  jobId?: string;
  status: 'running' | 'completed' | 'failed';
  message: string;
  collapsed: boolean;
  steps: DeleteStep[];
}
