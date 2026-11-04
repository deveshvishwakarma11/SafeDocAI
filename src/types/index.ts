export type DocumentStatus = 'queued' | 'uploading' | 'processing' | 'success' | 'failed' | 'cancelled' | 'superseded';
export type DocumentType = 'pdf' | 'jpg' | 'jpeg' | 'png' | 'other';
export type DocumentCategory =
  | 'Marksheet / Degree'
  | 'Utility Bill'
  | 'Identity Document'
  | 'Bank Statement / Passbook'
  | 'Insurance Policy'
  | 'Receipt / Invoice'
  | 'Travel Ticket'
  | 'Application Form'
  | 'General Document';

export interface ExtractedField {
  id: string;
  document_id: number;
  field_name: string;        // canonical field key e.g. "roll_number"
  display_label: string;      // "Roll Number"
  field_value: string;       // "2023CS001"
  evidence_text: string;     // verbatim sentence or context
  page_number?: number;
  confidence: number;        // 0.0 to 1.0
  value_type: 'identifier' | 'date' | 'amount' | 'name' | 'code' | 'text' | 'address';
  source_start?: number;
  source_end?: number;
  created_at: string;
}

export interface DocumentChunk {
  chunk_id: string;          // doc_{id}_chunk_{idx}
  document_id: number;
  file_name: string;
  page_number: number;
  chunk_index: number;
  text: string;
  token_count?: number;
}

export interface StoredDocument {
  id: number;
  profile_id: string;
  file_name: string;
  file_type: DocumentType;
  category: DocumentCategory;
  file_path: string;
  content_hash: string;      // SHA-256
  artifact_stem: string;     // collision-safe unique stem
  status: DocumentStatus;
  error_message?: string;
  upload_timestamp: string;
  processed_at?: string;
  file_size_bytes: number;
  page_count: number;
  raw_text: string;
  data_url?: string; // base64 data url (set right after upload)
  file_url?: string; // managed-store URL (/api/documents/:id/file) for previews that survive reloads
  fields: ExtractedField[];
  chunks: DocumentChunk[];
}

export type QueryRoute = 'exact_field' | 'semantic_rag' | 'document_retrieval' | 'clarification' | 'out_of_scope' | 'greeting';

export interface SourceEvidence {
  document_id: number;
  file_name: string;
  field_name?: string;
  value?: string;
  snippet: string;
  page_number?: number;
  confidence?: number;
}

export interface ClarificationOption {
  label: string;
  value: string;
  document_id?: number;
}

export interface QueryResult {
  route: QueryRoute;
  detected_language: 'en' | 'hi' | 'hinglish';
  intent: string;
  answer: string;
  evidence: SourceEvidence[];
  clarification_options?: ClarificationOption[];
  is_insufficient_evidence?: boolean;
  latency_ms: number;
  execution_plan?: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  route?: QueryRoute;
  intent?: string;
  evidence?: SourceEvidence[];
  clarification_options?: ClarificationOption[];
  is_insufficient_evidence?: boolean;
  is_cancelled?: boolean;
}

export interface IngestionJob {
  job_id: string;
  file_name: string;
  file_size: number;
  stage: 'queued' | 'uploading' | 'ocr_parsing' | 'understanding' | 'validating_evidence' | 'indexing' | 'completed' | 'failed' | 'cancelled';
  progress: number; // 0 to 100
  error?: string;
  document_id?: number;
}

export interface RuntimeDiagnostics {
  sqlite_status: 'healthy' | 'degraded' | 'offline';
  sqlite_records_count: number;
  sqlite_metadata_count: number;
  vector_engine_status: 'healthy' | 'indexing' | 'idle';
  vector_chunks_count: number;
  local_llm_status: 'connected_ollama' | 'unavailable';
  ocr_engine_status: 'active_eng_hin' | 'active_eng_only' | 'unavailable';
  network_egress_status: 'zero_egress_local' | 'offline_verified';
  storage_bytes_used: number;
}
