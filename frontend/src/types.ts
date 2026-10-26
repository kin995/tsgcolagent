export type Status = 'active' | 'done' | 'pending' | 'standby' | 'running' | 'completed' | 'waiting' | string;

export interface Metrics {
  pass_rate?: number | null; avg_exec?: number | null; exec_count?: number; exec_count_status?: string;
  tir?: number | null; tir_comparable?: number; tir_improved?: number; tir_excluded?: number; tir_status?: string;
  knowledge_occurrence?: number; knowledge_use_frequency?: number; knowledge_record_ids?: string[];
  tests_passed?: boolean; collected_count?: number; line_coverage?: number; branch_coverage?: number;
  duration_seconds?: number | null; sfc?: number; ae?: number; mutation_score?: number; tsq?: number; assertion_count?: number;
  effective_assertions?: number; effective_mutants?: number; mutation_reliability?: string;
}

export interface Task {
  id: string; name: string; source?: string; output_dir?: string; summary_path?: string; graph_path?: string;
  scope?: 'file' | 'class' | 'project' | string;
  test_path?: string; status?: string; grade: string; round: number; round_label: string; iterations?: number;
  stop_reason?: string; metrics: Metrics; grade_history?: unknown[]; reports_count?: number;
  round_strategy?: string; round_status?: string;
  failure_report?: string;
  failure?: {
    category?: string; failed_tests?: string[]; weak_flags?: string[]; threshold_failures?: Record<string, number>;
    reason?: string; error?: string; previous_solutions?: string[]; suggested_solutions?: string[];
    attempted_measures?: string[]; knowledge_history?: unknown[]; failure_report?: string;
    stage?: string; knowledge_record_id?: string;
  };
  updated_at?: string; experiment_id?: string; experiment_label?: string;
  dataset_id?: string; dataset_name?: string;
  duration_seconds?: number; started_at?: string; finished_at?: string; attempt_count?: number; timing_history?: unknown[];
  test_artifact_available?: boolean;
}

export interface FlowEvent {
  event_id?: string; event_type?: string; source_agent?: string; target_agent?: string; iteration?: number;
  related_nodes?: string[]; related_edges?: string[]; payload?: Record<string, unknown>; timestamp?: string;
  experiment_id?: string; experiment_label?: string; task_id?: string; task_name?: string;
}

export interface InternalTask { name: string; status: Status; event_count: number; last_event?: FlowEvent | null }
export interface AgentState {
  id: string; name: string; status: Status; modules: string[]; internal_tasks?: InternalTask[];
  current_action?: string; interaction_role?: 'executed' | 'received'; last_event?: FlowEvent; event_count?: number;
}

export interface StateNode {
  id: string; type?: string; name?: string; visited?: boolean; evidenced?: boolean; line?: number;
  coverage_status?: string; mutation_status?: string; assertion_effective?: boolean; repair_flag?: boolean; defect_score?: number;
}
export interface StateEdge { id?: string; source: string; target: string; type?: string; visited?: boolean }

export interface CatalogModule { id: string; name: string; description: string }
export interface CatalogAgent { id: string; name: string; mission: string; inputs: string[]; outputs: string[]; modules: CatalogModule[] }
export interface CatalogLanguage { id: string; name: string; source_extensions: string[]; test_framework: string; coverage_tools: string[]; mutation_tools: string[]; analysis: string[]; generated_test: string; evidence: string[] }
export interface CatalogPhase { id: string; name: string; owner: string; description: string; artifacts: string[]; status?: Status; event_count?: number; last_event?: FlowEvent }
export interface CatalogScope { id: string; name: string; entry: string; description: string }
export interface CatalogMetric { id: string; name: string; description: string }
export interface CatalogExecutionPolicy { id: string; name: string; value: string; description: string }
export interface SystemCatalog { name: string; runtime: string; languages: CatalogLanguage[]; scopes: CatalogScope[]; phases: CatalogPhase[]; agents: CatalogAgent[]; quality_metrics: CatalogMetric[]; execution_policies?: CatalogExecutionPolicy[] }

export interface RuntimeSnapshot {
  schema_version: number; generated_at: string;
  experiment: { id: string; output_dir: string; label: string; registered_at: string; status: string; report_path?: string | null; started_at?: string; finished_at?: string; duration_seconds?: number } | null;
  experiments: Array<{ id: string; parent_id?: string; output_dir: string; label: string; batch_report_path?: string | null; status?: string; language?: string; scope?: string; total_tasks?: number; completed_execution?: number; not_completed_execution?: number; progress?: number; queue_position?: number; started_at?: string; finished_at?: string; duration_seconds?: number; duration_source?: string }>;
  portfolio?: { view: string; experiments: RuntimeSnapshot['experiments']; counts: Record<string, number> };
  overview: {
    page_title?: string; language?: string; model_name?: string; model_usage?: { recorded_tasks: number; calls: number; prompt_tokens: number; completion_tokens: number; total_tokens: number; duration_seconds: number } | null;
    mode?: string; status: string; scope?: string; total_tasks?: number; completed_execution?: number;
    not_completed_execution?: number; grade_counts?: Record<string, number>;
    followup_task_count?: number; followup_attempts?: number; quality_averages?: Record<string, number | string | unknown[]>;
    started_at?: string; finished_at?: string; duration_seconds?: number; active_session_started_at?: string;
    duration_source?: string;
    accumulated_duration_before_session?: number; session_duration_seconds?: number; dataset_count?: number; completed_datasets?: number;
    task_workers?: number; active_task_count?: number; queued_task_count?: number;
  };
  parallel_execution?: {
    task_workers: number; llm_concurrency: number; python_parallel: boolean; java_parallel: boolean;
    active_task_count: number; queued_task_count: number; knowledge_store?: string;
    active_tasks: Array<{ id?: string; name?: string; round_label?: string; updated_at?: string }>;
  };
  catalog: SystemCatalog; phase_states: CatalogPhase[];
  toolchain: Array<{ name: string; status: string; value?: unknown }>;
  agents: AgentState[]; current_task?: Task | null; tasks: Task[];
  state_model: { version?: number; iteration?: number; nodes: StateNode[]; edges: StateEdge[]; metadata?: Record<string, unknown> };
  flow_events: FlowEvent[]; failures?: Record<string, unknown>;
  control?: { can_start?: boolean; process_id?: number; process_running?: boolean; log_path?: string; queue_position?: number; queue_size?: number; execution_status?: string };
  assistant?: { enabled: boolean; provider: string; model: string; base_url?: string; backend_managed: boolean; api_key_configured: boolean };
}

export interface KnowledgeRecord {
  id: string; status: string; language?: string; agent?: string; action?: string; failure_category?: string;
  accepted?: boolean; reward?: number; quality_delta?: Record<string, number>; source?: string; solution?: string;
  usage?: string;
  topic?: string; record_type?: string; learning_stage?: string; validation_count?: number; validation_failure_count?: number;
  support_count?: number; independent_task_count?: number;
  source_type?: 'run_record' | 'historical_baseline' | 'dataset_summary' | 'post_import_learning'; dataset?: string;
}
export interface KnowledgeSnapshot {
  path?: string; total: number; counts: Record<string, number>; records: KnowledgeRecord[];
  scope?: 'imported_run' | 'global';
  availability?: {
    complete: boolean; note: string;
    recovered_snapshot_records: number; recovered_journal_records: number;
    run_record_count: number; baseline_record_count: number; dataset_summary_count: number;
    post_import_learning_count?: number;
    missing_datasets: string[];
  };
}

export interface KnowledgeLearning { record_id?: string; topic?: string; status?: string; validation_count?: number; adoption_requirement?: string }
export interface PendingLearning { id: string; status: string; attempts?: number; created_at?: string; last_error?: string }
export interface ChatMessage { id: string; role: 'user' | 'assistant'; content: string; created_at: string; agent?: string; sources?: string[]; model_used?: boolean; knowledge_used?: boolean; response_mode?: string; needs_learning?: boolean; pending_learning?: PendingLearning | null; learning?: KnowledgeLearning | null; context?: { experiment_id?: string; task_id?: string; label?: string } }
export interface Conversation { id: string; title: string; created_at: string; updated_at: string; bound_experiment?: string; bound_task?: string; context_label?: string; message_count?: number; last_message?: string; messages?: ChatMessage[] }
export interface ChatReply { conversation: Conversation; message: ChatMessage; answer: { agent: string; answer: string; sources: string[]; model_used: boolean; knowledge_used?: boolean; response_mode?: string; needs_learning?: boolean; pending_learning?: PendingLearning | null; read_only: boolean; knowledge_learning?: KnowledgeLearning | null } }
export interface Artifact {
  path: string; name: string; size: number; content?: string; truncated?: boolean; kind?: 'state_model' | 'data' | 'html'; data?: unknown;
  hidden_internal_metrics?: boolean;
  state_model?: RuntimeSnapshot['state_model'] & { source_path?: string };
}
export interface ExperimentRequest { mode: 'single' | 'directory' | 'datasets'; source: string; output: string; language: 'auto' | 'python' | 'java' | 'all'; label?: string; llm: boolean; llm_provider?: string; llm_model?: string; llm_base_url?: string; llm_api_key?: string }
