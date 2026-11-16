from pydantic import BaseModel, Field


class RunMetadata(BaseModel):
    run_name: str
    mode: str = "crag"
    timestamp: str
    crag_enabled: bool
    model: str
    temperature: float | None = None
    judge_model: str
    judge_temperature: float | None = None
    top_k: int


class JudgeEvaluationPrompt(BaseModel):
    faithfulness_reason: str
    faithfulness_score: int = Field(..., ge=0, le=5)

    relevance_reason: str
    relevance_score: int = Field(..., ge=0, le=5)


class CRAGTelemetry(BaseModel):
    web_search_triggered: bool
    docs_after_grading: int
    grader_tokens: int = 0
    grader_cost_usd: float = 0.0


class GenerationEvaluationItem(BaseModel):
    id: str | None = None
    query: str
    source_doc: str | None = None
    question_type: str | None = None
    top_k: int
    context: str | None = None
    answer: str | None = None
    error: str | None = None
    generation_latency_ms: float = 0.0
    judge_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    rag_tokens: int = 0
    rag_cost_usd: float = 0.0
    judge_tokens: int = 0
    judge_cost_usd: float = 0.0
    total_tokens: int = 0
    cost_usd: float = 0.0
    tool_call_count: int = 0
    tools_used: dict[str, int] = Field(default_factory=dict)
    escalated: bool | None = None
    judge_result: JudgeEvaluationPrompt
    crag_telemetry: CRAGTelemetry | None = None


class QuestionTypeMetrics(BaseModel):
    count: int
    valid_count: int
    error_count: int
    avg_faithfulness_all: float
    avg_faithfulness_valid: float
    avg_relevance_all: float
    avg_relevance_valid: float
    avg_generation_latency_ms: float
    p95_generation_latency_ms: float
    avg_tool_calls: float
    escalation_count: int | None = None
    escalation_rate: float | None = None


class GenerationSummary(BaseModel):
    avg_faithfulness: float
    avg_relevance: float
    all_questions_avg_faithfulness: float = 0.0
    all_questions_avg_relevance: float = 0.0
    avg_multiturn_faithfulness: float | None = None
    avg_multiturn_relevance: float | None = None
    error_count: int = 0
    error_rate: float = 0.0
    avg_generation_latency_ms: float
    avg_judge_latency_ms: float
    avg_total_latency_ms: float
    p95_generation_latency_ms: float
    p95_total_latency_ms: float
    total_rag_cost_usd: float = 0.0
    avg_rag_tokens_per_query: float = 0.0
    total_grader_cost_usd: float = 0.0
    avg_grader_tokens_per_query: float = 0.0
    total_judge_cost_usd: float = 0.0
    total_run_cost_usd: float = 0.0
    avg_tokens_per_query: float = 0.0
    avg_tool_calls_per_query: float = 0.0
    escalation_count: int | None = None
    escalation_rate: float | None = None
    by_question_type: dict[str, QuestionTypeMetrics] = Field(default_factory=dict)


class GenerationEvaluationReport(BaseModel):
    metadata: RunMetadata
    total_questions: int
    total_valid: int
    total_multiturn: int
    summary: GenerationSummary
    detailed_results: list[GenerationEvaluationItem]
