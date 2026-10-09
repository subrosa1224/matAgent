"""Master orchestration layer for the multi-agent system."""

from materials_screening.master.application_contracts import (
    MultiAgentArtifact,
    SubAgentUiRegistry,
    SubAgentUiSpec,
    UnifiedConversationContext,
    UnifiedResultEnvelope,
    UnifiedResultReference,
)
from materials_screening.master.application_feasibility import (
    ApplicationCandidate,
    prioritize_uv_candidates,
    render_uv_candidate_queue,
)
from materials_screening.master.artifact_registry import ArtifactRegistry
from materials_screening.master.cross_agent import (
    CrossAgentCandidate,
    CrossAgentResult,
    LiteratureDatabaseCoordinator,
    LiteratureMaterialClueExtractor,
    MaterialClue,
)
from materials_screening.master.data_analysis_handoffs import (
    DataAnalysisCrossAgentCoordinator,
    DataAnalysisHandoff,
    MaterialLookupHandoff,
    SourcePartition,
    SourcePartitionedSynthesis,
    SourceSynthesisSection,
    render_source_partitioned_synthesis,
)
from materials_screening.master.literature_evidence_trial import (
    LiteratureEvidenceTrialResult,
    LiteratureEvidenceTrialService,
    render_literature_evidence_trial,
)
from materials_screening.master.master_context import MasterAgentContext
from materials_screening.master.master_graph_builder import (
    MASTER_GRAPH_NAME,
    MasterGraphInput,
    compile_master_graph,
    create_master_agent_builder,
)
from materials_screening.master.master_runner import MasterAgentRunner
from materials_screening.master.master_settings import MasterAgentSettings
from materials_screening.master.master_state import MasterAgentState
from materials_screening.master.result_renderers import (
    ResultRendererRegistry,
    default_renderer_registry,
    render_markdown_result,
)
from materials_screening.master.sub_agent_executor import (
    SubAgentExecutionOutcome,
    SubAgentExecutor,
)
from materials_screening.master.sub_agent_registry import SubAgentRegistry
from materials_screening.master.sub_agent_spec import (
    SubAgentCall,
    SubAgentDefinition,
    SubAgentResultEnvelope,
    SubAgentSpec,
)
from materials_screening.master.ui_controller import (
    MasterUiUpdate,
    stream_master_request,
)
from materials_screening.master.ui_plugins import (
    SubAgentUiPlugin,
    SubAgentUiPluginRegistry,
)

__all__ = [
    "MASTER_GRAPH_NAME",
    "ApplicationCandidate",
    "ArtifactRegistry",
    "CrossAgentCandidate",
    "CrossAgentResult",
    "DataAnalysisCrossAgentCoordinator",
    "DataAnalysisHandoff",
    "LiteratureDatabaseCoordinator",
    "LiteratureEvidenceTrialResult",
    "LiteratureEvidenceTrialService",
    "LiteratureMaterialClueExtractor",
    "MaterialClue",
    "MaterialLookupHandoff",
    "MultiAgentArtifact",
    "ResultRendererRegistry",
    "MasterAgentContext",
    "MasterAgentRunner",
    "MasterAgentSettings",
    "MasterAgentState",
    "MasterGraphInput",
    "MasterUiUpdate",
    "SubAgentCall",
    "SubAgentDefinition",
    "SubAgentExecutionOutcome",
    "SubAgentExecutor",
    "SubAgentRegistry",
    "SubAgentResultEnvelope",
    "SubAgentSpec",
    "SubAgentUiRegistry",
    "SubAgentUiSpec",
    "SourcePartition",
    "SourcePartitionedSynthesis",
    "SourceSynthesisSection",
    "SubAgentUiPlugin",
    "SubAgentUiPluginRegistry",
    "UnifiedConversationContext",
    "UnifiedResultEnvelope",
    "UnifiedResultReference",
    "compile_master_graph",
    "create_master_agent_builder",
    "default_renderer_registry",
    "render_markdown_result",
    "prioritize_uv_candidates",
    "render_literature_evidence_trial",
    "render_source_partitioned_synthesis",
    "render_uv_candidate_queue",
    "stream_master_request",
]
