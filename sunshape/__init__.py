from sunshape.codec import SunShapeBlockCodec, SunShapeQuantized, _generate_rotation
from sunshape.diagnose import (
    diagnose_model,
    diagnose_layer,
    diagnose_from_traces,
    layer_selection_score,
    rank_layers,
    ModelDiagnostic,
)
from sunshape.hf import (
    SunShapeBundle,
    SunShapeCausalLM,
    SunShapeConfig,
    SunShapeForCausalLM,
    TraceConfig,
    default_block_dim,
    extract_trace_artifact,
    fit_bundle_from_traces,
    fit_sunshape_bundle,
    infer_attention_capable_layers,
    load_model_and_tokenizer,
    load_trace_artifact,
    prepare_sunshape_model,
    resolve_mode_alias,
    save_trace_artifact,
)
from sunshape.integration.vllm import (
    SunShapeVLLMConfig,
    SunShapeVLLMHandle,
    SunShapeVLLMRuntimeState,
    build_launch_env,
    export_bundle_for_vllm,
    load_runtime_state,
    prepare_vllm_bundle,
    runtime_notes,
)
from sunshape.methods import (
    MethodSpec,
    SUNSHAPE_PRIMARY_METHODS,
    build_cache_factory,
    canonical_method_name,
    default_method_grid,
    method_family,
    parse_method_spec,
)
from sunshape.metrics import build_tlsunshape_metric, log_shape
from sunshape.eval import run_cache_eval, save_eval_outputs, load_eval_texts
from sunshape.server import SunShapeRuntime, build_runtime, serve_runtime
from sunshape.stats import CompressionStats, build_compression_stats, load_trace_meta
from sunshape.turbo_cache import TurboQuantCache
from sunshape.triton_kernels import sunshape_attention_scores, _torch_attention_scores
from sunshape.turbo_baseline import (
    TurboQuantMSE,
    TurboQuantProd,
    QJL,
    turbo_1bit_quantize,
    qjl_1bit_quantize,
    compute_logit_mse,
    compute_kl_attention,
)

__version__ = "0.1.0"
