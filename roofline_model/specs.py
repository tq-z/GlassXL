"""Offline architecture registry for the coarse MoE roofline estimator."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionProfile:
    kind: str
    projection_params: int
    pair_flops: float
    kv_elements_per_token: float
    fixed_state_elements: float = 0.0
    fixed_state_flops: float = 0.0
    index_scan_flops_per_item: float = 0.0
    index_elements_per_token: float = 0.0
    window: int = 0
    sparse_top_k: int = 0
    compression: int = 1


@dataclass(frozen=True)
class LayerGroup:
    name: str
    count: int
    layer_indices: str
    attention: AttentionProfile
    ffn_kind: str


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    aliases: tuple[str, ...]
    source_urls: tuple[str, ...]
    hidden_size: int
    num_layers: int
    dense_intermediate_size: int
    expert_intermediate_size: int
    num_experts: int
    top_k: int
    shared_experts: int
    shared_intermediate_size: int
    routed_width: int
    moe_common_active_params: int
    layer_groups: tuple[LayerGroup, ...]

    @property
    def routed_activation_width(self) -> int:
        return self.routed_width

    @property
    def dense_active_params(self) -> int:
        return 3 * self.hidden_size * self.dense_intermediate_size

    @property
    def routed_expert_params(self) -> int:
        return 3 * self.routed_width * self.expert_intermediate_size

    @property
    def shared_active_params(self) -> int:
        return (
            3
            * self.hidden_size
            * self.shared_intermediate_size
            * self.shared_experts
        )

    @property
    def moe_active_params(self) -> int:
        return (
            self.routed_expert_params * self.top_k
            + self.shared_active_params
            + self.moe_common_active_params
        )


def _gqa(hidden: int, q_heads: int, kv_heads: int, head_dim: int, *, gated: bool = False) -> AttentionProfile:
    q_dim = q_heads * head_dim
    kv_dim = kv_heads * head_dim
    q_multiplier = 2 if gated else 1
    params = hidden * (q_multiplier * q_dim + 2 * kv_dim) + q_dim * hidden
    return AttentionProfile(
        kind="gated_full" if gated else "gqa_full",
        projection_params=params,
        pair_flops=4.0 * q_heads * head_dim,
        kv_elements_per_token=2.0 * kv_heads * head_dim,
    )


def _mla(
    hidden: int,
    heads: int,
    q_rank: int,
    kv_rank: int,
    qk_nope: int,
    rope_dim: int,
    v_dim: int,
    *,
    kind: str = "mla_full",
    gated: bool = False,
) -> AttentionProfile:
    qk_dim = qk_nope + rope_dim
    params = (
        hidden * q_rank
        + q_rank * heads * qk_dim
        + hidden * (kv_rank + rope_dim)
        + kv_rank * heads * (qk_nope + v_dim)
        + heads * v_dim * hidden
    )
    if gated:
        params += hidden * heads * v_dim
    return AttentionProfile(
        kind=kind,
        projection_params=params,
        pair_flops=2.0 * heads * (kv_rank + rope_dim + v_dim),
        kv_elements_per_token=float(kv_rank + rope_dim),
    )


GLM45_ATTN = _gqa(5120, 96, 8, 128)
GLM52_MLA = _mla(6144, 64, 2048, 512, 192, 64, 256, kind="dsa_shared")
GLM52_FRESH = AttentionProfile(
    **{
        **GLM52_MLA.__dict__,
        "kind": "dsa_fresh",
        "index_scan_flops_per_item": 2.0 * 32 * 128,
        "index_elements_per_token": 128.0,
        "sparse_top_k": 2048,
    }
)
GLM52_SHARED = AttentionProfile(
    **{
        **GLM52_MLA.__dict__,
        "kind": "dsa_shared",
        "sparse_top_k": 2048,
    }
)
DEEPSEEK_V3_ATTN = _mla(7168, 128, 1536, 512, 128, 64, 128)
KIMI_K2_ATTN = _mla(7168, 64, 1536, 512, 128, 64, 128)
KIMI_K3_MLA = _mla(7168, 96, 1536, 512, 128, 64, 128, gated=True)
KIMI_K3_KDA = AttentionProfile(
    kind="kda",
    projection_params=4 * 7168 * 7168,
    pair_flops=0.0,
    kv_elements_per_token=0.0,
    fixed_state_elements=96 * 128 * 128 + 3 * 12_288 * 3,
    fixed_state_flops=4.0 * 96 * 128 * 128,
)
QWEN38_FULL = _gqa(8192, 64, 4, 256, gated=True)
QWEN38_DELTA = AttentionProfile(
    kind="delta_net",
    projection_params=4 * 8192 * 8192,
    pair_flops=0.0,
    kv_elements_per_token=0.0,
    fixed_state_elements=128 * 128 * 128 + 20_480 * 3,
    fixed_state_flops=4.0 * 128 * 128 * 128,
)
QWEN3_ATTN = _gqa(4096, 64, 4, 128)
V4_BASE_PARAMS = 7168 * (2 * 128 * 512 + 2 * 512)
V4_HCA = AttentionProfile(
    kind="hca",
    projection_params=V4_BASE_PARAMS,
    pair_flops=4.0 * 128 * 512,
    kv_elements_per_token=512.0,
    window=128,
    compression=128,
)
V4_CSA = AttentionProfile(
    kind="csa",
    projection_params=V4_BASE_PARAMS,
    pair_flops=4.0 * 128 * 512,
    kv_elements_per_token=512.0,
    index_scan_flops_per_item=2.0 * 64 * 128,
    index_elements_per_token=128.0 / 4.0,
    window=128,
    sparse_top_k=1024,
    compression=4,
)


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="glm-5.2",
        display_name="GLM-5.2",
        aliases=("glm5.2", "glm52", "zai-org/glm-5.2"),
        source_urls=("https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json",),
        hidden_size=6144,
        num_layers=78,
        dense_intermediate_size=12288,
        expert_intermediate_size=2048,
        num_experts=256,
        top_k=8,
        shared_experts=1,
        shared_intermediate_size=2048,
        routed_width=6144,
        moe_common_active_params=0,
        layer_groups=(
            LayerGroup("fresh-index DSA + dense FFN", 3, "0-2", GLM52_FRESH, "dense"),
            LayerGroup("fresh-index DSA + MoE", 18, "6,10,...,74", GLM52_FRESH, "moe"),
            LayerGroup("reused-index DSA + MoE", 57, "remaining main layers", GLM52_SHARED, "moe"),
        ),
    ),
    ModelSpec(
        key="deepseek-v4",
        display_name="DeepSeek-V4-Pro-1.6T",
        aliases=("deepseekv4", "deepseek-v4-pro", "deepseek-ai/deepseek-v4-pro"),
        source_urls=("https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json",),
        hidden_size=7168,
        num_layers=61,
        dense_intermediate_size=0,
        expert_intermediate_size=3072,
        num_experts=384,
        top_k=6,
        shared_experts=1,
        shared_intermediate_size=3072,
        routed_width=7168,
        moe_common_active_params=0,
        layer_groups=(
            LayerGroup("HCA + MoE", 31, "0-1 and odd 3-59", V4_HCA, "moe"),
            LayerGroup("CSA + MoE", 30, "2 and even 4-60", V4_CSA, "moe"),
        ),
    ),
    ModelSpec(
        key="kimi-k3",
        display_name="Kimi-K3",
        aliases=("kimi k3", "kimik3", "moonshotai/kimi-k3"),
        source_urls=("https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json",),
        hidden_size=7168,
        num_layers=93,
        dense_intermediate_size=33792,
        expert_intermediate_size=3072,
        num_experts=896,
        top_k=16,
        shared_experts=2,
        shared_intermediate_size=6144,
        routed_width=3584,
        moe_common_active_params=2 * 7168 * 3584,
        layer_groups=(
            LayerGroup("KDA + dense FFN", 1, "0", KIMI_K3_KDA, "dense"),
            LayerGroup("KDA + LatentMoE", 68, "KDA layers excluding 0", KIMI_K3_KDA, "moe"),
            LayerGroup(
                "gated MLA + LatentMoE",
                24,
                "configured 24-layer gated-MLA schedule",
                KIMI_K3_MLA,
                "moe",
            ),
        ),
    ),
    ModelSpec(
        key="qwen3.8-2.4t",
        display_name="Qwen3.8-2.4T-A95B",
        aliases=("qwen3.8-2.4t-a95b", "qwen38-2.4t", "qwen/qwen3.8-2.4t-a95b"),
        source_urls=("https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/blob/main/config.json",),
        hidden_size=8192,
        num_layers=92,
        dense_intermediate_size=0,
        expert_intermediate_size=2048,
        num_experts=512,
        top_k=10,
        shared_experts=1,
        shared_intermediate_size=2048,
        routed_width=8192,
        moe_common_active_params=0,
        layer_groups=(
            LayerGroup("Gated DeltaNet + MoE", 69, "3 of every 4 layers", QWEN38_DELTA, "moe"),
            LayerGroup("gated full attention + MoE", 23, "every 4th layer", QWEN38_FULL, "moe"),
        ),
    ),
    ModelSpec(
        key="glm-4.5",
        display_name="GLM-4.5",
        aliases=("glm4.5", "glm45", "zai-org/glm-4.5"),
        source_urls=("https://huggingface.co/zai-org/GLM-4.5/blob/main/config.json",),
        hidden_size=5120,
        num_layers=92,
        dense_intermediate_size=12288,
        expert_intermediate_size=1536,
        num_experts=160,
        top_k=8,
        shared_experts=1,
        shared_intermediate_size=1536,
        routed_width=5120,
        moe_common_active_params=0,
        layer_groups=(
            LayerGroup("GQA + dense FFN", 3, "0-2", GLM45_ATTN, "dense"),
            LayerGroup("GQA + MoE", 89, "3-91", GLM45_ATTN, "moe"),
        ),
    ),
    ModelSpec(
        key="deepseek-v3",
        display_name="DeepSeek-V3",
        aliases=("deepseekv3", "deepseek-ai/deepseek-v3"),
        source_urls=("https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json",),
        hidden_size=7168,
        num_layers=61,
        dense_intermediate_size=18432,
        expert_intermediate_size=2048,
        num_experts=256,
        top_k=8,
        shared_experts=1,
        shared_intermediate_size=2048,
        routed_width=7168,
        moe_common_active_params=0,
        layer_groups=(
            LayerGroup("MLA + dense FFN", 3, "0-2", DEEPSEEK_V3_ATTN, "dense"),
            LayerGroup("MLA + MoE", 58, "3-60", DEEPSEEK_V3_ATTN, "moe"),
        ),
    ),
    ModelSpec(
        key="kimi-k2",
        display_name="Kimi-K2-Instruct",
        aliases=("kimi k2", "kimik2", "moonshotai/kimi-k2-instruct"),
        source_urls=("https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/config.json",),
        hidden_size=7168,
        num_layers=61,
        dense_intermediate_size=18432,
        expert_intermediate_size=2048,
        num_experts=384,
        top_k=8,
        shared_experts=1,
        shared_intermediate_size=2048,
        routed_width=7168,
        moe_common_active_params=0,
        layer_groups=(
            LayerGroup("MLA + dense FFN", 1, "0", KIMI_K2_ATTN, "dense"),
            LayerGroup("MLA + MoE", 60, "1-60", KIMI_K2_ATTN, "moe"),
        ),
    ),
    ModelSpec(
        key="qwen3-235b",
        display_name="Qwen3-235B-A22B",
        aliases=("qwen3-235b-a22b", "qwen/qwen3-235b-a22b"),
        source_urls=("https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json",),
        hidden_size=4096,
        num_layers=94,
        dense_intermediate_size=0,
        expert_intermediate_size=1536,
        num_experts=128,
        top_k=8,
        shared_experts=0,
        shared_intermediate_size=0,
        routed_width=4096,
        moe_common_active_params=0,
        layer_groups=(
            LayerGroup("GQA + MoE", 94, "0-93", QWEN3_ATTN, "moe"),
        ),
    ),
)


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", name.strip().lower()).strip("-")


_BY_ALIAS: dict[str, ModelSpec] = {}
for _spec in MODEL_SPECS:
    for _alias in (_spec.key, _spec.display_name, *_spec.aliases):
        _BY_ALIAS[_normalize(_alias)] = _spec


def resolve_model(name: str) -> ModelSpec:
    """Resolve a case/punctuation-insensitive model alias."""

    try:
        return _BY_ALIAS[_normalize(name)]
    except KeyError as error:
        supported = ", ".join(spec.key for spec in MODEL_SPECS)
        raise ValueError(f"unknown model {name!r}; supported models: {supported}") from error


def supported_models() -> tuple[str, ...]:
    return tuple(spec.key for spec in MODEL_SPECS)

