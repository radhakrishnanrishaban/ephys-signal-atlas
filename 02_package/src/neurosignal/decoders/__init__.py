"""Decoders: the one linear spec every evaluation uses (D-15, D-18), the pooled MLP (D-17), the LINK-paper baseline."""

from .baseline import paper_ridge_r2, spec_r2_on_paper_split  # noqa: F401
from .linear import SPEC, DecoderSpec, GramRidge, fit_decoder, make_xy, same_day_r2, session_design  # noqa: F401
from .mlp import MLP_SPEC, PooledDecoders, fit_mlp, fit_pca_joint, fit_pooled_decoders, score_pooled, score_sessions  # noqa: F401

__all__ = [
    "paper_ridge_r2",
    "spec_r2_on_paper_split",
    "SPEC",
    "DecoderSpec",
    "GramRidge",
    "fit_decoder",
    "make_xy",
    "same_day_r2",
    "session_design",
    "MLP_SPEC",
    "PooledDecoders",
    "fit_mlp",
    "fit_pca_joint",
    "fit_pooled_decoders",
    "score_pooled",
    "score_sessions",
]
