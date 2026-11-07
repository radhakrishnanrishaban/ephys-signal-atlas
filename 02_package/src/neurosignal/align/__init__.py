"""Cross-session alignment. Procrustes now; FA alignment and NoMAD-lite land here in M4."""

from .procrustes import align_sessions_sbp_procrustes  # noqa: F401

__all__ = ["align_sessions_sbp_procrustes"]
