"""
neurosignal.features.fixed — normaliser v2 (τ 2 s, reset each session) in integer arithmetic (EDA-11, EXP-006).

The device form of `causal_zscore_ewma_v2(carry=False)` followed by `causal_smooth_ewma`: int32 state,
int64 products, round-to-nearest shifts, saturation instead of overflow. Formats, from the LINK SBP range
(0.475 to 52.3; end-of-session variance 0.0146 to 17.5):

    x, μ        int32  Q16.16
    var         uint32 Q12.20, saturating
    constants   Q0.31  (α, 1 − α, α(1 − α), smoother 0.2, warm-up 1/n)
    σ = √var    Q.20   (64-bit integer sqrt of var · 2²⁰, at least 1 LSB)
    z           Q3.12, clipped to ±6
    y           int32  Q3.28 (one-pole smoother)
    output      int16  Q3.12 (the decoder reads output / 2¹²)

State per channel: μ, var, y = 12 bytes. The warm-up counter and the constants are shared by all channels.

Every integer primitive goes through `Ops`, which counts it once per channel as it runs, so the EXP-006
cost table comes from this code and not from a hand count. `isqrt_bitwise` and `div_round_bitwise` are the
digit-by-digit forms a core without sqrt or 64/32-bit divide hardware would run; they return the same
integers as the fast paths and exist to count those steps.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .causal import alpha_from_tau

__all__ = [
    "Q_X",
    "Q_V",
    "Q_Z",
    "STATE_BYTES_PER_CH",
    "KINDS",
    "Ops",
    "isqrt",
    "isqrt_bitwise",
    "div_round",
    "div_round_bitwise",
    "to_q16",
    "FixedPointNormalizer",
    "fixed_point_sessions",
    "normalize_sessions_sbp_fixed",
]

Q_X, Q_V, Q_C, Q_Z, Q_Y = 16, 20, 31, 12, 28  # fractional bits: x and μ, var, constants, z and output, smoother
I32_MIN, I32_MAX, U32_MAX = -(2**31), 2**31 - 1, 2**32 - 1
STATE_BYTES_PER_CH = 3 * 4  # μ int32, var uint32, y int32
KINDS = ("add", "mul", "shift", "cmp", "sqrt", "div")


def _q31(c: float) -> int:
    return int(round(c * 2**Q_C))


class Ops:
    """Integer primitives on int64 arrays (one lane per channel); each call counts once, i.e. once per channel."""

    def __init__(self):
        self.count: Dict[str, int] = dict.fromkeys(KINDS, 0)

    def add(self, a, b):
        self.count["add"] += 1
        return a + b

    def sub(self, a, b):
        self.count["add"] += 1
        return a - b

    def mul(self, a, b):
        self.count["mul"] += 1
        return a * b

    def shl(self, a, s: int):
        self.count["shift"] += 1
        return a << s

    def shr_round(self, a, s: int):
        """Round-to-nearest right shift: (a + 2^(s−1)) >> s."""
        self.count["add"] += 1
        self.count["shift"] += 1
        return (a + (1 << (s - 1))) >> s

    def at_most(self, a, hi: int):
        self.count["cmp"] += 1
        return np.minimum(a, hi)

    def at_least(self, a, lo: int):
        self.count["cmp"] += 1
        return np.maximum(a, lo)

    def sqrt(self, a):
        self.count["sqrt"] += 1
        return isqrt(a)

    def div_round(self, a, b):
        """round(a / b) for b > 0: one shift and one add for the rounding, then one division."""
        self.count["shift"] += 1
        self.count["add"] += 1
        self.count["div"] += 1
        return div_round(a, b)


def isqrt(n: np.ndarray) -> np.ndarray:
    """floor(√n) for 0 ≤ n < 2⁵³, exact (float sqrt, then one correction step each way)."""
    n = np.asarray(n, dtype=np.int64)
    r = np.floor(np.sqrt(n.astype(np.float64))).astype(np.int64)
    r -= (r * r > n).astype(np.int64)
    r += ((r + 1) * (r + 1) <= n).astype(np.int64)
    return r


def isqrt_bitwise(n: np.ndarray, ops: Ops, bits: int = 52) -> np.ndarray:
    """floor(√n) for 0 ≤ n < 2^bits (bits even), digit by digit: bits/2 rounds of add, compare, subtract, shift."""
    rem, res, bit = np.asarray(n, dtype=np.int64).copy(), np.zeros_like(n, dtype=np.int64), 1 << (bits - 2)
    for _ in range(bits // 2):
        trial = ops.add(res, bit)
        ge = rem >= trial
        ops.count["cmp"] += 1
        rem = np.where(ge, ops.sub(rem, trial), rem)
        res = res >> 1
        res = np.where(ge, ops.add(res, bit), res)
        bit >>= 2
        ops.count["shift"] += 2  # res >> 1, bit >> 2
    return res


def div_round(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """round(a / b), halves up, for b > 0."""
    return np.floor_divide(a + (b >> 1), b)


def div_round_bitwise(a: np.ndarray, b: np.ndarray, ops: Ops, qbits: int = 16) -> np.ndarray:
    """
    div_round(a, b) for b > 0 by restoring division on |a|, saturated to ±(2^qbits − 1): enough when the result is
    then clipped below 2^qbits, as z is (±6 · 2¹² = 24,576 < 2¹⁶). qbits rounds of shift, compare, subtract, set bit.
    Halves round up, as in div_round: floor((a + h) / b) with h = b >> 1 is −floor((|a| + b − 1 − h) / b) for a < 0.
    """
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    neg = a < 0
    ops.count["cmp"] += 1
    h = b >> 1
    ops.count["shift"] += 1
    bias = np.where(neg, b - 1 - h, h)
    ops.count["add"] += 2                       # b − 1 − h
    num = ops.add(np.where(neg, -a, a), bias)
    ops.count["add"] += 1                       # the negation
    big = num >= ops.shl(b, qbits)
    ops.count["cmp"] += 1
    q, rem = np.zeros_like(num), np.where(big, 0, num)
    for i in reversed(range(qbits)):
        t = ops.shl(b, i)
        ge = rem >= t
        ops.count["cmp"] += 1
        rem = np.where(ge, ops.sub(rem, t), rem)
        q = np.where(ge, ops.add(q, 1 << i), q)
    q = np.where(big, 2**qbits - 1, q)
    ops.count["add"] += 1                       # sign restore
    return np.where(neg, -q, q)


def to_q16(X: np.ndarray) -> Tuple[np.ndarray, int]:
    """Float SBP → int32 Q16.16 (rounded, saturated); returns (x, number of saturated samples)."""
    raw = np.round(np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0) * 2**Q_X)
    sat = int(np.count_nonzero((raw < I32_MIN) | (raw > I32_MAX)))
    return np.clip(raw, I32_MIN, I32_MAX).astype(np.int64), sat


class FixedPointNormalizer:
    """
    One bin at a time, all channels at once: call start_session() at each session start, then update(x) with
    x an int Q16.16 vector; returns the int16 Q3.12 output. Same algorithm as the float reference: causal Welford
    warm-up (running-variance form, var += (δ(x − μ) − var) / n), then var = (1 − α) var + α(1 − α) δ², z clipped,
    one-pole smoother restarted each session.

    ops_warmup / ops_steady count the per-channel primitives in each phase; ops_shared counts the per-bin work all
    channels share (the warm-up reciprocal 1/n). saturated counts clamps other than the ±clip; clipped counts
    channel-bins at ±clip after the warm-up (the scored bins).
    """

    def __init__(self, n_channels: int, tau_s: float = 2.0, warmup: int = 300, clip: float = 6.0, alpha_smooth: float = 0.2):
        alpha = alpha_from_tau(tau_s)
        self.a, self.one_minus_a, self.a_one_minus_a = _q31(alpha), _q31(1.0 - alpha), _q31(alpha * (1.0 - alpha))
        self.smooth = _q31(alpha_smooth)
        self.z_max = int(round(clip * 2**Q_Z))
        self.warmup = int(warmup)
        self.ops_warmup, self.ops_steady, self.ops_shared = Ops(), Ops(), Ops()
        self.bins = {"warmup": 0, "steady": 0}
        self.saturated, self.clipped = 0, 0
        self.n_channels = n_channels
        self.start_session()

    def start_session(self) -> None:
        self.mu = np.zeros(self.n_channels, dtype=np.int64)
        self.var = np.zeros(self.n_channels, dtype=np.int64)
        self.y = np.zeros(self.n_channels, dtype=np.int64)
        self.n = 0

    def _saturate(self, a: np.ndarray, o: Ops) -> np.ndarray:
        self.saturated += int(np.count_nonzero(a > U32_MAX))
        return o.at_most(a, U32_MAX)

    def update(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.int64)
        if self.n < self.warmup:
            o = self.ops_warmup
            self.bins["warmup"] += 1
            self.n += 1
            self.ops_shared.count["div"] += 1
            r = (2**Q_C + self.n // 2) // self.n  # 1/n in Q0.31, one division per bin for all channels
            d = o.sub(x, self.mu)
            self.mu = o.add(self.mu, o.shr_round(o.mul(d, r), Q_C))
            e = self._saturate(o.shr_round(o.mul(d, o.sub(x, self.mu)), 2 * Q_X - Q_V), o)
            var = o.add(self.var, o.shr_round(o.mul(o.sub(e, self.var), r), Q_C))
            self.var = o.at_least(self._saturate(var, o), 0)
        else:
            o = self.ops_steady
            self.bins["steady"] += 1
            d = o.sub(x, self.mu)
            self.mu = o.add(self.mu, o.shr_round(o.mul(d, self.a), Q_C))
            d2 = self._saturate(o.shr_round(o.mul(d, d), 2 * Q_X - Q_V), o)
            # (1 − α) + α(1 − α) < 1, so the sum stays below 2⁶³ and the result below 2³²
            self.var = o.shr_round(o.add(o.mul(self.var, self.one_minus_a), o.mul(d2, self.a_one_minus_a)), Q_C)
        sigma = o.at_least(o.sqrt(o.shl(self.var, Q_V)), 1)           # √var in Q.20
        z = o.div_round(o.shl(o.sub(x, self.mu), Q_V + Q_Z - Q_X), sigma)  # Q16 · 2¹⁶ / Q20 → Q3.12
        z = o.at_least(o.at_most(z, self.z_max), -self.z_max)
        if o is self.ops_steady:
            self.clipped += int(np.count_nonzero(np.abs(z) == self.z_max))
        self.y = o.add(self.y, o.shr_round(o.mul(o.sub(o.shl(z, Q_Y - Q_Z), self.y), self.smooth), Q_C))
        return o.shr_round(self.y, Q_Y - Q_Z)

    def ops_per_channel_bin(self) -> Dict[str, Dict[str, float]]:
        """Primitives per channel per bin in each phase (data-independent, so every bin of a phase counts the same)."""
        return {
            phase: {k: ops.count[k] / max(self.bins[phase], 1) for k in KINDS}
            for phase, ops in (("warmup", self.ops_warmup), ("steady", self.ops_steady))
        }


def fixed_point_sessions(
    sessions_sbp: List[np.ndarray],
    tau_s: float = 2.0,
    warmup: int = 300,
    clip: float = 6.0,
    alpha_smooth: float = 0.2,
) -> Tuple[List[np.ndarray], FixedPointNormalizer, int]:
    """Run one normaliser over each session (reset at each start); returns (int16 outputs, normaliser, input saturations)."""
    norm, out, input_sat = None, [], 0
    for sbp in sessions_sbp:
        x, sat = to_q16(sbp)
        input_sat += sat
        if norm is None:
            norm = FixedPointNormalizer(x.shape[1], tau_s=tau_s, warmup=warmup, clip=clip, alpha_smooth=alpha_smooth)
        norm.start_session()
        out.append(np.stack([norm.update(row) for row in x]).astype(np.int16))
    return out, norm, input_sat


def normalize_sessions_sbp_fixed(
    sessions_sbp: List[np.ndarray],
    tau_s: float = 2.0,
    warmup: int = 300,
    clip: float = 6.0,
    alpha_smooth: float = 0.2,
) -> List[np.ndarray]:
    """What the decoder reads: the fixed-point output as float, output / 2¹², per session."""
    out, _, _ = fixed_point_sessions(sessions_sbp, tau_s=tau_s, warmup=warmup, clip=clip, alpha_smooth=alpha_smooth)
    return [o.astype(np.float64) / 2**Q_Z for o in out]
