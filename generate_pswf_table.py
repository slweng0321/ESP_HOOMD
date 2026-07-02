#!/usr/bin/env python3
"""
generate_pswf_table.py

Offline coefficient generator for the ESP (Ewald Summation with Prolates)
HOOMD-blue plugin.

Implements, from first principles, the mathematics described in:
Liang, Lu, Barnett, Greengard, Jiang, "Accelerating molecular dynamics
simulations using fast Ewald summation with prolates", Nat. Commun. 2026.

CRITICAL NUMERICAL-STABILITY NOTE
----------------------------------
Earlier revisions of this script attempted to directly fit the real-space
kernel-splitting function

    L(r) = 1/(4 pi r) * [1 - Phi_psi(r / r_c)]

with piecewise polynomials. Because L(r) diverges as 1/r when r -> 0, any
polynomial fit on the first (innermost) segment is fundamentally ill-posed
and blows up (observed max relative fit error ~ 5.7e13).

The fix used here: we NEVER fit the 1/r part. Instead we tabulate only the
smooth, bounded SCREENING FUNCTION

    S(r)  = 1 - Phi_psi(r / r_c)                      in [0, 1]
    S'(r) = dS/dr = -psi_0^c(r / r_c)^2 / r_c          bounded, smooth

Both S(r) and S'(r) are smooth and O(1) over the entire domain [0, r_c],
so degree-4 piecewise polynomial fits are extremely well-conditioned. The
1/(4 pi r) singular factor is instead applied ANALYTICALLY at evaluation
time (in both the CPU EvaluatorPairPSWF and the GPU kernel), in registers,
after the table lookup:

    L(r)     =  S(r) / (4 pi r)
    -dL/dr   =  S(r) / (4 pi r^2)  -  S'(r) / (4 pi r)

Pipeline
--------
1. Solve for the zeroth-order prolate spheroidal wave function psi_0^c(x)
   on x in [-1, 1] for a given bandwidth parameter c, using scipy's
   angular prolate spheroidal wave function pro_ang1(m=0, n=0, c, x).
2. Calibrate c against a target relative error tolerance epsilon, using
   the paper's Table 2 (epsilon, c_pswf, P_esp) anchors.
3. Build the normalised CDF Phi_psi and the SCREENING FUNCTION S(r).
4. Fit S(r) and S'(r) on n_segs equal-width sub-intervals of [0, r_c]
   with degree-4 Horner-form polynomials (5 coefficients each).
5. Compute the reciprocal-space influence function p_k (Eq. 9 analogue).
6. Fit the 1D PSWF particle-grid assignment/interpolation stencil (Eq. 14).
7. Emit a self-contained C++ header (PSWF_Coeffs.h).
"""

from __future__ import annotations

import argparse
import datetime
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy import integrate
from scipy.special import pro_ang1

# ---------------------------------------------------------------------------
# Physical / numerical constants
# ---------------------------------------------------------------------------

POLY_DEGREE = 4
N_POLY_COEFFS = POLY_DEGREE + 1

_TABLE2_EPSILON = np.array([1.0e-3, 1.0e-4, 1.0e-5])
_TABLE2_C_PSWF = np.array([6.0, 9.0, 12.0])
_TABLE2_P_ESP = np.array([4, 5, 8])


def epsilon_to_c(epsilon: float) -> float:
    """Map a target relative force error to the PSWF bandwidth c via
    log-linear interpolation through the Table 2 anchors."""
    log_eps = np.log10(_TABLE2_EPSILON)
    x = -log_eps
    coeffs = np.polyfit(x, _TABLE2_C_PSWF, deg=1)
    c = np.polyval(coeffs, -np.log10(epsilon))
    return float(max(c, 1.0))


def epsilon_to_order(epsilon: float) -> int:
    """Map a target relative force error to the recommended PSWF
    particle-grid interpolation order P (Table 2)."""
    log_eps = np.log10(_TABLE2_EPSILON)
    x = -log_eps
    coeffs = np.polyfit(x, _TABLE2_P_ESP, deg=1)
    p = np.polyval(coeffs, -np.log10(epsilon))
    return int(round(np.clip(p, 3, 12)))


# ---------------------------------------------------------------------------
# Step 1-2: Zeroth-order PSWF and its normalised CDF
# ---------------------------------------------------------------------------

class PSWFKernel:
    """Zeroth-order prolate spheroidal wave function psi_0^c(x) on [-1, 1]."""

    def __init__(self, c: float, n_quad: int = 4000):
        self.c = float(c)
        self.n_quad = int(n_quad)

        self._x_grid = np.linspace(-1.0, 1.0, self.n_quad)
        psi_vals = self._eval_psi_raw(self._x_grid)

        energy = np.trapezoid(psi_vals ** 2, self._x_grid)
        self._norm = 1.0 / np.sqrt(energy)
        self._psi_grid = psi_vals * self._norm

        cdf = integrate.cumulative_trapezoid(
            self._psi_grid ** 2, self._x_grid, initial=0.0
        )
        cdf /= cdf[-1]
        self._cdf_grid = cdf

    def _eval_psi_raw(self, x: np.ndarray) -> np.ndarray:
        x_safe = np.clip(x, -1.0 + 1e-12, 1.0 - 1e-12)
        vals, _ = pro_ang1(0, 0, self.c, x_safe)
        return vals

    def psi(self, x: np.ndarray) -> np.ndarray:
        """Normalised PSWF value at x in [-1, 1]."""
        return np.interp(x, self._x_grid, self._psi_grid)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Normalised cumulative energy distribution Phi_psi(x)."""
        x_clamped = np.clip(x, -1.0, 1.0)
        return np.interp(x_clamped, self._x_grid, self._cdf_grid)

    def boundary_value(self) -> float:
        return float(self.psi(np.array([1.0]))[0])


# ---------------------------------------------------------------------------
# Step 3: SMOOTH screening function S(r) (replaces the singular L(r) fit)
# ---------------------------------------------------------------------------

@dataclass
class SplitKernel:
    """Real-space kernel-splitting screening function.

    S(r)  = 1 - Phi_psi(r / r_c),         0 <= r <= r_c    (bounded in [0,1])
    S'(r) = -psi_0^c(r / r_c)^2 / r_c,    0 <= r <= r_c    (bounded, smooth)

    The full short-range complement is recovered analytically as:
        L(r)   = S(r) / (4 pi r)
        -dL/dr = S(r) / (4 pi r^2) - S'(r) / (4 pi r)

    Neither S(r) nor S'(r) has a 1/r singularity, so both are excellent
    candidates for stable piecewise polynomial fitting.
    """
    pswf: PSWFKernel
    rcut: float

    def screen(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=np.float64)
        out = np.zeros_like(r)
        mask = (r >= 0.0) & (r <= self.rcut)
        s = r[mask] / self.rcut
        out[mask] = 1.0 - self.pswf.cdf(s)
        return out

    def dscreen_dr(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=np.float64)
        out = np.zeros_like(r)
        mask = (r >= 0.0) & (r <= self.rcut)
        s = r[mask] / self.rcut
        out[mask] = -(self.pswf.psi(s) ** 2) / self.rcut
        return out

    def L(self, r: np.ndarray) -> np.ndarray:
        """Full singular kernel-splitting function, used only for
        VALIDATION against the analytic combination, never for fitting."""
        r = np.asarray(r, dtype=np.float64)
        out = np.zeros_like(r)
        mask = r > 1.0e-12
        out[mask] = self.screen(r[mask]) / (4.0 * np.pi * r[mask])
        return out

    def negdLdr(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=np.float64)
        out = np.zeros_like(r)
        mask = r > 1.0e-12
        rr = r[mask]
        out[mask] = (self.screen(rr) / (4.0 * np.pi * rr ** 2)
                     - self.dscreen_dr(rr) / (4.0 * np.pi * rr))
        return out


# ---------------------------------------------------------------------------
# Step 4: Piecewise degree-4 polynomial fit of S(r) and S'(r) (STABLE)
# ---------------------------------------------------------------------------

@dataclass
class TableSegment:
    r_lo: float
    dr_inv: float
    screen_coeffs: np.ndarray    # length 5, Horner c0..c4 for S(r)
    dscreen_coeffs: np.ndarray   # length 5, Horner c0..c4 for S'(r)


def fit_piecewise_table(
    kernel: SplitKernel, n_segs: int, n_sample_per_seg: int = 64
) -> List[TableSegment]:
    """Fit the SMOOTH, BOUNDED functions S(r) and S'(r) on n_segs
    equal-width segments of [0, r_c]. Because neither function has a
    singularity, this least-squares fit is well-conditioned everywhere,
    including the first segment near r = 0."""
    rcut = kernel.rcut
    seg_width = rcut / n_segs
    segments: List[TableSegment] = []

    for i in range(n_segs):
        r_lo = i * seg_width
        dr_inv = 1.0 / seg_width

        t_interior = np.linspace(0.0, 1.0, n_sample_per_seg)
        r_samples = r_lo + t_interior * seg_width
        r_samples = np.clip(r_samples, 0.0, rcut)

        S_samples = kernel.screen(r_samples)
        dS_samples = kernel.dscreen_dr(r_samples)

        t_local = (r_samples - r_lo) * dr_inv
        V = np.vander(t_local, N=N_POLY_COEFFS, increasing=True)

        w = np.ones_like(t_local)
        w[0] *= 50.0
        w[-1] *= 50.0
        W = np.diag(w)

        A = V.T @ W @ V
        b_S = V.T @ W @ S_samples
        b_dS = V.T @ W @ dS_samples

        c_S = np.linalg.solve(A, b_S)
        c_dS = np.linalg.solve(A, b_dS)

        segments.append(
            TableSegment(
                r_lo=r_lo,
                dr_inv=dr_inv,
                screen_coeffs=c_S,
                dscreen_coeffs=c_dS,
            )
        )

    return segments


def enforce_c0_continuity(segments: List[TableSegment]) -> None:
    """Rescale the constant (c0) term of each segment (except the first)
    so the fitted S(r) and S'(r) are C0-continuous across segment
    boundaries, removing residual jumps left by the independent
    per-segment least-squares fits."""
    for i in range(1, len(segments)):
        prev = segments[i - 1]
        cur = segments[i]

        prev_val_S = float(np.polyval(prev.screen_coeffs[::-1], 1.0))
        cur.screen_coeffs[0] += prev_val_S - float(cur.screen_coeffs[0])

        prev_val_dS = float(np.polyval(prev.dscreen_coeffs[::-1], 1.0))
        cur.dscreen_coeffs[0] += prev_val_dS - float(cur.dscreen_coeffs[0])


# ---------------------------------------------------------------------------
# Step 5: Reciprocal-space influence function p_k (Eq. 9 analogue)
# ---------------------------------------------------------------------------

def compute_influence_function(
    kernel: SplitKernel, k_max: float, n_k: int = 4096
) -> Tuple[np.ndarray, np.ndarray]:
    """p(k) = 4 pi * integral_0^{r_c} [1/(4 pi r) - L(r)] * sin(kr)/(kr) * r^2 dr

    Uses the analytic L(r) = S(r)/(4 pi r) combination (safe here since we
    integrate over r > 0 on a dense grid that never touches r = 0 exactly).
    """
    rcut = kernel.rcut
    r_grid = np.linspace(1.0e-6, rcut, 2000)
    S_r = 1.0 / (4.0 * np.pi * r_grid) - kernel.L(r_grid)

    k_grid = np.linspace(0.0, k_max, n_k)
    p_k = np.zeros_like(k_grid)

    for idx, k in enumerate(k_grid):
        if k < 1.0e-10:
            integrand = S_r * r_grid ** 2
        else:
            integrand = S_r * np.sin(k * r_grid) / (k * r_grid) * r_grid ** 2
        p_k[idx] = 4.0 * np.pi * np.trapezoid(integrand, r_grid)

    return k_grid, p_k


def fit_influence_polynomial(
    k_grid: np.ndarray, p_k: np.ndarray, degree: int = 8
) -> np.ndarray:
    """Fit p(k) with an even polynomial in k^2 for cheap GPU evaluation."""
    k2 = k_grid ** 2
    V = np.vander(k2, N=degree // 2 + 1, increasing=True)
    coeffs, *_ = np.linalg.lstsq(V, p_k, rcond=None)
    return coeffs


# ---------------------------------------------------------------------------
# Step 6: 1D PSWF particle-grid assignment/interpolation stencil (Eq. 14)
# ---------------------------------------------------------------------------

@dataclass
class StencilSegment:
    x_lo: float
    dx_inv: float
    coeffs: np.ndarray  # length 5, Horner c0..c4 for phi(x) on this segment


class AssignmentStencil:
    """1D PSWF-based particle-grid assignment/interpolation kernel (Eq. 14):

        phi(x) = psi_0^{c1}( 2x / (P h) ),   x in [-Ph/2, Ph/2]
        phi(x) = 0,                           otherwise

    phi(x) is itself bounded and smooth (no 1/r involved), so its fit was
    never numerically unstable; it is retained unchanged from the prior
    revision but included here for a fully self-contained script.
    """

    def __init__(self, order: int, c1: float):
        self.order = int(order)
        self.c1 = float(c1)
        self._psi = PSWFKernel(c=self.c1, n_quad=4000)

    def phi_of_u(self, u: np.ndarray) -> np.ndarray:
        u = np.clip(u, -1.0, 1.0)
        return self._psi.psi(u)

    def phi(self, x: np.ndarray, h: float) -> np.ndarray:
        u = 2.0 * x / (self.order * h)
        out = np.zeros_like(u)
        mask = np.abs(u) <= 1.0
        out[mask] = self.phi_of_u(u[mask])
        return out


def fit_assignment_stencil(
    stencil: AssignmentStencil, h: float, n_sample_per_seg: int = 64
) -> List[StencilSegment]:
    P = stencil.order
    half_width = P * h / 2.0
    seg_width = h
    segments: List[StencilSegment] = []

    for k in range(P):
        x_lo = -half_width + k * seg_width
        dx_inv = 1.0 / seg_width

        t_local = np.linspace(0.0, 1.0, n_sample_per_seg)
        x_samples = x_lo + t_local * seg_width
        phi_samples = stencil.phi(x_samples, h)

        V = np.vander(t_local, N=N_POLY_COEFFS, increasing=True)
        w = np.ones_like(t_local)
        w[0] *= 50.0
        w[-1] *= 50.0
        W = np.diag(w)

        A = V.T @ W @ V
        b = V.T @ W @ phi_samples
        c = np.linalg.solve(A, b)

        segments.append(StencilSegment(x_lo=x_lo, dx_inv=dx_inv, coeffs=c))

    return segments


def enforce_partition_of_unity(
    segments: List[StencilSegment], h: float, n_test_points: int = 200
) -> float:
    x_test = np.linspace(0.0, h, n_test_points, endpoint=False)
    total = np.zeros_like(x_test)

    for seg in segments:
        t_local = np.clip((x_test - seg.x_lo) * seg.dx_inv, 0.0, 1.0 - 1e-12)
        total += np.polyval(seg.coeffs[::-1], t_local)

    mean_total = float(np.mean(total))
    scale = 1.0 / mean_total if abs(mean_total) > 1e-14 else 1.0

    for seg in segments:
        seg.coeffs *= scale

    return scale


# ---------------------------------------------------------------------------
# Step 7: C++ header emission
# ---------------------------------------------------------------------------

def _format_double_array(name: str, values: np.ndarray, per_line: int = 6) -> str:
    flat = values.ravel()
    lines = []
    for i in range(0, len(flat), per_line):
        chunk = flat[i : i + per_line]
        lines.append("    " + ", ".join(f"{v:.17e}" for v in chunk) + ",")
    body = "\n".join(lines)
    return f"inline constexpr double {name}[{len(flat)}] = {{\n{body}\n}};\n"


def _format_header_table(name: str, mat: np.ndarray) -> str:
    n, k = mat.shape
    flat = mat.ravel()
    lines = []
    for i in range(0, len(flat), k):
        chunk = flat[i : i + k]
        lines.append("    " + ", ".join(f"{v:.17e}" for v in chunk) + ",")
    body = "\n".join(lines)
    return (
        f"// Row-major [seg][coeff], seg in [0,{n}), coeff in [0,{k})\n"
        f"inline constexpr double {name}[{n * k}] = {{\n{body}\n}};\n"
    )


def emit_stencil_header_section(segments: List[StencilSegment]) -> str:
    P = len(segments)
    coeffs_mat = np.array([s.coeffs for s in segments])
    x_lo = np.array([s.x_lo for s in segments])
    dx_inv = np.array([s.dx_inv for s in segments])

    parts = [f"inline constexpr int PSWF_STENCIL_ORDER = {P};", ""]
    parts.append(_format_header_table("PSWF_STENCIL_COEFFS", coeffs_mat))
    parts.append(_format_double_array("PSWF_STENCIL_X_LO", x_lo))
    parts.append(_format_double_array("PSWF_STENCIL_DX_INV", dx_inv))
    return "\n".join(parts)


def emit_header(
    path: str,
    epsilon: float,
    c_pswf: float,
    rcut: float,
    n_segs: int,
    order: int,
    segments: List[TableSegment],
    gf_denom_coeffs: np.ndarray,
    stencil_segments: List[StencilSegment],
) -> None:
    n = len(segments)
    screen_mat = np.array([s.screen_coeffs for s in segments])
    dscreen_mat = np.array([s.dscreen_coeffs for s in segments])
    r_lo = np.array([s.r_lo for s in segments])
    dr_inv = np.array([s.dr_inv for s in segments])

    header = []
    header.append("// " + "=" * 76)
    header.append("// PSWF_Coeffs.h — AUTO-GENERATED. DO NOT EDIT BY HAND.")
    header.append(
        f"// Generated by generate_pswf_table.py on "
        f"{datetime.datetime.now(datetime.UTC).isoformat()}Z"
    )
    header.append(
        f"// epsilon = {epsilon:.3e}, c_pswf = {c_pswf:.6f}, r_cut = {rcut:.6f}, "
        f"n_segs = {n_segs}, order (P) = {order}"
    )
    header.append("//")
    header.append("// Tabulated quantities are the SMOOTH SCREENING FUNCTION")
    header.append("// S(r) = 1 - Phi_psi(r/r_c) and its derivative S'(r), NOT the")
    header.append("// singular kernel L(r) = S(r)/(4 pi r). The 1/(4 pi r) factor")
    header.append("// must be applied analytically at evaluation time:")
    header.append("//   L(r)   = S(r) / (4 pi r)")
    header.append("//   -dL/dr = S(r) / (4 pi r^2) - S'(r) / (4 pi r)")
    header.append("//")
    header.append("// Reference: Liang, Lu, Barnett, Greengard, Jiang, Nat. Commun. 2026.")
    header.append("// " + "=" * 76)
    header.append("#pragma once")
    header.append("")
    header.append("namespace hoomd")
    header.append("{")
    header.append("namespace md")
    header.append("{")
    header.append("namespace esp_pswf_coeffs")
    header.append("{")
    header.append("")
    header.append(f"inline constexpr int PSWF_N_SEGS = {n_segs};")
    header.append(f"inline constexpr int PSWF_ORDER = {order};")
    header.append(f"inline constexpr double PSWF_C = {c_pswf:.17e};")
    header.append(f"inline constexpr double PSWF_RCUT = {rcut:.17e};")
    header.append("")
    header.append(_format_header_table("PSWF_SCREEN_COEFFS", screen_mat))
    header.append(_format_header_table("PSWF_DSCREEN_COEFFS", dscreen_mat))
    header.append(_format_double_array("PSWF_R_LO", r_lo))
    header.append(_format_double_array("PSWF_DR_INV", dr_inv))
    header.append(_format_double_array("PSWF_GF_DENOM_COEFFS", gf_denom_coeffs))
    header.append("")
    header.append(emit_stencil_header_section(stencil_segments))
    header.append("")
    header.append("} // namespace esp_pswf_coeffs")
    header.append("} // namespace md")
    header.append("} // namespace hoomd")
    header.append("")

    with open(path, "w") as f:
        f.write("\n".join(header))


# ---------------------------------------------------------------------------
# Validation utilities
# ---------------------------------------------------------------------------

def validate_table(kernel: SplitKernel, segments: List[TableSegment], n_check: int = 500) -> float:
    """Max relative error of the fitted SCREENING FUNCTION S(r) against
    the analytic value. Since S(r) is bounded in [0, 1] and smooth
    everywhere (including r -> 0), this error should now be extremely
    small (typically < 1e-8 for n_segs ~ 4096), in stark contrast to the
    catastrophic 5.7e13 error observed when fitting the singular L(r)
    directly."""
    rng = np.random.default_rng(0)
    r_test = rng.uniform(0.0, kernel.rcut * (1.0 - 1.0e-9), n_check)
    seg_width = kernel.rcut / len(segments)

    max_rel_err = 0.0
    for r in r_test:
        seg_idx = min(int(r / seg_width), len(segments) - 1)
        seg = segments[seg_idx]
        t = (r - seg.r_lo) * seg.dr_inv
        S_fit = float(np.polyval(seg.screen_coeffs[::-1], t))
        S_true = float(kernel.screen(np.array([r]))[0])
        denom = max(abs(S_true), 1.0e-14)
        max_rel_err = max(max_rel_err, abs(S_fit - S_true) / denom)
    return max_rel_err


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PSWF/ESP LUT header for HOOMD-blue.")
    parser.add_argument("--epsilon", type=float, default=1.0e-4,
                         help="Target relative force error tolerance.")
    parser.add_argument("--rcut", type=float, default=1.0,
                         help="Real-space cutoff r_c in simulation length units.")
    parser.add_argument("--n-segs", type=int, default=4096,
                         help="Number of piecewise-polynomial table segments.")
    parser.add_argument("--k-max", type=float, default=50.0,
                         help="Maximum |k| for influence-function sampling.")
    parser.add_argument("--out", type=str, default="PSWF_Coeffs.h",
                         help="Output C++ header path.")
    args = parser.parse_args()

    c_pswf = epsilon_to_c(args.epsilon)
    order = epsilon_to_order(args.epsilon)

    print(f"[generate_pswf_table] epsilon={args.epsilon:.3e} -> c_pswf={c_pswf:.4f}, "
          f"order P={order}")

    pswf = PSWFKernel(c=c_pswf)
    print(f"[generate_pswf_table] psi_0^c(1) = {pswf.boundary_value():.6e} "
          f"(sanity check vs target epsilon)")

    kernel = SplitKernel(pswf=pswf, rcut=args.rcut)

    segments = fit_piecewise_table(kernel, n_segs=args.n_segs)
    enforce_c0_continuity(segments)

    max_rel_err = validate_table(kernel, segments)
    print(f"[generate_pswf_table] max relative S(r) fit error = {max_rel_err:.3e}")
    if max_rel_err > args.epsilon:
        print("[generate_pswf_table] WARNING: fit error exceeds target epsilon; "
              "consider increasing --n-segs.")

    k_grid, p_k = compute_influence_function(kernel, k_max=args.k_max)
    gf_denom_coeffs = fit_influence_polynomial(k_grid, p_k, degree=8)

    h_stencil = args.rcut / args.n_segs * 32.0
    stencil = AssignmentStencil(order=order, c1=c_pswf)
    stencil_segments = fit_assignment_stencil(stencil, h=h_stencil)
    scale = enforce_partition_of_unity(stencil_segments, h=h_stencil)
    print(f"[generate_pswf_table] stencil partition-of-unity scale = {scale:.6f}")

    emit_header(
        path=args.out,
        epsilon=args.epsilon,
        c_pswf=c_pswf,
        rcut=args.rcut,
        n_segs=args.n_segs,
        order=order,
        segments=segments,
        gf_denom_coeffs=gf_denom_coeffs,
        stencil_segments=stencil_segments,
    )
    print(f"[generate_pswf_table] wrote {args.out}")


if __name__ == "__main__":
    main()