# Copyright (c) 2024-2026 ESP Plugin Contributors.
# Released under the BSD 3-Clause License.

# hoomd_esp/esp.py
#
# Python-level wrappers for the ESP (Ewald Summation with Prolates) plugin.
# Provides two public classes for the ESP API:
#
#   hoomd_esp.esp.Spectral - long-range mesh contribution
#     (wraps ESPForceCompute via md.long_range)
#   hoomd_esp.esp.Local - short-range real-space contribution
#     (wraps PotentialPair<EvaluatorPairPSWF>)
#
# Bug fix in this revision
# --------------------------
# Local._sync_params_from_spectral() previously pushed "n_segs" and
# "table_ptr" into each type-pair's params dict, sourced from
# Spectral._cpp_obj.getTableSize()/getTablePtr(). Those C++ accessors return
# metadata about ESPForceCompute's CPU-side introspection table
# (m_pswf_table_cpu), NOT a valid GPU device pointer -- EvaluatorPairPSWF
# has been rewritten to read its screening-function tables directly from
# the compile-time PSWF_Coeffs.h header instead, and its param_type no
# longer has "n_segs"/"table_ptr" fields at all. Continuing to pass them
# here would silently be ignored by the new param_type(dict) constructor,
# but is removed entirely to avoid confusion and stale documentation.
#
# The synced params are now exactly: kappa, rcut, alpha, V_cut (the last
# fixed at 0.0, since L(r) vanishes at r_cut by construction of the PSWF
# splitting -- there is no cutoff-energy discontinuity to shift away).
#
# Typical usage
# -------------
# import hoomd
# import hoomd_esp.esp as esp
#
# sim = hoomd.Simulation(device=hoomd.device.GPU())
# sim.create_state_from_gsd("init.gsd")
#
# nl = hoomd.md.nlist.Cell(buffer=0.4)
#
# spectral = esp.Spectral(
#     nlist=nl,
#     default_r_cut=2.0,
#     resolution=(64, 64, 64),
#     order=6,
#     kappa=1.0,
# )
#
# local = esp.Local(
#     nlist=nl,
#     default_r_cut=2.0,
#     spectral=spectral,
# )
# local.params[("A", "A")] = {}  # all params come from Spectral
# local.r_cut[("A", "A")] = 2.0
#
# integrator = hoomd.md.Integrator(dt=0.002)
# integrator.forces.append(spectral)
# integrator.forces.append(local)
# sim.operations.integrator = integrator

from __future__ import annotations

import warnings
from typing import Optional, Sequence, Tuple

import hoomd
import hoomd.md

# ---------------------------------------------------------------------------
# Try to import the compiled C++ extension. If it is not found we raise a
# clear ImportError rather than letting users see a cryptic AttributeError.
# ---------------------------------------------------------------------------
try:
    from . import _esp  # compiled pybind11 extension (esp_plugin.so)
except ImportError as _exc:
    raise ImportError(
        "hoomd_esp: compiled extension '_esp' not found. "
        "Please build the plugin with 'cmake --build' before importing."
    ) from _exc

# ---------------------------------------------------------------------------
# Module-level constants (kept in sync with ESPForceCompute.h)
# ---------------------------------------------------------------------------

#: Maximum supported PSWF interpolation order P. Mirrors ESP_MAX_ORDER.
MAX_ORDER: int = 8

#: Default number of look-up table segments for the CPU introspection table
#: (ESPForceCompute.table property). Mirrors ESPForceCompute.h's
#: ESP_TABLE_SEGMENTS default, which itself mirrors PSWF_Coeffs.h's
#: PSWF_N_SEGS. NOT used by EvaluatorPairPSWF any more (that evaluator
#: reads PSWF_Coeffs.h directly, with no runtime segment count).
DEFAULT_TABLE_SEGMENTS: int = 512

#: Minimum table segments allowed for the introspection table.
MIN_TABLE_SEGMENTS: int = 16

# ===========================================================================
# Utility helpers
# ===========================================================================

def _check_positive_int(val: int, name: str) -> None:
    if not isinstance(val, int) or val <= 0:
        raise ValueError(f"ESP: '{name}' must be a positive integer, got {val!r}.")


def _check_positive_float(val: float, name: str) -> None:
    if not (isinstance(val, (int, float)) and val > 0):
        raise ValueError(f"ESP: '{name}' must be a positive number, got {val!r}.")


def _check_nonneg_float(val: float, name: str) -> None:
    if not (isinstance(val, (int, float)) and val >= 0):
        raise ValueError(f"ESP: '{name}' must be >= 0, got {val!r}.")


def _validate_resolution(res: Sequence[int]) -> Tuple[int, int, int]:
    """Validate and unpack a (Nx, Ny, Nz) mesh-resolution tuple."""
    res = tuple(int(v) for v in res)
    if len(res) != 3:
        raise ValueError(f"ESP: 'resolution' must be a 3-tuple (Nx, Ny, Nz), got length {len(res)}.")
    for i, (v, label) in enumerate(zip(res, ("Nx", "Ny", "Nz"))):
        _check_positive_int(v, f"resolution[{i}] ({label})")
    return res  # type: ignore[return-value]


def _estimate_kappa(r_cut: float, resolution: Tuple[int, int, int]) -> float:
    """Rough default kappa = 3.2 / r_cut."""
    return 3.2 / r_cut


# ===========================================================================
# class Spectral
# ===========================================================================

class Spectral(hoomd.md.force.Force):
    """Long-range Coulomb force via the ESP mesh method.

    Wraps ``ESPForceCompute`` (C++/CUDA) and exposes it with the ESP public API.

    The reciprocal-space contribution is computed on a regular mesh using a
    PSWF-based charge-assignment kernel of order *P* and the corresponding
    optimal influence function. The short-range complement must be added via
    :class:`Local`.

    Parameters
    ----------
    nlist:
        Neighbor list used for the real-space exclusion correction.
    default_r_cut:
        Real-space cutoff radius *r_c* (in simulation length units). Must
        equal the PSWF_RCUT baked into PSWF_Coeffs.h at build time.
    resolution:
        Global mesh size ``(Nx, Ny, Nz)``. Each dimension should be even;
        powers of two are required when running with MPI domain decomposition.
    order:
        PSWF interpolation order *P* (integer, 1 <= P <= 8).
        Higher values give smaller mesh error at the cost of a wider stencil.
        Recommended: 6 for ~10^-8 accuracy.
    kappa:
        Ewald splitting parameter kappa [1/length]. If ``None``, a
        heuristic default of ``3.2 / r_cut`` is used.
    alpha:
        Debye-Hueckel screening constant [1/length]. Set to 0 (default)
        for pure Coulomb; set > 0 for Yukawa-screened electrostatics.
    n_table:
        Number of piecewise-polynomial segments used for the CPU-side
        introspection table (``Spectral.table`` / plotting). Does NOT
        affect the runtime-critical GPU/CPU evaluators, which always read
        the fixed-resolution compile-time PSWF_Coeffs.h table.
    nlist_buffer:
        Extra skin distance added to the neighbor-list cutoff to account for
        particle motion between list rebuilds. Defaults to 10% of r_cut.

    Attributes
    ----------
    kappa : float
        Splitting parameter actually in use (may differ from the constructor
        argument if the heuristic default was applied).
    resolution : tuple[int, int, int]
        Mesh dimensions ``(Nx, Ny, Nz)``.
    order : int
        PSWF interpolation order.
    r_cut : float
        Real-space cutoff.
    alpha : float
        Debye screening parameter (0 = pure Coulomb).
    q_sum : float
        Total system charge Sum q_i (available after the first time step).
    q2_sum : float
        Sum q_i^2 (available after the first time step).

    Example
    -------
    .. code-block:: python

        spectral = esp.Spectral(
            nlist=nl,
            default_r_cut=2.0,
            resolution=(64, 64, 64),
            order=6,
            kappa=1.0,
        )
        integrator.forces.append(spectral)
    """

    def __init__(
        self,
        nlist: hoomd.md.nlist.NList,
        default_r_cut: float,
        resolution: Sequence[int] = (32, 32, 32),
        order: int = 6,
        kappa: Optional[float] = None,
        alpha: float = 0.0,
        n_table: int = DEFAULT_TABLE_SEGMENTS,
        nlist_buffer: Optional[float] = None,
    ) -> None:
        super().__init__()

        # -- validate inputs --------------------------------------------------
        _check_positive_float(default_r_cut, "default_r_cut")
        res = _validate_resolution(resolution)

        if not (isinstance(order, int) and 1 <= order <= MAX_ORDER):
            raise ValueError(f"ESP: 'order' must be an integer in [1, {MAX_ORDER}], got {order!r}.")

        if kappa is None:
            kappa = _estimate_kappa(default_r_cut, res)
            warnings.warn(
                f"ESP: 'kappa' not specified; using heuristic kappa = {kappa:.4f} "
                f"(= 3.2 / r_cut). For production runs, tune kappa explicitly.",
                stacklevel=2,
            )
        else:
            _check_positive_float(kappa, "kappa")

        _check_nonneg_float(alpha, "alpha")

        if not (isinstance(n_table, int) and n_table >= MIN_TABLE_SEGMENTS):
            raise ValueError(f"ESP: 'n_table' must be an integer >= {MIN_TABLE_SEGMENTS}, got {n_table!r}.")

        # -- store parameters ---------------------------------------------------
        self._nlist = nlist
        self._r_cut = float(default_r_cut)
        self._resolution = res
        self._order = int(order)
        self._kappa = float(kappa)
        self._alpha = float(alpha)
        self._n_table = int(n_table)

        if nlist_buffer is None:
            nlist_buffer = 0.1 * self._r_cut
        self._nlist_buffer = float(nlist_buffer)

        # -- C++ object (created lazily in _attach) ------------------------------
        self._cpp_obj: Optional[_esp.ESPForceCompute] = None

        # -- back-reference used by esp.Local -------------------------------------
        self._pair: Optional["Local"] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def kappa(self) -> float:
        """Ewald splitting parameter kappa [1/length]."""
        if self._cpp_obj is not None:
            return self._cpp_obj.getKappa()
        return self._kappa

    @property
    def resolution(self) -> Tuple[int, int, int]:
        """Global mesh dimensions (Nx, Ny, Nz)."""
        if self._cpp_obj is not None:
            return tuple(self._cpp_obj.getResolution())  # type: ignore[return-value]
        return self._resolution

    @property
    def order(self) -> int:
        """PSWF interpolation order P."""
        if self._cpp_obj is not None:
            return self._cpp_obj.getOrder()
        return self._order

    @property
    def r_cut(self) -> float:
        """Real-space cutoff radius r_c."""
        if self._cpp_obj is not None:
            return self._cpp_obj.getRCut()
        return self._r_cut

    @property
    def alpha(self) -> float:
        """Debye screening parameter alpha (0 = pure Coulomb)."""
        if self._cpp_obj is not None:
            return self._cpp_obj.getAlpha()
        return self._alpha

    @property
    def q_sum(self) -> float:
        """Total system charge Sum q_i (updated each time step)."""
        if self._cpp_obj is None:
            raise RuntimeError("ESP: q_sum not available before simulation attachment.")
        return self._cpp_obj.getQSum()

    @property
    def q2_sum(self) -> float:
        """Sum q_i^2 (updated each time step)."""
        if self._cpp_obj is None:
            raise RuntimeError("ESP: q2_sum not available before simulation attachment.")
        return self._cpp_obj.getQ2Sum()

    # ------------------------------------------------------------------
    # Parameter update at runtime
    # ------------------------------------------------------------------

    def set_params(
        self,
        *,
        resolution: Optional[Sequence[int]] = None,
        order: Optional[int] = None,
        kappa: Optional[float] = None,
        r_cut: Optional[float] = None,
        alpha: Optional[float] = None,
        n_table: Optional[int] = None,
    ) -> None:
        """Update ESP parameters at runtime and trigger re-initialisation.

        All keyword arguments are optional; only the supplied values are
        updated. The change takes effect on the *next* call to
        ``computeForces()``.

        Parameters
        ----------
        resolution:
            New mesh dimensions ``(Nx, Ny, Nz)``.
        order:
            New PSWF interpolation order.
        kappa:
            New splitting parameter.
        r_cut:
            New real-space cutoff.
        alpha:
            New Debye screening constant.
        n_table:
            New CPU introspection look-up table size.

        Raises
        ------
        RuntimeError
            If called before the Simulation is attached.
        """
        if self._cpp_obj is None:
            raise RuntimeError(
                "ESP: 'set_params()' called before the Simulation was attached. "
                "Pass initial values to the Spectral constructor instead."
            )

        if resolution is not None:
            self._resolution = _validate_resolution(resolution)
        if order is not None:
            if not (isinstance(order, int) and 1 <= order <= MAX_ORDER):
                raise ValueError(f"ESP: invalid order {order!r}.")
            self._order = int(order)
        if kappa is not None:
            _check_positive_float(kappa, "kappa")
            self._kappa = float(kappa)
        if r_cut is not None:
            _check_positive_float(r_cut, "r_cut")
            self._r_cut = float(r_cut)
        if alpha is not None:
            _check_nonneg_float(alpha, "alpha")
            self._alpha = float(alpha)
        if n_table is not None:
            if not (isinstance(n_table, int) and n_table >= MIN_TABLE_SEGMENTS):
                raise ValueError(f"ESP: invalid n_table {n_table!r}.")
            self._n_table = int(n_table)

        # Push updated parameters to C++ and mark for re-initialisation.
        nx, ny, nz = self._resolution
        self._cpp_obj.setParams(
            nx, ny, nz,
            self._order,
            self._kappa,
            self._r_cut,
            self._alpha,
            self._n_table,
        )

        # Propagate r_cut/kappa/alpha changes to the associated Local evaluator.
        if self._pair is not None:
            self._pair._sync_params_from_spectral()

    def invalidate(self) -> None:
        """Force a full re-initialisation on the next compute step.

        Useful after e.g. box-resize operations that do not automatically
        trigger a recompute of the influence function.
        """
        if self._cpp_obj is not None:
            self._cpp_obj.invalidate()

    # ------------------------------------------------------------------
    # HOOMD lifecycle
    # ------------------------------------------------------------------

    def _attach_hook(self) -> None:
        """Create the C++ ESPForceCompute and register it with HOOMD."""
        sim: hoomd.Simulation = self._simulation  # type: ignore[assignment]
        sysdef = sim.state._cpp_sys_def
        nlist_cpp = self._nlist._cpp_obj
        group_cpp = sim.state.all_group._cpp_obj  # act on all particles

        self._cpp_obj = _esp.ESPForceCompute(sysdef, nlist_cpp, group_cpp)

        nx, ny, nz = self._resolution
        self._cpp_obj.setParams(
            nx, ny, nz,
            self._order,
            self._kappa,
            self._r_cut,
            self._alpha,
            self._n_table,
        )

        # Hand the underlying C++ object to the base class so HOOMD can
        # call computeForces() each time step.
        self._cpp_force = self._cpp_obj  # type: ignore[attr-defined]

    def _detach_hook(self) -> None:
        """Release the C++ object when detached from a Simulation."""
        self._cpp_obj = None
        self._cpp_force = None  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"esp.Spectral("
            f"resolution={self._resolution}, "
            f"order={self._order}, "
            f"kappa={self._kappa:.4g}, "
            f"r_cut={self._r_cut}, "
            f"alpha={self._alpha})"
        )


# ===========================================================================
# class Local
# ===========================================================================

class Local(hoomd.md.pair.Pair):
    """Short-range real-space complement for the ESP mesh method.

    Wraps ``PotentialPair<EvaluatorPairPSWF>`` (C++) and must always be used
    together with :class:`Spectral`. The pair evaluator reads S(r)/S'(r)
    directly from the compile-time PSWF_Coeffs.h table baked into the C++
    extension at build time -- it does NOT read a runtime table pointer.
    Only the physical scalars (kappa, r_c, alpha, V_cut) are synced from the
    associated :class:`Spectral` object, so ``params`` dictionaries for each
    type-pair contain **no user-settable fields**.

    Parameters
    ----------
    nlist:
        Neighbor list (should be the same instance passed to :class:`Spectral`).
    default_r_cut:
        Default real-space cutoff for all type pairs. Must match the value
        given to :class:`Spectral` (and the PSWF_RCUT compiled into the
        extension).
    spectral:
        The associated :class:`Spectral` instance. The pair evaluator will
        pull kappa, r_c, and alpha from this object.
    default_r_on:
        Inner smoothing radius (0 = sharp cutoff, which is correct for ESP
        since L(r) already goes smoothly to zero at r_c).
    mode:
        Shifting mode: ``"none"`` (default, correct for ESP) or ``"shift"``.
        ESP's L(r) is constructed to vanish at r_c, so no additional shifting
        is needed; V_cut is always synced as 0.0.

    Example
    -------
    .. code-block:: python

        local = esp.Local(
            nlist=nl,
            default_r_cut=2.0,
            spectral=spectral,
        )
        # Register every type-pair that carries a charge.
        for t1 in sim.state.particle_types:
            for t2 in sim.state.particle_types:
                local.params[(t1, t2)] = {}
                local.r_cut[(t1, t2)] = 2.0
        integrator.forces.append(local)
    """

    _accepted_modes = ("none",)  # ESP pair does not need XPLOR smoothing

    def __init__(
        self,
        nlist: hoomd.md.nlist.NList,
        default_r_cut: float,
        spectral: Optional[Spectral] = None,
        coulomb: Optional[Spectral] = None,
        default_r_on: float = 0.0,
        mode: str = "none",
    ) -> None:
        if spectral is None and coulomb is not None:
            warnings.warn(
                "The 'coulomb' parameter is deprecated; use 'spectral' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            spectral = coulomb
        if spectral is None:
            raise ValueError("'spectral' (or deprecated 'coulomb') must be provided.")
        if not isinstance(spectral, Spectral):
            raise TypeError(f"ESP: 'spectral' must be an esp.Spectral instance, got {type(spectral).__name__!r}.")

        if mode not in ("none",):
            raise ValueError("ESP: Pair mode must be 'none' for ESP (L(r) already vanishes at r_c).")

        super().__init__(
            nlist=nlist,
            default_r_cut=default_r_cut,
            default_r_on=default_r_on,
            mode=mode,
        )

        self._spectral = spectral
        spectral._pair = self  # cross-reference for runtime param sync

    # ------------------------------------------------------------------
    # params validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_params(params: dict) -> dict:
        """ESP pair parameters carry no user-settable fields.

        Accepts an empty dict ``{}``; all physical parameters (kappa, r_c,
        alpha) are sourced from the associated :class:`Spectral` object at
        attach time.
        """
        if params and set(params.keys()) - {"_reserved"}:
            warnings.warn(
                "ESP: Local.params accepts an empty dict {}; all ESP parameters "
                "are read from the associated Spectral object. "
                f"Ignoring keys: {set(params.keys())!r}.",
                stacklevel=3,
            )
        return {}

    # ------------------------------------------------------------------
    # HOOMD lifecycle
    # ------------------------------------------------------------------

    def _attach_hook(self) -> None:
        """Create PotentialPair<EvaluatorPairPSWF> and push scalar params."""
        super()._attach_hook()
        self._sync_params_from_spectral()

    def _sync_params_from_spectral(self) -> None:
        """Push current ESP scalar parameters from Spectral into the C++ pair object.

        This method is called:
          - once at attach time (after both Spectral and Local are attached),
          - whenever Spectral.set_params() updates kappa, r_cut, or alpha.

        BUG FIX (this revision): previously also read
        ``cpp_spectral.getTableSize()`` and ``cpp_spectral.getTablePtr()``
        and pushed them as "n_segs"/"table_ptr" into each type-pair's
        params. EvaluatorPairPSWF::param_type no longer has those fields --
        it reads its S(r)/S'(r) tables directly from the compile-time
        PSWF_Coeffs.h header instead of any runtime pointer. Only the four
        physical scalars below are synced now.
        """
        if self._cpp_obj is None:
            return  # not yet attached; will be called again in _attach_hook
        if self._spectral._cpp_obj is None:
            raise RuntimeError(
                "ESP: Local._sync_params_from_spectral() called but the associated "
                "Spectral object is not attached. Attach Spectral before Local."
            )

        cpp_spectral = self._spectral._cpp_obj
        kappa = cpp_spectral.getKappa()
        r_cut = cpp_spectral.getRCut()
        alpha = cpp_spectral.getAlpha()

        # L(r) vanishes at r_cut by construction of the PSWF splitting
        # kernel (S(1) == 0 in PSWF_Coeffs.h's normalisation), so no
        # energy-shift constant is needed; V_cut is always 0.0.
        v_cut = 0.0

        # Build param_type-compatible dict for each registered type pair.
        # The C++ evaluator reads: kappa, rcut, alpha, V_cut.
        for key in list(self.params.keys()):
            self.params[key] = dict(
                kappa=kappa,
                rcut=r_cut,
                alpha=alpha,
                V_cut=v_cut,
            )

    def _detach_hook(self) -> None:
        super()._detach_hook()
        self._spectral._pair = None

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"esp.Local(r_cut={self._spectral._r_cut}, "
            f"kappa={self._spectral._kappa:.4g}, "
            f"alpha={self._spectral._alpha})"
        )


# ===========================================================================
# Convenience factory
# ===========================================================================

def make_esp_forces(
    nlist: hoomd.md.nlist.NList,
    r_cut: float,
    resolution: Sequence[int] = (32, 32, 32),
    order: int = 6,
    kappa: Optional[float] = None,
    alpha: float = 0.0,
    n_table: int = DEFAULT_TABLE_SEGMENTS,
) -> Tuple[Spectral, Local]:
    """Create a coupled pair of ESP long-range (Spectral) and short-range (Local) forces.

    This is the recommended entry point for ESP simulations.

    Parameters
    ----------
    nlist:
        Shared neighbor list.
    r_cut:
        Real-space cutoff (same for both components).
    resolution:
        Mesh dimensions ``(Nx, Ny, Nz)``.
    order:
        PSWF interpolation order P.
    kappa:
        Splitting parameter kappa. Heuristic default if ``None``.
    alpha:
        Debye screening constant (0 = pure Coulomb).
    n_table:
        CPU introspection look-up table size.

    Returns
    -------
    spectral : esp.Spectral
        The long-range mesh force.
    local : esp.Local
        The short-range real-space complement.

    Example
    -------
    .. code-block:: python

        spectral, local = esp.make_esp_forces(
            nlist=nl, r_cut=2.0, resolution=(64, 64, 64), order=6, kappa=1.0
        )
        integrator.forces.extend([spectral, local])
        # Register type pairs:
        for t1 in sim.state.particle_types:
            for t2 in sim.state.particle_types:
                local.params[(t1, t2)] = {}
                local.r_cut[(t1, t2)] = 2.0
    """
    spectral = Spectral(
        nlist=nlist,
        default_r_cut=r_cut,
        resolution=resolution,
        order=order,
        kappa=kappa,
        alpha=alpha,
        n_table=n_table,
    )
    local = Local(nlist=nlist, default_r_cut=r_cut, spectral=spectral)
    return spectral, local


# ===========================================================================
# __all__ / backward-compatible aliases
# ===========================================================================

Coulomb = Spectral
Pair = Local

__all__ = [
    "Spectral",
    "Local",
    "make_esp_forces",
    "Coulomb",
    "Pair",
    "MAX_ORDER",
    "DEFAULT_TABLE_SEGMENTS",
    "MIN_TABLE_SEGMENTS",
]