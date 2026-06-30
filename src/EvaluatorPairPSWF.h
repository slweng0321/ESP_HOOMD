// Copyright (c) 2024-2026 ESP Plugin Contributors.
// Released under the BSD 3-Clause License.
//
// EvaluatorPairPSWF.h
//
// Pair potential evaluator for the real-space short-range correction term
// of the ESP (Ewald Summation with Prolates) method.
//
// Physical background
// -------------------
// The full Coulomb potential is split as:
//
//   1/(4π r) = L(r) + S(r)
//
// where S(r) is the smooth long-range part handled by the mesh (reciprocal
// space), and L(r) is the short-range part handled by this pair evaluator.
//
// For a PSWF splitting kernel ψ with bandwidth parameter α, L(r) is defined
// by the complement of the cumulative distribution of ψ:
//
//   L(r) = 1/(4π r) * [1 - Φ_ψ(r / r_c)]          (Eq. 6, paper)
//
// where Φ_ψ is the CDF of ψ normalised so Φ_ψ(1) = 1, and r_c is the
// real-space cutoff.
//
// Because L(r) has no analytic closed form in terms of standard special
// functions, it is evaluated via a piecewise-polynomial look-up table that
// is pre-built by ESPForceCompute::buildPSWFTable() on the CPU and passed to
// this evaluator via the param_type::table_ptr pointer.
//
// Interface contract with HOOMD-blue
// ------------------------------------
// This header follows EXACTLY the same interface pattern as
// EvaluatorPairEwald.h so that it can be plugged into HOOMD's
// PotentialPair<> template machinery without any additional framework changes.
//
// Key design constraints enforced below:
//   1. The constructor is marked DEVICE so it can be called from both CPU
//      code and CUDA kernels.
//   2. evalForceAndEnergy() is DEVICE and performs only arithmetic / table
//      lookups — no dynamic allocation, no virtual calls, no exceptions.
//   3. The param_type struct is 16-byte aligned (required by HOOMD for safe
//      __constant__ / Texture memory placement on GPUs).
//   4. param_type::load_shared / allocate_shared follow the HOOMD shared-
//      memory staging protocol used by PotentialPairGPU.
//
// Look-up table memory layout
// ----------------------------
// The table is a flat Scalar array owned by ESPForceCompute and uploaded to
// the GPU via GPUArray<Scalar>.  Each segment s occupies 12 consecutive
// Scalar elements (chosen for 16-byte alignment at 4-byte Scalar):
//
//   offset  0..4  : potential coefficients c[0..4]  (Horner form, L(r))
//   offset  5..9  : force coefficients     f[0..4]  (Horner form, -dL/dr)
//   offset  10    : r_lo  (left edge of segment, physical distance)
//   offset  11    : dr_inv (1 / segment width)
//
// Within evalForceAndEnergy() the segment index is computed as:
//   seg = clamp( floor(r * n_segs / r_c), 0, n_segs-1 )
// and the normalised intra-segment coordinate as:
//   t = (r - table[seg*12 + 10]) * table[seg*12 + 11]
//
// Energy shift
// ------------
// When energy_shift == true the potential is shifted so that V(r_cut) = 0.
// The shift value V_cut is pre-computed and stored in param_type::V_cut so
// that no additional table lookup is required at the cutoff boundary.
//
// Reference
// ---------
//   Liang, Lu, Barnett, Greengard, Jiang,
//   "Accelerating molecular dynamics simulations using fast Ewald summation
//    with prolates", Nat. Commun. 2026.
//   https://doi.org/10.1038/s41467-026-73232-8

#ifndef __PAIR_EVALUATOR_PSWF_H__
#define __PAIR_EVALUATOR_PSWF_H__

// pybind11 is NOT available inside device (CUDA/HIP) code.
#ifndef __HIPCC__
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <sstream>
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

// ── Device / host qualifier macros ───────────────────────────────────────────
// Mirrors EvaluatorPairEwald.h exactly so this header compiles transparently
// in both nvcc/hipcc and host-only translation units.
#ifdef __HIPCC__
#define DEVICE     __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#endif

// ── Number of Scalar elements per table segment ───────────────────────────────
// 5 potential coefficients + 5 force coefficients + r_lo + dr_inv = 12.
// Must match the layout produced by ESPForceCompute::buildPSWFTable() and
// the flat-array upload in ESPForceCompute::initializeFFT().
// Override with -DESP_TABLE_STRIDE=N only if buildPSWFTable is also changed.
#ifndef ESP_TABLE_STRIDE
#define ESP_TABLE_STRIDE 12
#endif

namespace hoomd
{
namespace md
{

// =============================================================================
// EvaluatorPairPSWF
// =============================================================================

/*!
 * \class EvaluatorPairPSWF
 * \brief Evaluates the real-space short-range correction L(r) for the ESP
 *        long-range electrostatics method using a piecewise-polynomial
 *        look-up table.
 *
 * The class is designed to be instantiated inside a tight neighbour-list loop
 * on both CPU and GPU.  All member functions are either DEVICE or HOSTDEVICE
 * to allow seamless nvcc/hipcc compilation.
 *
 * \par Usage in HOOMD PotentialPair machinery
 * \code
 *   // In Python / C++ pair force setup:
 *   auto pair = std::make_shared<PotentialPair<EvaluatorPairPSWF>>(...);
 *   EvaluatorPairPSWF::param_type p;
 *   p.kappa      = 0.34;
 *   p.alpha      = 0.0;
 *   p.rcut       = 1.2;
 *   p.n_segs     = 512;
 *   p.table_ptr  = gpu_table_device_ptr;
 *   p.V_cut      = L_evaluated_at_rcut;
 *   pair->setParams(typeA, typeB, p);
 * \endcode
 */
class EvaluatorPairPSWF
    {
    public:

    // =========================================================================
    // param_type
    // =========================================================================

    /*!
     * \brief Per-type-pair parameters for EvaluatorPairPSWF.
     *
     * This struct is stored in HOOMD's per-type-pair parameter table and may
     * be placed in GPU __constant__ memory.  The 16-byte alignment attribute
     * is set at the end of the struct body (matching EvaluatorPairEwald.h).
     */
    struct param_type
        {
        Scalar       kappa;      //!< Ewald splitting parameter κ [1/length]
        Scalar       alpha;      //!< Debye screening κ_D [1/length]; 0 = off
        Scalar       rcut;       //!< Real-space cutoff r_c [length]
        Scalar       V_cut;      //!< L(r_c): precomputed energy-shift reference

        //! Number of piecewise segments in the L(r) look-up table.
        //! Must equal ESPForceCompute::m_n_table_segments.
        unsigned int n_segs;

        //! Device pointer to the flat Scalar look-up table produced by
        //! ESPForceCompute::buildPSWFTable().  Each segment occupies
        //! ESP_TABLE_STRIDE Scalar elements (see file header for layout).
        //!
        //! On CPU this points into the host-side mirror of GPUArray<Scalar>;
        //! on GPU this is the device-side pointer.  ESPForceCompute is
        //! responsible for keeping this pointer valid for the lifetime of any
        //! PotentialPair that uses this evaluator.
        const Scalar* table_ptr;

        // ── HOOMD shared-memory staging protocol ─────────────────────────────
        // PotentialPairGPU calls these two methods to determine how much
        // shared memory to reserve for parameters.  Because table_ptr is a
        // device pointer (not inline data), we do NOT stage the table in
        // shared memory; the struct itself fits comfortably in registers.

        DEVICE void load_shared(char*& /*ptr*/,
                                unsigned int& /*available_bytes*/)
            {
            // No-op: table data is accessed via global memory / L1 cache.
            // The small struct fields are broadcast from __constant__ memory.
            }

        HOSTDEVICE void allocate_shared(char*& /*ptr*/,
                                        unsigned int& /*available_bytes*/) const
            {
            // No-op: reports zero shared-memory bytes required.
            }

#ifdef ENABLE_HIP
        //! Advise the HIP/CUDA runtime on memory access patterns.
        void set_memory_hints() const
            {
            // table_ptr is managed by GPUArray; HOOMD handles prefetch/advise
            // at that level.  No additional hints are needed.
            }
#endif // ENABLE_HIP

        // ── Host-only members ─────────────────────────────────────────────────
#ifndef __HIPCC__

        //! Default constructor: zero-initialised; evalForceAndEnergy() will
        //! return false immediately until the struct is properly populated.
        param_type()
            : kappa(Scalar(0.0)),
              alpha(Scalar(0.0)),
              rcut(Scalar(0.0)),
              V_cut(Scalar(0.0)),
              n_segs(0u),
              table_ptr(nullptr)
            { }

        /*!
         * \brief Construct from a Python dict.
         *
         * Required keys:
         *   "kappa"     : float  — Ewald splitting parameter
         *   "rcut"      : float  — Real-space cutoff
         *   "V_cut"     : float  — L(r_c) for energy-shift
         *   "n_segs"    : int    — Number of table segments
         *   "table_ptr" : int    — Device pointer cast to Python int
         *                          (obtained via GPUArray handle.ptr)
         *
         * Optional keys:
         *   "alpha"     : float  — Debye screening (default 0.0)
         *
         * \note "table_ptr" is passed as a Python integer (uintptr_t).
         *       The caller (esp.py) must ensure the GPUArray owning the
         *       table remains alive for the duration of the simulation.
         */
        param_type(pybind11::dict v, bool /*managed*/ = false)
            {
            kappa = v["kappa"].cast<Scalar>();
            alpha = v.contains("alpha") ? v["alpha"].cast<Scalar>()
                                        : Scalar(0.0);
            rcut  = v["rcut"].cast<Scalar>();
            V_cut = v["V_cut"].cast<Scalar>();
            n_segs = v["n_segs"].cast<unsigned int>();

            if (v.contains("table_ptr") && !v["table_ptr"].is_none())
                {
                uintptr_t raw = v["table_ptr"].cast<uintptr_t>();
                table_ptr = reinterpret_cast<const Scalar*>(raw);
                }
            else
                {
                table_ptr = nullptr;
                }

            if (kappa <= Scalar(0.0))
                throw std::invalid_argument(
                    "EvaluatorPairPSWF: kappa must be positive.");
            if (rcut <= Scalar(0.0))
                throw std::invalid_argument(
                    "EvaluatorPairPSWF: rcut must be positive.");
            if (n_segs == 0u)
                throw std::invalid_argument(
                    "EvaluatorPairPSWF: n_segs must be > 0.");
            }

        //! Serialise to Python dict (inverse of the constructor above).
        pybind11::dict asDict() const
            {
            pybind11::dict v;
            v["kappa"]     = kappa;
            v["alpha"]     = alpha;
            v["rcut"]      = rcut;
            v["V_cut"]     = V_cut;
            v["n_segs"]    = n_segs;
            v["table_ptr"] = reinterpret_cast<uintptr_t>(table_ptr);
            return v;
            }

#endif // !__HIPCC__

        // ── Alignment attribute ───────────────────────────────────────────────
        // 8-byte alignment for float32 builds; 16-byte for float64 builds.
        // This mirrors the attribute in EvaluatorPairEwald::param_type and is
        // required by HOOMD's GPU parameter-upload path.
        }
#if HOOMD_LONGREAL_SIZE == 32
    __attribute__((aligned(8)));
#else
    __attribute__((aligned(16)));
#endif

    // =========================================================================
    // Constructor
    // =========================================================================

    /*!
     * \brief Construct the evaluator for a specific particle pair.
     *
     * \param _rsq     Squared distance between particles i and j.
     * \param _rcutsq  Squared cutoff distance r_c^2.
     * \param _params  Reference to the per-type-pair parameter struct.
     *
     * The constructor stores only its arguments; all computation is deferred
     * to evalForceAndEnergy() to keep the constructor inlinable and register-
     * pressure minimal at the call site.
     */
    DEVICE EvaluatorPairPSWF(Scalar _rsq,
                              Scalar _rcutsq,
                              const param_type& _params)
        : rsq(_rsq),
          rcutsq(_rcutsq),
          kappa(_params.kappa),
          alpha(_params.alpha),
          rcut(_params.rcut),
          V_cut(_params.V_cut),
          n_segs(_params.n_segs),
          table_ptr(_params.table_ptr),
          qiqj(Scalar(0.0))
        { }

    // =========================================================================
    // Charge handling
    // =========================================================================

    //! This potential uses partial charges on every particle.
    DEVICE static bool needsCharge()
        {
        return true;
        }

    /*!
     * \brief Accept charges from the PotentialPair driver.
     * \param qi Charge of particle i.
     * \param qj Charge of particle j.
     */
    DEVICE void setCharge(Scalar qi, Scalar qj)
        {
        qiqj = qi * qj;
        }

    // =========================================================================
    // Core evaluation
    // =========================================================================

    /*!
     * \brief Evaluate the PSWF real-space force and energy for this pair.
     *
     * \param[out] force_divr  Force divided by r: |F|/r (positive = repulsive).
     * \param[out] pair_eng    Pair potential energy V(r) = q_i q_j L(r).
     * \param[in]  energy_shift  If true, subtract V(r_c) = q_i q_j V_cut.
     *
     * \return true  if the pair was within the cutoff and was evaluated.
     *         false if rsq >= rcutsq, qiqj == 0, or table_ptr is null.
     *
     * \par Degree-4 Horner evaluation (critical path)
     *
     * For each pair the algorithm performs:
     *
     *  1. Guard check  (rsq, qiqj, table_ptr, n_segs).
     *  2. r = 1 / fast::rsqrt(rsq).
     *  3. seg = clamp(floor(r * n_segs / rcut), 0, n_segs-1).
     *  4. base = table_ptr + seg * ESP_TABLE_STRIDE.
     *  5. r_lo  = base[10];  dr_inv = base[11];  t = (r - r_lo) * dr_inv.
     *  6. Horner: L(r)      = base[0..4] evaluated at t.
     *  7. Horner: -dL/dr    = base[5..9] evaluated at t.
     *  8. Debye correction  (if alpha != 0): multiply by exp(-alpha*r),
     *                        chain-rule the derivative.
     *  9. pair_eng = qiqj * L;  if energy_shift: pair_eng -= qiqj * V_cut.
     * 10. force_divr = qiqj * (-dLdr) * rinv.
     *
     * \par Register budget
     * The critical path uses ~14 scalar registers (t, L, dLdr, r, rinv,
     * qiqj, alpha, V_cut + 4 base-pointer loads + seg), keeping SM occupancy
     * high on both NVIDIA Ampere and AMD RDNA2 architectures.
     */
    DEVICE bool evalForceAndEnergy(Scalar& force_divr,
                                   Scalar& pair_eng,
                                   bool    energy_shift)
        {
        // ── Guard ─────────────────────────────────────────────────────────────
        if (rsq    >= rcutsq          ||
            qiqj   == Scalar(0.0)     ||
            table_ptr == nullptr      ||
            n_segs == 0u)
            {
            return false;
            }

        // ── Distance ──────────────────────────────────────────────────────────
        // fast::rsqrt maps to __frsqrt_rn on GPU (1 ULP) and std::sqrt on CPU.
        const Scalar rinv = fast::rsqrt(rsq);
        const Scalar r    = Scalar(1.0) / rinv;

        // ── Segment index ─────────────────────────────────────────────────────
        // Multiply by (n_segs / rcut) — the compiler treats this as a constant
        // when n_segs and rcut are loop-invariant, hoisting it automatically.
        const Scalar seg_f = r * (Scalar(n_segs) / rcut);
        unsigned int seg   = static_cast<unsigned int>(seg_f);
        if (seg >= n_segs)
            seg = n_segs - 1u;

        // ── Table base pointer ────────────────────────────────────────────────
        // A contiguous 12-element read; on Ampere this fits in two 128-bit
        // cache-line loads, minimising L2 traffic in the neighbour loop.
        const Scalar* base = table_ptr
                             + seg * static_cast<unsigned int>(ESP_TABLE_STRIDE);

        // ── Intra-segment normalised coordinate t ∈ [0, 1) ───────────────────
        const Scalar r_lo   = base[10];
        const Scalar dr_inv = base[11];
        const Scalar t      = (r - r_lo) * dr_inv;

        // ── Horner evaluation: L(r) ───────────────────────────────────────────
        // L(r) ≈ c0 + t*(c1 + t*(c2 + t*(c3 + t*c4)))
        // Coefficients: base[0] = c0 (constant term), base[4] = c4 (leading).
        Scalar L = base[4];
        L = base[3] + t * L;
        L = base[2] + t * L;
        L = base[1] + t * L;
        L = base[0] + t * L;

        // ── Horner evaluation: -dL/dr ─────────────────────────────────────────
        // Stored as force coefficients f0..f4 at base[5..9], same Horner form.
        // Note: the table stores -dL/dr (positive for repulsive L), so no sign
        // flip is needed before assigning to force_divr.
        Scalar dLdr_neg = base[9];
        dLdr_neg = base[8] + t * dLdr_neg;
        dLdr_neg = base[7] + t * dLdr_neg;
        dLdr_neg = base[6] + t * dLdr_neg;
        dLdr_neg = base[5] + t * dLdr_neg;

        // ── Debye-Hückel / Yukawa screening correction ────────────────────────
        // When alpha > 0 the Coulomb kernel is replaced by exp(-alpha*r)/r.
        // The PSWF table was built for the bare Coulomb splitting (alpha = 0).
        // We apply the Yukawa envelope here as a multiplicative factor so a
        // single table serves both screened and unscreened simulations.
        //
        //   L_D(r) = L(r) * exp(-alpha * r)
        //  -dL_D/dr = exp(-alpha*r) * [-dL/dr + alpha * L]
        //
        // Branch: compiler eliminates this block entirely when alpha == 0.
        if (alpha != Scalar(0.0))
            {
            const Scalar expfac = fast::exp(-alpha * r);
            // Chain rule: d/dr [L * exp(-alpha*r)]
            //           = (dL/dr)*exp(-alpha*r) + L*(-alpha)*exp(-alpha*r)
            // In our sign convention (-dL/dr stored):
            // -dL_D/dr = exp(-alpha*r) * (-dL/dr) + exp(-alpha*r) * alpha * L
            dLdr_neg = expfac * (dLdr_neg + alpha * L);
            L        = L * expfac;
            }

        // ── Energy ────────────────────────────────────────────────────────────
        pair_eng = qiqj * L;

        if (energy_shift)
            {
            // V_cut = L(r_c) [Debye-corrected if alpha != 0] was precomputed
            // and stored in param_type::V_cut by ESPForceCompute::buildPSWFTable().
            // Subtracting it ensures V(r_c) = 0 without an extra table lookup.
            pair_eng -= qiqj * V_cut;
            }

        // ── Force ─────────────────────────────────────────────────────────────
        // PotentialPair expects force_divr = F / r = -dV/dr / r.
        //   V = q_i * q_j * L(r)  =>  -dV/dr = q_i*q_j * (-dL/dr)
        //   force_divr = q_i*q_j * (-dL/dr) / r = qiqj * dLdr_neg * rinv
        force_divr = qiqj * dLdr_neg * rinv;

        return true;
        }

    // =========================================================================
    // Long-range correction integrals (isotropic tail)
    // =========================================================================

    /*!
     * \brief Pressure long-range correction integral.
     *
     * The PSWF short-range complement L(r) decays exponentially beyond r_c
     * (construction property of the PSWF splitting), so the tail correction
     * is negligible.  Returns 0, matching EvaluatorPairEwald.
     */
    DEVICE Scalar evalPressureLRCIntegral()
        {
        return Scalar(0.0);
        }

    /*!
     * \brief Energy long-range correction integral.
     *
     * Same reasoning as evalPressureLRCIntegral(): exponential decay beyond
     * r_c makes the tail contribution negligible.
     */
    DEVICE Scalar evalEnergyLRCIntegral()
        {
        return Scalar(0.0);
        }

    // =========================================================================
    // Host-only metadata
    // =========================================================================

#ifndef __HIPCC__

    /*!
     * \brief Canonical name used in HOOMD's type registry and Python API.
     *
     * Referenced in esp.py:
     * \code
     *   pair_pswf = hoomd.md.pair.Pair(nlist, default_r_cut=rcut,
     *                                  type_pair="pswf")
     * \endcode
     */
    static std::string getName()
        {
        return std::string("pswf");
        }

    /*!
     * \brief Shape specification (not applicable to a point-charge potential).
     *
     * Called by the HOOMD serialisation framework for shape-dependent
     * potentials.  Throws to prevent silent misuse in rigid-body workflows.
     */
    std::string getShapeSpec() const
        {
        throw std::runtime_error(
            "EvaluatorPairPSWF: shape specification is not defined "
            "for a scalar point-charge pair potential.");
        }

#endif // !__HIPCC__

    // =========================================================================
    // Protected data members
    // =========================================================================

    protected:

    Scalar        rsq;        //!< Squared pair distance (set in constructor)
    Scalar        rcutsq;     //!< Squared cutoff r_c^2 (set in constructor)
    Scalar        kappa;      //!< Ewald splitting κ (stored for diagnostics)
    Scalar        alpha;      //!< Debye screening κ_D (0 = no screening)
    Scalar        rcut;       //!< Physical cutoff r_c [length]
    Scalar        V_cut;      //!< L(r_c): energy-shift reference
    unsigned int  n_segs;     //!< Number of look-up table segments
    const Scalar* table_ptr;  //!< Device/host pointer to the flat table array
    Scalar        qiqj;       //!< Charge product q_i * q_j (set in setCharge)

    }; // class EvaluatorPairPSWF

} // namespace md
} // namespace hoomd

// ── Clean up local macros ─────────────────────────────────────────────────────
// DEVICE and HOSTDEVICE are intentionally NOT undefined here, matching the
// convention of EvaluatorPairEwald.h and every other HOOMD pair evaluator.
// ESP_TABLE_STRIDE is undefined to avoid polluting downstream translation
// units that may want to include this header with a different stride.
#undef ESP_TABLE_STRIDE

#endif // __PAIR_EVALUATOR_PSWF_H__
