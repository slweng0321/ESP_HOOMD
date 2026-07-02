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
//   1/(4*pi*r) = L(r) + S_mesh(r)
//
// where S_mesh(r) is the smooth long-range part handled by the reciprocal-
// space mesh, and L(r) is the short-range part handled by this pair
// evaluator:
//
//   L(r) = S(r) / (4*pi*r)         (S(r) is the PSWF screening function)
//   -dL/dr = S(r) / (4*pi*r^2) - S'(r) / (4*pi*r)
//
// IMPORTANT ARCHITECTURE FIX (this revision)
// --------------------------------------------
// Earlier revisions of this evaluator stored a RUNTIME device pointer
// (param_type::table_ptr) to a combined, already-singular L(r)/-dL/dr
// look-up table built by ESPForceCompute::buildPSWFTable(), with a
// separate n_segs and V_cut per type-pair. This has been REMOVED. Reasons:
//
//   1. It duplicated the PSWF table in two independently-maintained
//      formats: the raw S(r)/S'(r) tables in PSWF_Coeffs.h (used by
//      ESPForceComputeGPU.cu's mesh-exclusion-correction kernel) and the
//      combined L(r)/-dL/dr table built at runtime by buildPSWFTable().
//      Any mismatch between rcut/n_segs/kappa on the two paths silently
//      produced two DIFFERENT numerical approximations of the same
//      physical kernel -- a correctness bug that is invisible in a single
//      short unit test but shows up as energy drift over long trajectories.
//   2. ESPForceCompute::getTablePtr() returns a pointer into the CPU-side
//      std::vector<ESPTableEntry> (m_pswf_table_cpu.data()), NOT a device
//      pointer. Passing that value through esp.py's uintptr_t bridge and
//      dereferencing it inside a CUDA/HIP kernel (as the old GPU pair
//      evaluator did) is undefined behaviour -- an illegal-address fault
//      waiting to happen on any GPU run.
//
// This evaluator now reads S(r)/S'(r) directly from the SAME compile-time
// PSWF_Coeffs.h tables used by ESPForceComputeGPU.cu and ESPForceCompute.cc,
// via Horner's method, and applies the 1/(4*pi*r) singular factor
// analytically at evaluation time. param_type is reduced to the three
// physical scalars (kappa, alpha, rcut) plus the precomputed energy-shift
// constant V_cut -- no pointers, no table size, no lifetime coupling to any
// GPUArray.
//
// Interface contract with HOOMD-blue
// ------------------------------------
// This header follows the same interface pattern as EvaluatorPairEwald.h
// so it plugs into HOOMD's PotentialPair<> template machinery without any
// additional framework changes.
//
// Reference
// ---------
// Liang, Lu, Barnett, Greengard, Jiang,
// "Accelerating molecular dynamics simulations using fast Ewald summation
// with prolates", Nat. Commun. 2026. https://doi.org/10.1038/s41467-026-73232-8
//
#ifndef __PAIR_EVALUATOR_PSWF_H__
#define __PAIR_EVALUATOR_PSWF_H__

// pybind11 is NOT available inside device (CUDA/HIP) code.
#ifndef __HIPCC__
#include <pybind11/pybind11.h>
#include <string>
#include <stdexcept>
#include <cmath>
#endif

#include "hoomd/HOOMDMath.h"
#include "PSWF_Coeffs.h"

// ── Device / host qualifier macros ──────────────────────────────────────────
#ifdef __HIPCC__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#endif

namespace hoomd
    {
namespace md
    {

namespace detail
    {
//! Shared Horner evaluator for a 5-coefficient PSWF table segment.
//! Identical in structure to gpu_horner5() in ESPForceComputeGPU.cu and
//! horner5() in ESPForceCompute.cc -- all three call sites must stay
//! bit-for-bit consistent.
DEVICE inline Scalar pswf_horner5(const double* coeffs, unsigned int seg, Scalar t)
    {
    const double* c = coeffs + static_cast<size_t>(seg) * 5u;
    Scalar val = static_cast<Scalar>(c[4]);
    val = static_cast<Scalar>(c[3]) + t * val;
    val = static_cast<Scalar>(c[2]) + t * val;
    val = static_cast<Scalar>(c[1]) + t * val;
    val = static_cast<Scalar>(c[0]) + t * val;
    return val;
    }

//! Look up S(r) and S'(r) from the compile-time PSWF_Coeffs.h tables and
//! combine them analytically with 1/(4*pi*r) to produce L(r) and -dL/dr.
//! Shared by EvaluatorPairPSWF::evalForceAndEnergy() and used to precompute
//! V_cut on the host.
DEVICE inline void pswf_eval_L(Scalar r, Scalar rcut, Scalar& out_L, Scalar& out_negdLdr)
    {
    using namespace hoomd::md::esp_pswf_coeffs;

    constexpr double ESP_INV_4PI = 0.079577471545947673;

    if (r < Scalar(1.0e-12) || r > rcut)
        {
        out_L = Scalar(0.0);
        out_negdLdr = Scalar(0.0);
        return;
        }

    Scalar s = r / rcut;
    s = s < Scalar(0.0) ? Scalar(0.0) : (s > Scalar(1.0) - Scalar(1.0e-9) ? Scalar(1.0) - Scalar(1.0e-9) : s);
    unsigned int seg = static_cast<unsigned int>(s * static_cast<Scalar>(PSWF_N_SEGS));
    seg = (seg >= static_cast<unsigned int>(PSWF_N_SEGS)) ? static_cast<unsigned int>(PSWF_N_SEGS) - 1u : seg;

    const Scalar r_lo = static_cast<Scalar>(PSWF_SEG_R_LO[seg]);
    const Scalar dr_inv = static_cast<Scalar>(PSWF_SEG_DR_INV[seg]);
    const Scalar t = (r - r_lo) * dr_inv;

    const Scalar S = pswf_horner5(PSWF_SCREEN_COEFFS, seg, t);
    const Scalar dS = pswf_horner5(PSWF_DSCREEN_COEFFS, seg, t);

    const Scalar inv_4pi_r = static_cast<Scalar>(ESP_INV_4PI) / r;
    const Scalar inv_4pi_r2 = inv_4pi_r / r;

    out_L = S * inv_4pi_r;
    out_negdLdr = S * inv_4pi_r2 - dS * inv_4pi_r;
    }
    } // namespace detail

// =============================================================================
// EvaluatorPairPSWF
// =============================================================================

/*!
 * \class EvaluatorPairPSWF
 * \brief Evaluates the real-space short-range correction L(r) for the ESP
 *        long-range electrostatics method, reading S(r)/S'(r) directly from
 *        the compile-time PSWF_Coeffs.h tables (no runtime table pointer).
 *
 * \par Usage in HOOMD PotentialPair machinery
 * \code
 * EvaluatorPairPSWF::param_type p;
 * p.kappa = 0.34;
 * p.alpha = 0.0;
 * p.rcut = 1.0;          // MUST equal PSWF_RCUT baked into PSWF_Coeffs.h
 * p.V_cut = 0.0;          // usually 0: L(r) already vanishes at r_c
 * pair->setParams(typeA, typeB, p);
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
     * Reduced to plain scalars only (no device pointers, no table size) --
     * safe to place in GPU __constant__ memory verbatim, and free of any
     * cross-object lifetime coupling to ESPForceCompute's GPUArray members.
     */
    struct param_type
        {
        Scalar kappa;   //!< Ewald splitting parameter kappa [1/length] (diagnostics only)
        Scalar alpha;   //!< Debye screening kappa_D [1/length]; 0 = off (diagnostics only)
        Scalar rcut;    //!< Real-space cutoff r_c [length]; MUST equal PSWF_RCUT
        Scalar V_cut;   //!< L(r_c): precomputed energy-shift reference (usually 0)

        DEVICE void load_shared(char*& /*ptr*/, unsigned int& /*available_bytes*/) { }

        HOSTDEVICE void allocate_shared(char*& /*ptr*/, unsigned int& /*available_bytes*/) const { }

#ifdef ENABLE_HIP
        void set_memory_hints() const { }
#endif

#ifndef __HIPCC__
        param_type() : kappa(Scalar(0.0)), alpha(Scalar(0.0)), rcut(Scalar(0.0)), V_cut(Scalar(0.0)) { }

        //! Construct from a Python dict.
        //!
        //! Required key: "rcut".
        //! Optional keys: "kappa" (default 0), "alpha" (default 0),
        //!                "V_cut" (default 0).
        //!
        //! Note: "table_ptr" / "n_segs" are no longer consumed. If present
        //! (e.g. from an older esp.py), they are silently ignored -- the
        //! evaluator now always reads PSWF_Coeffs.h directly.
        param_type(pybind11::dict v, bool /*managed*/ = false)
            {
            rcut = v.contains("rcut") ? v["rcut"].cast<Scalar>() : Scalar(0.0);
            kappa = v.contains("kappa") ? v["kappa"].cast<Scalar>() : Scalar(0.0);
            alpha = v.contains("alpha") ? v["alpha"].cast<Scalar>() : Scalar(0.0);
            V_cut = v.contains("V_cut") ? v["V_cut"].cast<Scalar>() : Scalar(0.0);
            }

        pybind11::dict asDict() const
            {
            pybind11::dict v;
            v["kappa"] = kappa;
            v["alpha"] = alpha;
            v["rcut"] = rcut;
            v["V_cut"] = V_cut;
            return v;
            }
#endif
        } __attribute__((aligned(16)));

    // =========================================================================
    // Construction (called once per pair, inside the neighbour-list loop)
    // =========================================================================

    DEVICE EvaluatorPairPSWF(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
        : rsq(_rsq), rcutsq(_rcutsq), kappa(_params.kappa), alpha(_params.alpha), rcut(_params.rcut),
          V_cut(_params.V_cut), qiqj(Scalar(0.0))
        {
        }

    //! ESP requires per-particle charge (not diameter).
    DEVICE static bool needsCharge()
        {
        return true;
        }

    DEVICE void setCharge(Scalar qi, Scalar qj)
        {
        qiqj = qi * qj;
        }

    //! ESP does not use particle diameters.
    DEVICE static bool needsDiameter()
        {
        return false;
        }

    DEVICE void setDiameter(Scalar, Scalar) { }

    //! Whether this evaluator needs charges to be nonzero to skip work.
    DEVICE bool areInteractionParametersUseless() const
        {
        return rcut < Scalar(1.0e-12);
        }

    // =========================================================================
    // Force / energy evaluation
    // =========================================================================

    /*!
     * \brief Evaluate the pairwise force and energy at the stored rsq.
     *
     * \param force_divr Output: F/r (accumulated into net force via rij)
     * \param pair_eng   Output: pairwise energy contribution
     * \param energy_shift Whether to subtract V_cut so V(r_cut) = 0
     * \return true if this pair is within cutoff and qi*qj != 0
     */
    DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& pair_eng, bool energy_shift)
        {
        if (rsq >= rcutsq || qiqj == Scalar(0.0) || rcut < Scalar(1.0e-12))
            return false;

        const Scalar r = sqrtf(rsq);
        const Scalar rinv = Scalar(1.0) / r;

        Scalar Lr, dLdr_neg;
        detail::pswf_eval_L(r, rcut, Lr, dLdr_neg);

        pair_eng = qiqj * Lr;
        if (energy_shift)
            pair_eng -= qiqj * V_cut;

        // PotentialPair expects force_divr = F / r = -dV/dr / r.
        // V = q_i * q_j * L(r)  =>  -dV/dr = q_i*q_j * (-dL/dr)
        // force_divr = q_i*q_j * (-dL/dr) / r = qiqj * dLdr_neg * rinv
        force_divr = qiqj * dLdr_neg * rinv;

        return true;
        }

    // =========================================================================
    // Long-range correction integrals (isotropic tail)
    // =========================================================================

    //! L(r) decays to exactly zero at r_c by construction (PSWF splitting
    //! property), so the tail correction beyond r_c is exactly zero, not
    //! merely negligible.
    DEVICE Scalar evalPressureLRCIntegral()
        {
        return Scalar(0.0);
        }

    DEVICE Scalar evalEnergyLRCIntegral()
        {
        return Scalar(0.0);
        }

    // =========================================================================
    // Host-only metadata
    // =========================================================================

#ifndef __HIPCC__
    static std::string getName()
        {
        return std::string("pswf");
        }

    std::string getShapeSpec() const
        {
        throw std::runtime_error(
            "EvaluatorPairPSWF: shape specification is not defined for a scalar point-charge pair potential.");
        }
#endif

    protected:
    Scalar rsq;      //!< Squared pair distance (set in constructor)
    Scalar rcutsq;   //!< Squared cutoff r_c^2 (set in constructor)
    Scalar kappa;    //!< Ewald splitting kappa (stored for diagnostics only)
    Scalar alpha;    //!< Debye screening kappa_D (0 = no screening; diagnostics only)
    Scalar rcut;     //!< Physical cutoff r_c [length]
    Scalar V_cut;    //!< L(r_c): energy-shift reference
    Scalar qiqj;     //!< Charge product q_i * q_j (set in setCharge)
    };              // class EvaluatorPairPSWF

    } // namespace md
    } // namespace hoomd

#endif // __PAIR_EVALUATOR_PSWF_H__