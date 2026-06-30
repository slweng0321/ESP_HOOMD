// Copyright (c) 2024-2026 ESP Plugin Contributors.
// Released under the BSD 3-Clause License.
//
// ESPForceCompute.h
//
// Implements the Ewald Summation with Prolates (ESP) method for long-range
// electrostatics in HOOMD-blue. The splitting kernel is based on Prolate
// Spheroidal Wave Functions (PSWF) rather than Gaussians, yielding superior
// spectral accuracy per mesh point compared to standard PPPM.
//
// Reference:
//   Liang, Lu, Barnett, Greengard, Jiang,
//   "Accelerating molecular dynamics simulations using fast Ewald summation
//    with prolates", Nat. Commun. 2026.
//   https://doi.org/10.1038/s41467-026-73232-8

#ifndef __ESP_FORCE_COMPUTE_H__
#define __ESP_FORCE_COMPUTE_H__

// ── Standard library ─────────────────────────────────────────────────────────
#include <memory>
#include <vector>

// ── HOOMD-blue core headers ───────────────────────────────────────────────────
#include "hoomd/ForceCompute.h"
#include "hoomd/GPUArray.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleGroup.h"

// ── HOOMD-blue MD headers (neighbor list lives in the md component) ───────────
#include "NeighborList.h"

// ── MPI / distributed-FFT support ────────────────────────────────────────────
#ifdef ENABLE_MPI
#include "CommunicatorGrid.h"
#include "hoomd/extern/dfftlib/src/dfft_host.h"
#endif

// ── CPU-side FFT (KISS FFT, always available as HOOMD fallback) ───────────────
#include "hoomd/extern/kiss_fftnd.h"

// ── pybind11 (for Python-visible getter/setter methods) ──────────────────────
#include <pybind11/pybind11.h>

namespace hoomd
{
namespace md
{

// ─────────────────────────────────────────────────────────────────────────────
// Compile-time constants
// ─────────────────────────────────────────────────────────────────────────────

//! Numerical epsilon for the optimal influence function denominator
const Scalar ESP_EPS_HOC(1.0e-7);

//! Maximum supported PSWF interpolation order P (Eq. (8) in paper).
//! Values 4–8 cover the practical accuracy range (10^{-4}–10^{-12}).
const unsigned int ESP_MAX_ORDER = 8;

//! Number of piecewise-polynomial segments used to represent L(r) on [0, r_c].
//! 512 segments give sub-ULP interpolation error at double precision.
const unsigned int ESP_TABLE_SEGMENTS = 512;

//! Polynomial degree within each segment of the L(r) look-up table.
//! Degree-4 (quintic Hermite) ensures C^2 continuity and force smoothness.
const unsigned int ESP_TABLE_POLY_DEGREE = 4;

// ─────────────────────────────────────────────────────────────────────────────
// Auxiliary POD struct: one segment of the piecewise-polynomial table
// ─────────────────────────────────────────────────────────────────────────────

//! \brief Stores coefficients for one piecewise-polynomial segment of L(r)
//!        and its negative derivative -dL/dr (the force table).
//!
//! The polynomial is evaluated in Horner form:
//!   L(r) ≈ c[0] + c[1]*t + c[2]*t^2 + c[3]*t^3 + c[4]*t^4
//! where t = (r - r_lo) / dr_seg, t ∈ [0, 1).
//!
//! Packing into Scalar4 keeps the struct 16-byte aligned and allows
//! coalesced reads in both CUDA and CPU code paths.
struct ESPTableEntry
{
    Scalar4 potential_coeffs; //!< c[0..3] of L(r) in Horner form
    Scalar  potential_coeff4; //!< c[4] of L(r) (quintic term)
    Scalar4 force_coeffs;     //!< c[0..3] of -dL/dr in Horner form
    Scalar  force_coeff4;     //!< c[4] of -dL/dr (quintic term)
    Scalar  r_lo;             //!< Left boundary of this segment
    Scalar  dr_inv;           //!< 1 / (segment width) for fast normalisation
    Scalar  _pad0;            //!< Padding for 16-byte alignment
    Scalar  _pad1;
};

// ─────────────────────────────────────────────────────────────────────────────
// Main class declaration
// ─────────────────────────────────────────────────────────────────────────────

/*! \class ESPForceCompute
 *  \brief Computes long-range Coulomb interactions via the ESP (Ewald
 *         Summation with Prolates) method.
 *
 *  The class mirrors the architecture of PPPMForceCompute but replaces:
 *    1. The Gaussian splitting kernel  → PSWF-based kernel χ_α(x).
 *    2. The B-spline charge-assignment → PSWF piecewise-polynomial stencil.
 *    3. The Gaussian influence function→ PSWF Fourier-space influence function.
 *    4. The erfc(κr)/r short-range pair→ PSWF-derived L(r) via look-up table.
 *
 *  The five-step algorithm (spread → FFT → scale → iFFT → gather) and the
 *  MPI/domain-decomposition infrastructure are inherited unchanged from the
 *  HOOMD-blue PPPM framework.
 */
class PYBIND11_EXPORT ESPForceCompute : public ForceCompute
{
public:
    // ── Construction / destruction ────────────────────────────────────────────

    /*! \param sysdef   Shared pointer to the system definition.
     *  \param nlist    Neighbor list used for the real-space pair sum.
     *  \param group    Particle group to apply forces to.
     */
    ESPForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                    std::shared_ptr<NeighborList>      nlist,
                    std::shared_ptr<ParticleGroup>     group);

    virtual ~ESPForceCompute();

    // ── Parameter setter / getters ────────────────────────────────────────────

    /*! \brief Set all ESP parameters and trigger re-initialisation.
     *
     *  \param nx       Global mesh points along x.
     *  \param ny       Global mesh points along y.
     *  \param nz       Global mesh points along z.
     *  \param order    PSWF interpolation order P (1 ≤ P ≤ ESP_MAX_ORDER).
     *  \param kappa    Ewald splitting parameter κ  [1/length].
     *  \param rcut     Real-space cutoff r_c        [length].
     *  \param alpha    Optional Debye screening parameter (0 = no screening).
     *  \param n_table  Number of look-up table segments (default ESP_TABLE_SEGMENTS).
     */
    virtual void setParams(unsigned int nx,
                           unsigned int ny,
                           unsigned int nz,
                           unsigned int order,
                           Scalar       kappa,
                           Scalar       rcut,
                           Scalar       alpha    = Scalar(0.0),
                           unsigned int n_table  = ESP_TABLE_SEGMENTS);

    //! \brief Force recomputation of influence function and look-up tables
    //!        on the next call to computeForces().
    void invalidate() { m_need_initialize = true; }

    // ── Python-visible read-only accessors ────────────────────────────────────

    /// Returns (Nx, Ny, Nz) of the global mesh as a Python tuple.
    pybind11::tuple getResolution() const
    {
        pybind11::list val;
        val.append(m_global_dim.x);
        val.append(m_global_dim.y);
        val.append(m_global_dim.z);
        return pybind11::tuple(val);
    }

    unsigned int getOrder()  const { return static_cast<unsigned int>(m_order); }
    Scalar       getKappa()  const { return m_kappa; }
    Scalar       getRCut()   const { return m_rcut;  }
    Scalar       getAlpha()  const { return m_alpha; }

    //! Returns the total system charge Σ q_i (after MPI reduction).
    Scalar getQSum()  const { return m_q;  }

    //! Returns Σ q_i^2 (used for self-energy correction).
    Scalar getQ2Sum() const { return m_q2; }

    //! Returns the number of look-up table segments currently allocated.
    unsigned int getTableSize() const { return m_n_table_segments; }

    // ── Main entry point (overrides ForceCompute) ─────────────────────────────

    /*! \brief Computes forces, energies, and virial contributions for all
     *         particles in m_group at the current time step.
     *
     *  Calls, in order:
     *    1. setupMesh()              (only on first call / after parameter change)
     *    2. initializeFFT()          (only on first call / after box change)
     *    3. setupCoeffs()            (only on first call)
     *    4. buildPSWFTable()         (only on first call)
     *    5. computeInfluenceFunction()
     *    6. assignParticles()
     *    7. updateMeshes()           (FFT → scale → iFFT)
     *    8. interpolateForces()
     *    9. computePE()
     *   10. computeVirial()
     *   11. fixExclusions()
     *
     *  \param timestep  Current simulation step (used for Autotuner state).
     */
    void computeForces(uint64_t timestep) override;

    // ── MPI ghost communication flags ─────────────────────────────────────────
#ifdef ENABLE_MPI
    /*! \brief Requests charge (and optionally body) ghost data from the
     *         HOOMD communicator whenever exclusions or body corrections are
     *         active.
     */
    virtual CommFlags getRequestedCommFlags(uint64_t timestep) override
    {
        CommFlags flags = ForceCompute::getRequestedCommFlags(timestep);
        bool correct_body = m_nlist->getFilterBody();
        if (m_nlist->getExclusionsSet() || correct_body)
        {
            flags[comm_flag::charge] = 1;
        }
        if (correct_body)
        {
            flags[comm_flag::body] = 1;
        }
        return flags;
    }
#endif

protected:
    // =========================================================================
    // ── Shared pointers to HOOMD objects ─────────────────────────────────────
    // =========================================================================

    std::shared_ptr<NeighborList>  m_nlist; //!< Neighbor list for real-space sum
    std::shared_ptr<ParticleGroup> m_group; //!< Particle group acted upon

    // =========================================================================
    // ── Mesh / grid geometry ──────────────────────────────────────────────────
    // =========================================================================

    uint3    m_mesh_points;    //!< Local mesh subdivisions (after MPI decomp)
    uint3    m_global_dim;     //!< Global mesh dimensions (Nx, Ny, Nz)
    uint3    m_n_ghost_cells;  //!< Ghost cells per axis for stencil overlap
    uint3    m_grid_dim;       //!< Total grid size including ghost cells
    Scalar3  m_ghost_width;    //!< Physical width of the ghost layer [length]
    unsigned int m_ghost_offset; //!< Linear index offset caused by ghost cells
    unsigned int m_n_cells;      //!< Total inner cells on this rank
    unsigned int m_n_inner_cells;//!< Inner cells visible to FFT (no ghosts)
    unsigned int m_radius;       //!< Half-stencil width = ceil(P/2)

    // =========================================================================
    // ── Physical parameters ───────────────────────────────────────────────────
    // =========================================================================

    Scalar m_kappa;   //!< Ewald splitting parameter κ  [1/length]
    Scalar m_rcut;    //!< Real-space cutoff r_c          [length]
    int    m_order;   //!< PSWF interpolation order P (controls stencil width)
    Scalar m_alpha;   //!< Debye screening length^{-1}    [1/length], 0 = off
    Scalar m_q;       //!< Total charge Σ q_i (reduced across MPI ranks)
    Scalar m_q2;      //!< Sum of squares Σ q_i^2

    // =========================================================================
    // ── State flags ───────────────────────────────────────────────────────────
    // =========================================================================

    bool m_need_initialize;    //!< Triggers full re-init on next compute
    bool m_params_set;         //!< True once setParams() has been called
    bool m_box_changed;        //!< True if box changed → FFT plan must rebuild
    bool m_ptls_added_removed; //!< True if particle count changed

    Scalar m_body_energy;      //!< Rigid-body self-energy correction

    // =========================================================================
    // ── PSWF stencil (charge assignment / force interpolation) ───────────────
    // =========================================================================

    //! PSWF piecewise-polynomial coefficients for charge assignment and
    //! force interpolation on the mesh.  Layout mirrors m_rho_coeff in PPPM:
    //!   m_rho_coeff[l + iorder * (2*P+1)]  for l ∈ [-P, P], iorder ∈ [0, P).
    //! Size: P * (2*P + 1)  Scalars.
    GPUArray<Scalar> m_rho_coeff;

    //! PSWF-based Green's function denominator coefficients (replaces gf_b
    //! in PPPM).  Size: P  Scalars.
    GPUArray<Scalar> m_gf_b;

    // =========================================================================
    // ── Reciprocal-space arrays ───────────────────────────────────────────────
    // =========================================================================

    //! Optimal influence function G̃(k): real-valued, defined on the local
    //! portion of k-space.  Size: m_n_inner_cells.
    GPUArray<Scalar>  m_inf_f;

    //! k-vector mesh Scalar3(kx, ky, kz) for each inner cell.
    //! Size: m_n_inner_cells.
    GPUArray<Scalar3> m_k;

    //! Short-wavelength cutoff k* squared (for DC term exclusion).
    Scalar m_qstarsq;

    //! Virial tensor contributions in k-space (6 independent components per
    //! inner cell).  Size: 6 * m_n_inner_cells.
    GPUArray<Scalar> m_virial_mesh;

    // =========================================================================
    // ── Real-space look-up table for L(r) = -dV_real/dr / q_i q_j ───────────
    // =========================================================================

    //! Number of segments in the piecewise-polynomial table for L(r).
    unsigned int m_n_table_segments;

    //! CPU-side table of ESPTableEntry structs.  Populated once in
    //! buildPSWFTable(); subsequently uploaded to m_pswf_table_gpu.
    std::vector<ESPTableEntry> m_pswf_table_cpu;

    //! GPU-side look-up table for L(r), interleaved as flat Scalar array for
    //! coalesced device reads.
    //! Layout (per segment s, polynomial coefficients c[0..4]):
    //!   potential: m_pswf_table_gpu[ s * 12 + 0 .. 4 ]
    //!   force    : m_pswf_table_gpu[ s * 12 + 5 .. 9 ]
    //!   r_lo     : m_pswf_table_gpu[ s * 12 + 10 ]
    //!   dr_inv   : m_pswf_table_gpu[ s * 12 + 11 ]
    GPUArray<Scalar> m_pswf_table_gpu;

    //! Precomputed PSWF self-energy constant:
    //!   I_pswf = ∫_0^∞ ψ(r/r_c)^2 dr / r_c
    //! Used to replace the PPPM Gaussian self-energy correction
    //!   E_self = -κ/√π · Σ q_i^2
    //! with the correct ESP analogue.
    Scalar m_pswf_self_energy_const;

    // =========================================================================
    // ── CPU-side FFT (KISS FFT) ───────────────────────────────────────────────
    // =========================================================================

    kiss_fftnd_cfg m_kiss_fft;     //!< Forward FFT plan (CPU path)
    kiss_fftnd_cfg m_kiss_ifft;    //!< Inverse FFT plan (CPU path)
    bool           m_kiss_fft_initialized; //!< Guard for KISS FFT lifecycle

    // =========================================================================
    // ── CPU-side mesh buffers (kiss_fft / host-dfft path) ────────────────────
    // =========================================================================

    //! Complex charge density mesh; ghost-padded.
    //! Size: m_n_cells + m_ghost_offset.
    GPUArray<kiss_fft_cpx> m_mesh;

    //! Fourier-transformed charge mesh: ρ̃(k).
    //! Size: m_n_inner_cells.
    GPUArray<kiss_fft_cpx> m_fourier_mesh;

    //! G̃(k) · ik_x · ρ̃(k): x-component of force mesh in reciprocal space.
    GPUArray<kiss_fft_cpx> m_fourier_mesh_G_x;
    //! G̃(k) · ik_y · ρ̃(k): y-component.
    GPUArray<kiss_fft_cpx> m_fourier_mesh_G_y;
    //! G̃(k) · ik_z · ρ̃(k): z-component.
    GPUArray<kiss_fft_cpx> m_fourier_mesh_G_z;

    //! Inverse-transformed force mesh, x-component; ghost-padded.
    GPUArray<kiss_fft_cpx> m_inv_fourier_mesh_x;
    //! Inverse-transformed force mesh, y-component; ghost-padded.
    GPUArray<kiss_fft_cpx> m_inv_fourier_mesh_y;
    //! Inverse-transformed force mesh, z-component; ghost-padded.
    GPUArray<kiss_fft_cpx> m_inv_fourier_mesh_z;

    // =========================================================================
    // ── MPI / distributed-FFT members ────────────────────────────────────────
    // =========================================================================

#ifdef ENABLE_MPI
    //! Distributed (host) FFT plan for the forward transform ρ → ρ̃.
    dfft_plan m_dfft_plan_forward;

    //! Distributed (host) FFT plan for the inverse transforms F̃ → F.
    dfft_plan m_dfft_plan_inverse;

    //! True once the host dfft plans have been created.
    bool m_dfft_initialized;

    //! Ghost-cell communicator: propagates charge mesh halos to neighbouring
    //! ranks before the forward FFT (forward = src → ghost direction).
    std::unique_ptr<CommunicatorGrid<kiss_fft_cpx>> m_grid_comm_forward;

    //! Ghost-cell communicator: propagates force mesh halos back after the
    //! inverse FFT (reverse = ghost → src direction).
    std::unique_ptr<CommunicatorGrid<kiss_fft_cpx>> m_grid_comm_reverse;
#endif

    // =========================================================================
    // ── Protected virtual pipeline methods ───────────────────────────────────
    // =========================================================================

    //! \brief (Re)compute mesh geometry: ghost cell counts, grid dimensions,
    //!        and the per-rank local mesh_points.  Called once per setParams().
    void setupMesh();

    //! \brief Allocate mesh GPUArrays and create FFT plans.
    //!        Called once per setParams() or when the simulation box changes.
    virtual void initializeFFT();

    //! \brief Compute the PSWF-based optimal influence function G̃(k).
    //!
    //! Replaces the Gaussian-based influence function of PPPM.  The function
    //! evaluates:
    //!   G̃(k) = |k|^{-2} · |ψ̂(k r_c / σ_0)|^2
    //!           / |Σ_{m∈Z^3} ψ̂((k + 2πm/h) r_c / σ_0)|^2
    //! where ψ̂ is the Fourier transform of the PSWF splitting kernel χ_α.
    virtual void computeInfluenceFunction();

    //! \brief Assign particle charges to the mesh using PSWF stencil.
    //!        Equivalent to PPPMForceCompute::assignParticles(), but uses
    //!        m_rho_coeff populated by compute_pswf_rho_coeff().
    virtual void assignParticles();

    //! \brief Forward FFT, multiply by influence function, inverse FFT.
    //!        Populates m_inv_fourier_mesh_{x,y,z}.
    virtual void updateMeshes();

    //! \brief Gather mesh forces back onto particles using PSWF stencil.
    virtual void interpolateForces();

    //! \brief Compute the reciprocal-space energy contribution.
    //!
    //! Returns E_k = (1/2V) Σ_k G̃(k) |ρ̃(k)|^2
    //! minus the PSWF self-energy correction:
    //!   E_self = -m_pswf_self_energy_const · Σ_i q_i^2
    virtual Scalar computePE();

    //! \brief Compute the reciprocal-space virial tensor (6 components).
    virtual void computeVirial();

    //! \brief Correct forces and energies on excluded pairs by subtracting
    //!        the mesh contribution and adding back the direct Coulomb term.
    //!        Uses the PSWF look-up table L(r) for the short-range correction.
    virtual void fixExclusions();

    //! \brief Recompute m_rho_coeff and m_gf_b from PSWF coefficients.
    //!        Called once per setParams().  Replaces compute_rho_coeff() and
    //!        compute_gf_denom() in the PPPM code.
    virtual void setupCoeffs();

    //! \brief Compute the rigid-body self-energy correction m_body_energy.
    virtual void computeBodyCorrection();

    // ── Signal slots ──────────────────────────────────────────────────────────

    //! Called by ParticleData signal when N changes.
    void slotGlobalParticleNumberChange() { m_ptls_added_removed = true; }

    //! Called by BoxDim signal when box dimensions change.
    void setBoxChange() { m_box_changed = true; }

private:
    // =========================================================================
    // ── Private PSWF coefficient helpers ─────────────────────────────────────
    // =========================================================================

    //! \brief Populate m_rho_coeff with PSWF piecewise-polynomial coefficients.
    //!
    //! The coefficients encode the PSWF kernel evaluated on the stencil offsets
    //! l ∈ {-⌊P/2⌋, …, ⌈P/2⌉} for each polynomial degree iorder ∈ [0, P).
    //! Layout identical to PPPMForceCompute::compute_rho_coeff() so that the
    //! GPU kernels (gpu_assign_particles, gpu_compute_forces) need no changes.
    void compute_pswf_rho_coeff();

    //! \brief Populate m_gf_b with the PSWF Green's function denominator
    //!        coefficients, analogous to PPPMForceCompute::compute_gf_denom().
    void compute_pswf_gf_denom();

    //! \brief Evaluate the PSWF Green's function denominator at (x, y, z)
    //!        using Horner's method.  Mirrors PPPMForceCompute::gf_denom().
    Scalar gf_denom_pswf(Scalar x, Scalar y, Scalar z) const;

    // =========================================================================
    // ── Private look-up table builder ─────────────────────────────────────────
    // =========================================================================

    //! \brief Build the piecewise-polynomial look-up table for L(r) and
    //!        its negative derivative -dL/dr on [0, r_cut].
    //!
    //! Called once in setupCoeffs() after m_kappa, m_rcut, m_order are set.
    //! Fills m_pswf_table_cpu and uploads to m_pswf_table_gpu.
    //! Also computes m_pswf_self_energy_const via Gauss-Legendre quadrature.
    void buildPSWFTable();

    //! \brief Evaluate the PSWF splitting kernel χ_α(x) at a single point x.
    //!        Used internally during table construction (CPU only).
    //! \param x  Dimensionless argument x = r * κ.
    Scalar evalPSWFKernel(Scalar x) const;

    //! \brief Evaluate L(r) = (1/(4πr)) * (1 - CDF_ψ(r/r_c)) via quadrature,
    //!        plus the 1/r Coulomb term.  Used to fill table entries.
    //! \param r  Physical distance.
    Scalar evalL_direct(Scalar r) const;

    //! \brief Evaluate -dL/dr for the force table.
    Scalar evalDL_direct(Scalar r) const;

    //! \brief Compute PSWF self-energy constant via numerical integration:
    //!   I = ∫_0^{r_c} ψ^2(r/r_c) / r_c  dr
    Scalar computePSWFSelfEnergyConst() const;

    // =========================================================================
    // ── Private geometry helpers ──────────────────────────────────────────────
    // =========================================================================

    //! \brief Compute ghost cell counts for current m_order and m_mesh_points.
    uint3 computeGhostCellNum() const;

    //! \brief Compute virial contributions from the mesh (k-space only).
    void computeVirialMesh();

    //! \brief RMS force error estimate; mirrors PPPMForceCompute::rms().
    Scalar rms(Scalar h, Scalar prd, Scalar natoms) const;

    // =========================================================================
    // ── Private state guard ───────────────────────────────────────────────────
    // =========================================================================

    //! True once m_kiss_fft and m_kiss_ifft have been allocated.
    bool m_kiss_fft_initialized_priv;
};

// ─────────────────────────────────────────────────────────────────────────────
// Free function: export Python bindings (called from module.cc)
// ─────────────────────────────────────────────────────────────────────────────

//! \brief Register ESPForceCompute with pybind11.
//! \param m  The pybind11 module object (hoomd.md or the plugin root module).
void export_ESPForceCompute(pybind11::module& m);

} // namespace md
} // namespace hoomd

#endif // __ESP_FORCE_COMPUTE_H__
