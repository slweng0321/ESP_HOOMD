// Copyright (c) 2024-2026 ESP Plugin Contributors.
// Released under the BSD 3-Clause License.
//
// ESPForceCompute.h
//
// CPU-path ForceCompute plugin implementing the Ewald Summation with Prolates
// (ESP) method for HOOMD-blue.
//
// Design philosophy
// -----------------
// * Mirrors PPPMForceCompute's five-step pipeline
//   (spread → FFT → scale → iFFT → gather).
// * Replaces every Gaussian / B-spline piece with PSWF-based equivalents:
//     - Reciprocal mesh coefficients  ← compute_pswf_rho_coeff()
//     - Influence function G̃(k)      ← computeInfluenceFunction()
//     - Short-range potential L(r)    ← buildPSWFTable()
//     - Self-energy correction        ← computePSWFSelfEnergyConst()
// * The lookup table (m_pswf_table_cpu / m_pswf_table_gpu) stores
//   degree-4 piecewise polynomials for L(r) and -dL/dr so that no
//   PSWF evaluation occurs in the inner force loop.
//
// Reference:
//   Liang, Lu, Barnett, Greengard, Jiang,
//   "Accelerating molecular dynamics simulations using fast Ewald summation
//    with prolates", Nat. Commun. 2026.
//   https://doi.org/10.1038/s41467-026-73232-8

#pragma once

#include "hoomd/ForceCompute.h"
#include "hoomd/GPUArray.h"
#include "hoomd/Index1D.h"
#include "hoomd/MeshDefinition.h"
#include "hoomd/ParticleGroup.h"

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#include "hoomd/extern/dfftlib/src/dfft_host.h"
#include <mpi.h>
#endif

#include "hoomd/extern/kiss_fft/tools/kiss_fftnd.h"

#include <pybind11/pybind11.h>

#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

namespace hoomd
{
namespace md
{

// ============================================================================
// Compile-time constants
// ============================================================================

/// Maximum allowed PSWF interpolation order.
constexpr int ESP_MAX_ORDER = 12;

/// Degree of the piecewise polynomial used for the PSWF table.
constexpr unsigned int ESP_TABLE_POLY_DEGREE = 4u;

/// Default number of piecewise-polynomial segments for the L(r) lookup table.
constexpr unsigned int ESP_TABLE_SEGMENTS = 4096;

// ============================================================================
// ESPTableEntry — one segment of the L(r) / (-dL/dr) lookup table
//
// Memory layout (12 Scalars per segment, matches GPU flat upload):
//   [0..4]   potential polynomial coefficients c0…c4
//   [5..9]   force     polynomial coefficients c0…c4
//   [10]     r_lo  (left endpoint of segment)
//   [11]     dr_inv (= 1 / (r_hi - r_lo), segment reciprocal width)
// No padding is included so the struct maps directly to a flat Scalar array.
// ============================================================================
struct ESPTableEntry
    {
    Scalar4 potential_coeffs;   // c0, c1, c2, c3 of L(r) polynomial
    Scalar  potential_coeff4;   // c4 of L(r) polynomial
    Scalar4 force_coeffs;       // c0, c1, c2, c3 of (-dL/dr) polynomial
    Scalar  force_coeff4;       // c4 of (-dL/dr) polynomial
    Scalar  r_lo;               // left edge of this segment [MD length units]
    Scalar  dr_inv;             // 1 / segment width
    };
static_assert(sizeof(ESPTableEntry) == 12 * sizeof(Scalar),
              "ESPTableEntry size mismatch — update GPU upload code if changed.");

// ============================================================================
// ESPForceCompute
// ============================================================================

class PYBIND11_EXPORT ESPForceCompute : public ForceCompute
    {
    public:
    // ----------------------------------------------------------------
    // Construction / destruction
    // ----------------------------------------------------------------

    /// Construct a new ESPForceCompute.
    ///
    /// @param sysdef    Owning system definition.
    /// @param nlist     Neighbour list (required for exclusion correction).
    /// @param group     Particle group to which long-range forces are applied.
    ESPForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                    std::shared_ptr<NeighborList>      nlist,
                    std::shared_ptr<ParticleGroup>     group);

    ~ESPForceCompute() override;

    // ----------------------------------------------------------------
    // Parameter setup
    // ----------------------------------------------------------------

    /// Set all ESP algorithm parameters.
    ///
    /// Must be called before the first computeForces() invocation.
    ///
    /// @param nx        Global mesh points in x.
    /// @param ny        Global mesh points in y.
    /// @param nz        Global mesh points in z.
    /// @param order     PSWF interpolation order P ∈ [1, ESP_MAX_ORDER].
    /// @param kappa     Ewald screening parameter κ [1/length].
    /// @param rcut      Real-space cutoff r_c.
    /// @param alpha     Debye-Hückel screening length α  (0 = plain Coulomb).
    /// @param n_table   Number of piecewise-polynomial table segments (≥16).
    void setParams(unsigned int nx,
                   unsigned int ny,
                   unsigned int nz,
                   unsigned int order,
                   Scalar       kappa,
                   Scalar       rcut,
                   Scalar       alpha   = Scalar(0.0),
                   unsigned int n_table = ESP_TABLE_SEGMENTS);

    // ----------------------------------------------------------------
    // Inspectors (Python-accessible)
    // ----------------------------------------------------------------
    uint3        getResolution() const { return m_global_dim; }
    unsigned int getOrder()      const { return static_cast<unsigned int>(m_order); }
    Scalar       getKappa()      const { return m_kappa;  }
    Scalar       getRCut()       const { return m_rcut;   }
    Scalar       getAlpha()      const { return m_alpha;  }
    Scalar       getQSum()       const { return m_q;      }
    Scalar       getQ2Sum()      const { return m_q2;     }
    unsigned int getTableSize()  const { return m_n_table_segments; }
    uintptr_t    getTablePtr()   const;
    const std::vector<ESPTableEntry>& getTable() const { return m_pswf_table_cpu; }

    /// Force lazy re-initialisation on the next compute step.
    void invalidate() { m_need_initialize = true; }

    // ----------------------------------------------------------------
    // Signal handlers (connected in constructor)
    // ----------------------------------------------------------------
    void setBoxChange()                    { m_box_changed        = true; }
    void slotGlobalParticleNumberChange()  { m_ptls_added_removed = true; }

    protected:
    // ----------------------------------------------------------------
    // Main compute entry point
    // ----------------------------------------------------------------
    void computeForces(uint64_t timestep) override;

    // ================================================================
    // Step 0: initialisation helpers (called once per param/box change)
    // ================================================================

    /// Compute ghost-cell dimensions and domain geometry.
    void setupMesh();

    /// (Re-)allocate mesh GPUArrays and FFT plans.
    void initializeFFT();

    /// Compute PSWF rho coefficients, gf_b, build table, and self-energy constant.
    void setupCoeffs();

    // ================================================================
    // PSWF coefficient builders
    // ================================================================

    /// Compute piecewise-polynomial charge-assignment (rho) coefficients
    /// from the PSWF kernel and store them in m_rho_coeff.
    void compute_pswf_rho_coeff();

    /// Compute the aliasing-denominator coefficients m_gf_b.
    void compute_pswf_gf_denom();

    /// Evaluate the aliasing denominator at dimensionless squared
    /// half-wavevectors (x, y, z) = (kx·hx/2)^2, etc.
    Scalar gf_denom_pswf(Scalar x, Scalar y, Scalar z) const;

    // ================================================================
    // Step 1: Influence function  G̃(k)
    // ================================================================

    /// Fill m_inf_f and m_k with the ESP optimal influence function values.
    void computeInfluenceFunction();

    // ================================================================
    // Step 2: Charge assignment
    // ================================================================

    /// Spread particle charges onto m_mesh using PSWF weights.
    void assignParticles();

    // ================================================================
    // Step 3: Forward FFT → scale by G̃ → inverse FFT
    // ================================================================

    /// Forward FFT, multiply by G̃(k), then IFFT three force-component meshes.
    void updateMeshes();

    // ================================================================
    // Step 4: Force interpolation
    // ================================================================

    /// Gather electric force from the real-space inverse meshes.
    void interpolateForces();

    // ================================================================
    // Step 5: Corrections
    // ================================================================

    /// Compute reciprocal-space energy (including self-energy subtraction).
    Scalar computePE();

    /// Fill m_virial_mesh with per-k virial contributions.
    void computeVirialMesh();

    /// Sum virial mesh and assign to m_virial.
    void computeVirial();

    /// Subtract mesh double-counting and add real-space remainder
    /// for all excluded pairs (Newton-III safe).
    void fixExclusions();

    /// Rigid-body intramolecular self-energy correction (stub; extend as needed).
    void computeBodyCorrection();

    // ================================================================
    // PSWF kernel and lookup table
    // ================================================================

    /// Evaluate the normalised PSWF kernel χ_α(x) at x ∈ [0, 1].
    /// Returns (1-x^2)^P · exp_approx(β(1-x^2)).
    Scalar evalPSWFKernel(Scalar x) const;

    /// Compute L(r) = (1 - CDF_ψ(r/r_c)) / (4π r) by direct numerical
    /// integration.  Used during table construction; not called in inner loops.
    Scalar evalL_direct(Scalar r) const;

    /// Compute -dL/dr by centred finite difference of evalL_direct().
    /// Used during table construction; not called in inner loops.
    Scalar evalDL_direct(Scalar r) const;

    /// Build the piecewise-polynomial lookup table for L(r) and -dL/dr,
    /// populating m_pswf_table_cpu and uploading to m_pswf_table_gpu.
    void buildPSWFTable();

    /// Upload the CPU PSWF table into the flat GPU buffer.
    void uploadPSWFTable();

    /// Return the PSWF self-energy constant I = ∫_0^1 ψ^2(u) du / r_c.
    Scalar computePSWFSelfEnergyConst() const;

    // ================================================================
    // Geometry helpers
    // ================================================================

    /// Return the number of ghost cells needed per direction.
    uint3 computeGhostCellNum() const;

    /// RMS force error estimate (Hockney-Eastwood style).
    /// @param h      Grid spacing.
    /// @param prd    Box length in that direction.
    /// @param natoms Number of atoms.
    Scalar rms(Scalar h, Scalar prd, Scalar natoms) const;

    // ================================================================
    // Particle data references
    // ================================================================
    std::shared_ptr<NeighborList>  m_nlist;   ///< Neighbour list (exclusions)
    std::shared_ptr<ParticleGroup> m_group;   ///< Target particle group

    // ================================================================
    // Mesh dimensions
    // ================================================================
    uint3  m_mesh_points;    ///< Local mesh points (= global_dim / MPI-decomp)
    uint3  m_global_dim;     ///< Global mesh dimensions (Nx, Ny, Nz)
    uint3  m_n_ghost_cells;  ///< Ghost-cell halos per direction
    uint3  m_grid_dim;       ///< mesh_points + 2 * ghost cells (padded array)
    Scalar3 m_ghost_width;   ///< Ghost halo width in simulation length units
    size_t m_ghost_offset;   ///< Linear offset into m_mesh where inner data start
    size_t m_n_cells;        ///< Total padded mesh size  (m_grid_dim product)
    size_t m_n_inner_cells;  ///< Inner mesh size (m_mesh_points product)

    // ================================================================
    // ESP algorithm parameters
    // ================================================================
    unsigned int m_radius;  ///< Stencil half-width = (m_order + 1) / 2
    Scalar       m_kappa;   ///< Ewald screening parameter
    Scalar       m_rcut;    ///< Real-space cutoff
    int          m_order;   ///< PSWF interpolation order P
    Scalar       m_alpha;   ///< DH inverse screening length (0 = pure Coulomb)
    Scalar       m_pswf_c;   ///< PSWF compact-support parameter c
    Scalar       m_q;       ///< Total charge (sum of q_i in group)
    Scalar       m_q2;      ///< Sum of q_i^2 (needed for self-energy)

    // ================================================================
    // State flags
    // ================================================================
    bool m_need_initialize;     ///< Re-initialise before next compute step
    bool m_params_set;          ///< setParams() has been called
    bool m_box_changed;         ///< Box change since last initialise
    bool m_ptls_added_removed;  ///< Particle count changed

    // ================================================================
    // Energy accumulators
    // ================================================================
    Scalar m_body_energy;         ///< Rigid-body self-correction energy
    Scalar m_qstarsq;             ///< Nyquist k^2 (DC guard for G̃)
    Scalar m_pswf_self_energy_const; ///< I = ∫ ψ^2 du / r_c

    // ================================================================
    // Lookup table storage
    // ================================================================
    unsigned int                m_n_table_segments;  ///< Number of table segments
    std::vector<ESPTableEntry>  m_pswf_table_cpu;    ///< CPU copy of lookup table
    GPUArray<Scalar>            m_pswf_table_gpu;    ///< Flat GPU buffer (12 Scalars/seg)

    // ================================================================
    // Mesh GPU buffers
    // ================================================================
    GPUArray<kiss_fft_cpx>  m_mesh;               ///< Real-space charge density
    GPUArray<kiss_fft_cpx>  m_fourier_mesh;        ///< ρ̃(k)
    GPUArray<kiss_fft_cpx>  m_fourier_mesh_G_x;    ///< G̃·ρ̃·ik_x
    GPUArray<kiss_fft_cpx>  m_fourier_mesh_G_y;    ///< G̃·ρ̃·ik_y
    GPUArray<kiss_fft_cpx>  m_fourier_mesh_G_z;    ///< G̃·ρ̃·ik_z
    GPUArray<kiss_fft_cpx>  m_inv_fourier_mesh_x;  ///< IFFT of G̃·ρ̃·ik_x
    GPUArray<kiss_fft_cpx>  m_inv_fourier_mesh_y;
    GPUArray<kiss_fft_cpx>  m_inv_fourier_mesh_z;
    GPUArray<Scalar>        m_inf_f;               ///< Influence function G̃(k)
    GPUArray<Scalar3>       m_k;                   ///< k-vector for each cell
    GPUArray<Scalar>        m_virial_mesh;          ///< Per-k virial (6-component)

    // ================================================================
    // PSWF coefficient arrays
    // ================================================================

    /// Charge-assignment (rho) coefficients.
    /// Layout identical to PPPMForceCompute::m_rho_coeff:
    ///   size = order * (2*order + 1)
    ///   m_rho_coeff[(offset - nlower) + degree * (2*order+1)] = c_d^{offset}
    GPUArray<Scalar> m_rho_coeff;

    /// Aliasing-denominator coefficients (degree m_order-1 polynomial in k).
    /// Layout: m_gf_b[l] = (-1)^l · β^l / (2l)!
    GPUArray<Scalar> m_gf_b;

    // ================================================================
    // FFT handles (kiss_fft — serial single-node path)
    // ================================================================
    kiss_fftnd_cfg m_kiss_fft;               ///< Forward FFT plan
    kiss_fftnd_cfg m_kiss_ifft;              ///< Inverse FFT plan
    bool           m_kiss_fft_initialized;
    bool           m_kiss_fft_initialized_priv;

#ifdef ENABLE_MPI
    // ================================================================
    // Distributed FFT handles (dfft — MPI path)
    // ================================================================
    dfft_plan m_dfft_plan_forward;
    dfft_plan m_dfft_plan_inverse;
    bool      m_dfft_initialized;

    std::unique_ptr<CommunicatorGrid<kiss_fft_cpx>> m_grid_comm_forward;
    std::unique_ptr<CommunicatorGrid<kiss_fft_cpx>> m_grid_comm_reverse;
#endif
    };

// ============================================================================
// Python binding declaration (defined in ESPForceCompute.cc)
// ============================================================================
void export_ESPForceCompute(pybind11::module& m);

} // namespace md
} // namespace hoomd
