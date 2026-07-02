// Copyright (c) 2024-2026 ESP Plugin Contributors.
// Released under the BSD 3-Clause License.
//
// ESPForceCompute.h
//
// CPU-path ForceCompute plugin implementing the Ewald Summation with Prolates
// (ESP) method for HOOMD-blue.
//
// Design notes for this revision
// --------------------------------
//  * PSWF_Coeffs.h is now included directly by this header. Both the CPU
//    real-space correction path (computeRealSpaceCorrection(),
//    computePSWFSelfEnergyConst()) and the GPU kernels in
//    ESPForceComputeGPU.cu draw from the SAME compile-time-generated
//    S(r)/S'(r) Horner tables -- eliminating any possibility of the CPU and
//    GPU paths silently diverging onto two different numerical
//    approximations of the splitting kernel.
//  * ESPTableEntry / m_pswf_table_cpu / m_pswf_table_gpu are retained
//    strictly for Python-side introspection (ESPForceCompute.table
//    property, used for validation/plotting in esp.py and unit tests).
//    They are populated by evaluating the analytic split
//    L(r) = S(r)/(4*pi*r), -dL/dr = S(r)/(4*pi*r^2) - S'(r)/(4*pi*r) at
//    table nodes -- they are NOT the runtime-critical path any more (that
//    role is filled by EvaluatorPairPSWF and the GPU kernels reading
//    PSWF_Coeffs.h directly).
//  * All device memory is owned via hoomd::GPUArray<T> (v5 API). No raw
//    CUDA/HIP allocation calls appear in this header or its .cc/.cu
//    counterparts.
//
#ifndef ESP_FORCE_COMPUTE_H_
#define ESP_FORCE_COMPUTE_H_

#include "hoomd/ForceCompute.h"
#include "hoomd/GPUArray.h"
#include "hoomd/GPUFlags.h"
#include "hoomd/Index1D.h"
#include "hoomd/ParticleGroup.h"
#include "hoomd/md/CommunicatorGrid.h"
#include "hoomd/md/NeighborList.h"

#include "PSWF_Coeffs.h"

#ifdef ENABLE_MPI
#include "hoomd/extern/dfftlib/src/dfft_host.h"
#include <mpi.h>
#endif

#include "hoomd/extern/kiss_fft/tools/kiss_fftnd.h"

#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace hoomd
    {
namespace md
    {

constexpr int ESP_MAX_ORDER = 12;
constexpr unsigned int ESP_TABLE_POLY_DEGREE = 4u;

// The CPU-side introspection table now mirrors the GPU compile-time table's
// segment count by default, so ESPForceCompute.table and the GPU kernels
// report numerically identical results at matching r values.
constexpr unsigned int ESP_TABLE_SEGMENTS
    = static_cast<unsigned int>(hoomd::md::esp_pswf_coeffs::PSWF_N_SEGS);

//! One segment of the ESP short-range lookup table, exposed to Python for
//! validation/plotting purposes only. Stores the fully-combined L(r)/-dL/dr
//! Horner coefficients (i.e. the 1/(4*pi*r) factor has ALREADY been folded
//! in at construction time by buildPSWFTable() -- unlike the raw S(r)/S'(r)
//! tables in PSWF_Coeffs.h, which keep the singular factor separate for
//! numerical stability during runtime evaluation).
struct alignas(16) ESPTableEntry
    {
    Scalar4 potential_coeffs;   //!< L(r) Horner coefficients c[0..3]
    Scalar4 force_coeffs;       //!< -dL/dr Horner coefficients c[0..3]
    Scalar potential_coeff4;    //!< L(r) Horner coefficient c[4]
    Scalar force_coeff4;        //!< -dL/dr Horner coefficient c[4]
    Scalar r_lo;                //!< Left edge of segment, physical distance
    Scalar dr_inv;              //!< 1 / segment width
    };

static_assert(sizeof(ESPTableEntry) % 16 == 0, "ESPTableEntry must be 16-byte aligned.");

class PYBIND11_EXPORT ESPForceCompute : public ForceCompute
    {
    public:
    ESPForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                     std::shared_ptr<NeighborList> nlist,
                     std::shared_ptr<ParticleGroup> group);

    ~ESPForceCompute() override;

    void setParams(unsigned int nx,
                    unsigned int ny,
                    unsigned int nz,
                    unsigned int order,
                    Scalar kappa,
                    Scalar rcut,
                    Scalar alpha = Scalar(0.0),
                    unsigned int n_table = ESP_TABLE_SEGMENTS);

    uint3 getResolution() const { return m_global_dim; }
    unsigned int getOrder() const { return static_cast<unsigned int>(m_order); }
    Scalar getKappa() const { return m_kappa; }
    Scalar getRCut() const { return m_rcut; }
    Scalar getAlpha() const { return m_alpha; }
    Scalar getQSum() const { return m_q; }
    Scalar getQ2Sum() const { return m_q2; }
    unsigned int getTableSize() const { return m_n_table_segments; }
    uintptr_t getTablePtr() const;
    const std::vector<ESPTableEntry>& getTable() const { return m_pswf_table_cpu; }
    ESPTableEntry getTableEntry(unsigned int i) const;

    void invalidate() { m_need_initialize = true; }

    void computeForces(uint64_t timestep);

#ifdef ENABLE_MPI
    CommFlags getRequestedCommFlags(uint64_t timestep) override;
#endif

    protected:
    void compute(uint64_t timestep) override;

    void setupMesh();
    void initializeFFT();
    void setupCoeffs();

    void compute_pswf_rho_coeff();
    void compute_pswf_gf_denom();
    Scalar gf_denom_pswf(Scalar x, Scalar y, Scalar z) const;

    void computeInfluenceFunction();
    void assignParticles();
    void updateMeshes();
    void interpolateForces();
    Scalar computePE();
    void computeVirialMesh();
    void computeVirial();
    void fixExclusions();
    void computeBodyCorrection();

    //! Evaluate the smooth PSWF screening function S(r/r_c) directly from
    //! the PSWF_Coeffs.h table (shared with the GPU path). Used only when
    //! (re)building the introspection table below; NOT used in any
    //! per-timestep hot loop on the CPU path (real-space correction is
    //! delegated to GPU-resident kernels in the GPU build, or to
    //! EvaluatorPairPSWF's own independent lookup in the CPU
    //! PotentialPair<> instantiation).
    Scalar evalScreen(Scalar r) const;
    Scalar evalDScreen(Scalar r) const;

    //! Fully-combined short-range kernel and its negative derivative,
    //! obtained by analytically re-applying 1/(4*pi*r) to evalScreen()/
    //! evalDScreen() above. Retained for backward-compatible naming with
    //! earlier plugin revisions.
    Scalar evalPSWFKernel(Scalar r) const;
    Scalar evalL_direct(Scalar r) const;
    Scalar evalDL_direct(Scalar r) const;

    void buildPSWFTable();
    void uploadPSWFTable();

    void computePSWFSelfEnergyConst();

    void computeForceMesh();
    void computeFieldMesh();
    void computeInverseMesh();
    void computeRealSpaceCorrection();

    std::shared_ptr<NeighborList> m_nlist;
    std::shared_ptr<ParticleGroup> m_group;

    bool m_need_initialize;
    bool m_box_changed;
    bool m_ptls_added_removed;

    uint3 m_global_dim;
    uint3 m_mesh_points;
    uint3 m_grid_dim;
    uint3 m_n_ghost_cells;

    unsigned int m_order;
    unsigned int m_n_table_segments;

    Scalar m_kappa;
    Scalar m_rcut;
    Scalar m_alpha;
    Scalar m_q;
    Scalar m_q2;
    Scalar m_qstar;
    Scalar m_body_energy;
    Scalar m_self_energy_const;

    Scalar m_pswf_c;

    GPUArray<Scalar> m_rho_coeff;
    GPUArray<Scalar> m_gf_b;
    GPUArray<Scalar> m_inf_f;
    GPUArray<Scalar3> m_k;
    GPUArray<hoomd::CScalar> m_mesh;
    GPUArray<hoomd::CScalar> m_mesh_scratch;
    GPUArray<Scalar> m_inv_fourier_mesh_x;
    GPUArray<Scalar> m_inv_fourier_mesh_y;
    GPUArray<Scalar> m_inv_fourier_mesh_z;
    GPUArray<Scalar> m_virial_mesh;
    GPUArray<Scalar> m_virial;
    GPUArray<Scalar> m_sum_partial;
    GPUArray<Scalar> m_sum_virial_partial;
    GPUArray<Scalar> m_sum_virial;

    std::vector<ESPTableEntry> m_pswf_table_cpu;
    GPUArray<Scalar> m_pswf_table_gpu;

    kiss_fftnd_cfg m_kiss_fft;
    kiss_fftnd_cfg m_kiss_ifft;

    bool m_kiss_fft_initialized;

#ifdef ENABLE_MPI
    dfft_plan m_dfft_plan_forward;
    dfft_plan m_dfft_plan_inverse;
    std::unique_ptr<CommunicatorGrid<hoomd::CScalar>> m_grid_comm_forward;
    std::unique_ptr<CommunicatorGrid<hoomd::CScalar>> m_grid_comm_reverse;
#endif
    uint3 computeGhostCellNum();
    };

void export_ESPForceCompute(pybind11::module& m);

    } // namespace md
    } // namespace hoomd

#endif // ESP_FORCE_COMPUTE_H_