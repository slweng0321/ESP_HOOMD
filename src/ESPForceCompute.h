// Copyright (c) 2024-2026 ESP Plugin Contributors.
// Released under the BSD 3-Clause License.
//
// ESPForceCompute.h
//
// CPU-path ForceCompute plugin implementing the Ewald Summation with Prolates
// (ESP) method for HOOMD-blue.

#pragma once

#include "hoomd/ForceCompute.h"
#include "hoomd/GPUArray.h"
#include "hoomd/Index1D.h"
#include "hoomd/ParticleGroup.h"
#include "hoomd/md/NeighborList.h" // Updated path
#include "hoomd/GPUFlags.h" // Added for GPUFlags
#include "hoomd/md/CommunicatorGrid.h" // Corrected for CommunicatorGrid

#ifdef ENABLE_MPI
#include "hoomd/extern/dfftlib/src/dfft_host.h"
#include <mpi.h>
#endif

#include "hoomd/extern/kiss_fft/tools/kiss_fftnd.h"

#include <pybind11/pybind11.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

namespace hoomd
    {
namespace md
    {

constexpr int ESP_MAX_ORDER = 12;
constexpr unsigned int ESP_TABLE_POLY_DEGREE = 4u;
constexpr unsigned int ESP_TABLE_SEGMENTS = 4096;

struct alignas(2 * sizeof(Scalar)) ESPTableEntry
    {
    Scalar4 potential_coeffs;
    Scalar  potential_coeff4;
    Scalar4 force_coeffs;
    Scalar  force_coeff4;
    Scalar  r_lo;
    Scalar  dr_inv;
    };

static_assert(sizeof(ESPTableEntry) == 12 * sizeof(Scalar),
              "ESPTableEntry size mismatch — update GPU upload code if changed.");

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

    void invalidate() { m_need_initialize = true; }

    void setBoxChange() { m_box_changed = true; }
    void slotGlobalParticleNumberChange() { m_ptls_added_removed = true; }

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

    Scalar evalPSWFKernel(Scalar x) const;
    Scalar evalL_direct(Scalar r) const;
    Scalar evalDL_direct(Scalar r) const;
    void buildPSWFTable();
    void uploadPSWFTable();

    void computePSWFSelfEnergyConst();

    protected:
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
    GPUArray<kiss_fft_cpx> m_mesh;
    GPUArray<kiss_fft_cpx> m_mesh_scratch;
    GPUArray<kiss_fft_cpx> m_inv_fourier_mesh_x;
    GPUArray<kiss_fft_cpx> m_inv_fourier_mesh_y;
    GPUArray<kiss_fft_cpx> m_inv_fourier_mesh_z;
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
    std::unique_ptr<hoomd::md::CommunicatorGrid<kiss_fft_cpx>> m_grid_comm_forward;
    std::unique_ptr<hoomd::md::CommunicatorGrid<kiss_fft_cpx>> m_grid_comm_reverse;
#endif
    uint3 computeGhostCellNum();
    };

    } // namespace md
    } // namespace hoomd
