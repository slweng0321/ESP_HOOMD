// Copyright (c) 2024-2026 ESP Plugin Contributors.
// Released under the BSD 3-Clause License.
//
// ESPForceComputeGPU.cuh
//
// C++ interface declarations for the CUDA kernels implemented in
// ESPForceComputeGPU.cu. This header is included by the host-only
// ESPForceComputeGPU.cc/.h translation units and must NOT be compiled by
// nvcc directly (it contains no device code, only host-callable launcher
// prototypes).
//
// Design notes
// ------------
//  * The large PSWF short-range screening coefficient tables
//    (PSWF_SCREEN_COEFFS / PSWF_DSCREEN_COEFFS / PSWF_SEG_R_LO /
//    PSWF_SEG_DR_INV) are now baked directly into the .cu translation unit
//    via a compile-time #include of PSWF_Coeffs.h. They are therefore no
//    longer passed as runtime device pointers through this interface --
//    eliminating an entire class of pointer-lifetime bugs and matching the
//    read-only-cache (__ldg) access pattern used internally.
//
//  * gpu_esp_fix_exclusions() now takes an explicit `virial_pitch` argument.
//    Callers MUST pass ForceCompute::m_virial's true pitch (obtained via
//    GPUArray::getPitch(), which equals N, the number of particles). This
//    fixes the previous bug where the kernel hardcoded the pitch to 1,
//    causing out-of-bounds writes into d_virial whenever N > 1.
//
#pragma once

#include "ESPForceCompute.h"
#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

namespace hoomd
    {
namespace md
    {
namespace kernel
    {

//! Upload the Green's-function denominator polynomial coefficients into
//! __constant__ memory. Must be called once whenever kappa/rcut/mesh
//! parameters change (small table: 5 doubles).
hipError_t gpu_esp_build_gf_denom(uint3 mesh_dim,
                                   Scalar3 box_L,
                                   Scalar rcut,
                                   const double* d_gf_denom_coeffs,
                                   Scalar* d_denom,
                                   unsigned int block_size);

//! Launch the reciprocal-space influence-function kernel.
hipError_t gpu_esp_build_influence(uint3 mesh_dim,
                                    Scalar3 box_L,
                                    Scalar kappa,
                                    Scalar alpha,
                                    const Scalar* d_denom,
                                    Scalar* d_inf_f,
                                    Scalar3* d_kvec,
                                    unsigned int block_size);

//! Upload the 1D PSWF assignment/interpolation stencil coefficients into
//! __constant__ memory. Must be called once whenever the interpolation
//! order changes (small table: <= ESP_MAX_ORDER_CU * 5 doubles).
hipError_t gpu_esp_upload_stencil_coeffs(const double* h_stencil_coeffs,
                                          const double* h_stencil_x_lo,
                                          const double* h_stencil_dx_inv,
                                          unsigned int order);

//! Launch the charge-spreading (mesh assignment) kernel.
hipError_t gpu_esp_assign_particles(uint3 mesh_dim,
                                     uint3 grid_dim,
                                     unsigned int group_size,
                                     const unsigned int* d_group_idx,
                                     const Scalar4* d_postype,
                                     const Scalar* d_charge,
                                     hoomd::CScalar* d_mesh,
                                     unsigned int order,
                                     const BoxDim& box,
                                     Scalar3 h_grid,
                                     unsigned int block_size);

//! Launch the influence-function application kernel (mesh *= inf_f).
hipError_t gpu_esp_apply_influence(unsigned int n_cells,
                                    const Scalar* d_inf_f,
                                    hoomd::CScalar* d_mesh,
                                    unsigned int block_size);

//! Launch the force-interpolation kernel (mesh field -> particle forces).
hipError_t gpu_esp_interpolate_forces(unsigned int n_particles,
                                       unsigned int group_size,
                                       const unsigned int* d_group_idx,
                                       const Scalar4* d_postype,
                                       const Scalar* d_charge,
                                       const Scalar* d_inv_mesh_x,
                                       const Scalar* d_inv_mesh_y,
                                       const Scalar* d_inv_mesh_z,
                                       uint3 grid_dim,
                                       unsigned int order,
                                       const BoxDim& box,
                                       Scalar3 h_grid,
                                       Scalar4* d_force,
                                       unsigned int block_size);

//! Launch the excluded-pair correction kernel.
//!
//! \param N              Number of particles (ForceCompute::m_pdata->getN()).
//! \param d_pos          Device positions array.
//! \param d_charge       Device charges array.
//! \param d_force        Device force array (read-write, atomic accumulate).
//! \param d_virial       Device virial array (read-write, atomic accumulate).
//! \param virial_pitch   REQUIRED: the true pitch of d_virial (== N). Obtain
//!                       via ForceCompute::m_virial.getPitch() on the host.
//!                       Passing any other value (e.g. the previous
//!                       hardcoded 1) will cause out-of-bounds writes for
//!                       any virial component beyond index 0.
//! \param d_n_ex         Per-particle exclusion counts.
//! \param d_ex_list      Flattened exclusion list.
//! \param ex_list_indexer Index2D indexer for d_ex_list.
//! \param rcut           Real-space cutoff (must match PSWF_RCUT baked into
//!                       PSWF_Coeffs.h).
//! \param box            Simulation box (for minimum-image convention).
//! \param block_size     CUDA block size (from Autotuner).
//!
//! \note The PSWF screening coefficients (S(r), S'(r), segment bounds) are
//!       NOT passed here: they are compiled directly into the kernel from
//!       PSWF_Coeffs.h and accessed via __ldg() for read-only-cache
//!       throughput.
hipError_t gpu_esp_fix_exclusions(unsigned int N,
                                   const Scalar4* d_pos,
                                   const Scalar* d_charge,
                                   Scalar4* d_force,
                                   Scalar* d_virial,
                                   size_t virial_pitch,
                                   const unsigned int* d_n_ex,
                                   const unsigned int* d_ex_list,
                                   Index2D ex_list_indexer,
                                   Scalar rcut,
                                   const BoxDim& box,
                                   unsigned int block_size);

    } // namespace kernel
    } // namespace md
    } // namespace hoomd