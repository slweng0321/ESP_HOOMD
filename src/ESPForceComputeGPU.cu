// Copyright (c) 2024-2026 ESP Plugin Contributors.
// Released under the BSD 3-Clause License.
//
// ESPForceComputeGPU.cu
//
// CUDA kernel implementations for the ESP (Ewald Summation with Prolates)
// particle-mesh + short-range-correction pipeline.
//
// Memory-placement strategy (HPC optimisation notes)
// ----------------------------------------------------
//  * Small, per-simulation constant tables (Green's-function denominator
//    polynomial: 5 doubles: 1D PSWF assignment/interpolation stencil:
//    <= ESP_MAX_ORDER_CU * 5 doubles) are mirrored into __constant__ memory
//    via hipMemcpyToSymbol() once per parameter change. These are broadcast
//    to every thread in a warp at full bandwidth.
//
//  * The large PSWF short-range screening tables baked into PSWF_Coeffs.h
//    (PSWF_N_SEGS = 4096 segments x 5 Horner coefficients x 8 bytes =
//    ~160 KB PER TABLE) are FAR too large for the 64 KB __constant__ memory
//    budget available on essentially all NVIDIA/AMD GPUs. They are instead
//    left in the compiled binary's read-only .rodata / __device__ global
//    segment and accessed exclusively through __ldg() (or the implicit
//    read-only cache path taken by "const __restrict__" pointers on
//    architectures >= sm_35), routing all lookups through the GPU's
//    read-only data cache instead of consuming general L1/L2 global-memory
//    bandwidth in the tight neighbour-list loop.
//
// Bug fixes in this revision
// ---------------------------
//  * gpu_esp_fix_exclusions_kernel: virial_pitch was previously hardcoded to
//    1, causing out-of-bounds writes into d_virial for any system with
//    N > 1 particles (the true pitch is N, i.e. m_virial.getPitch()). This
//    revision takes the pitch as a runtime argument and uses it exclusively.
//
#include "ESPForceComputeGPU.cuh"
#include "PSWF_Coeffs.h"

#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"

#include <cmath>

#define ESP_MAX_ORDER_CU 12
#define ESP_INV_4PI 0.079577471545947673

namespace hoomd
    {
namespace md
    {
namespace kernel
    {

using namespace hoomd::md::esp_pswf_coeffs;

// ============================================================================
// __constant__ memory: small, per-simulation polynomial tables.
// ============================================================================
__constant__ double d_c_gf_denom_coeffs[5];
__constant__ double d_c_stencil_coeffs[ESP_MAX_ORDER_CU * 5];
__constant__ double d_c_stencil_x_lo[ESP_MAX_ORDER_CU];
__constant__ double d_c_stencil_dx_inv[ESP_MAX_ORDER_CU];

// ============================================================================
// Horner evaluation of a 5-coefficient segment, shared by all table types
// (screening function, its derivative, and the 1D stencil). All reads are
// routed through __ldg() to force the read-only data cache path regardless
// of whether the backing storage is __constant__ or plain __device__/global.
// ============================================================================
__device__ __forceinline__ Scalar gpu_horner5(const double* __restrict__ coeffs, unsigned int seg, Scalar t)
    {
    const double* c = coeffs + static_cast<size_t>(seg) * 5u;
    Scalar val = static_cast<Scalar>(__ldg(&c[4]));
    val = static_cast<Scalar>(__ldg(&c[3])) + t * val;
    val = static_cast<Scalar>(__ldg(&c[2])) + t * val;
    val = static_cast<Scalar>(__ldg(&c[1])) + t * val;
    val = static_cast<Scalar>(__ldg(&c[0])) + t * val;
    return val;
    }

// ============================================================================
// evalPSWFKernel: looks up the SMOOTH screening function S(r) and its
// derivative S'(r) directly from the compile-time PSWF_Coeffs.h tables
// (PSWF_SCREEN_COEFFS / PSWF_DSCREEN_COEFFS / PSWF_SEG_R_LO /
// PSWF_SEG_DR_INV), then analytically combines them with 1/(4*pi*r) in
// registers to produce L(r) and -dL/dr. No singular quantity is ever
// fitted or stored in the table itself -- this is what eliminates the
// numerical singularity at small r.
//
//   L(r)    = S(r) / (4*pi*r)
//   -dL/dr  = S(r) / (4*pi*r^2) - S'(r) / (4*pi*r)
// ============================================================================
__device__ __forceinline__ void evalPSWFKernel(Scalar r, Scalar rcut, Scalar& out_L, Scalar& out_negdLdr)
    {
    if (r < Scalar(1.0e-12) || r > rcut)
        {
        out_L = Scalar(0.0);
        out_negdLdr = Scalar(0.0);
        return;
        }

    Scalar s = r / rcut;
    s = fminf(fmaxf(s, Scalar(0.0)), Scalar(1.0) - Scalar(1.0e-9));
    unsigned int seg = static_cast<unsigned int>(s * static_cast<Scalar>(PSWF_N_SEGS));
    seg = (seg >= static_cast<unsigned int>(PSWF_N_SEGS)) ? static_cast<unsigned int>(PSWF_N_SEGS) - 1u : seg;

    const Scalar seg_r_lo = static_cast<Scalar>(__ldg(&PSWF_SEG_R_LO[seg]));
    const Scalar seg_dr_inv = static_cast<Scalar>(__ldg(&PSWF_SEG_DR_INV[seg]));
    const Scalar t = (r - seg_r_lo) * seg_dr_inv;

    const Scalar S = gpu_horner5(PSWF_SCREEN_COEFFS, seg, t);
    const Scalar dS = gpu_horner5(PSWF_DSCREEN_COEFFS, seg, t);

    const Scalar inv_4pi_r = static_cast<Scalar>(ESP_INV_4PI) / r;
    const Scalar inv_4pi_r2 = inv_4pi_r / r;

    out_L = S * inv_4pi_r;
    out_negdLdr = S * inv_4pi_r2 - dS * inv_4pi_r;
    }

//! Direct (unsplit) Coulomb evaluator, kept for reference/validation paths
//! (e.g. computing the exclusion double-counting subtraction below).
__device__ __forceinline__ Scalar evalL_direct(Scalar r)
    {
    return static_cast<Scalar>(ESP_INV_4PI) / r;
    }

__device__ __forceinline__ Scalar evalDL_direct(Scalar r)
    {
    return -static_cast<Scalar>(ESP_INV_4PI) / (r * r);
    }

// ============================================================================
// 1D PSWF assignment/interpolation stencil weight. phi(x) was never
// singular, so its __constant__-resident table needs no analytic-split fix;
// __ldg() is still used for consistency and because __constant__ reads that
// are not uniform across the warp fall back to the same cache hierarchy.
// ============================================================================
__device__ __forceinline__ Scalar gpu_eval_stencil_weight(unsigned int k, Scalar x_local)
    {
    const Scalar x_lo = static_cast<Scalar>(__ldg(&d_c_stencil_x_lo[k]));
    const Scalar dx_inv = static_cast<Scalar>(__ldg(&d_c_stencil_dx_inv[k]));
    Scalar t = (x_local - x_lo) * dx_inv;
    t = fminf(fmaxf(t, Scalar(0.0)), Scalar(1.0));

    const double* c = d_c_stencil_coeffs + static_cast<size_t>(k) * 5u;
    Scalar val = static_cast<Scalar>(__ldg(&c[4]));
    val = static_cast<Scalar>(__ldg(&c[3])) + t * val;
    val = static_cast<Scalar>(__ldg(&c[2])) + t * val;
    val = static_cast<Scalar>(__ldg(&c[1])) + t * val;
    val = static_cast<Scalar>(__ldg(&c[0])) + t * val;
    return val;
    }

// ============================================================================
// Kernel: reciprocal-space influence function p_k (Eq. 9 analogue).
// ============================================================================
__global__ void gpu_esp_build_influence_kernel(uint3 dim,
                                                Scalar3 box_L,
                                                Scalar kappa,
                                                Scalar alpha,
                                                Scalar* d_inf_f,
                                                Scalar3* d_kvec)
    {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int n = dim.x * dim.y * dim.z;
    if (idx >= n)
        return;

    const unsigned int ix = idx % dim.x;
    const unsigned int iy = (idx / dim.x) % dim.y;
    const unsigned int iz = idx / (dim.x * dim.y);

    const int kx = (ix <= dim.x / 2) ? int(ix) : int(ix) - int(dim.x);
    const int ky = (iy <= dim.y / 2) ? int(iy) : int(iy) - int(dim.y);
    const int kz = (iz <= dim.z / 2) ? int(iz) : int(iz) - int(dim.z);

    const Scalar twopi = Scalar(2.0) * Scalar(M_PI);
    const Scalar kxp = twopi * Scalar(kx) / box_L.x;
    const Scalar kyp = twopi * Scalar(ky) / box_L.y;
    const Scalar kzp = twopi * Scalar(kz) / box_L.z;
    const Scalar k2 = kxp * kxp + kyp * kyp + kzp * kzp;
    d_kvec[idx] = make_scalar3(kxp, kyp, kzp);

    if (k2 < Scalar(1.0e-20))
        {
        d_inf_f[idx] = Scalar(0.0);
        return;
        }

    // Green's-function denominator: Horner form via small __constant__ table.
    Scalar b_k = static_cast<Scalar>(d_c_gf_denom_coeffs[4]);
    b_k = static_cast<Scalar>(d_c_gf_denom_coeffs[3]) + k2 * b_k;
    b_k = static_cast<Scalar>(d_c_gf_denom_coeffs[2]) + k2 * b_k;
    b_k = static_cast<Scalar>(d_c_gf_denom_coeffs[1]) + k2 * b_k;
    b_k = static_cast<Scalar>(d_c_gf_denom_coeffs[0]) + k2 * b_k;
    b_k = fmaxf(b_k, Scalar(1.0e-12));

    const Scalar exp_term = expf(-k2 / (Scalar(4.0) * kappa * kappa));
    const Scalar screening = (alpha > Scalar(0.0)) ? expf(-alpha * sqrtf(k2)) : Scalar(1.0);
    d_inf_f[idx] = screening * exp_term / (k2 * b_k);
    }

// ============================================================================
// Kernel: charge spreading (Horner-based PSWF stencil, __constant__ memory).
// ============================================================================
__global__ void gpu_esp_assign_particles_kernel(uint3 mesh_dim,
                                                 uint3 grid_dim,
                                                 unsigned int group_size,
                                                 const unsigned int* __restrict__ d_group_idx,
                                                 const Scalar4* __restrict__ d_postype,
                                                 const Scalar* __restrict__ d_charge,
                                                 hoomd::CScalar* d_mesh,
                                                 unsigned int order,
                                                 BoxDim box,
                                                 Scalar3 h_grid)
    {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= group_size)
        return;

    const unsigned int pidx = d_group_idx[tid];
    const Scalar4 postype = d_postype[pidx];
    const Scalar q = d_charge[pidx];
    if (q == Scalar(0.0))
        return;

    const Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    const Scalar3 frac = box.makeFraction(pos);

    const Scalar gx = frac.x * Scalar(mesh_dim.x);
    const Scalar gy = frac.y * Scalar(mesh_dim.y);
    const Scalar gz = frac.z * Scalar(mesh_dim.z);

    const int ix0 = int(floorf(gx)) - int(order / 2);
    const int iy0 = int(floorf(gy)) - int(order / 2);
    const int iz0 = int(floorf(gz)) - int(order / 2);

    const Scalar fx = gx - floorf(gx);
    const Scalar fy = gy - floorf(gy);
    const Scalar fz = gz - floorf(gz);

    Scalar wx[ESP_MAX_ORDER_CU], wy[ESP_MAX_ORDER_CU], wz[ESP_MAX_ORDER_CU];

#pragma unroll
    for (unsigned int k = 0; k < ESP_MAX_ORDER_CU; ++k)
        {
        if (k >= order)
            break;
        const Scalar xl = (fx - Scalar(k) + Scalar(order / 2)) * h_grid.x;
        const Scalar yl = (fy - Scalar(k) + Scalar(order / 2)) * h_grid.y;
        const Scalar zl = (fz - Scalar(k) + Scalar(order / 2)) * h_grid.z;
        wx[k] = gpu_eval_stencil_weight(k, xl);
        wy[k] = gpu_eval_stencil_weight(k, yl);
        wz[k] = gpu_eval_stencil_weight(k, zl);
        }

    for (unsigned int kx = 0; kx < order; ++kx)
        {
        const int cx = ((ix0 + int(kx)) % int(grid_dim.x) + int(grid_dim.x)) % int(grid_dim.x);
        for (unsigned int ky = 0; ky < order; ++ky)
            {
            const int cy = ((iy0 + int(ky)) % int(grid_dim.y) + int(grid_dim.y)) % int(grid_dim.y);
            const Scalar wxy = wx[kx] * wy[ky];
            for (unsigned int kz = 0; kz < order; ++kz)
                {
                const int cz = ((iz0 + int(kz)) % int(grid_dim.z) + int(grid_dim.z)) % int(grid_dim.z);
                const size_t cell = (size_t(cz) * grid_dim.y + cy) * grid_dim.x + cx;
                atomicAdd(&d_mesh[cell].x, static_cast<Scalar>(q * wxy * wz[kz]));
                }
            }
        }
    }

// ============================================================================
// Kernel: apply reciprocal-space influence function to the mesh.
// ============================================================================
__global__ void gpu_esp_apply_influence_kernel(unsigned int n_cells,
                                                const Scalar* __restrict__ d_inf_f,
                                                hoomd::CScalar* d_mesh)
    {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells)
        return;
    const Scalar f = __ldg(&d_inf_f[idx]);
    d_mesh[idx].x *= f;
    d_mesh[idx].y *= f;
    }

// ============================================================================
// Kernel: interpolate mesh field back to particle forces.
// ============================================================================
__global__ void gpu_esp_interpolate_forces_kernel(unsigned int n_particles,
                                                   unsigned int group_size,
                                                   const unsigned int* __restrict__ d_group_idx,
                                                   const Scalar4* __restrict__ d_postype,
                                                   const Scalar* __restrict__ d_charge,
                                                   const Scalar* __restrict__ d_inv_mesh_x,
                                                   const Scalar* __restrict__ d_inv_mesh_y,
                                                   const Scalar* __restrict__ d_inv_mesh_z,
                                                   uint3 grid_dim,
                                                   unsigned int order,
                                                   BoxDim box,
                                                   Scalar3 h_grid,
                                                   Scalar4* d_force)
    {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= group_size)
        return;

    const unsigned int pidx = d_group_idx[tid];
    if (pidx >= n_particles)
        return;

    const Scalar4 postype = d_postype[pidx];
    const Scalar q = d_charge[pidx];
    if (q == Scalar(0.0))
        return;

    const Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    const Scalar3 frac = box.makeFraction(pos);

    const Scalar gx = frac.x * Scalar(grid_dim.x);
    const Scalar gy = frac.y * Scalar(grid_dim.y);
    const Scalar gz = frac.z * Scalar(grid_dim.z);

    const int ix0 = int(floorf(gx)) - int(order / 2);
    const int iy0 = int(floorf(gy)) - int(order / 2);
    const int iz0 = int(floorf(gz)) - int(order / 2);

    const Scalar fx = gx - floorf(gx);
    const Scalar fy = gy - floorf(gy);
    const Scalar fz = gz - floorf(gz);

    Scalar wx[ESP_MAX_ORDER_CU], wy[ESP_MAX_ORDER_CU], wz[ESP_MAX_ORDER_CU];

#pragma unroll
    for (unsigned int k = 0; k < ESP_MAX_ORDER_CU; ++k)
        {
        if (k >= order)
            break;
        const Scalar xl = (fx - Scalar(k) + Scalar(order / 2)) * h_grid.x;
        const Scalar yl = (fy - Scalar(k) + Scalar(order / 2)) * h_grid.y;
        const Scalar zl = (fz - Scalar(k) + Scalar(order / 2)) * h_grid.z;
        wx[k] = gpu_eval_stencil_weight(k, xl);
        wy[k] = gpu_eval_stencil_weight(k, yl);
        wz[k] = gpu_eval_stencil_weight(k, zl);
        }

    Scalar3 field = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));

    for (unsigned int kx = 0; kx < order; ++kx)
        {
        const int cx = ((ix0 + int(kx)) % int(grid_dim.x) + int(grid_dim.x)) % int(grid_dim.x);
        for (unsigned int ky = 0; ky < order; ++ky)
            {
            const int cy = ((iy0 + int(ky)) % int(grid_dim.y) + int(grid_dim.y)) % int(grid_dim.y);
            const Scalar wxy = wx[kx] * wy[ky];
            for (unsigned int kz = 0; kz < order; ++kz)
                {
                const int cz = ((iz0 + int(kz)) % int(grid_dim.z) + int(grid_dim.z)) % int(grid_dim.z);
                const size_t cell = (size_t(cz) * grid_dim.y + cy) * grid_dim.x + cx;
                const Scalar w = wxy * wz[kz];
                field.x += w * __ldg(&d_inv_mesh_x[cell]);
                field.y += w * __ldg(&d_inv_mesh_y[cell]);
                field.z += w * __ldg(&d_inv_mesh_z[cell]);
                }
            }
        }

    atomicAdd(&d_force[pidx].x, q * field.x);
    atomicAdd(&d_force[pidx].y, q * field.y);
    atomicAdd(&d_force[pidx].z, q * field.z);
    }

// ============================================================================
// Kernel: exclusion correction.
//
// BUG FIX: virial_pitch is now taken as a runtime parameter and MUST equal
// the true pitch of ForceCompute::m_virial (== N, obtained via
// GPUArray::getPitch() on the host side). The previous implementation
// hardcoded this to 1, which silently wrote out of bounds into d_virial for
// every system with N > 1 -- corrupting adjacent GPU memory and producing
// wrong / non-deterministic virial (and sometimes energy/force) values.
//
// The screening table itself is looked up directly from the compile-time
// PSWF_Coeffs.h arrays (via evalPSWFKernel above), NOT from any runtime
// device pointer -- eliminating the previous host-to-device table upload
// entirely for this kernel and removing an entire class of pointer-lifetime
// bugs.
// ============================================================================
__global__ void gpu_esp_fix_exclusions_kernel(unsigned int n,
                                               const Scalar4* __restrict__ d_pos,
                                               const Scalar* __restrict__ d_charge,
                                               Scalar4* d_force,
                                               Scalar* d_virial,
                                               size_t virial_pitch,
                                               const unsigned int* __restrict__ d_n_ex,
                                               const unsigned int* __restrict__ d_ex_list,
                                               Index2D ex_list_indexer,
                                               Scalar rcut,
                                               BoxDim box)
    {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    const Scalar4 pi4 = d_pos[i];
    const Scalar3 pi = make_scalar3(pi4.x, pi4.y, pi4.z);
    const Scalar qi = d_charge[i];

    Scalar4 f_local = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
    Scalar v_local[6] = {0, 0, 0, 0, 0, 0};

    const unsigned int n_ex_i = d_n_ex[i];
    for (unsigned int ex = 0; ex < n_ex_i; ++ex)
        {
        const unsigned int j = d_ex_list[ex_list_indexer(ex, i)];
        if (j >= n)
            continue;

        const Scalar4 pj4 = d_pos[j];
        Scalar3 rij = make_scalar3(pj4.x, pj4.y, pj4.z) - pi;
        rij = box.minImage(rij);
        const Scalar rsq = dot(rij, rij);
        if (rsq < Scalar(1.0e-20) || rsq >= rcut * rcut)
            continue;

        const Scalar r = sqrtf(rsq);
        const Scalar invr = Scalar(1.0) / r;
        const Scalar qj = d_charge[j];
        const Scalar qiqj = qi * qj;

        Scalar Lr, negdLdr;
        evalPSWFKernel(r, rcut, Lr, negdLdr);

        // Excluded pairs must have their FULL bare-Coulomb interaction
        // (which is already implicitly included by the reciprocal-space
        // mesh sum for ALL pairs, excluded or not) subtracted back out,
        // leaving only the short-range correction L(r) as the net
        // contribution for this excluded pair.
        const Scalar coulomb_full = evalL_direct(r);
        const Scalar dcoulomb_full = -evalDL_direct(r); // == 1/(4*pi*r^2)

        const Scalar dE = qiqj * (Lr - coulomb_full);
        const Scalar dF_divr = qiqj * (negdLdr - dcoulomb_full) * invr;
        const Scalar3 f = dF_divr * rij;

        f_local.x -= f.x;
        f_local.y -= f.y;
        f_local.z -= f.z;
        f_local.w += Scalar(0.5) * dE;

        v_local[0] += Scalar(0.5) * rij.x * f.x;
        v_local[1] += Scalar(0.5) * rij.x * f.y;
        v_local[2] += Scalar(0.5) * rij.x * f.z;
        v_local[3] += Scalar(0.5) * rij.y * f.y;
        v_local[4] += Scalar(0.5) * rij.y * f.z;
        v_local[5] += Scalar(0.5) * rij.z * f.z;
        }

    atomicAdd(&d_force[i].x, f_local.x);
    atomicAdd(&d_force[i].y, f_local.y);
    atomicAdd(&d_force[i].z, f_local.z);
    atomicAdd(&d_force[i].w, f_local.w);

    // FIX: use the CORRECT pitch (== N) passed in by the host, instead of
    // the previous hardcoded value of 1, which caused an out-of-bounds
    // write for every virial component beyond index 0 whenever N > 1.
    for (int c = 0; c < 6; ++c)
        atomicAdd(&d_virial[static_cast<size_t>(c) * virial_pitch + i], v_local[c]);
    }

// ============================================================================
// Host launchers
// ============================================================================

hipError_t gpu_esp_build_gf_denom(uint3, Scalar3, Scalar, const double* d_gf_denom_coeffs, Scalar*, unsigned int)
    {
    return hipMemcpyToSymbol(HIP_SYMBOL(d_c_gf_denom_coeffs), d_gf_denom_coeffs, 5 * sizeof(double));
    }

hipError_t gpu_esp_build_influence(uint3 mesh_dim,
                                    Scalar3 box_L,
                                    Scalar kappa,
                                    Scalar alpha,
                                    const Scalar*,
                                    Scalar* d_inf_f,
                                    Scalar3* d_kvec,
                                    unsigned int block_size)
    {
    const unsigned int n = mesh_dim.x * mesh_dim.y * mesh_dim.z;
    const unsigned int grid_size = (n + block_size - 1) / block_size;
    gpu_esp_build_influence_kernel<<<grid_size, block_size>>>(mesh_dim, box_L, kappa, alpha, d_inf_f, d_kvec);
    return hipPeekAtLastError();
    }

hipError_t gpu_esp_upload_stencil_coeffs(const double* h_stencil_coeffs,
                                          const double* h_stencil_x_lo,
                                          const double* h_stencil_dx_inv,
                                          unsigned int order)
    {
    if (order > ESP_MAX_ORDER_CU)
        return hipErrorInvalidValue;
    hipError_t err;
    err = hipMemcpyToSymbol(HIP_SYMBOL(d_c_stencil_coeffs), h_stencil_coeffs, order * 5 * sizeof(double));
    if (err != hipSuccess)
        return err;
    err = hipMemcpyToSymbol(HIP_SYMBOL(d_c_stencil_x_lo), h_stencil_x_lo, order * sizeof(double));
    if (err != hipSuccess)
        return err;
    return hipMemcpyToSymbol(HIP_SYMBOL(d_c_stencil_dx_inv), h_stencil_dx_inv, order * sizeof(double));
    }

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
                                     unsigned int block_size)
    {
    const unsigned int grid_size = (group_size + block_size - 1) / block_size;
    gpu_esp_assign_particles_kernel<<<grid_size, block_size>>>(
        mesh_dim, grid_dim, group_size, d_group_idx, d_postype, d_charge, d_mesh, order, box, h_grid);
    return hipPeekAtLastError();
    }

hipError_t gpu_esp_apply_influence(unsigned int n_cells,
                                    const Scalar* d_inf_f,
                                    hoomd::CScalar* d_mesh,
                                    unsigned int block_size)
    {
    const unsigned int grid_size = (n_cells + block_size - 1) / block_size;
    gpu_esp_apply_influence_kernel<<<grid_size, block_size>>>(n_cells, d_inf_f, d_mesh);
    return hipPeekAtLastError();
    }

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
                                       unsigned int block_size)
    {
    const unsigned int grid_size = (group_size + block_size - 1) / block_size;
    gpu_esp_interpolate_forces_kernel<<<grid_size, block_size>>>(n_particles,
                                                                  group_size,
                                                                  d_group_idx,
                                                                  d_postype,
                                                                  d_charge,
                                                                  d_inv_mesh_x,
                                                                  d_inv_mesh_y,
                                                                  d_inv_mesh_z,
                                                                  grid_dim,
                                                                  order,
                                                                  box,
                                                                  h_grid,
                                                                  d_force);
    return hipPeekAtLastError();
    }

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
                                   unsigned int block_size)
    {
    const unsigned int grid_size = (N + block_size - 1) / block_size;
    gpu_esp_fix_exclusions_kernel<<<grid_size, block_size>>>(
        N, d_pos, d_charge, d_force, d_virial, virial_pitch, d_n_ex, d_ex_list, ex_list_indexer, rcut, box);
    return hipPeekAtLastError();
    }

    } // namespace kernel
    } // namespace md
    } // namespace hoomd

#undef ESP_MAX_ORDER_CU
#undef ESP_INV_4PI