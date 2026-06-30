// Copyright (c) 2024-2026 ESP Plugin Contributors.
// Released under the BSD 3-Clause License.
//
// ESPForceComputeGPU.cc
//
// GPU override for the ESP force pipeline.

#include "ESPForceComputeGPU.h"

#include <algorithm>
#include <cmath>

namespace hoomd
    {
namespace md
    {

namespace
    {
__device__ inline Scalar gpu_eval_gf_denom(const Scalar* gf_b, Scalar x, Scalar y, Scalar z)
    {
    Scalar denom = Scalar(0.0);
    const Scalar s = x + y + z;
    for (int l = 0; l < 12; ++l)
        denom = denom * s + gf_b[l];
    return std::max(denom, Scalar(1.0e-12));
    }

__global__ void gpu_build_gf_denom_kernel(uint3 dim,
                                          Scalar box_lx,
                                          Scalar box_ly,
                                          Scalar box_lz,
                                          Scalar rcut,
                                          const Scalar* gf_b,
                                          Scalar* denom_out)
    {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int n = dim.x * dim.y * dim.z;
    if (idx >= n)
        return;

    const unsigned int ix = idx % dim.x;
    const unsigned int iy = (idx / dim.x) % dim.y;
    const unsigned int iz = idx / (dim.x * dim.y);

    const int kx = (ix <= dim.x / 2) ? static_cast<int>(ix)
                                     : static_cast<int>(ix) - static_cast<int>(dim.x);
    const int ky = (iy <= dim.y / 2) ? static_cast<int>(iy)
                                     : static_cast<int>(iy) - static_cast<int>(dim.y);
    const int kz = (iz <= dim.z / 2) ? static_cast<int>(iz)
                                     : static_cast<int>(iz) - static_cast<int>(dim.z);

    const Scalar twopi = Scalar(2.0) * Scalar(M_PI);
    const Scalar kx_phys = twopi * Scalar(kx) / box_lx;
    const Scalar ky_phys = twopi * Scalar(ky) / box_ly;
    const Scalar kz_phys = twopi * Scalar(kz) / box_lz;

    const Scalar x = (kx_phys * rcut) * (kx_phys * rcut);
    const Scalar y = (ky_phys * rcut) * (ky_phys * rcut);
    const Scalar z = (kz_phys * rcut) * (kz_phys * rcut);
    denom_out[idx] = gpu_eval_gf_denom(gf_b, x, y, z);
    }

__global__ void gpu_build_influence_kernel(uint3 dim,
                                           Scalar box_lx,
                                           Scalar box_ly,
                                           Scalar box_lz,
                                           Scalar kappa,
                                           Scalar alpha,
                                           const Scalar* denom,
                                           Scalar* inf_f,
                                           Scalar3* kvec)
    {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int n = dim.x * dim.y * dim.z;
    if (idx >= n)
        return;

    const unsigned int ix = idx % dim.x;
    const unsigned int iy = (idx / dim.x) % dim.y;
    const unsigned int iz = idx / (dim.x * dim.y);

    const int kx = (ix <= dim.x / 2) ? static_cast<int>(ix)
                                     : static_cast<int>(ix) - static_cast<int>(dim.x);
    const int ky = (iy <= dim.y / 2) ? static_cast<int>(iy)
                                     : static_cast<int>(iy) - static_cast<int>(dim.y);
    const int kz = (iz <= dim.z / 2) ? static_cast<int>(iz)
                                     : static_cast<int>(iz) - static_cast<int>(dim.z);

    const Scalar twopi = Scalar(2.0) * Scalar(M_PI);
    const Scalar kx_phys = twopi * Scalar(kx) / box_lx;
    const Scalar ky_phys = twopi * Scalar(ky) / box_ly;
    const Scalar kz_phys = twopi * Scalar(kz) / box_lz;

    const Scalar k2 = kx_phys * kx_phys + ky_phys * ky_phys + kz_phys * kz_phys;
    kvec[idx] = make_scalar3(kx_phys, ky_phys, kz_phys);

    if (k2 < Scalar(1.0e-20))
        {
        inf_f[idx] = Scalar(0.0);
        return;
        }

    const Scalar k = std::sqrt(k2);
    const Scalar exp_term = std::exp(-k2 / (Scalar(4.0) * kappa * kappa));
    const Scalar screening = (alpha > Scalar(0.0)) ? std::exp(-alpha * k) : Scalar(1.0);
    inf_f[idx] = screening * exp_term / (k2 * denom[idx]);
    }

__global__ void gpu_fix_exclusions_kernel(unsigned int n,
                                          const Scalar4* pos,
                                          const Scalar* chg,
                                          Scalar4* force,
                                          Scalar* virial,
                                          const unsigned int* nex,
                                          const unsigned int* exlist,
                                          Index2D ex_idx,
                                          const ESPTableEntry* table,
                                          unsigned int n_segs,
                                          Scalar rcut,
                                          const BoxDim box)
    {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    const Scalar4 pi4 = pos[i];
    const Scalar3 pi = make_scalar3(pi4.x, pi4.y, pi4.z);
    const Scalar qi = chg[i];

    for (unsigned int ex = 0; ex < nex[i]; ++ex)
        {
        const unsigned int j = exlist[ex_idx(ex, i)];
        if (j <= i || j >= n)
            continue;

        const Scalar4 pj4 = pos[j];
        const Scalar3 pj = make_scalar3(pj4.x, pj4.y, pj4.z);
        const Scalar qj = chg[j];

        Scalar3 rij = pj - pi;
        rij = box.minImage(rij);
        const Scalar rsq = dot(rij, rij);
        if (rsq < Scalar(1.0e-20))
            continue;

        const Scalar r = std::sqrt(rsq);
        const Scalar s = std::min(Scalar(1.0) - Scalar(1.0e-9), r / rcut);
        const unsigned int seg = min(static_cast<unsigned int>(s * Scalar(n_segs)), n_segs - 1u);
        const ESPTableEntry e = table[seg];
        const Scalar t = (r - e.r_lo) * e.dr_inv;

        const Scalar pc[5] = {e.potential_coeffs.x,
                              e.potential_coeffs.y,
                              e.potential_coeffs.z,
                              e.potential_coeffs.w,
                              e.potential_coeff4};
        const Scalar fc[5] = {e.force_coeffs.x,
                              e.force_coeffs.y,
                              e.force_coeffs.z,
                              e.force_coeffs.w,
                              e.force_coeff4};

        Scalar Lr = pc[4];
        Scalar mdLdr = fc[4];
        for (int k = 3; k >= 0; --k)
            {
            Lr = Lr * t + pc[k];
            mdLdr = mdLdr * t + fc[k];
            }

        const Scalar qiqj = qi * qj;
        const Scalar invr = Scalar(1.0) / r;
        const Scalar force_divr = qiqj * (-invr * invr + mdLdr) * invr;
        const Scalar3 f = force_divr * rij;

        atomicAdd(&force[i].x, -f.x);
        atomicAdd(&force[i].y, -f.y);
        atomicAdd(&force[i].z, -f.z);
        atomicAdd(&force[j].x, f.x);
        atomicAdd(&force[j].y, f.y);
        atomicAdd(&force[j].z, f.z);

        atomicAdd(&force[i].w, Scalar(0.5) * qiqj * (invr - Lr));
        atomicAdd(&force[j].w, Scalar(0.5) * qiqj * (invr - Lr));

        const size_t vpit = 1;
        const Scalar vxx = Scalar(0.5) * rij.x * f.x;
        const Scalar vxy = Scalar(0.5) * rij.x * f.y;
        const Scalar vxz = Scalar(0.5) * rij.x * f.z;
        const Scalar vyy = Scalar(0.5) * rij.y * f.y;
        const Scalar vyz = Scalar(0.5) * rij.y * f.z;
        const Scalar vzz = Scalar(0.5) * rij.z * f.z;

        atomicAdd(&virial[0 * vpit + i], vxx);
        atomicAdd(&virial[0 * vpit + j], vxx);
        atomicAdd(&virial[1 * vpit + i], vxy);
        atomicAdd(&virial[1 * vpit + j], vxy);
        atomicAdd(&virial[2 * vpit + i], vxz);
        atomicAdd(&virial[2 * vpit + j], vxz);
        atomicAdd(&virial[3 * vpit + i], vyy);
        atomicAdd(&virial[3 * vpit + j], vyy);
        atomicAdd(&virial[4 * vpit + i], vyz);
        atomicAdd(&virial[4 * vpit + j], vyz);
        atomicAdd(&virial[5 * vpit + i], vzz);
        atomicAdd(&virial[5 * vpit + j], vzz);
        }
    }

} // namespace

/*! \brief Construct a GPU ESP force compute. */
ESPForceComputeGPU::ESPForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                                       std::shared_ptr<NeighborList> nlist,
                                       std::shared_ptr<ParticleGroup> group)
    : ESPForceCompute(sysdef, nlist, group),
      m_sum(m_exec_conf),
      m_block_size(256)
    {
    m_tuner_assign.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                          m_exec_conf,
                                          "esp_assign"));
    m_tuner_update.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                          m_exec_conf,
                                          "esp_update_mesh"));
    m_tuner_force.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                         m_exec_conf,
                                         "esp_force"));
    m_tuner_influence.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                             m_exec_conf,
                                             "esp_influence"));

    m_autotuners.insert(m_autotuners.end(),
                        {m_tuner_assign, m_tuner_update, m_tuner_force, m_tuner_influence});

    m_local_fft = true;
    m_cufft_initialized = false;
    m_cuda_dfft_initialized = false;
    }

ESPForceComputeGPU::~ESPForceComputeGPU()
    {
    if (m_local_fft && m_cufft_initialized)
        {
        CHECK_HIPFFT_ERROR(hipfftDestroy(m_hipfft_plan));
        }
#ifdef ENABLE_MPI
    else if (m_cuda_dfft_initialized)
        {
        dfft_destroy_plan(m_dfft_plan_forward);
        dfft_destroy_plan(m_dfft_plan_inverse);
        }
#endif
    }

/*! \brief Setup FFT plans and allocate GPU mesh arrays. */
void ESPForceComputeGPU::initializeFFT()
    {
    ESPForceCompute::initializeFFT();

#ifdef ENABLE_MPI
    m_local_fft = !m_pdata->getDomainDecomposition();

    if (!m_local_fft)
        {
        m_gpu_grid_comm_forward = std::shared_ptr<CommunicatorGridGPUComplex>(
            new CommunicatorGridGPUComplex(m_sysdef,
                                           make_uint3(m_mesh_points.x,
                                                      m_mesh_points.y,
                                                      m_mesh_points.z),
                                           make_uint3(m_grid_dim.x, m_grid_dim.y, m_grid_dim.z),
                                           m_n_ghost_cells,
                                           true));
        m_gpu_grid_comm_reverse = std::shared_ptr<CommunicatorGridGPUComplex>(
            new CommunicatorGridGPUComplex(m_sysdef,
                                           make_uint3(m_mesh_points.x,
                                                      m_mesh_points.y,
                                                      m_mesh_points.z),
                                           make_uint3(m_grid_dim.x, m_grid_dim.y, m_grid_dim.z),
                                           m_n_ghost_cells,
                                           false));

        int gdim[3];
        int pdim[3];
        Index3D decomp_idx = m_pdata->getDomainDecomposition()->getDomainIndexer();
        pdim[0] = decomp_idx.getD();
        pdim[1] = decomp_idx.getH();
        pdim[2] = decomp_idx.getW();
        gdim[0] = m_mesh_points.z * pdim[0];
        gdim[1] = m_mesh_points.y * pdim[1];
        gdim[2] = m_mesh_points.x * pdim[2];
        int embed[3];
        embed[0] = m_mesh_points.z + 2 * m_n_ghost_cells.z;
        embed[1] = m_mesh_points.y + 2 * m_n_ghost_cells.y;
        embed[2] = m_mesh_points.x + 2 * m_n_ghost_cells.x;
        m_ghost_offset
            = (m_n_ghost_cells.z * embed[1] + m_n_ghost_cells.y) * embed[2] + m_n_ghost_cells.x;
        uint3 pcoord = m_pdata->getDomainDecomposition()->getGridPos();
        int pidx[3];
        pidx[0] = pcoord.z;
        pidx[1] = pcoord.y;
        pidx[2] = pcoord.x;
        int row_m = 0;
        ArrayHandle<unsigned int> h_cart_ranks(m_pdata->getDomainDecomposition()->getCartRanks(),
                                               access_location::host,
                                               access_mode::read);
#ifndef USE_HOST_DFFT
        dfft_cuda_create_plan(&m_dfft_plan_forward,
                              3,
                              gdim,
                              embed,
                              NULL,
                              pdim,
                              pidx,
                              row_m,
                              0,
                              1,
                              m_exec_conf->getMPICommunicator(),
                              (int*)h_cart_ranks.data);
        dfft_cuda_create_plan(&m_dfft_plan_inverse,
                              3,
                              gdim,
                              NULL,
                              embed,
                              pdim,
                              pidx,
                              row_m,
                              0,
                              1,
                              m_exec_conf->getMPICommunicator(),
                              (int*)h_cart_ranks.data);
#else
        dfft_create_plan(&m_dfft_plan_forward,
                         3,
                         gdim,
                         embed,
                         NULL,
                         pdim,
                         pidx,
                         row_m,
                         0,
                         1,
                         m_exec_conf->getMPICommunicator(),
                         (int*)h_cart_ranks.data);
        dfft_create_plan(&m_dfft_plan_inverse,
                         3,
                         gdim,
                         NULL,
                         embed,
                         pdim,
                         pidx,
                         row_m,
                         0,
                         1,
                         m_exec_conf->getMPICommunicator(),
                         (int*)h_cart_ranks.data);
#endif
        m_cuda_dfft_initialized = true;
        }
#endif

    if (m_local_fft)
        {
        CHECK_HIPFFT_ERROR(hipfftPlan3d(&m_hipfft_plan,
                                        m_mesh_points.z,
                                        m_mesh_points.y,
                                        m_mesh_points.x,
                                        HIPFFT_C2C));
        m_cufft_initialized = true;
        }

    GPUArray<hipfftComplex> mesh(m_n_cells, m_exec_conf);
    m_mesh.swap(mesh);

    unsigned int inv_mesh_elements = m_n_cells;
    GPUArray<hipfftComplex> mesh_scratch(inv_mesh_elements, m_exec_conf);
    m_mesh_scratch.swap(mesh_scratch);

    GPUArray<hipfftComplex> inv_fourier_mesh_x(inv_mesh_elements, m_exec_conf);
    m_inv_fourier_mesh_x.swap(inv_fourier_mesh_x);
    GPUArray<hipfftComplex> inv_fourier_mesh_y(inv_mesh_elements, m_exec_conf);
    m_inv_fourier_mesh_y.swap(inv_fourier_mesh_y);
    GPUArray<hipfftComplex> inv_fourier_mesh_z(inv_mesh_elements, m_exec_conf);
    m_inv_fourier_mesh_z.swap(inv_fourier_mesh_z);

    unsigned int n_blocks
        = (m_mesh_points.x * m_mesh_points.y * m_mesh_points.z + m_block_size - 1)
          / m_block_size;
    GPUArray<Scalar> sum_partial(n_blocks, m_exec_conf);
    m_sum_partial.swap(sum_partial);
    GPUArray<Scalar> sum_virial_partial(6 * n_blocks, m_exec_conf);
    m_sum_virial_partial.swap(sum_virial_partial);
    GPUArray<Scalar> sum_virial(6, m_exec_conf);
    m_sum_virial.swap(sum_virial);
    }

/*! \brief Assign particle charges to the GPU mesh. */
void ESPForceComputeGPU::assignParticles()
    {
    ArrayHandle<Scalar4> d_postype(m_pdata->getPositions(),
                                   access_location::device,
                                   access_mode::read);
    ArrayHandle<hipfftComplex> d_mesh(m_mesh, access_location::device, access_mode::overwrite);
    ArrayHandle<hipfftComplex> d_mesh_scratch(m_mesh_scratch,
                                              access_location::device,
                                              access_mode::overwrite);
    ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);
    unsigned int group_size = m_group->getNumMembers();
    ArrayHandle<Scalar> d_rho_coeff(m_rho_coeff, access_location::device, access_mode::read);

    m_exec_conf->setDevice();
    m_tuner_assign->begin();
    unsigned int block_size = m_tuner_assign->getParam()[0];

    kernel::gpu_assign_particles(m_mesh_points,
                                 m_n_ghost_cells,
                                 m_grid_dim,
                                 group_size,
                                 d_index_array.data,
                                 d_postype.data,
                                 d_charge.data,
                                 d_mesh.data,
                                 d_mesh_scratch.data,
                                 (unsigned int)m_mesh.getNumElements(),
                                 m_order,
                                 m_pdata->getBox(),
                                 block_size,
                                 d_rho_coeff.data,
                                 m_exec_conf->dev_prop);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_assign->end();
    }

/*! \brief Update the reciprocal mesh on GPU. */
void ESPForceComputeGPU::updateMeshes()
    {
    if (m_local_fft)
        {
        ArrayHandle<hipfftComplex> d_mesh(m_mesh, access_location::device, access_mode::readwrite);
        m_tuner_update->begin();
        CHECK_HIPFFT_ERROR(hipfftExecC2C(m_hipfft_plan, d_mesh.data, d_mesh.data, HIPFFT_FORWARD));
        m_tuner_update->end();
        }
#ifdef ENABLE_MPI
    else
        {
        m_exec_conf->msg->notice(8) << "charge.esp: Ghost cell update" << std::endl;
        m_gpu_grid_comm_forward->communicate(m_mesh);
        m_exec_conf->msg->notice(8) << "charge.esp: Distributed FFT mesh" << std::endl;
#ifndef USE_HOST_DFFT
        ArrayHandle<hipfftComplex> d_mesh(m_mesh, access_location::device, access_mode::read);
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            dfft_cuda_check_errors(&m_dfft_plan_forward, 1);
        else
            dfft_cuda_check_errors(&m_dfft_plan_forward, 0);
        dfft_cuda_execute(d_mesh.data + m_ghost_offset,
                          d_mesh.data + m_ghost_offset,
                          0,
                          &m_dfft_plan_forward);
#else
        ArrayHandle<hipfftComplex> h_mesh(m_mesh, access_location::host, access_mode::read);
        dfft_execute((cpx_t*)(h_mesh.data + m_ghost_offset),
                     (cpx_t*)(h_mesh.data + m_ghost_offset),
                     0,
                     m_dfft_plan_forward);
#endif
        }
#endif
    }

/*! \brief Interpolate forces from the inverse mesh. */
void ESPForceComputeGPU::interpolateForces()
    {
    ArrayHandle<Scalar4> d_postype(m_pdata->getPositions(),
                                   access_location::device,
                                   access_mode::read);
    ArrayHandle<Scalar4> d_force(m_pdata->getForces(), access_location::device, access_mode::readwrite);
    ArrayHandle<hipfftComplex> d_inv_fourier_mesh_x(m_inv_fourier_mesh_x,
                                                    access_location::device,
                                                    access_mode::read);
    ArrayHandle<hipfftComplex> d_inv_fourier_mesh_y(m_inv_fourier_mesh_y,
                                                    access_location::device,
                                                    access_mode::read);
    ArrayHandle<hipfftComplex> d_inv_fourier_mesh_z(m_inv_fourier_mesh_z,
                                                    access_location::device,
                                                    access_mode::read);
    ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);
    ArrayHandle<Scalar> d_rho_coeff(m_rho_coeff, access_location::device, access_mode::read);
    unsigned int group_size = m_group->getNumMembers();

    m_tuner_force->begin();
    unsigned int block_size = m_tuner_force->getParam()[0];
    kernel::gpu_compute_forces(m_pdata->getNParticles(),
                               group_size,
                               d_postype.data,
                               d_force.data,
                               d_inv_fourier_mesh_x.data,
                               d_inv_fourier_mesh_y.data,
                               d_inv_fourier_mesh_z.data,
                               m_grid_dim,
                               m_n_ghost_cells,
                               d_charge.data,
                               m_pdata->getBox(),
                               m_order,
                               d_index_array.data,
                               d_rho_coeff.data,
                               block_size,
                               m_local_fft,
                               (unsigned int)m_mesh.getNumElements());
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_force->end();
    }

/*! \brief Compute the influence function on GPU. */
void ESPForceComputeGPU::computeInfluenceFunction()
    {
    computeInfluenceFunctionGPU();
    }

/*! \brief Compute reciprocal-space potential energy. */
Scalar ESPForceComputeGPU::computePE()
    {
    return ESPForceCompute::computePE();
    }

/*! \brief Correct excluded-pair forces on GPU. */
void ESPForceComputeGPU::fixExclusions()
    {
    fixExclusionsGPU();
    }

/*! \brief Build the GPU influence function from the PSWF table. */
void ESPForceComputeGPU::computeInfluenceFunctionGPU()
    {
    launchInfluenceFunctionKernel();
    }

void ESPForceComputeGPU::launchInfluenceFunctionKernel()
    {
    m_exec_conf->setDevice();

    ArrayHandle<Scalar> d_inf_f(m_inf_f, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar3> d_k(m_k, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_denom(m_sum_partial, access_location::device, access_mode::overwrite);

    const uint3 dim = m_mesh_points;
    const unsigned int n = dim.x * dim.y * dim.z;
    const unsigned int block_size = 256;
    const unsigned int grid_size = (n + block_size - 1) / block_size;

    launchGFDenominatorKernel();
    gpu_build_influence_kernel<<<grid_size, block_size>>>(dim,
                                                           m_pdata->getBox().getL().x,
                                                           m_pdata->getBox().getL().y,
                                                           m_pdata->getBox().getL().z,
                                                           m_kappa,
                                                           m_alpha,
                                                           d_denom.data,
                                                           d_inf_f.data,
                                                           d_k.data);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    }

void ESPForceComputeGPU::launchGFDenominatorKernel()
    {
    m_exec_conf->setDevice();

    ArrayHandle<Scalar> d_gf_b(m_gf_b, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_denom(m_sum_partial, access_location::device, access_mode::overwrite);

    const uint3 dim = m_mesh_points;
    const unsigned int n = dim.x * dim.y * dim.z;
    const unsigned int block_size = 256;
    const unsigned int grid_size = (n + block_size - 1) / block_size;

    gpu_build_gf_denom_kernel<<<grid_size, block_size>>>(dim,
                                                          m_pdata->getBox().getL().x,
                                                          m_pdata->getBox().getL().y,
                                                          m_pdata->getBox().getL().z,
                                                          m_rcut,
                                                          d_gf_b.data,
                                                          d_denom.data);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    }

/*! \brief Apply excluded-pair corrections on the GPU. */
void ESPForceComputeGPU::fixExclusionsGPU()
    {
    m_exec_conf->setDevice();

    if (!m_nlist->getExclusionsSet())
        return;

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_chg(m_pdata->getCharges(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_virial(m_virial, access_location::device, access_mode::readwrite);
    ArrayHandle<unsigned int> d_nex(m_nlist->getNExArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_exlist(m_nlist->getExListArray(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_table_flat(m_pswf_table_gpu, access_location::device, access_mode::read);

    const unsigned int n = m_pdata->getN();
    const unsigned int block_size = 128;
    const unsigned int grid_size = (n + block_size - 1) / block_size;
    const Index2D ex_idx = m_nlist->getExListIndexer();
    const ESPTableEntry* table = reinterpret_cast<const ESPTableEntry*>(d_table_flat.data);

    gpu_fix_exclusions_kernel<<<grid_size, block_size>>>(n,
                                                         d_pos.data,
                                                         d_chg.data,
                                                         d_force.data,
                                                         d_virial.data,
                                                         d_nex.data,
                                                         d_exlist.data,
                                                         ex_idx,
                                                         table,
                                                         m_n_table_segments,
                                                         m_rcut,
                                                         m_pdata->getBox());
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    }

void ESPForceComputeGPU::computeBodyCorrection()
    {
    refreshChargeDependentState();
    ESPForceCompute::computeBodyCorrection();
    }

void ESPForceComputeGPU::refreshChargeDependentState()
    {
    m_q  = Scalar(0.0);
    m_q2 = Scalar(0.0);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);
    for (unsigned int idx = 0; idx < m_pdata->getN(); ++idx)
        {
        const Scalar q = h_charge.data[idx];
        m_q += q;
        m_q2 += q * q;
        }
    }

    } // end namespace md
    } // end namespace hoomd

#endif // ENABLE_HIP
