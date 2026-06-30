// Copyright (c) 2024-2026 ESP Plugin Contributors.
// Released under the BSD 3-Clause License.
//
// ESPForceComputeGPU.cc
//
// GPU override for the ESP force pipeline.

#include "ESPForceComputeGPU.h"

#ifdef ENABLE_HIP
#include "ESPForceComputeGPU.cuh"

namespace hoomd
    {
namespace md
    {

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

    GPUArray<hipfftComplex> mesh(m_n_cells + m_ghost_offset, m_exec_conf);
    m_mesh.swap(mesh);

    unsigned int inv_mesh_elements = m_n_cells + m_ghost_offset;
    GPUArray<hipfftComplex> mesh_scratch(inv_mesh_elements, m_exec_conf);
    m_mesh_scratch.swap(mesh_scratch);

    GPUArray<hipfftComplex> inv_fourier_mesh_x(inv_mesh_elements, m_exec_conf);
    m_inv_fourier_mesh_x.swap(inv_fourier_mesh_x);
    GPUArray<hipfftComplex> inv_fourier_mesh_y(inv_mesh_elements, m_exec_conf);
    m_inv_fourier_mesh_y.swap(inv_fourier_mesh_y);
    GPUArray<hipfftComplex> inv_fourier_mesh_z(inv_mesh_elements, m_exec_conf);
    m_inv_fourier_mesh_z.swap(inv_fourier_mesh_z);

    unsigned int n_blocks
        = (m_mesh_points.x * m_mesh_points.y * m_mesh_points.z) / m_block_size + 1;
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
    // TODO(ESP): The gpu_compute_influence_function kernel computes the
    // Gaussian influence function using arggauss = 0.25*dot2/kappa^2.
    // For ESP, this must be replaced by the PSWF Fourier transform:
    //   pswf_hat(k*rc/sigma0) = prod_d sinc(k_d*h_d/2)^P * compact(k_d*h_d/2)
    // Until the kernel is patched, the reciprocal-space forces use the
    // Gaussian approximation and will exhibit O(exp(-kappa^2*rc^2)) error.
    // See ESPForceCompute::computeInfluenceFunction() for the correct formula.
    m_tuner_influence->begin();
    kernel::gpu_compute_influence_function(m_mesh_points,
                                           m_global_dim,
                                           m_inf_f.data(),
                                           m_k.data(),
                                           m_pdata->getGlobalBox(),
                                           m_local_fft,
                                           make_uint3(0, 0, 0),
                                           make_uint3(1, 1, 1),
                                           Scalar(0.0),
                                           m_kappa,
                                           m_alpha,
                                           m_gf_b.data(),
                                           m_order,
                                           m_tuner_influence->getParam()[0]);
    m_tuner_influence->end();
    }

/*! \brief Compute reciprocal-space potential energy. */
Scalar ESPForceComputeGPU::computePE()
    {
    return ESPForceCompute::computePE();
    }

/*! \brief Correct excluded-pair forces on GPU. */
void ESPForceComputeGPU::fixExclusions()
    {
    ESPForceCompute::fixExclusions();
    // TODO(ESP): gpu_fix_exclusions uses erfc(kappa*r)/r from Gaussian splitting.
    // For ESP, the exclusion correction must subtract the PSWF-derived L(r) from
    // m_pswf_table_gpu instead of erfc(kappa*r)/r. This causes a systematic error
    // proportional to |L_ewald(r) - L_esp(r)| for excluded pairs.
    // Fix: add a new gpu_fix_exclusions_pswf kernel that reads m_pswf_table_gpu.
    }

    } // end namespace md
    } // end namespace hoomd

#endif // ENABLE_HIP// Copyright (c) 2025 Shih-Lun Weng.
// Part of RxMC, released under the BSD 3-Clause License.

/*! \file ESPForceComputeGPU.cc
    \brief Defines the custom GPU PPPM force compute.
*/

#include "HelpFunctions/ESPForceComputeGPU.h"

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {

#ifdef ENABLE_HIP
ESPForceComputeGPU::ESPForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                     std::shared_ptr<NeighborList> nlist,
                                                     std::shared_ptr<ParticleGroup> group)
    : PPPMForceComputeGPU(sysdef, nlist, group)
    {
    }

uint3 ESPForceComputeGPU::computeGhostCellNumCustom() const
    {
    uint3 n_ghost_cells = make_uint3(0, 0, 0);
#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        Index3D di = m_pdata->getDomainDecomposition()->getDomainIndexer();
        n_ghost_cells.x = (di.getW() > 1) ? m_radius : 0;
        n_ghost_cells.y = (di.getH() > 1) ? m_radius : 0;
        n_ghost_cells.z = (di.getD() > 1) ? m_radius : 0;
        }
#endif

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        Scalar r_buff = m_nlist->getRBuff() / 2.0;
        const BoxDim& box = m_pdata->getBox();
        Scalar3 cell_width = box.getNearestPlaneDistance()
                             / make_scalar3(m_mesh_points.x, m_mesh_points.y, m_mesh_points.z);

        if (n_ghost_cells.x)
            n_ghost_cells.x += (unsigned int)(r_buff / cell_width.x) + 1;
        if (n_ghost_cells.y)
            n_ghost_cells.y += (unsigned int)(r_buff / cell_width.y) + 1;
        if (n_ghost_cells.z)
            n_ghost_cells.z += (unsigned int)(r_buff / cell_width.z) + 1;
        }
#endif
    return n_ghost_cells;
    }

void ESPForceComputeGPU::refreshChargeDependentState()
    {
    m_q = getQSum();
    m_q2 = getQ2Sum();

    if (m_nlist->getFilterBody())
        {
        computeBodyCorrection();
        }
    }

void ESPForceComputeGPU::computeForces(uint64_t timestep)
    {
    if (m_need_initialize || m_ptls_added_removed)
        {
        if (!m_params_set)
            {
            m_exec_conf->msg->error()
                << "custom_pppm: parameters must be set before run()" << std::endl;
            throw std::runtime_error("Error computing custom PPPM GPU forces");
            }

        setupMesh();
        setupCoeffs();
        computeInfluenceFunction();

        if (m_nlist->getFilterBody())
            {
            m_exec_conf->msg->notice(2)
                << "custom_pppm: calculating rigid body correction (N^2)" << std::endl;
            computeBodyCorrection();
            }

        m_need_initialize = false;
        m_ptls_added_removed = false;
        }

    bool ghost_cell_num_changed = false;
    uint3 n_ghost_cells = computeGhostCellNumCustom();
    if (m_n_ghost_cells.x != n_ghost_cells.x || m_n_ghost_cells.y != n_ghost_cells.y
        || m_n_ghost_cells.z != n_ghost_cells.z)
        ghost_cell_num_changed = true;

    if (m_box_changed || ghost_cell_num_changed)
        {
        if (ghost_cell_num_changed)
            {
            setupMesh();
            }
        computeInfluenceFunction();
        m_box_changed = false;
        }

    // Phase-3 optimization hook for RxMC:
    // keep the PPPM charge cache consistent after charge/type moves, but avoid
    // the global-particle-number path that reallocates mesh/FFT structures.
    refreshChargeDependentState();

    assignParticles();
    updateMeshes();

    PDataFlags flags = this->m_pdata->getFlags();
    computePE();
    interpolateForces();

    if (flags[pdata_flag::pressure_tensor])
        {
        computeVirial();
        }
    else
        {
        for (unsigned int i = 0; i < 6; ++i)
            {
            m_external_virial[i] = Scalar(0.0);
            }
        }

    if (m_nlist->getExclusionsSet())
        {
        m_nlist->compute(timestep);
        fixExclusions();
        }
    }
#endif

namespace detail
    {
void export_ESPForceComputeGPU(pybind11::module& m)
    {
#ifdef ENABLE_HIP
    pybind11::class_<ESPForceComputeGPU,
                     PPPMForceComputeGPU,
                     std::shared_ptr<ESPForceComputeGPU>>(m, "ESPForceComputeGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<NeighborList>,
                            std::shared_ptr<ParticleGroup>>());
#else
    (void)m;
#endif
    }
    } // namespace detail

    } // namespace md
    } // namespace hoomd
