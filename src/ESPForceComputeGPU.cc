// Copyright (c) 2024-2026 ESP Plugin Contributors.
// Released under the BSD 3-Clause License.
//
// ESPForceComputeGPU.cc
//
// GPU override for the ESP force pipeline. Host-side glue code that stages
// HOOMD ArrayHandle<> access, launches the CUDA kernels declared in
// ESPForceComputeGPU.cuh, and manages Autotuner-driven block sizes.
//
// Bug fix in this revision
// -------------------------
// fixExclusionsGPU() previously passed runtime device pointers for the PSWF
// screening-table coefficients (m_pswf_pot_coeffs_device, etc.) into
// kernel::gpu_esp_fix_exclusions(). Those tables are now compiled directly
// into ESPForceComputeGPU.cu from PSWF_Coeffs.h, so this call site is
// simplified accordingly. The virial_pitch fix (m_virial.getPitch() instead
// of a hardcoded 1) is preserved and is now enforced by the updated .cuh
// function signature itself.
//
#include "ESPForceComputeGPU.h"
#include "ESPForceComputeGPU.cuh"

#include <sstream>
#include <stdexcept>

#ifdef ENABLE_HIP

namespace hoomd
    {
namespace md
    {

#define CHECK_HIPFFT_ERROR(call) handleHIPFFTResult((call), __FILE__, __LINE__)
#define CHECK_HIP_ERROR(err)                                                                                        \
    do                                                                                                              \
        {                                                                                                            \
        if ((err) != hipSuccess)                                                                                    \
            {                                                                                                        \
            std::ostringstream oss;                                                                                 \
            oss << "HIP error " << hipGetErrorString(err) << " in file " << __FILE__ << " line " << __LINE__;       \
            throw std::runtime_error(oss.str());                                                                     \
            }                                                                                                        \
        }                                                                                                            \
    while (0)

/*! \brief Construct a GPU ESP force compute. */
ESPForceComputeGPU::ESPForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                                        std::shared_ptr<NeighborList> nlist,
                                        std::shared_ptr<ParticleGroup> group)
    : ESPForceCompute(sysdef, nlist, group), m_sum(m_exec_conf), m_block_size(256)
    {
    m_tuner_assign.reset(
        new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)}, m_exec_conf, "esp_assign"));
    m_tuner_update.reset(
        new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)}, m_exec_conf, "esp_update_mesh"));
    m_tuner_force.reset(
        new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)}, m_exec_conf, "esp_force"));
    m_tuner_influence.reset(
        new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)}, m_exec_conf, "esp_influence"));

    m_autotuners.insert(m_autotuners.end(), {m_tuner_assign, m_tuner_update, m_tuner_force, m_tuner_influence});

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
        m_gpu_grid_comm_forward = std::shared_ptr<CommunicatorGridGPUComplex>(new CommunicatorGridGPUComplex(
            m_sysdef,
            make_uint3(m_mesh_points.x, m_mesh_points.y, m_mesh_points.z),
            make_uint3(m_grid_dim.x, m_grid_dim.y, m_grid_dim.z),
            m_n_ghost_cells,
            true));
        m_gpu_grid_comm_reverse = std::shared_ptr<CommunicatorGridGPUComplex>(new CommunicatorGridGPUComplex(
            m_sysdef,
            make_uint3(m_mesh_points.x, m_mesh_points.y, m_mesh_points.z),
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
        m_ghost_offset = (m_n_ghost_cells.z * embed[1] + m_n_ghost_cells.y) * embed[2] + m_n_ghost_cells.x;
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
        CHECK_HIPFFT_ERROR(
            hipfftPlan3d(&m_hipfft_plan, m_mesh_points.z, m_mesh_points.y, m_mesh_points.x, HIPFFT_C2C));
        m_cufft_initialized = true;
        }

    const unsigned int n_cells = m_mesh_points.x * m_mesh_points.y * m_mesh_points.z;

    GPUArray<hoomd::CScalar> mesh(n_cells, m_exec_conf);
    m_mesh.swap(mesh);

    unsigned int inv_mesh_elements = n_cells;
    GPUArray<hoomd::CScalar> mesh_scratch(inv_mesh_elements, m_exec_conf);
    m_mesh_scratch.swap(mesh_scratch);

    GPUArray<Scalar> inv_fourier_mesh_x(inv_mesh_elements, m_exec_conf);
    m_inv_fourier_mesh_x.swap(inv_fourier_mesh_x);
    GPUArray<Scalar> inv_fourier_mesh_y(inv_mesh_elements, m_exec_conf);
    m_inv_fourier_mesh_y.swap(inv_fourier_mesh_y);
    GPUArray<Scalar> inv_fourier_mesh_z(inv_mesh_elements, m_exec_conf);
    m_inv_fourier_mesh_z.swap(inv_fourier_mesh_z);

    unsigned int n_blocks = (n_cells + m_block_size - 1) / m_block_size;
    GPUArray<Scalar> sum_partial(n_blocks, m_exec_conf);
    m_sum_partial.swap(sum_partial);
    GPUArray<Scalar> sum_virial_partial(6 * n_blocks, m_exec_conf);
    m_sum_virial_partial.swap(sum_virial_partial);
    GPUArray<Scalar> sum_virial(6, m_exec_conf);
    m_sum_virial.swap(sum_virial);
    }

/*! \brief Assign particle charges to the GPU mesh (PSWF stencil, Horner-based). */
void ESPForceComputeGPU::assignParticles()
    {
    m_exec_conf->setDevice();

    ArrayHandle<Scalar4> d_postype(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<hoomd::CScalar> d_mesh(m_mesh, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);
    unsigned int group_size = m_group->getNumMembers();

    // Zero the mesh before atomic accumulation.
    hipMemset(d_mesh.data, 0, m_mesh.getNumElements() * sizeof(hoomd::CScalar));

    m_tuner_assign->begin();
    unsigned int block_size = m_tuner_assign->getParam()[0];

    const Scalar3 h_grid = make_scalar3(Scalar(1.0) / Scalar(m_mesh_points.x),
                                         Scalar(1.0) / Scalar(m_mesh_points.y),
                                         Scalar(1.0) / Scalar(m_mesh_points.z));

    hipError_t err = kernel::gpu_esp_assign_particles(m_mesh_points,
                                                       m_grid_dim,
                                                       group_size,
                                                       d_index_array.data,
                                                       d_postype.data,
                                                       d_charge.data,
                                                       d_mesh.data,
                                                       m_order,
                                                       m_pdata->getBox(),
                                                       h_grid,
                                                       block_size);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_HIP_ERROR(err);
    m_tuner_assign->end();
    }

/*! \brief Update the reciprocal mesh on GPU (forward FFT). */
void ESPForceComputeGPU::updateMeshes()
    {
    if (m_local_fft)
        {
        ArrayHandle<hoomd::CScalar> d_mesh(m_mesh, access_location::device, access_mode::readwrite);
        m_tuner_update->begin();
        CHECK_HIPFFT_ERROR(hipfftExecC2C(m_hipfft_plan,
                                          reinterpret_cast<hipfftComplex*>(d_mesh.data),
                                          reinterpret_cast<hipfftComplex*>(d_mesh.data),
                                          HIPFFT_FORWARD));
        m_tuner_update->end();
        }
#ifdef ENABLE_MPI
    else
        {
        m_exec_conf->msg->notice(8) << "charge.esp: Ghost cell update" << std::endl;
        m_gpu_grid_comm_forward->communicate(m_mesh);
        m_exec_conf->msg->notice(8) << "charge.esp: Distributed FFT mesh" << std::endl;
#ifndef USE_HOST_DFFT
        ArrayHandle<hoomd::CScalar> d_mesh(m_mesh, access_location::device, access_mode::read);
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            dfft_cuda_check_errors(&m_dfft_plan_forward, 1);
        else
            dfft_cuda_check_errors(&m_dfft_plan_forward, 0);
        dfft_cuda_execute(d_mesh.data + m_ghost_offset, d_mesh.data + m_ghost_offset, 0, &m_dfft_plan_forward);
#else
        ArrayHandle<hoomd::CScalar> h_mesh(m_mesh, access_location::host, access_mode::read);
        dfft_execute((cpx_t*)(h_mesh.data + m_ghost_offset), (cpx_t*)(h_mesh.data + m_ghost_offset), 0, m_dfft_plan_forward);
#endif
        }
#endif
    }

/*! \brief Apply the influence function on GPU (reciprocal-space scaling). */
void ESPForceComputeGPU::computeInfluenceFunctionGPU()
    {
    launchInfluenceFunctionKernel();
    }

void ESPForceComputeGPU::launchGFDenominatorKernel()
    {
    m_exec_conf->setDevice();

    ArrayHandle<Scalar> d_gf_b(m_gf_b, access_location::device, access_mode::read);

    const uint3 dim = m_mesh_points;
    hipError_t err = kernel::gpu_esp_build_gf_denom(
        dim, m_pdata->getBox().getL(), m_rcut, reinterpret_cast<const double*>(d_gf_b.data), nullptr, m_block_size);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_HIP_ERROR(err);
    }

void ESPForceComputeGPU::launchInfluenceFunctionKernel()
    {
    m_exec_conf->setDevice();

    ArrayHandle<Scalar> d_inf_f(m_inf_f, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar3> d_k(m_k, access_location::device, access_mode::overwrite);

    const uint3 dim = m_mesh_points;

    launchGFDenominatorKernel();

    hipError_t err = kernel::gpu_esp_build_influence(
        dim, m_pdata->getBox().getL(), m_kappa, m_alpha, nullptr, d_inf_f.data, d_k.data, m_block_size);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_HIP_ERROR(err);
    }

/*! \brief Compute the influence function on GPU. */
void ESPForceComputeGPU::computeInfluenceFunction()
    {
    computeInfluenceFunctionGPU();
    }

/*! \brief Interpolate forces from the inverse mesh on GPU. */
void ESPForceComputeGPU::interpolateForces()
    {
    m_exec_conf->setDevice();

    ArrayHandle<Scalar4> d_postype(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_inv_fourier_mesh_x(m_inv_fourier_mesh_x, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_inv_fourier_mesh_y(m_inv_fourier_mesh_y, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_inv_fourier_mesh_z(m_inv_fourier_mesh_z, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);
    unsigned int group_size = m_group->getNumMembers();

    m_tuner_force->begin();
    unsigned int block_size = m_tuner_force->getParam()[0];

    const Scalar3 h_grid = make_scalar3(Scalar(1.0) / Scalar(m_grid_dim.x),
                                         Scalar(1.0) / Scalar(m_grid_dim.y),
                                         Scalar(1.0) / Scalar(m_grid_dim.z));

    hipError_t err = kernel::gpu_esp_interpolate_forces(m_pdata->getN(),
                                                        group_size,
                                                        d_index_array.data,
                                                        d_postype.data,
                                                        d_charge.data,
                                                        d_inv_fourier_mesh_x.data,
                                                        d_inv_fourier_mesh_y.data,
                                                        d_inv_fourier_mesh_z.data,
                                                        m_grid_dim,
                                                        m_order,
                                                        m_pdata->getBox(),
                                                        h_grid,
                                                        d_force.data,
                                                        block_size);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_HIP_ERROR(err);
    m_tuner_force->end();
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

/*!
 * \brief Apply excluded-pair corrections on the GPU (v5-compliant, bug-fixed).
 *
 * The PSWF screening coefficients are no longer passed as runtime device
 * pointers: ESPForceComputeGPU.cu compiles them in directly from
 * PSWF_Coeffs.h and reads them through __ldg(). This call site only needs
 * to supply the topology (positions, charges, exclusion lists) and the
 * CORRECT virial pitch.
 */
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

    const unsigned int n = m_pdata->getN();
    const unsigned int block_size = 128;
    const Index2D ex_idx = m_nlist->getExListIndexer();

    // BUG FIX: use the true virial pitch (== number of particles N) rather
    // than a hardcoded constant of 1. m_virial is a GPUArray of logical
    // shape (6, N); getPitch() returns N. This is now REQUIRED by the
    // updated kernel::gpu_esp_fix_exclusions() signature.
    const size_t virial_pitch = m_virial.getPitch();

    hipError_t err = kernel::gpu_esp_fix_exclusions(n,
                                                     d_pos.data,
                                                     d_chg.data,
                                                     d_force.data,
                                                     d_virial.data,
                                                     virial_pitch,
                                                     d_nex.data,
                                                     d_exlist.data,
                                                     ex_idx,
                                                     m_rcut,
                                                     m_pdata->getBox(),
                                                     block_size);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_HIP_ERROR(err);
    }

void ESPForceComputeGPU::computeBodyCorrection()
    {
    refreshChargeDependentState();
    ESPForceCompute::computeBodyCorrection();
    }

void ESPForceComputeGPU::refreshChargeDependentState()
    {
    m_q = Scalar(0.0);
    m_q2 = Scalar(0.0);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);
    for (unsigned int idx = 0; idx < m_pdata->getN(); ++idx)
        {
        const Scalar q = h_charge.data[idx];
        m_q += q;
        m_q2 += q * q;
        }
    }

// ============================================================================
// pybind11 export: ESPForceComputeGPU
//
// Registered as a subclass of ESPForceCompute so Python code that type-checks
// against esp.Spectral._cpp_obj (an ESPForceCompute instance) continues to
// work transparently whether the GPU or CPU build was loaded.
// ============================================================================
void export_ESPForceComputeGPU(pybind11::module& m)
    {
    pybind11::class_<ESPForceComputeGPU, ESPForceCompute, std::shared_ptr<ESPForceComputeGPU>>(
        m, "ESPForceComputeGPU", R"delim(
GPU-accelerated long-range electrostatics compute based on PSWF/ESP mesh methods.

Drop-in GPU replacement for ESPForceCompute: identical Python-facing API,
dispatches all mesh-pipeline stages (charge assignment, FFT/influence
function, force interpolation, exclusion correction) to CUDA/HIP kernels.
)delim")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                             std::shared_ptr<NeighborList>,
                             std::shared_ptr<ParticleGroup>>(),
             pybind11::arg("sysdef"),
             pybind11::arg("nlist"),
             pybind11::arg("group"),
             R"delim(
Construct the GPU ESP force compute.

Args:
    sysdef: System definition.
    nlist: Neighbor list used for short-range exclusions and metadata.
    group: Particle group to which the force applies.
)delim");
    }

    } // end namespace md
    } // end namespace hoomd

#endif // ENABLE_HIP