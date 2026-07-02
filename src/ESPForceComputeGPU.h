// Copyright (c) 2024-2026 ESP Plugin Contributors.
// Released under the BSD 3-Clause License.
//
// ESPForceComputeGPU.h
//
// GPU override for the ESPForceCompute mesh + short-range-correction
// pipeline. Implements the HOOMD-blue v5.x GPU ForceCompute override
// pattern: base-class virtual hooks are overridden here to dispatch to
// CUDA/HIP kernels declared in ESPForceComputeGPU.cuh, while all memory
// lifetime is managed exclusively through hoomd::GPUArray<T> (v5 API) --
// no raw cudaMalloc/cudaFree, and no reliance on deprecated v3/v4
// GPUArray::acquire()-style handles.
//
#pragma once

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "ESPForceCompute.h"
#include "hoomd/Autotuner.h"
#include "hoomd/GPUArray.h"
#include "hoomd/GPUFlags.h"

#ifdef ENABLE_HIP
#include <hipfft/hipfft.h>
#endif

#include <memory>
#include <sstream>
#include <stdexcept>

namespace hoomd
    {
namespace md
    {

/*!
 * \class ESPForceComputeGPU
 * \brief GPU-accelerated implementation of the ESP (Ewald Summation with
 *        Prolates) long-range electrostatics force compute.
 *
 * Overrides the mesh-pipeline stages of ESPForceCompute (assignment,
 * FFT/influence-function application, force interpolation, exclusion
 * correction) with CUDA/HIP kernel dispatches. All device buffers are
 * owned via hoomd::GPUArray<T> (v5 API), which handles host/device
 * synchronization and lifetime automatically -- consistent with every
 * other GPU ForceCompute in HOOMD-blue v5.1+ (e.g. PPPMForceComputeGPU).
 *
 * The large PSWF short-range screening tables are NOT stored as GPUArray
 * members on this class: they are compiled directly into
 * ESPForceComputeGPU.cu from PSWF_Coeffs.h and accessed via __ldg(), so no
 * separate upload/lifetime management is required for them here.
 */
class PYBIND11_EXPORT ESPForceComputeGPU : public ESPForceCompute
    {
    public:
    ESPForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                        std::shared_ptr<NeighborList> nlist,
                        std::shared_ptr<ParticleGroup> group);

    ~ESPForceComputeGPU() override;

    protected:
    // ── ESPForceCompute pipeline-stage overrides ────────────────────────────
    void initializeFFT();
    void assignParticles();
    void updateMeshes();
    void interpolateForces();
    void computeInfluenceFunction();
    Scalar computePE();
    void fixExclusions();
    void computeBodyCorrection();

    // ── GPU-specific implementation helpers ─────────────────────────────────
    void computeInfluenceFunctionGPU();
    void launchInfluenceFunctionKernel();
    void launchGFDenominatorKernel();
    void fixExclusionsGPU();
    void refreshChargeDependentState();

#ifdef ENABLE_HIP
    //! Translate a hipfftResult into a thrown std::runtime_error with
    //! file/line context, mirroring HOOMD's CHECK_CUDA_ERROR() convention
    //! for hipFFT return codes (which are not covered by hipError_t).
    inline void handleHIPFFTResult(hipfftResult result, const char* file, unsigned int line) const
        {
        if (result != HIPFFT_SUCCESS)
            {
            std::ostringstream oss;
            oss << "HIPFFT returned error " << result << " in file " << file << " line " << line << std::endl;
            throw std::runtime_error(oss.str());
            }
        }
#endif

    private:
    // ── Autotuners (HOOMD v5 Autotuner<1> API: block-size search only) ──────
    std::shared_ptr<Autotuner<1>> m_tuner_assign;
    std::shared_ptr<Autotuner<1>> m_tuner_update;
    std::shared_ptr<Autotuner<1>> m_tuner_force;
    std::shared_ptr<Autotuner<1>> m_tuner_influence;

#ifdef ENABLE_HIP
    hipfftHandle m_hipfft_plan;
#endif
    bool m_local_fft;
    bool m_cufft_initialized;
    bool m_cuda_dfft_initialized;

#ifdef ENABLE_MPI
    using CommunicatorGridGPUComplex = CommunicatorGridGPU<hoomd::CScalar>;
    std::shared_ptr<CommunicatorGridGPUComplex> m_gpu_grid_comm_forward;
    std::shared_ptr<CommunicatorGridGPUComplex> m_gpu_grid_comm_reverse;
    dfft_plan m_dfft_plan_forward;
    dfft_plan m_dfft_plan_inverse;
#endif

    // ── Device-resident mesh buffers (v5 GPUArray<T>: RAII + auto sync) ─────
    GPUArray<hoomd::CScalar> m_mesh;
    GPUArray<hoomd::CScalar> m_mesh_scratch;
    GPUArray<Scalar> m_inv_fourier_mesh_x;
    GPUArray<Scalar> m_inv_fourier_mesh_y;
    GPUArray<Scalar> m_inv_fourier_mesh_z;

    GPUFlags<Scalar> m_sum;
    GPUArray<Scalar> m_sum_partial;
    GPUArray<Scalar> m_sum_virial_partial;
    GPUArray<Scalar> m_sum_virial;
    unsigned int m_block_size;
    };

    } // namespace md
    } // namespace hoomd