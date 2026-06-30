// Copyright (c) 2024-2026 ESP Plugin Contributors.
// Released under the BSD 3-Clause License.
//
// ESPForceComputeGPU.h
//
// GPU override for the ESPForceCompute mesh pipeline.

#pragma once

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "ESPForceCompute.h"
#include "hoomd/Autotuner.h"
#include "hoomd/GPUFlags.h"

#ifdef ENABLE_HIP
#include <hipfft.h>
#endif

#include <memory>
#include <sstream>

namespace hoomd
    {
namespace md
    {

class PYBIND11_EXPORT ESPForceComputeGPU : public ESPForceCompute
    {
    public:
    ESPForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                       std::shared_ptr<NeighborList> nlist,
                       std::shared_ptr<ParticleGroup> group);

    ~ESPForceComputeGPU() override;

    protected:
    void initializeFFT();
    void assignParticles();
    void updateMeshes();
    void interpolateForces();
    void computeInfluenceFunction();
    Scalar computePE();
    void fixExclusions();

    void computeInfluenceFunctionGPU();
    void launchInfluenceFunctionKernel();
    void launchGFDenominatorKernel();
    void fixExclusionsGPU();
    void refreshChargeDependentState();

#ifdef ENABLE_HIP
    inline void handleHIPFFTResult(hipfftResult result,
                                   const char* file,
                                   unsigned int line) const
        {
        if (result != HIPFFT_SUCCESS)
            {
            std::ostringstream oss;
            oss << "HIPFFT returned error " << result << " in file " << file
                << " line " << line << std::endl;
            throw std::runtime_error(oss.str());
            }
        }
#endif

    private:
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
    using CommunicatorGridGPUComplex = CommunicatorGridGPU<hipfftComplex>;
    std::shared_ptr<CommunicatorGridGPUComplex> m_gpu_grid_comm_forward;
    std::shared_ptr<CommunicatorGridGPUComplex> m_gpu_grid_comm_reverse;
    dfft_plan m_dfft_plan_forward;
    dfft_plan m_dfft_plan_inverse;
#endif

    GPUArray<hipfftComplex> m_mesh;
    GPUArray<hipfftComplex> m_mesh_scratch;
    GPUArray<hipfftComplex> m_inv_fourier_mesh_x;
    GPUArray<hipfftComplex> m_inv_fourier_mesh_y;
    GPUArray<hipfftComplex> m_inv_fourier_mesh_z;

    GPUFlags<Scalar> m_sum;
    GPUArray<Scalar> m_sum_partial;
    GPUArray<Scalar> m_sum_virial_partial;
    GPUArray<Scalar> m_sum_virial;
    unsigned int m_block_size;
    };

    } // namespace md
    } // namespace hoomd
