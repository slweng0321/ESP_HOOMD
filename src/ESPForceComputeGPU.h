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

#include <sstream>

namespace hoomd
    {
namespace md
    {
/*! \brief GPU-accelerated ESP force compute.
 */
class PYBIND11_EXPORT ESPForceComputeGPU : public ESPForceCompute
    {
    public:
    /*! \brief Construct a GPU ESP force compute.
     */
    ESPForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                       std::shared_ptr<NeighborList> nlist,
                       std::shared_ptr<ParticleGroup> group);

    ~ESPForceComputeGPU() override;

    protected:
    /*! \brief Setup FFT plans and allocate GPU mesh arrays. */
    void initializeFFT() override;

    /*! \brief Assign particle charges to the GPU mesh. */
    void assignParticles() override;

    /*! \brief Update the reciprocal mesh on GPU. */
    void updateMeshes() override;

    /*! \brief Interpolate forces from the inverse mesh. */
    void interpolateForces() override;

    /*! \brief Compute the influence function on GPU. */
    void computeInfluenceFunction() override;

    /*! \brief Compute reciprocal-space potential energy. */
    Scalar computePE() override;

    /*! \brief Correct excluded-pair forces on GPU. */
    void fixExclusions() override;

    /*! \brief Build the GPU influence function from the PSWF table. */
    void computeInfluenceFunctionGPU();

    /*! \brief Launch the GPU kernel that builds the influence function. */
    void launchInfluenceFunctionKernel();

    /*! \brief Launch the GPU kernel that evaluates the PSWF denominator. */
    void launchGFDenominatorKernel();

    /*! \brief Apply excluded-pair corrections on the GPU. */
    void fixExclusionsGPU();

    void refreshChargeDependentState();

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

    private:
    std::shared_ptr<Autotuner<1>> m_tuner_assign;
    std::shared_ptr<Autotuner<1>> m_tuner_update;
    std::shared_ptr<Autotuner<1>> m_tuner_force;
    std::shared_ptr<Autotuner<1>> m_tuner_influence;

    hipfftHandle m_hipfft_plan;
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

#endif // ENABLE_HIP
