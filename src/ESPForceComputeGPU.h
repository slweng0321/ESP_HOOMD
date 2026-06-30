// Copyright (c) 2025 Shih-Lun Weng.
// Part of RxMC, released under the BSD 3-Clause License.

/*! \file ESPForceComputeGPU.h
    \brief Declares the GPU-accelerated ESPForceCompute class
*/

#pragma once

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "HelpFunctions/ESPForceCompute.h"
#include "hoomd/md/PPPMForceComputeGPU.h"

namespace hoomd
    {
namespace md
    {
/*! \ingroup updaters
    \brief GPU-accelerated custom PPPM force compute
    
    This class provides GPU acceleration for ESPForceCompute,
    leveraging HIP for NVIDIA and AMD GPUs while maintaining correctness
    during reactive Monte Carlo simulations.
*/
class PYBIND11_EXPORT ESPForceComputeGPU : public PPPMForceComputeGPU
    {
    public:
    //! Constructor
    /*! \param sysdef System definition
        \param nlist Neighbor list for PPPM calculations
        \param group Particle group to apply force to
    */
    ESPForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                              std::shared_ptr<NeighborList> nlist,
                              std::shared_ptr<ParticleGroup> group);

    //! GPU-accelerated force computation
    /*! \param timestep Current simulation timestep
    */
    void computeForces(uint64_t timestep) override;

    protected:
    //! GPU implementation of charge-dependent state refresh
    void refreshChargeDependentState();

    //! GPU implementation of ghost cell computation
    uint3 computeGhostCellNumCustom() const;
    };

namespace detail
    {
//! Python export function
void export_ESPForceComputeGPU(pybind11::module& m);
    } // namespace detail

    } // namespace md
    } // namespace hoomd
