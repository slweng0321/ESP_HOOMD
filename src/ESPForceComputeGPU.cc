// Copyright (c) 2025 Shih-Lun Weng.
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
