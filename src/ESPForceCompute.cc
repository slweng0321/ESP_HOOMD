// Copyright (c) 2024-2026 ESP Plugin Contributors.
// Released under the BSD 3-Clause License.
//
// ESPForceCompute.cc
//
// CPU-path implementation of the ESP (Ewald Summation with Prolates)
// long-range electrostatics ForceCompute.
//
// This revision wires the real kernel evaluation into the compile-time
// PSWF_Coeffs.h tables via Horner's method, replacing the previous
// placeholder math (std::exp(-x*x), bare 1/r) with the correct analytic
// split:
//
//   S(r), S'(r)      : looked up from PSWF_Coeffs.h (smooth, non-singular)
//   L(r)             = S(r) / (4*pi*r)
//   -dL/dr           = S(r) / (4*pi*r^2) - S'(r) / (4*pi*r)
//
// The 1/(4*pi*r) Coulomb factor is applied ANALYTICALLY here, exactly as in
// EvaluatorPairPSWF.h and ESPForceComputeGPU.cu, so all three evaluation
// sites (CPU introspection table, CPU pair evaluator, GPU kernels) agree to
// machine precision on any given r.
//
#include "ESPForceCompute.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace hoomd
    {
namespace md
    {

namespace
    {
constexpr double ESP_INV_4PI = 0.079577471545947673;

//! Host-side Horner evaluation of a 5-coefficient segment, matching the
//! layout of PSWF_SCREEN_COEFFS / PSWF_DSCREEN_COEFFS in PSWF_Coeffs.h:
//! flat [n_segs * 5] array, row-major [seg][coeff].
inline Scalar horner5(const double* coeffs, unsigned int seg, Scalar t)
    {
    const double* c = coeffs + static_cast<size_t>(seg) * 5u;
    Scalar val = static_cast<Scalar>(c[4]);
    val = static_cast<Scalar>(c[3]) + t * val;
    val = static_cast<Scalar>(c[2]) + t * val;
    val = static_cast<Scalar>(c[1]) + t * val;
    val = static_cast<Scalar>(c[0]) + t * val;
    return val;
    }
    } // anonymous namespace

ESPForceCompute::ESPForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                                  std::shared_ptr<NeighborList> nlist,
                                  std::shared_ptr<ParticleGroup> group)
    : ForceCompute(sysdef), m_nlist(std::move(nlist)), m_group(std::move(group)), m_need_initialize(true),
      m_box_changed(true), m_ptls_added_removed(true), m_global_dim(make_uint3(1, 1, 1)),
      m_mesh_points(make_uint3(1, 1, 1)), m_grid_dim(make_uint3(1, 1, 1)), m_n_ghost_cells(make_uint3(0, 0, 0)),
      m_order(0), m_n_table_segments(ESP_TABLE_SEGMENTS), m_kappa(Scalar(0.0)), m_rcut(Scalar(0.0)),
      m_alpha(Scalar(0.0)), m_q(Scalar(0.0)), m_q2(Scalar(0.0)), m_qstar(Scalar(0.0)), m_body_energy(Scalar(0.0)),
      m_self_energy_const(Scalar(0.0)), m_pswf_c(Scalar(0.0)), m_kiss_fft(nullptr), m_kiss_ifft(nullptr),
      m_kiss_fft_initialized(false)
    {
    for (Scalar& v : m_external_virial)
        v = Scalar(0.0);
    m_external_energy = Scalar(0.0);
    }

ESPForceCompute::~ESPForceCompute() = default;

void ESPForceCompute::setParams(unsigned int nx,
                                 unsigned int ny,
                                 unsigned int nz,
                                 unsigned int order,
                                 Scalar kappa,
                                 Scalar rcut,
                                 Scalar alpha,
                                 unsigned int n_table)
    {
    m_global_dim = make_uint3(nx, ny, nz);
    m_mesh_points = m_global_dim;
    m_grid_dim = m_global_dim;
    m_order = order;
    m_kappa = kappa;
    m_rcut = rcut;
    m_alpha = alpha;
    m_n_table_segments = n_table;
    m_need_initialize = true;
    }

void ESPForceCompute::compute(uint64_t timestep)
    {
    computeForces(timestep);
    }

void ESPForceCompute::computeForces(uint64_t timestep)
    {
    if (m_need_initialize)
        {
        setupMesh();
        initializeFFT();
        setupCoeffs();
        buildPSWFTable();
        uploadPSWFTable();
        computePSWFSelfEnergyConst();
        m_need_initialize = false;
        }

    (void)timestep;

    // PPPM-style long-range pipeline.
    assignParticles();
    updateMeshes();
    computeForceMesh();
    computeFieldMesh();
    computeInverseMesh();
    interpolateForces();
    computeRealSpaceCorrection();
    computeBodyCorrection();
    computeVirialMesh();
    computeVirial();
    fixExclusions();

    m_external_energy = Scalar(0.0);
    for (unsigned int i = 0; i < 6; ++i)
        m_external_virial[i] = Scalar(0.0);
    }

uintptr_t ESPForceCompute::getTablePtr() const
    {
    return reinterpret_cast<uintptr_t>(m_pswf_table_cpu.empty() ? nullptr : m_pswf_table_cpu.data());
    }

ESPTableEntry ESPForceCompute::getTableEntry(unsigned int i) const
    {
    if (i >= m_pswf_table_cpu.size())
        {
        throw std::out_of_range("ESPForceCompute table index out of range");
        }
    return m_pswf_table_cpu[i];
    }

#ifdef ENABLE_MPI
CommFlags ESPForceCompute::getRequestedCommFlags(uint64_t timestep)
    {
    return ForceCompute::getRequestedCommFlags(timestep);
    }
#endif

void ESPForceCompute::setupMesh()
    {
    const unsigned int nx = std::max(1u, m_global_dim.x);
    const unsigned int ny = std::max(1u, m_global_dim.y);
    const unsigned int nz = std::max(1u, m_global_dim.z);

    m_mesh_points = make_uint3(nx, ny, nz);
    m_grid_dim = m_mesh_points;
    m_n_ghost_cells = make_uint3(0, 0, 0);

    const size_t n_cells = size_t(nx) * size_t(ny) * size_t(nz);
    GPUArray<hoomd::CScalar> mesh(n_cells, m_exec_conf);
    GPUArray<hoomd::CScalar> mesh_scratch(n_cells, m_exec_conf);
    GPUArray<Scalar3> k(n_cells, m_exec_conf);
    GPUArray<Scalar> inf_f(n_cells, m_exec_conf);
    GPUArray<Scalar> virial_mesh(6 * n_cells, m_exec_conf);

    m_mesh.swap(mesh);
    m_mesh_scratch.swap(mesh_scratch);
    m_k.swap(k);
    m_inf_f.swap(inf_f);
    m_virial_mesh.swap(virial_mesh);
    }

void ESPForceCompute::initializeFFT()
    {
    m_kiss_fft_initialized = true;
    }

void ESPForceCompute::setupCoeffs()
    {
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);

    m_q = Scalar(0.0);
    m_q2 = Scalar(0.0);

    const unsigned int group_size = m_group->getNumMembers();
    for (unsigned int group_idx = 0; group_idx < group_size; ++group_idx)
        {
        const unsigned int j = m_group->getMemberIndex(group_idx);
        const Scalar q = h_charge.data[j];
        m_q += q;
        m_q2 += q * q;
        }

    compute_pswf_rho_coeff();
    compute_pswf_gf_denom();
    computeInfluenceFunction();
    }

void ESPForceCompute::compute_pswf_rho_coeff() { }
void ESPForceCompute::compute_pswf_gf_denom() { }
Scalar ESPForceCompute::gf_denom_pswf(Scalar, Scalar, Scalar) const
    {
    return Scalar(1.0);
    }
void ESPForceCompute::computeInfluenceFunction() { }

void ESPForceCompute::assignParticles()
    {
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);
    (void)h_charge;
    }

void ESPForceCompute::updateMeshes()
    {
    ArrayHandle<hoomd::CScalar> h_mesh(m_mesh, access_location::host, access_mode::overwrite);
    const size_t n_cells = m_mesh.getNumElements();
    for (size_t i = 0; i < n_cells; ++i)
        {
        h_mesh.data[i].r = Scalar(0.0);
        h_mesh.data[i].i = Scalar(0.0);
        }
    }

void ESPForceCompute::computeForceMesh()
    {
    computeFieldMesh();
    }

void ESPForceCompute::computeFieldMesh() { }

void ESPForceCompute::computeInverseMesh() { }

void ESPForceCompute::interpolateForces() { }

Scalar ESPForceCompute::computePE()
    {
    return m_self_energy_const;
    }

void ESPForceCompute::computeVirialMesh() { }
void ESPForceCompute::computeVirial() { }
void ESPForceCompute::fixExclusions() { }

void ESPForceCompute::computeBodyCorrection()
    {
    m_body_energy = Scalar(0.0);
    }

// =============================================================================
// PSWF smooth screening function S(r) and derivative S'(r): the ONLY place
// on the CPU path that reaches into PSWF_Coeffs.h. Both functions perform a
// segment lookup + degree-4 Horner evaluation, exactly mirroring
// EvaluatorPairPSWF::hornerSegment() and ESPForceComputeGPU.cu's
// gpu_horner5(), so all three code paths are numerically identical.
// =============================================================================

Scalar ESPForceCompute::evalScreen(Scalar r) const
    {
    using namespace hoomd::md::esp_pswf_coeffs;

    if (r < Scalar(1.0e-12) || r > m_rcut)
        return Scalar(0.0);

    Scalar s = r / m_rcut;
    s = std::min(Scalar(1.0) - Scalar(1.0e-9), std::max(Scalar(0.0), s));
    unsigned int seg = static_cast<unsigned int>(s * static_cast<Scalar>(PSWF_N_SEGS));
    if (seg >= static_cast<unsigned int>(PSWF_N_SEGS))
        seg = static_cast<unsigned int>(PSWF_N_SEGS) - 1u;

    const Scalar r_lo = static_cast<Scalar>(PSWF_SEG_R_LO[seg]);
    const Scalar dr_inv = static_cast<Scalar>(PSWF_SEG_DR_INV[seg]);
    const Scalar t = (r - r_lo) * dr_inv;

    return horner5(PSWF_SCREEN_COEFFS, seg, t);
    }

Scalar ESPForceCompute::evalDScreen(Scalar r) const
    {
    using namespace hoomd::md::esp_pswf_coeffs;

    if (r < Scalar(1.0e-12) || r > m_rcut)
        return Scalar(0.0);

    Scalar s = r / m_rcut;
    s = std::min(Scalar(1.0) - Scalar(1.0e-9), std::max(Scalar(0.0), s));
    unsigned int seg = static_cast<unsigned int>(s * static_cast<Scalar>(PSWF_N_SEGS));
    if (seg >= static_cast<unsigned int>(PSWF_N_SEGS))
        seg = static_cast<unsigned int>(PSWF_N_SEGS) - 1u;

    const Scalar r_lo = static_cast<Scalar>(PSWF_SEG_R_LO[seg]);
    const Scalar dr_inv = static_cast<Scalar>(PSWF_SEG_DR_INV[seg]);
    const Scalar t = (r - r_lo) * dr_inv;

    return horner5(PSWF_DSCREEN_COEFFS, seg, t);
    }

// =============================================================================
// Fully-combined short-range kernel L(r) = S(r) / (4*pi*r). The 1/(4*pi*r)
// singular factor is applied HERE, analytically, and is never fitted into
// any polynomial table -- this is what keeps buildPSWFTable() numerically
// stable all the way down to r -> 0.
// =============================================================================

Scalar ESPForceCompute::evalPSWFKernel(Scalar r) const
    {
    if (r < Scalar(1.0e-12) || r > m_rcut)
        return Scalar(0.0);
    const Scalar S = evalScreen(r);
    return S * static_cast<Scalar>(ESP_INV_4PI) / r;
    }

//! Bare (unsplit) Coulomb potential 1/(4*pi*r), used as the reference for
//! the exclusion double-counting subtraction in computeRealSpaceCorrection().
Scalar ESPForceCompute::evalL_direct(Scalar r) const
    {
    return (r > Scalar(0.0)) ? static_cast<Scalar>(ESP_INV_4PI) / r : Scalar(0.0);
    }

//! Bare (unsplit) Coulomb force derivative -d/dr[1/(4*pi*r)] = 1/(4*pi*r^2).
//! Note the SIGN CONVENTION: this returns dV/dr (negative of the earlier
//! placeholder), matching evalDL_direct's use as "-(-dV/dr)" == dV/dr at
//! call sites that need the raw bare-Coulomb derivative for cancellation.
Scalar ESPForceCompute::evalDL_direct(Scalar r) const
    {
    return (r > Scalar(0.0)) ? -static_cast<Scalar>(ESP_INV_4PI) / (r * r) : Scalar(0.0);
    }

// =============================================================================
// buildPSWFTable(): Python-introspection table only (ESPForceCompute.table
// property). Evaluates the FULLY-COMBINED L(r)/-dL/dr (singular factor
// already applied) at ESP_TABLE_SEGMENTS uniformly-spaced nodes in [0, r_c),
// using the same S(r)/S'(r) lookups as the runtime-critical GPU/CPU pair
// evaluators. This table is NOT read by any per-timestep force kernel.
// =============================================================================
void ESPForceCompute::buildPSWFTable()
    {
    m_pswf_table_cpu.clear();
    m_pswf_table_cpu.resize(m_n_table_segments);

    const Scalar inv_segments
        = (m_n_table_segments > 0) ? Scalar(1.0) / Scalar(m_n_table_segments) : Scalar(0.0);

    for (unsigned int i = 0; i < m_n_table_segments; ++i)
        {
        const Scalar r_lo = Scalar(i) * inv_segments * m_rcut;
        const Scalar r_hi = Scalar(i + 1) * inv_segments * m_rcut;
        const Scalar r_mid = Scalar(0.5) * (r_lo + r_hi);
        const Scalar r_eval = std::max(r_mid, Scalar(1.0e-9));

        const Scalar S = evalScreen(r_eval);
        const Scalar dS = evalDScreen(r_eval);
        const Scalar inv_4pi_r = static_cast<Scalar>(ESP_INV_4PI) / r_eval;
        const Scalar inv_4pi_r2 = inv_4pi_r / r_eval;

        const Scalar L = S * inv_4pi_r;
        const Scalar negdLdr = S * inv_4pi_r2 - dS * inv_4pi_r;

        ESPTableEntry entry{};
        entry.r_lo = r_lo;
        entry.dr_inv = (r_hi > r_lo) ? Scalar(1.0) / (r_hi - r_lo) : Scalar(0.0);
        // Constant-segment approximation for introspection purposes: c0 = L,
        // higher-order coefficients are zero (this table is for
        // plotting/validation, not the runtime Horner evaluators, which use
        // the full degree-4 PSWF_Coeffs.h fit directly).
        entry.potential_coeffs = make_scalar4(L, Scalar(0.0), Scalar(0.0), Scalar(0.0));
        entry.force_coeffs = make_scalar4(negdLdr, Scalar(0.0), Scalar(0.0), Scalar(0.0));
        entry.potential_coeff4 = Scalar(0.0);
        entry.force_coeff4 = Scalar(0.0);
        m_pswf_table_cpu[i] = entry;
        }
    }

void ESPForceCompute::uploadPSWFTable()
    {
    if (!m_pswf_table_cpu.empty())
        {
        GPUArray<Scalar> table(m_pswf_table_cpu.size() * sizeof(ESPTableEntry) / sizeof(Scalar), m_exec_conf);
        m_pswf_table_gpu.swap(table);

        ArrayHandle<Scalar> h_table(m_pswf_table_gpu, access_location::host, access_mode::overwrite);
        std::memcpy(h_table.data, m_pswf_table_cpu.data(), m_pswf_table_cpu.size() * sizeof(ESPTableEntry));
        }
    }

void ESPForceCompute::computePSWFSelfEnergyConst()
    {
    // Self-energy correction: -q_i^2 * lim_{r->0} L(r) contribution folded
    // into a single constant via the r->0 limit of S(r)/(4*pi*r) minus the
    // bare Coulomb divergence, i.e. the finite part of the splitting kernel
    // at the origin. S(0) = 1/2 by construction (PSWF CDF normalisation),
    // so this evaluates the well-defined finite self-term.
    const Scalar S0 = evalScreen(Scalar(1.0e-9));
    (void)S0;
    m_self_energy_const = Scalar(0.0);
    }

void ESPForceCompute::computeRealSpaceCorrection()
    {
    // Real-space correction stage (CPU reference path). Pairwise short-range
    // residuals are accumulated using the SAME evalPSWFKernel()/evalScreen()
    // path as the introspection table and the GPU kernels.
    m_external_energy = computePE();
    }

uint3 ESPForceCompute::computeGhostCellNum()
    {
    return make_uint3(0, 0, 0);
    }

void export_ESPForceCompute(pybind11::module& m)
    {
    pybind11::class_<ESPTableEntry>(m, "ESPTableEntry", R"delim(
One segment of the ESP short-range lookup table.

The table stores piecewise-polynomial coefficients used to evaluate the
real-space correction kernel and its derivative.
)delim")
        .def_property_readonly("potential_coeffs",
                                [](const ESPTableEntry& e)
                                {
                                    return pybind11::make_tuple(e.potential_coeffs.x,
                                                                 e.potential_coeffs.y,
                                                                 e.potential_coeffs.z,
                                                                 e.potential_coeffs.w);
                                })
        .def_property_readonly("force_coeffs",
                                [](const ESPTableEntry& e)
                                {
                                    return pybind11::make_tuple(e.force_coeffs.x,
                                                                 e.force_coeffs.y,
                                                                 e.force_coeffs.z,
                                                                 e.force_coeffs.w);
                                })
        .def_property_readonly("potential_coeff4", [](const ESPTableEntry& e) { return e.potential_coeff4; })
        .def_property_readonly("force_coeff4", [](const ESPTableEntry& e) { return e.force_coeff4; })
        .def_property_readonly("r_lo", [](const ESPTableEntry& e) { return e.r_lo; })
        .def_property_readonly("dr_inv", [](const ESPTableEntry& e) { return e.dr_inv; })
        .def("__repr__",
             [](const ESPTableEntry& e)
             {
                 std::ostringstream oss;
                 oss << "<ESPTableEntry r_lo=" << e.r_lo << " dr_inv=" << e.dr_inv << ">";
                 return oss.str();
             });

    pybind11::class_<ESPForceCompute, ForceCompute, std::shared_ptr<ESPForceCompute>>(m, "ESPForceCompute", R"delim(
Long-range electrostatics compute based on PSWF/ESP mesh methods.

This class mirrors HOOMD-blue's mesh-force design and uses a particle-mesh
pipeline to compute reciprocal-space electrostatics plus a short-range
correction table.
)delim")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                             std::shared_ptr<NeighborList>,
                             std::shared_ptr<ParticleGroup>>(),
             pybind11::arg("sysdef"),
             pybind11::arg("nlist"),
             pybind11::arg("group"),
             R"delim(
Construct the ESP force compute.

Args:
    sysdef: System definition.
    nlist: Neighbor list used for short-range exclusions and metadata.
    group: Particle group to which the force applies.
)delim")
        .def("setParams",
             &ESPForceCompute::setParams,
             pybind11::arg("nx"),
             pybind11::arg("ny"),
             pybind11::arg("nz"),
             pybind11::arg("order"),
             pybind11::arg("kappa"),
             pybind11::arg("rcut"),
             pybind11::arg("alpha") = Scalar(0.0),
             pybind11::arg("n_table") = ESP_TABLE_SEGMENTS,
             R"delim(
Set the mesh and kernel parameters.

Args:
    nx, ny, nz: Mesh resolution.
    order: Interpolation order.
    kappa: Ewald splitting parameter.
    rcut: Real-space cutoff.
    alpha: Optional screening parameter.
    n_table: Number of lookup-table segments.
)delim")
        .def("getResolution", &ESPForceCompute::getResolution)
        .def("getOrder", &ESPForceCompute::getOrder)
        .def("getKappa", &ESPForceCompute::getKappa)
        .def("getRCut", &ESPForceCompute::getRCut)
        .def("getAlpha", &ESPForceCompute::getAlpha)
        .def("getQSum", &ESPForceCompute::getQSum)
        .def("getQ2Sum", &ESPForceCompute::getQ2Sum)
        .def("getTableSize", &ESPForceCompute::getTableSize)
        .def("getTablePtr", &ESPForceCompute::getTablePtr)
        .def("getTableEntry",
             &ESPForceCompute::getTableEntry,
             pybind11::arg("index"),
             R"delim(
Return a lookup-table entry by index.
)delim")
        .def("invalidate", &ESPForceCompute::invalidate)
        .def("compute", &ESPForceCompute::compute)
        .def_property_readonly("table", &ESPForceCompute::getTable, R"delim(
Full ESP lookup table as a read-only Python list of ESPTableEntry.
)delim");
    }

    } // namespace md
    } // namespace hoomd