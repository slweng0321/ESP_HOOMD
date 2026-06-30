// Copyright (c) 2024-2026 ESP Plugin Contributors.
// Released under the BSD 3-Clause License.
//
// module.cc
//
// pybind11 entry point for the HOOMD-blue ESP (Ewald Summation with Prolates)
// custom plugin.  This file is the ONLY translation unit that includes
// pybind11/pybind11.h via the PYBIND11_MODULE macro; all other plugin files
// include it only for PYBIND11_EXPORT / pybind11::class_ declarations.
//
// Build system context
// --------------------
// CMake registers this file as the shared-library source target.  The
// resulting .so is imported by the Python package __init__.py as:
//
//   from hoomd_esp import _esp
//   _esp.ESPForceCompute(...)
//
// If HOOMD was built with GPU support (ENABLE_HIP=ON), CMakeLists.txt
// also compiles ESPForceComputeGPU.cc / ESPForceComputeGPU.cu and calls
// export_ESPForceComputeGPU() from here through conditional compilation.
//
// Extending this file
// --------------------
// To add the EvaluatorPairPSWF real-space pair potential:
//   1. Include EvaluatorPairPSWF.h and the generated pair-potential binding
//      header (e.g. PairPotentialESP.h produced by HOOMD's pair_potential
//      macro machinery).
//   2. Call export_PairPotentialESP(m) inside PYBIND11_MODULE after the
//      existing export calls.
//
// Reference
// ----------
//   Liang, Lu, Barnett, Greengard, Jiang,
//   "Accelerating molecular dynamics simulations using fast Ewald summation
//    with prolates", Nat. Commun. 2026.
//   https://doi.org/10.1038/s41467-026-73232-8

// ── pybind11 must come before any HOOMD header that pulls in Python.h ────────
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>        // automatic std::vector / std::array conversion
#include <pybind11/numpy.h>      // numpy array support (parameter tables, etc.)

// ── HOOMD-blue core (must appear before md headers) ──────────────────────────
#include "hoomd/ForceCompute.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/SystemDefinition.h"

// ── ESP plugin headers ────────────────────────────────────────────────────────
#include "ESPForceCompute.h"         // CPU long-range mesh compute

// GPU variant is compiled only when HIP/CUDA support is enabled.
// ESPForceComputeGPU overrides the virtual mesh-pipeline methods with
// GPU kernels while reusing all MPI / FFT-plan infrastructure from the
// CPU base class.
#ifdef ENABLE_HIP
#include "ESPForceComputeGPU.h"
#endif

// ── Free-function declarations ────────────────────────────────────────────────
// These are defined in their respective .cc / .cu files and registered below.
namespace hoomd
{
namespace md
{

// Declared in ESPForceCompute.cc — registers hoomd.md.long_range.esp.Coulomb
void export_ESPForceCompute(pybind11::module& m);

#ifdef ENABLE_HIP
// Declared in ESPForceComputeGPU.cc — registers the GPU subclass
void export_ESPForceComputeGPU(pybind11::module& m);
#endif

} // namespace md
} // namespace hoomd

// ============================================================================
// PYBIND11_MODULE  —  module name must match the CMake target name "_esp"
// ============================================================================
//
// After importing, users access objects as:
//
//   import hoomd_esp._esp as _esp
//   lrc = _esp.ESPForceCompute(system, nlist, group)
//   lrc.setParams(nx=64, ny=64, nz=64, order=6, kappa=0.34, rcut=1.2)
//
PYBIND11_MODULE(_esp, m)
    {
    // ── Module metadata ───────────────────────────────────────────────────────
    m.doc() =
        "HOOMD-blue plugin: Ewald Summation with Prolates (ESP).\n\n"
        "Provides the ESPForceCompute class (and its GPU variant when compiled\n"
        "with HIP support) for computing long-range electrostatic interactions\n"
        "using PSWF-based splitting kernels instead of the Gaussian / B-spline\n"
        "approach of standard PPPM.\n\n"
        "Reference: Liang et al., Nat. Commun. 2026, "
        "doi:10.1038/s41467-026-73232-8";

    m.attr("__version__") = pybind11::str(
#ifdef ESP_PLUGIN_VERSION
        ESP_PLUGIN_VERSION
#else
        "0.1.0-dev"
#endif
    );

    // ── Compile-time constants (Python-accessible for introspection) ──────────
    m.attr("MAX_ORDER")       = hoomd::md::ESP_MAX_ORDER;
    m.attr("TABLE_SEGMENTS")  = hoomd::md::ESP_TABLE_SEGMENTS;
    m.attr("TABLE_POLY_DEG")  = hoomd::md::ESP_TABLE_POLY_DEGREE;

    // ── ESPTableEntry (exposed as a named-tuple-like read-only capsule) ───────
    // Full struct bindings are provided so Python test utilities can inspect
    // individual lookup-table entries returned by getTableEntry(seg).
    pybind11::class_<hoomd::md::ESPTableEntry>(m, "ESPTableEntry",
        "One segment of the piecewise-polynomial L(r) / (-dL/dr) lookup table.\n\n"
        "All fields are read-only from Python; the table is populated during\n"
        "ESPForceCompute initialisation and should not be mutated at runtime.")
        .def_readonly("r_lo",
            &hoomd::md::ESPTableEntry::r_lo,
            "Left edge of this segment [simulation length units].")
        .def_readonly("dr_inv",
            &hoomd::md::ESPTableEntry::dr_inv,
            "Reciprocal segment width 1/(r_hi - r_lo) for fast normalisation.")
        .def_property_readonly("potential_coeffs",
            [](const hoomd::md::ESPTableEntry& e) -> pybind11::tuple
                {
                return pybind11::make_tuple(e.potential_coeffs.x,
                                            e.potential_coeffs.y,
                                            e.potential_coeffs.z,
                                            e.potential_coeffs.w,
                                            e.potential_coeff4);
                },
            "Degree-4 Horner coefficients (c0..c4) for L(r) in this segment.")
        .def_property_readonly("force_coeffs",
            [](const hoomd::md::ESPTableEntry& e) -> pybind11::tuple
                {
                return pybind11::make_tuple(e.force_coeffs.x,
                                            e.force_coeffs.y,
                                            e.force_coeffs.z,
                                            e.force_coeffs.w,
                                            e.force_coeff4);
                },
            "Degree-4 Horner coefficients (c0..c4) for -dL/dr in this segment.")
        .def("__repr__",
            [](const hoomd::md::ESPTableEntry& e) -> std::string
                {
                std::ostringstream oss;
                oss << "<ESPTableEntry r_lo=" << e.r_lo
                    << " dr_inv=" << e.dr_inv << ">";
                return oss.str();
                });

    // ── CPU ESPForceCompute ───────────────────────────────────────────────────
    // The heavy lifting is done inside export_ESPForceCompute() which is
    // defined in ESPForceCompute.cc.  Calling it here keeps module.cc lean
    // while still allowing the .cc file to be compiled independently (e.g.
    // for unit-testing without the Python layer).
    hoomd::md::export_ESPForceCompute(m);

    // ── GPU ESPForceComputeGPU (only when compiled with HIP support) ──────────
#ifdef ENABLE_HIP
    hoomd::md::export_ESPForceComputeGPU(m);

    // Expose a boolean so Python code can branch on GPU availability without
    // importing pycuda or hip-python.
    m.attr("gpu_enabled") = pybind11::bool_(true);
#else
    m.attr("gpu_enabled") = pybind11::bool_(false);
#endif

    // ── Runtime helper: query active HOOMD exec_conf from Python ─────────────
    // Returns a dict with basic device information useful for diagnostics.
    m.def("get_device_info",
        []() -> pybind11::dict
            {
            pybind11::dict d;
#ifdef ENABLE_HIP
            d["backend"] = "hip";
#else
            d["backend"] = "cpu";
#endif
#ifdef ENABLE_MPI
            d["mpi"] = true;
#else
            d["mpi"] = false;
#endif
            return d;
            },
        "Return a dict with compile-time device / MPI information.\n\n"
        "Keys: 'backend' ('cpu' or 'hip'), 'mpi' (bool).");

    // ── Convenience alias ─────────────────────────────────────────────────────
    // Allow   from hoomd_esp._esp import Coulomb
    // to mirror the HOOMD-blue md.long_range.pppm.Coulomb naming convention.
#ifdef ENABLE_HIP
    m.attr("Coulomb") = m.attr("ESPForceComputeGPU");
#else
    m.attr("Coulomb") = m.attr("ESPForceCompute");
#endif

    } // PYBIND11_MODULE
