// Copyright (c) 2024-2026 ESP Plugin Contributors.
// Released under the BSD 3-Clause License.
//
// module.cc
//
// pybind11 module entry point for the compiled _esp extension.
//
// Bug fix in this revision
// ---------------------------
// The previous revision of this file DECLARED export_PotentialPairPSWF()
// and export_PotentialPairPSWFGPU() as extern prototypes, matching the
// naming convention used for export_ESPForceCompute()/
// export_ESPForceComputeGPU() (which ARE defined in their own .cc files).
// However, CMakeLists.txt's source list confirms EvaluatorPairPSWF.h is
// ONLY ever included as a header -- there is no ESPPairPSWF.cc/.cu
// translation unit anywhere in the plugin that instantiates
// PotentialPair<EvaluatorPairPSWF> (or PotentialPairGPU<EvaluatorPairPSWF>)
// and defines those export functions. Declaring-but-never-defining them
// would produce an undefined-reference linker error the moment CMake
// tried to build _esp.
//
// FIX: PotentialPair<EvaluatorPairPSWF> is instantiated and exported
// DIRECTLY in this file (module.cc), which is the standard HOOMD-blue
// plugin pattern for evaluators that don't need their own translation
// unit (see e.g. how simple pair potentials are wired up in HOOMD's own
// md/module-md.cc). No new source file or CMakeLists.txt change is
// required.
//
#include <pybind11/pybind11.h>

#include "hoomd/md/PotentialPair.h"

#include "ESPForceCompute.h"
#include "EvaluatorPairPSWF.h"

#ifdef ENABLE_HIP
#include "ESPForceComputeGPU.h"
#include "hoomd/md/PotentialPairGPU.h"
#endif

namespace hoomd
    {
namespace md
    {

// ── CPU exports (always compiled) ───────────────────────────────────────────
void export_ESPForceCompute(pybind11::module& m);

// ── GPU exports (HIP/CUDA builds only) ──────────────────────────────────────
#ifdef ENABLE_HIP
void export_ESPForceComputeGPU(pybind11::module& m);
#endif

    } // namespace md
    } // namespace hoomd

PYBIND11_MODULE(_esp, m)
    {
    m.doc() = "HOOMD-blue ESP plugin: PSWF-based Ewald Summation with Prolates "
              "long-range mesh force and matching short-range pair correction.";

    using hoomd::md::EvaluatorPairPSWF;
    using hoomd::md::PotentialPair;

    // Long-range mesh contribution (always available).
    hoomd::md::export_ESPForceCompute(m);

    // Short-range real-space complement: PotentialPair<EvaluatorPairPSWF>
    // instantiated inline here (no separate .cc needed -- EvaluatorPairPSWF
    // is header-only and has no CUDA-specific code path of its own; the
    // GPU dispatch is handled entirely by HOOMD's generic PotentialPairGPU
    // template below).
    hoomd::md::detail::export_PotentialPair<EvaluatorPairPSWF>(m, "PotentialPairPSWF");

#ifdef ENABLE_HIP
    hoomd::md::export_ESPForceComputeGPU(m);

    using hoomd::md::PotentialPairGPU;
    hoomd::md::detail::export_PotentialPairGPU<EvaluatorPairPSWF>(m, "PotentialPairPSWFGPU");

    m.attr("gpu_enabled") = pybind11::bool_(true);
#else
    m.attr("gpu_enabled") = pybind11::bool_(false);
#endif
    }