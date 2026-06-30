# ===========================================================================
# CMakeLists.txt  —  ESP (Ewald Summation with Prolates) HOOMD-blue Plugin
# ===========================================================================
#
# Directory layout expected:
#
#   esp_plugin/
#   ├── CMakeLists.txt          ← this file
#   ├── hoomd_esp/
#   │   ├── __init__.py
#   │   └── esp.py
#   └── src/
#       ├── module.cc           ← pybind11 module entry point
#       ├── ESPForceCompute.h
#       ├── ESPForceCompute.cc  ← CPU implementation
#       ├── EvaluatorPairPSWF.h ← header-only GPU/CPU pair evaluator
#       ├── ESPForceComputeGPU.h       (optional GPU override)
#       ├── ESPForceComputeGPU.cc      (optional GPU override)
#       └── ESPForceComputeGPU.cu      (optional GPU CUDA kernels)
#
# Build (out-of-source):
#   cmake -B build -S . \
#         -DHOOMD_ROOT=/path/to/hoomd/install \
#         -DCMAKE_INSTALL_PREFIX=$(python3 -c "import site; print(site.getsitepackages()[0])")
#   cmake --build build -j$(nproc)
#   cmake --install build
#
# ===========================================================================

cmake_minimum_required(VERSION 3.18)

project(
    hoomd_esp
    VERSION 0.1.0
    DESCRIPTION "ESP (Ewald Summation with Prolates) plugin for HOOMD-blue"
    LANGUAGES CXX
)

# ---------------------------------------------------------------------------
# C++ standard — must match HOOMD-blue (C++17 since HOOMD 3.x)
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Build type default
# ---------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Build type (Release | RelWithDebInfo | Debug)" FORCE)
    message(STATUS "ESP: No build type selected, defaulting to Release")
endif()

# ---------------------------------------------------------------------------
# Option: HOOMD_ROOT — path to a HOOMD-blue *install* tree
#   (the directory that contains lib/python3.x/site-packages/hoomd/)
# If not set, we try to locate hoomd via Python's import system.
# ---------------------------------------------------------------------------
option(HOOMD_ROOT "" "")  # empty default → auto-detect

if(HOOMD_ROOT)
    # Explicit path: find hoomd CMake config under the install prefix
    list(PREPEND CMAKE_PREFIX_PATH "${HOOMD_ROOT}")
    message(STATUS "ESP: Searching for HOOMD under HOOMD_ROOT=${HOOMD_ROOT}")
endif()

# ---------------------------------------------------------------------------
# Locate HOOMD-blue via its exported CMake config
# (installed as <prefix>/lib/cmake/hoomd/hoomd-config.cmake)
# ---------------------------------------------------------------------------
find_package(HOOMD 4.0 REQUIRED CONFIG
    HINTS
        "${HOOMD_ROOT}/lib/cmake/hoomd"
        "${HOOMD_ROOT}/lib64/cmake/hoomd"
)
message(STATUS "ESP: Found HOOMD ${HOOMD_VERSION} at ${HOOMD_DIR}")

# ---------------------------------------------------------------------------
# Python + pybind11
# HOOMD's own CMake config already finds Python; we honour that choice
# so we always compile against the same interpreter.
# ---------------------------------------------------------------------------
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED CONFIG
    HINTS
        "${HOOMD_ROOT}/lib/cmake/pybind11"
        "${HOOMD_ROOT}/lib64/cmake/pybind11"
)

# ---------------------------------------------------------------------------
# MPI (optional — mirrors HOOMD's own MPI handling)
# ---------------------------------------------------------------------------
if(HOOMD_MPI)
    find_package(MPI REQUIRED)
    message(STATUS "ESP: MPI support enabled (${MPI_CXX_COMPILER})")
else()
    message(STATUS "ESP: Building without MPI support")
endif()

# ---------------------------------------------------------------------------
# CUDA / HIP (optional GPU support)
# ---------------------------------------------------------------------------
# We follow HOOMD's convention: if HOOMD was built with HIP support the
# ENABLE_HIP CMake variable is exported via hoomd-config.cmake.
# ---------------------------------------------------------------------------
set(ESP_ENABLE_GPU OFF)
set(ESP_GPU_SOURCES "")

if(HOOMD_GPU_PLATFORM STREQUAL "CUDA" OR ENABLE_HIP)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(ESP_ENABLE_GPU ON)
    message(STATUS "ESP: GPU support enabled (platform: ${HOOMD_GPU_PLATFORM})")

    # Collect CUDA source files
    file(GLOB ESP_GPU_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/ESPForceComputeGPU.cu"
    )
    if(NOT ESP_GPU_SOURCES)
        message(STATUS "ESP: No .cu files found — GPU kernels will not be compiled")
        set(ESP_ENABLE_GPU OFF)
    endif()
else()
    message(STATUS "ESP: Building CPU-only (no CUDA/HIP)")
endif()

# ---------------------------------------------------------------------------
# Collect source files
# ---------------------------------------------------------------------------
set(ESP_CPU_SOURCES
    src/ESPForceCompute.cc
    src/module.cc
)

# Optional GPU C++ wrapper (host-side glue for GPU kernels)
if(ESP_ENABLE_GPU AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/ESPForceComputeGPU.cc")
    list(APPEND ESP_CPU_SOURCES src/ESPForceComputeGPU.cc)
endif()

# ---------------------------------------------------------------------------
# Build the pybind11 extension module: _esp
# ---------------------------------------------------------------------------
if(ESP_ENABLE_GPU)
    # GPU build: combine .cc and .cu into a single shared library
    pybind11_add_module(_esp MODULE
        ${ESP_CPU_SOURCES}
        ${ESP_GPU_SOURCES}
    )

    # CUDA compile flags
    set_target_properties(_esp PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    # Mirror HOOMD's GPU arch targets (passed via HOOMD CMake config)
    if(DEFINED HOOMD_GPU_ARCHITECTURES)
        set_target_properties(_esp PROPERTIES
            CUDA_ARCHITECTURES "${HOOMD_GPU_ARCHITECTURES}"
        )
        message(STATUS "ESP: CUDA architectures: ${HOOMD_GPU_ARCHITECTURES}")
    else()
        # Safe default covering Turing (75), Ampere (80/86), Ada (89), Hopper (90)
        set_target_properties(_esp PROPERTIES
            CUDA_ARCHITECTURES "75;80;86;89;90"
        )
        message(WARNING
            "ESP: HOOMD_GPU_ARCHITECTURES not set; "
            "defaulting to 75;80;86;89;90.  Pass -DCMAKE_CUDA_ARCHITECTURES "
            "to override.")
    endif()

    target_compile_definitions(_esp PRIVATE ENABLE_HIP)

else()
    # CPU-only build
    pybind11_add_module(_esp MODULE ${ESP_CPU_SOURCES})
endif()

# ---------------------------------------------------------------------------
# Include directories
# ---------------------------------------------------------------------------
target_include_directories(_esp
    PRIVATE
        # Plugin headers live in src/
        "${CMAKE_CURRENT_SOURCE_DIR}/src"

        # HOOMD public headers (exported by hoomd CMake targets)
        # hoomd::hoomd exports HOOMD_INCLUDE_DIR via target properties;
        # we pull them through the target link below, but also add explicitly
        # for IDE navigation.
        $<TARGET_PROPERTY:hoomd::hoomd,INTERFACE_INCLUDE_DIRECTORIES>

        # HOOMD MD component headers (NeighborList.h, CommunicatorGrid.h …)
        $<TARGET_PROPERTY:hoomd::md,INTERFACE_INCLUDE_DIRECTORIES>

        # MPI headers (if enabled)
        $<$<BOOL:${HOOMD_MPI}>:${MPI_CXX_INCLUDE_PATH}>
)

# ---------------------------------------------------------------------------
# Link libraries
# ---------------------------------------------------------------------------
target_link_libraries(_esp
    PRIVATE
        hoomd::hoomd          # core: ForceCompute, GPUArray, BoxDim …
        hoomd::md             # md: NeighborList, CommunicatorGrid, PPPM …
        pybind11::module      # pybind11 module support
        $<$<BOOL:${HOOMD_MPI}>:MPI::MPI_CXX>
        $<$<BOOL:${ESP_ENABLE_GPU}>:CUDA::cufft>
        $<$<BOOL:${ESP_ENABLE_GPU}>:CUDA::cudart>
)

# ---------------------------------------------------------------------------
# Compile definitions
# ---------------------------------------------------------------------------
target_compile_definitions(_esp
    PRIVATE
        # Pass through MPI enable flag so our headers can use #ifdef ENABLE_MPI
        $<$<BOOL:${HOOMD_MPI}>:ENABLE_MPI>
        # GPU enable flag for EvaluatorPairPSWF.h DEVICE/HOSTDEVICE macros
        $<$<BOOL:${ESP_ENABLE_GPU}>:ENABLE_HIP>
        # Plugin version string (accessible via _esp.__version__ if exposed)
        ESP_PLUGIN_VERSION="${PROJECT_VERSION}"
)

# ---------------------------------------------------------------------------
# Compiler warning flags (C++ host code)
# ---------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(_esp PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:
            -Wall
            -Wextra
            -Wno-unused-parameter   # HOOMD headers have some unused params
            -Wno-sign-compare
        >
    )
endif()

# CUDA-specific flags: suppress verbose device function warnings from HOOMD headers
if(ESP_ENABLE_GPU)
    target_compile_options(_esp PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-relaxed-constexpr
            --expt-extended-lambda
            -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored
            # Match HOOMD's default: fast-math in Release
            $<$<CONFIG:Release>:-use_fast_math>
            # Relocatable device code needed for separate compilation
            --device-c
        >
    )
endif()

# ---------------------------------------------------------------------------
# Output placement — put _esp.so next to the Python package files
# so `import hoomd_esp.esp` finds it without extra sys.path manipulation.
# ---------------------------------------------------------------------------
set_target_properties(_esp PROPERTIES
    # Strip the "lib" prefix that CMake adds on Linux
    PREFIX ""
    # Place alongside the Python sources during the build
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/hoomd_esp"
    # Embed RPATH to HOOMD's lib directory so dlopen finds libhoomd.so
    BUILD_RPATH
        "$<TARGET_FILE_DIR:hoomd::hoomd>"
    INSTALL_RPATH
        "$ORIGIN/../hoomd;$ORIGIN"
    BUILD_WITH_INSTALL_RPATH OFF
)

# ---------------------------------------------------------------------------
# Install rules
# ---------------------------------------------------------------------------

# Determine the Python site-packages directory if not provided
if(NOT DEFINED CMAKE_INSTALL_PYTHON_SITE_PACKAGES)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
            "import site; print(site.getsitepackages()[0])"
        OUTPUT_VARIABLE _py_site
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _py_site_ret
    )
    if(_py_site_ret EQUAL 0)
        set(CMAKE_INSTALL_PYTHON_SITE_PACKAGES "${_py_site}" CACHE PATH
            "Python site-packages install directory")
    else()
        set(CMAKE_INSTALL_PYTHON_SITE_PACKAGES
            "${CMAKE_INSTALL_PREFIX}/lib/python3/site-packages"
            CACHE PATH "Python site-packages install directory (fallback)")
    endif()
endif()
message(STATUS "ESP: Install destination: ${CMAKE_INSTALL_PYTHON_SITE_PACKAGES}/hoomd_esp")

# Install the compiled extension
install(TARGETS _esp
    LIBRARY DESTINATION "${CMAKE_INSTALL_PYTHON_SITE_PACKAGES}/hoomd_esp"
)

# Install the Python package
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/hoomd_esp"
    DESTINATION "${CMAKE_INSTALL_PYTHON_SITE_PACKAGES}"
    FILES_MATCHING
        PATTERN "*.py"
    PATTERN "__pycache__" EXCLUDE
    PATTERN "*.pyc"       EXCLUDE
)

# ---------------------------------------------------------------------------
# Optional: generate compile_commands.json for clangd / IDE LSP
# ---------------------------------------------------------------------------
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "========== ESP Plugin Configuration ==========")
message(STATUS "  Version         : ${PROJECT_VERSION}")
message(STATUS "  HOOMD version   : ${HOOMD_VERSION}")
message(STATUS "  Build type      : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard    : C++${CMAKE_CXX_STANDARD}")
message(STATUS "  GPU support     : ${ESP_ENABLE_GPU}")
if(ESP_ENABLE_GPU)
    message(STATUS "  CUDA toolkit    : ${CUDAToolkit_VERSION}")
endif()
message(STATUS "  MPI support     : ${HOOMD_MPI}")
message(STATUS "  Python          : ${Python_EXECUTABLE}")
message(STATUS "  Install prefix  : ${CMAKE_INSTALL_PYTHON_SITE_PACKAGES}")
message(STATUS "==============================================")
message(STATUS "")
