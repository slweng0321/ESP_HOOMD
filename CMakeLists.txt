CMAKE_MINIMUM_REQUIRED(VERSION 3.15...3.27 FATAL_ERROR)

project(
    hoomd_esp
    VERSION 0.1.0
    DESCRIPTION "ESP (Ewald Summation with Prolates) plugin for HOOMD-blue"
    LANGUAGES CXX
)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

################################
## Build options

option(CONSTANT_PH_BUILD_SHARED "Build shared objects (plugin)." ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set a default build type when none is specified.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

################################
## Dependencies

find_package(HOOMD REQUIRED)
find_package(pybind11 REQUIRED)

message(STATUS "Found HOOMD ${HOOMD_VERSION}: ${HOOMD_INSTALL_PREFIX}/${PYTHON_SITE_INSTALL_DIR}")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${HOOMD_INSTALL_PREFIX} CACHE PATH "Installation prefix" FORCE)
endif()

include_directories(${HOOMD_INSTALL_PREFIX}/include)

################################
## Compiler settings

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(
        -Wall
        -Wextra
        -Wno-unknown-pragmas
        -Wno-deprecated-declarations
        -Wno-unused-result
    )
elseif(MSVC)
    add_compile_options(/W3 /permissive-)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#######################
## Process subdirectory

add_subdirectory(src)

message(STATUS "Configuration complete. Build type: ${CMAKE_BUILD_TYPE}")
