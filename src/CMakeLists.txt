# ESP_HOOMD/src/CMakeLists.txt
# Append ESP sources to the top-level _esp target.

set(ESP_PYTHON_FILES
    esp.py
    __init__.py
    version.py
)

if(NOT TARGET _esp)
    message(FATAL_ERROR "ESP: top-level target _esp must be defined before adding src/")
endif()

target_include_directories(_esp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_sources(_esp PRIVATE
    ESPForceCompute.cc
    module.cc
)

if(ESP_ENABLE_GPU)
    set_source_files_properties(ESPForceComputeGPU.cu PROPERTIES LANGUAGE CUDA)

    target_sources(_esp PRIVATE
        ESPForceComputeGPU.cc
        ESPForceComputeGPU.cu
    )
endif()

install(FILES ${ESP_PYTHON_FILES}
    DESTINATION ${CMAKE_INSTALL_PYTHON_SITE_PACKAGES}/hoomd_esp/
)
