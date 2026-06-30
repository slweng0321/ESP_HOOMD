set(COMPONENT_NAME hoomd_esp)

if(NOT BUILD_MD)
    message(WARNING "The HOOMD build configuration does not support MD. Skipping ${COMPONENT_NAME}.")
    return()
endif()

##############################
## Source setup

set(_${COMPONENT_NAME}_sources
    module.cc
    ESPForceCompute.cc
)

set(python_files
    __init__.py
    esp.py
    version.py
    )

#########################
## Build the module

hoomd_add_module(_${COMPONENT_NAME} SHARED ${_${COMPONENT_NAME}_sources} NO_EXTRAS)

# alias into the HOOMD namespace so plugins and symlinked components both work
add_library(HOOMD::_${COMPONENT_NAME} ALIAS _${COMPONENT_NAME})

target_include_directories(_${COMPONENT_NAME} PRIVATE
    ${HOOMD_INSTALL_PREFIX}/${PYTHON_SITE_INSTALL_DIR}/include/hoomd/extern/dfftlib/src
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

target_link_libraries(_${COMPONENT_NAME} PUBLIC
    HOOMD::_hoomd
    HOOMD::_md
    pybind11::module
    )

set_target_properties(_${COMPONENT_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    POSITION_INDEPENDENT_CODE ON
    VISIBILITY_INLINES_HIDDEN ON
    )

if(APPLE)
    set_target_properties(_${COMPONENT_NAME} PROPERTIES INSTALL_RPATH "@loader_path/..;@loader_path")
elseif(UNIX)
    set_target_properties(_${COMPONENT_NAME} PROPERTIES INSTALL_RPATH "\$ORIGIN/..;\$ORIGIN")
endif()

if(WIN32)
    set_target_properties(_${COMPONENT_NAME} PROPERTIES SUFFIX ".pyd")
endif()

install(TARGETS _${COMPONENT_NAME}
        LIBRARY DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${COMPONENT_NAME}
        RUNTIME DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${COMPONENT_NAME}
        ARCHIVE DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${COMPONENT_NAME}
        )

install(FILES ${python_files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${COMPONENT_NAME}
        )

copy_files_to_build("${python_files}" "hoomd-component-${COMPONENT_NAME}" "*.py")