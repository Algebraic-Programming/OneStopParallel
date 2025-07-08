# Copyright 2024 Huawei Technologies Co., Ltd.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# @author Toni Boehnlein, Benjamin Lozes, Pal Andras Papp, Raphael S. Steiner


cmake_minimum_required(VERSION 3.12)
project(OneStopParallel VERSION 1.0.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Ensure C++17 standard is strictly enforced

# Configure git hooks
add_subdirectory(.githooks)


# --- Compile Options Configuration ---
# Common compile options including strict warnings and error-on-warning
set(COMMON_CXX_WARNING_FLAGS
    "-Wall"
    "-Wextra"
    "-Wfatal-errors"
    "-Wconversion"
    "-Wsign-conversion"
    "-Wunused"
    "-Wunreachable-code"
    "-Wuninitialized"
    "-Werror"
)

# --- Project-specific Compile Flags ---
# An interface library to hold compile flags for executables built in this project.
# This avoids polluting the main OneStopParallel interface library with flags that
# should not be imposed on downstream consumers.
add_library(ProjectExecutableFlags INTERFACE)
target_compile_options(ProjectExecutableFlags INTERFACE ${COMMON_CXX_WARNING_FLAGS})

# Set build-type specific compile options using modern generator expressions
target_compile_options(ProjectExecutableFlags INTERFACE
    $<IF:$<CONFIG:Debug>,
        # Debug flags
        -g,
        # Else (not Debug)
        $<IF:$<CONFIG:RelWithDebInfo>,
            # RelWithDebInfo flags
            "-O2;-g;-DNDEBUG",
            # Release and other/custom flags
            "-O3;-DNDEBUG"
        >
    >
)

# --- Find External Libraries ---
# Find COPT libraries
find_package(COPT)


# Find Boost libraries
# Note: link_directories is deprecated. It's better to link against Boost targets directly
# using target_link_libraries in your specific executable/library targets.
find_package(Boost 1.71.0 OPTIONAL_COMPONENTS graph unit_test_framework)
# The modern CMake approach uses target-based properties, so global include_directories is avoided.
if(NOT Boost_FOUND)
    message(WARNING "Boost not found. Any features or tests depending on it will be disabled.")
endif()

# Allow user to manually specify the Eigen include path
find_package(Eigen3 3.4)
if(Eigen3_FOUND)
    message(STATUS "Eigen3 found!")
else()
    message(WARNING "Eigen3 not found. Some features may be disabled.")
endif()

# Find OpenMP
find_package(OpenMP)
if(OpenMP_FOUND)
    set(OSP_DEPENDS_ON_OPENMP ON)
    message(STATUS "OpenMP found, enabling OpenMP support.")
else()
    set(OSP_DEPENDS_ON_OPENMP OFF)
    message(WARNING "OpenMP not found. The library will be built without OpenMP support.")
endif()

# --- Library Definition ---
add_library(OneStopParallel INTERFACE)

target_include_directories(OneStopParallel INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# The OneStopParallel INTERFACE library itself should not impose compile flags
# on consumers. Project-internal executables will link to ProjectExecutableFlags
# to get the desired settings.

# Display status messages for build types
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Configuring for Debug build. Debug symbols enabled.")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Configuring for Release build. Optimization -O3 and NDEBUG enabled.")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    message(STATUS "Configuring for Release with Debug Info build. Optimization -O2 and debug symbols enabled.")
else()
    message(STATUS "Configuring for custom build type '${CMAKE_BUILD_TYPE}'. Applying default optimization and warnings.")
endif()


if(OpenMP_FOUND)
    target_link_libraries(OneStopParallel INTERFACE OpenMP::OpenMP_CXX)
endif()

if(Boost_graph_FOUND)
    set(OSP_DEPENDS_ON_BOOST_GRAPH ON)
    target_link_libraries(OneStopParallel INTERFACE Boost::graph)
else()
    set(OSP_DEPENDS_ON_BOOST_GRAPH OFF)
endif()
if (COPT_FOUND)
    target_link_libraries(OneStopParallel INTERFACE COPT::COPT)

    set_target_properties(OneStopParallel PROPERTIES
        BUILD_RPATH "${COPT_RPATHS}"
    )
endif()

if(Eigen3_FOUND)
    target_link_libraries(OneStopParallel INTERFACE Eigen3::Eigen)
    target_compile_definitions(OneStopParallel INTERFACE EIGEN_FOUND)
endif()

# --- Subdirectories for Source and Executables ---
add_subdirectory(apps)
add_subdirectory(apps/graph_generator)


# --- Build Tests Configuration ---
# Option to control whether tests are built.
set(OSP_DATA_DIR ${PROJECT_SOURCE_DIR}/data CACHE INTERNAL "Path to project data files")
# Defaults to ON, but will be OFF if CMAKE_BUILD_TYPE is Release.
option(BUILD_TESTS "Build tests for the project" ON)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(BUILD_TESTS OFF)
    message(STATUS "CMAKE_BUILD_TYPE is Release, so BUILD_TESTS is forced to OFF.")
endif()

if(BUILD_TESTS)
    # Check if Boost::unit_test_framework was found for tests to be enabled.
    if(NOT Boost_graph_FOUND)
        message(WARNING "Boost::graph component not found. Tests require it and will be disabled.")
    elseif(NOT Boost_unit_test_framework_FOUND)
        message(WARNING "Boost::unit_test_framework component not found. Tests require it and will be disabled.")
    elseif(NOT OpenMP_FOUND)
        message(WARNING "OpenMP not found. Tests require it and will be disabled.")
    else()
        enable_testing()
        add_subdirectory(tests)
        message(STATUS "Tests will be built.")
    endif()
else()
    message(STATUS "Tests will not be built (BUILD_TESTS is OFF).")
endif()


# --- Documentation Generation ---
# Add a custom target to build Doxygen documentation
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    add_custom_target(doc
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
    message(STATUS "Doxygen found. 'doc' target available for building documentation.")
else()
    message(STATUS "Doxygen not found. Documentation target 'doc' will not be available.")
endif()


# --- Installation ---
# Set a default installation prefix for local development if the user has not specified one.
# This avoids needing sudo for a typical build-and-install cycle.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/install" CACHE PATH "Default install path for developer builds" FORCE)
    message(STATUS "Setting default install prefix to ${CMAKE_INSTALL_PREFIX}")
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS OneStopParallel EXPORT OneStopParallelTargets)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/OneStopParallelConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(cmake/OneStopParallelConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/OneStopParallelConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OneStopParallel
)

install(EXPORT OneStopParallelTargets
    FILE OneStopParallelTargets.cmake
    NAMESPACE OneStopParallel::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OneStopParallel
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/OneStopParallelConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/OneStopParallelConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OneStopParallel
)
