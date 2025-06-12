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
project(one-stop-parallel VERSION 1.0.0 LANGUAGES CXX)

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

# Set compile options based on build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Configuring for Debug build. Debug symbols enabled.")
    # For Debug builds: enable debug symbols
    add_compile_options("-g" ${COMMON_CXX_WARNING_FLAGS})
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Configuring for Release build. Optimization -O3 and NDEBUG enabled.")
    # For Release builds: enable optimization and disable asserts
    add_compile_options("-O3" "-DNDEBUG" ${COMMON_CXX_WARNING_FLAGS})
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    message(STATUS "Configuring for Release with Debug Info build. Optimization -O2 and debug symbols enabled.")
    # For RelWithDebInfo builds: optimize and include debug info
    add_compile_options("-O2" "-g" "-DNDEBUG" ${COMMON_CXX_WARNING_FLAGS})
else()
    message(STATUS "Configuring for custom build type '${CMAKE_BUILD_TYPE}'. Applying default optimization and warnings.")
    # Default for other build types if not specified
    add_compile_options("-O3" "-DNDEBUG" ${COMMON_CXX_WARNING_FLAGS})
endif()


# --- Include Directories ---
# Global include directories. For more precise control, use target_include_directories in sub-CMakeLists.txt files.
include_directories(include)


# --- Find External Libraries ---

# Find COPT libraries
find_package(COPT)

# Find ICU libraries
find_package(ICU COMPONENTS data i18n uc REQUIRED)
include_directories(${ICU_INCLUDE_DIR}) # Use include_directories for global include paths

# Find Boost libraries
# Note: link_directories is deprecated. It's better to link against Boost targets directly
# using target_link_libraries in your specific executable/library targets.
find_package(Boost 1.71.0 REQUIRED COMPONENTS graph log log_setup OPTIONAL_COMPONENTS unit_test_framework)
include_directories(${Boost_INCLUDE_DIRS}) # Use include_directories for global include paths

# Allow user to manually specify the Eigen include path
find_package(Eigen3 3.4)
if(Eigen3_FOUND)
    message(STATUS "Eigen3 found!")
    set(DEIGEN ON)
    add_definitions(-DEIGEN_FOUND)
    get_target_property(eigen_incs Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)

    # Mark Eigen includes as SYSTEM to suppress warnings from third-party headers
    foreach(eigen_inc ${eigen_incs})
        include_directories(SYSTEM "${eigen_inc}")
    endforeach()
else()
    message(WARNING "Eigen3 not found. Some features may be disabled.")
endif()

# Find pthread library
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)


# --- Subdirectories for Source and Executables ---
add_subdirectory(src)
add_subdirectory(apps)
add_subdirectory(apps/graph_generator)


# --- Build Tests Configuration ---
# Option to control whether tests are built.
# Defaults to ON, but will be OFF if CMAKE_BUILD_TYPE is Release.
option(BUILD_TESTS "Build tests for the project" ON)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(BUILD_TESTS OFF)
    message(STATUS "CMAKE_BUILD_TYPE is Release, so BUILD_TESTS is forced to OFF.")
endif()

if(BUILD_TESTS)
    # Check if Boost::unit_test_framework was found for tests to be enabled.
    if(NOT Boost_unit_test_framework_FOUND)
        message(WARNING "Boost::unit_test_framework cannot be found. Tests are disabled even though BUILD_TESTS is ON.")
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
