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


list( APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" )

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # Configure git hooks
add_subdirectory(.githooks)

add_compile_options(
    "-O3" "-DNDEBUG" "-Wall" "-Wextra" "-Wfatal-errors" "-Wconversion" "-Wsign-conversion" "-Wunused" "-Wunreachable-code" "-Wuninitialized" "-Werror" # "-Wnrvo" # "-g" "-DNDEBUG" "-O3"
)

#Include directories
include_directories(include)


# Find COPT libraries
find_package(COPT)

# Find ICU libraries
find_package(ICU COMPONENTS data i18n uc)
include_directories(${ICU_INCLUDE_DIR})
link_directories(${ICU_LIBRARIES})

# Find Boost libraries
find_package(Boost 1.71.0 REQUIRED COMPONENTS graph log log_setup OPTIONAL_COMPONENTS unit_test_framework )
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIR})

# Allow user to manually specify the Eigen include path
find_package(Eigen3 3.4)

if(Eigen3_FOUND)
    message(STATUS "Eigen3 found!")
    set(DEIGEN ON)
    add_definitions(-DEIGEN_FOUND)
    get_target_property(eigen_incs Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)

    # Mark Eigen includes as SYSTEM to suppress warnings
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

#Add subdirectory for sources
add_subdirectory(src)


# Add subdirectory for tests
if( NOT Boost_unit_test_framework_FOUND )
    message( WARNING "Boost::unit_test_framework cannot be found. Tests are disabled" )
else()
    enable_testing()
    add_subdirectory(tests)
endif()

# Add the documentation command
find_package(Doxygen QUIET)
if( DOXYGEN_FOUND )
    add_custom_target(doc
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
endif( DOXYGEN_FOUND )

