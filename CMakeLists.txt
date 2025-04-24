cmake_minimum_required(VERSION 3.12)
project(OneStopParallel VERSION 0.1.0 LANGUAGES CXX)

list( APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" )

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure git hooks
add_subdirectory(.githooks)

add_compile_options(
    "-DNDEBUG" "-O3"
)

# Include directories
include_directories(include)
find_package(COPT)

# Find ICU libraries
find_package(ICU COMPONENTS data i18n uc)
include_directories(${ICU_INCLUDE_DIR})
link_directories(${ICU_LIBRARIES})

# Find Boost libraries
# set(Boost_USE_STATIC_LIBS ON)
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
    include_directories(${eigen_incs})
else()
    message(WARNING "Eigen3 not found. Some features may be disabled.")
endif()

if(COPT_FOUND)
    configure_file(main_config.json main_config.json COPYONLY)
else()
    configure_file(main_config_no_copt.json main_config_no_copt.json COPYONLY)
endif()


configure_file(main_partition_config.json main_partition_config.json COPYONLY)

# Find pthread library
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Add subdirectory for sources
add_subdirectory(src)

# Add the main executable
add_executable(OneStopParallel main.cpp)
target_link_libraries(OneStopParallel PRIVATE ${Boost_LIBRARIES})
if (COPT_FOUND)
    target_link_libraries(OneStopParallel PRIVATE COPT::COPT)
endif()
if(Eigen3_FOUND)
    target_link_libraries(OneStopParallel PRIVATE Eigen3::Eigen)
endif()


target_link_libraries(OneStopParallel PRIVATE BspSchedulingLibrary)

add_executable(OneStopParallel_Partition main_partition.cpp)
target_link_libraries(OneStopParallel_Partition PRIVATE ${Boost_LIBRARIES})
if (COPT_FOUND)
    target_link_libraries(OneStopParallel_Partition PRIVATE COPT::COPT)
endif()
if(Eigen3_FOUND)
    target_link_libraries(OneStopParallel_Partition PRIVATE Eigen3::Eigen)
endif()

target_link_libraries(OneStopParallel_Partition PRIVATE BspSchedulingLibrary)


# Add subdirectory for executables
add_subdirectory(executables)

# Add subdirectory for examples
add_subdirectory(examples)

# Add subdirectory for test_suite
add_subdirectory(test_suite)

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


