cmake_minimum_required(VERSION 3.12)
project(BSP-Scheduling VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure git hooks
add_subdirectory(.githooks)

add_compile_options(
    "-O3" "-DNDEBUG" "-Wall" "-Wfatal-errors" #"-g"
)

# Include directories
include_directories(include)

# Find Boost libraries
set(Boost_USE_STATIC_LIBS ON)
find_package(Boost 1.37.0 REQUIRED COMPONENTS graph log log_setup OPTIONAL_COMPONENTS unit_test_framework )
include_directories(${Boost_INCLUDE_DIRS})

configure_file(main_config.json main_config.json COPYONLY)

# Find pthread library
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Add subdirectory for sources
add_subdirectory(src)

# Add the main executable
add_executable(main main.cpp)
target_link_libraries(main PRIVATE BspSchedulingLibrary)

# Add subdirectory for examples
add_subdirectory(examples)


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


