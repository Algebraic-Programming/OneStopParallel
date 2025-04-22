cmake_minimum_required(VERSION 3.12)
project(one-stop-parallel VERSION 0.2.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # Configure git hooks
add_subdirectory(.githooks)

add_compile_options(
    "-DNDEBUG" "-O3" "-Wall" -Wextra "-Wfatal-errors" "-Wconversion" "-Wsign-conversion" "-Wunused" "-Wunreachable-code" "-Wuninitialized" "-Wnrvo" # "-g" "-DNDEBUG" "-O3"
)


#Include directories
include_directories(include)

# Find Boost libraries
# set(Boost_USE_STATIC_LIBS ON)
find_package(Boost 1.71.0 REQUIRED COMPONENTS graph log log_setup OPTIONAL_COMPONENTS unit_test_framework )
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIR})


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

