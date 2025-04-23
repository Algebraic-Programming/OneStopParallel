# Create an empty list for the tests
set( tests_list "" )

# Function to create a test
macro( _add_test name )
  # Add the test to the list
  list( APPEND tests_list test_${name} )
  # Create the test
  add_executable( test_${name} ${name}.cpp )
  target_link_libraries( test_${name} BspSchedulingLibrary Boost::unit_test_framework )
  target_include_directories( test_${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include )
  add_test( NAME test_${name} COMMAND test_${name} )
endmacro()

_add_test( coarser )

_add_test( bsp_schedulers_csr )

_add_test( cuthill_mckee )

_add_test( connected_components_part )


_add_test( approx_edge_reduction )

# Test: BspArchitecture
_add_test( bsp_architecture )

# Test: ComputationalDag
_add_test( ComputationalDag )

_add_test( kl )

# Test: DAG
_add_test( DAG )

# Test: boost_bfs
_add_test( boost_bfs )

# Test: intpower
_add_test( intpower )

# Test: minimal_matching
_add_test( minimal_matching )

# Test: scheduler_with_time_limit
_add_test( scheduler_with_time_limit )

# Test: set_operations
_add_test( set_operations )

# Test: sorts_and_arrangements
_add_test( sorts_and_arrangements )

# Test: uf_structures
_add_test( uf_structures )

# Test: balanced_coin_flips
_add_test( balanced_coin_flips )

# Test: min_sym_sub_sum
_add_test( min_sym_sub_sum )

# Test: bsp_schedulers
_add_test( bsp_schedulers )

# Test: bsp_improvementschedulers
_add_test( bsp_improvementschedulers )

# Test: schedule_node_permuter
_add_test( schedule_node_permuter )

# Test: loose_supersteps
_add_test( loose_supersteps )

# Test: hash_pair
_add_test( hash_pair )

# Test: simple_partitioners
_add_test( simple_partitioners )

# Test: coarsen_history
_add_test( coarsen_history )

# Test: coarse_refine_scheduler
_add_test( coarse_refine_scheduler )

# Test: filereader
_add_test( filereader )

# Test: dag_partitioners
_add_test( dag_partitioners )


# Test: schedulers with node/core types
_add_test( bsp_schedulers_with_node_types )

# Test: schedule model with vertical I/O
_add_test( bsp_mem_schedule_class )

# Test: ssp_scheduler_class
_add_test( ssp_scheduler_class )

# Test: ssp_schedulers
_add_test( ssp_schedulers )

# Test: ssp_schedulers_with_node_types
_add_test( ssp_schedulers_with_node_types )

# Test: SM_instance_scheduler

IF(Eigen3_FOUND)
_add_test( SM_instance_scheduler )
ENDIF()

# Custom target to compile all the tests
add_custom_target( build_tests DEPENDS ${tests_list} )
add_custom_target( run_tests COMMAND ${CMAKE_CTEST_COMMAND} DEPENDS build_tests )
add_custom_target( tests COMMAND ${CMAKE_CTEST_COMMAND} DEPENDS ${tests_list} )
