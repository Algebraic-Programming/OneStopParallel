# Create an empty list for the tests
set( tests_list "" )

# Function to create a test
macro( _add_test name )
  # Add the test to the list
  list( APPEND tests_list test_${name} )
  # Create the test
  add_executable( test_${name} ${name}.cpp )
  target_link_libraries( test_${name} BspSchedulingLibrary
   Boost::unit_test_framework
   OpenMP::OpenMP_CXX )
  target_include_directories( test_${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include )
  add_test( NAME test_${name} COMMAND test_${name} )
endmacro()


## graph implementations
_add_test( graph_vector_impl )

_add_test( graph_vector_edge_desc_impl )

_add_test( boost_graph_adaptor )

_add_test ( connected_components_part )
IF(Eigen3_FOUND)
_add_test ( sparse_matrix_impl )
ENDIF()


## graph algorithms
_add_test( directed_graph_util )

_add_test( directed_graph_top_sort )

_add_test( kl )

_add_test( hill_climbing )

## bsp model
_add_test( bsp_architecture )

_add_test( bsp_instance )

_add_test( bsp_schedule )

_add_test( bsp_schedule_recomp )

## pebbling model
_add_test( pebbling_schedule_class )

## auxiliary
_add_test( intpower )

_add_test( uf_structures )

_add_test( set_operations )

_add_test( sorts_and_arrangements )

_add_test( balanced_coin_flips )

_add_test( iterators )

_add_test( random_graph_gen )

## io
_add_test( filereader )


## scheduler 
if (COPT_FOUND)

#_add_test( ilp_bsp_scheduler )

endif()

_add_test( bsp_schedulers )

_add_test( bsp_schedulers_mem_const )

_add_test( bsp_improvementschedulers )

_add_test( coarser )

_add_test( compact_sparse_graph )

_add_test( wavefront_divider )

_add_test( cuthill_mckee )

## pebbling ILPs

if (COPT_FOUND)

#_add_test( ilp_pebbling_scheduler )

endif()

## coarsening

_add_test( stepbystep_coarsen_and_multilevel )

_add_test (heavy_edge_preprocessing)

# Custom target to compile all the tests
add_custom_target( build_tests DEPENDS ${tests_list} )
add_custom_target( run_tests COMMAND ${CMAKE_CTEST_COMMAND} DEPENDS build_tests )
add_custom_target( tests COMMAND ${CMAKE_CTEST_COMMAND} DEPENDS ${tests_list} )
