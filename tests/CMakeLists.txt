# Create an empty list for the tests
set( tests_list )

# Function to create a test
macro( _add_test name )
  # cmake-format: off
  cmake_parse_arguments(ARG "" "" "DATA" ${ARGN})
  # cmake-format: on
  # Add the test to the list
  list( APPEND tests_list test_${name} )
  # Create the test
  add_executable(test_${name} ${name}.cpp)
  target_link_libraries(test_${name} PRIVATE OneStopParallel Boost::unit_test_framework ProjectExecutableFlags)
  target_include_directories( test_${name} PRIVATE ${PROJECT_SOURCE_DIR}/include )

  if(ARG_DATA)
    target_compile_definitions(test_${name} PRIVATE OSP_TEST_DATA_DIR="${OSP_DATA_DIR}")
  endif()

  add_test( NAME test_${name} COMMAND test_${name} )
endmacro()


## graph implementations
_add_test( graph_vector_impl )

_add_test( graph_vector_edge_desc_impl )

_add_test( boost_graph_adaptor )

_add_test( graph_vector_adapter )

_add_test ( connected_components_part )

_add_test( compact_sparse_graph )

_add_test( compact_sparse_graph_edge_desc )

IF(Eigen3_FOUND)
_add_test ( sparse_matrix_impl )
_add_test( sptrsv )
ENDIF()


## graph algorithms

_add_test( directed_graph_algorithms )

_add_test( strongly_connected_components )

_add_test( directed_graph_util )

_add_test( directed_graph_top_sort )

_add_test( kl_total )

_add_test( kl_lambda )

_add_test( kl_util )

_add_test( kl_bsp_cost_policies )

_add_test( kl_bsp_comm_affinities )

#_add_test( kl_bsp_max_improver_test )

_add_test( kl_bsp_incremental_affinity_test )

_add_test( kl_cross_step_work_affinity_test )

_add_test( kl_max_bsp_affinity_test )

_add_test( kl_max_bsp_comm_affinity )

_add_test( heaps )

_add_test( kl_mem_constr )

_add_test( isomorphic_subgraph_scheduler )

_add_test( trimmed_group_scheduler )

_add_test( isomorphism_mapper )

_add_test( merkle_hash_computer )

_add_test( orbit_graph_processor )

_add_test( eft_subgraph_scheduler )

_add_test( wavefront_component_divider )

_add_test( hill_climbing )

# --- Standalone Debugging Executables ---
add_executable(debug_merkle_divider debug_merkle_divider.cpp)
target_link_libraries(debug_merkle_divider PRIVATE OneStopParallel ProjectExecutableFlags)


## bsp model
_add_test( bsp_architecture )

_add_test( bsp_instance )

_add_test( bsp_schedule )

_add_test( bsp_schedule_recomp )

_add_test( bsp_greedy_recomputer )

## pebbling model
_add_test( pebbling_schedule_class )

## partitioning model
_add_test( hypergraph_and_partition )

## auxiliary
_add_test( intpower )

_add_test( divisors )

_add_test( uf_structures )

_add_test( set_operations )

_add_test( sorts_and_arrangements )

_add_test( balanced_coin_flips )

_add_test( iterators )

_add_test( random_graph_gen )

_add_test( permutations )

_add_test( bit_mask )

_add_test( hash_pair )

## io
_add_test( filereader DATA )


## scheduler
if (COPT_FOUND)

#_add_test( ilp_bsp_scheduler )

#_add_test( ilp_hypergraph_partitioning )

endif()

_add_test( bsp_schedulers )

_add_test( max_bsp_schedulers )

_add_test( bsp_schedulers_mem_const )

_add_test( bsp_improvementschedulers )

_add_test( coarser )

_add_test( wavefront_divider )

_add_test( wavefront_scheduler )

_add_test( cuthill_mckee )

_add_test( maxbsp_converter_and_hc )

_add_test( cost_evaluation )

_add_test( subgraph )

## pebbling ILPs

if (COPT_FOUND)

#_add_test( ilp_pebbling_scheduler )

endif()

## coarsening

_add_test( coarser_util )

_add_test( stepbystep_coarsen_and_multilevel )

_add_test (heavy_edge_preprocessing)

# Custom target to compile all the tests
add_custom_target( build_tests DEPENDS ${tests_list} )
add_custom_target( run_tests COMMAND ${CMAKE_CTEST_COMMAND} DEPENDS build_tests )
