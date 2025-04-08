# Create an empty list for the tests
set( tests_list "" )

# Function to create a test
macro( _add_test name )
  # Add the test to the list
  list( APPEND tests_list test_${name} )
  # Create the test
  add_executable( test_${name} ${name}.cpp )
  target_link_libraries( test_${name} BspSchedulingLibrary
   Boost::unit_test_framework )
  target_include_directories( test_${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include )
  add_test( NAME test_${name} COMMAND test_${name} )
endmacro()


## graph implementations
_add_test( graph_vector_impl )

_add_test( graph_vector_edge_desc_impl )

_add_test( boost_graph_adaptor )


## graph algorithms
_add_test( directed_graph_util )


## bsp model
_add_test( bsp_architecture )

_add_test( bsp_instance )


## auxiliary
_add_test( intpower )

_add_test( set_operations )

_add_test( sorts_and_arrangements )

_add_test( balanced_coin_flips )

## io
_add_test( filereader )


## scheduler 

_add_test( bsp_schedulers )


# Custom target to compile all the tests
add_custom_target( build_tests DEPENDS ${tests_list} )
add_custom_target( run_tests COMMAND ${CMAKE_CTEST_COMMAND} DEPENDS build_tests )
add_custom_target( tests COMMAND ${CMAKE_CTEST_COMMAND} DEPENDS ${tests_list} )
