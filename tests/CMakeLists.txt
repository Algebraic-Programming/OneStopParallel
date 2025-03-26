# Create an empty list for the tests
set( tests_list "" )

# Function to create a test
macro( _add_test name )
  # Add the test to the list
  list( APPEND tests_list test_${name} )
  # Create the test
  add_executable( test_${name} ${name}.cpp )
  target_link_libraries( test_${name} #BspSchedulingLibrary
   Boost::unit_test_framework )
  target_include_directories( test_${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include )
  add_test( NAME test_${name} COMMAND test_${name} )
endmacro()



_add_test( graph_vector_impl )

_add_test( directed_graph_util )

_add_test( graph_vector_edge_desc_impl )


# Custom target to compile all the tests
add_custom_target( build_tests DEPENDS ${tests_list} )
add_custom_target( run_tests COMMAND ${CMAKE_CTEST_COMMAND} DEPENDS build_tests )
add_custom_target( tests COMMAND ${CMAKE_CTEST_COMMAND} DEPENDS ${tests_list} )
