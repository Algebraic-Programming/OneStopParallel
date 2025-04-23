# Create an empty list for the examples
set( examples_list "" )

# Function to create an example
macro( _add_example name )
  # Add the example to the list
  list( APPEND examples_list ${name} )
  # Create the example
  add_executable( ${name} ${name}.cpp )
  target_link_libraries( ${name} PUBLIC BspSchedulingLibrary )
  if (COPT_FOUND)
    target_link_libraries( ${name} PUBLIC COPT::COPT )
  endif()
endmacro()


IF(Eigen3_FOUND)
_add_example( post_incomplete_cholesky )
ENDIF()

_add_example( spmv_example )
_add_example( graph_analyser )
_add_example( gen_Erdos-Renyi_graph )
_add_example( gen_near_diag_random_graph )



# Custom target to compile all the examples
add_custom_target( build_examples DEPENDS ${examples_list} )
add_custom_target( examples DEPENDS ${examples_list} )
