# Create an empty list for the examples
set( graph_generator_list )

# Function to create an example
macro( _add_executable name)
  # Add the example to the list
  list( APPEND graph_generator_list ${name} )
  # Create the example
  add_executable( ${name} ${name}.cpp )
  target_link_libraries( ${name} PRIVATE OneStopParallel )
endmacro()

if(Eigen3_FOUND)
_add_executable( post_incomplete_cholesky )
endif()

_add_executable( gen_Erdos-Renyi_graph )

_add_executable( gen_near_diag_random_graph )

# Custom target to compile all the examples
add_custom_target( build_graph_generator DEPENDS ${graph_generator_list} )
