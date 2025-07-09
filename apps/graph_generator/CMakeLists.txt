# Create an empty list for the executables
set( graph_generator_list )

# Function to create an executable
macro( _add_executable name)
  # Add the executable to the list
  list( APPEND graph_generator_list ${name} )
  # Create the executable
  add_executable( ${name} ${name}.cpp )
  target_link_libraries( ${name} PRIVATE OneStopParallel ProjectExecutableFlags )
endmacro()

if(Eigen3_FOUND)
_add_executable( post_incomplete_cholesky )
endif()

_add_executable( gen_Erdos-Renyi_graph )

_add_executable( gen_near_diag_random_graph )

# Custom target to compile all the executables
add_custom_target( build_graph_generator DEPENDS ${graph_generator_list} )
