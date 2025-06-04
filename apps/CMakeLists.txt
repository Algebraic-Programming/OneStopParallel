# Create an empty list for the examples
set( executable_list "" )

# Function to create an example
macro( _add_executable name)
  # Add the example to the list
  list( APPEND executable_list ${name} )
  # Create the example
  add_executable( ${name} ${name}.cpp )
  target_link_libraries( ${name} PUBLIC BspSchedulingLibrary )
endmacro()

_add_executable( osp_turnus )

_add_executable( graph_analyser )


if (COPT_FOUND)
_add_executable( ilp_scheduler_app )
endif()

# Custom target to compile all the examples
add_custom_target( build_executables DEPENDS ${executable_list} )
add_custom_target( executable DEPENDS ${executable_list} )
