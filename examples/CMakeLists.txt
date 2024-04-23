# Create an empty list for the examples
set( examples_list "" )

# Function to create an example
macro( _add_example name )
  # Add the example to the list
  list( APPEND examples_list ${name} )
  # Create the example
  add_executable( ${name} ${name}.cpp )
  target_link_libraries( ${name} PUBLIC BspSchedulingLibrary )
endmacro()


# Example: write_dot
_add_example( example )


# Custom target to compile all the examples
add_custom_target( build_examples DEPENDS ${examples_list} )
add_custom_target( examples DEPENDS ${examples_list} )
