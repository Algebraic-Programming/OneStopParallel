# Create an empty list for the examples
set( test_suite_list "" )

# Function to create an example
macro( _add_test_suite name )
  # Add the example to the list
  list( APPEND test_suite_list ${name} )
  # Create the example
  add_executable( ${name} ${name}.cpp )
  target_link_libraries( ${name} PUBLIC BspSchedulingLibrary )
endmacro()

# Example: write_dot
_add_test_suite( test_suite_execution )

IF(Eigen3_FOUND)
_add_test_suite( test_suite_execution_SM )
ENDIF()

_add_test_suite( test_suite_bsp_schedulers )

add_executable( mem_sched_ts test_suite_mem_schedulers.cpp )
target_link_libraries( mem_sched_ts PUBLIC BspSchedulingLibrary )

# Custom target to compile all the examples
add_custom_target( build_test_suite DEPENDS ${test_suite_list} )
add_custom_target( test_suite DEPENDS ${test_suite_list} )


# Copies test_suite_config
configure_file(test_suite_config.json test_suite_config.json COPYONLY)
configure_file(run_test_suite.py run_test_suite.py COPYONLY)

configure_file(test_suite_config_SM.json test_suite_config_SM.json COPYONLY)
configure_file(run_test_suite_SM.py run_test_suite_SM.py COPYONLY)