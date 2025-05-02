# Copyright 2024 Huawei Technologies Co., Ltd.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# @author Toni Boehnlein, Benjamin Lozes, Pal Andras Papp, Raphael S. Steiner


find_path(COPT_INCLUDE_DIR
	NAMES coptcpp_pch.h # by looking for this header file
	HINTS ENV COPT_HOME # start looking from COPT_ROOT (by cmake convention), and then in COPT_HOME (the default installation directory proposed by COPT)
	PATH_SUFFIXES include/coptcpp_inc  # when inspecting a path, look inside the include directory
	DOC "NUMA include directory"

)

find_library( COPT_LIBRARY
	NAMES copt_cpp 
	HINTS ENV COPT_HOME 
    PATH_SUFFIXES lib # when inspecting a path, look inside the lib directory
    DOC "NUMA library"
)


# if the listed variables are set to existing paths, set the COPT_FOUND variable
# if not and the REQUIRED option was given when calling this find_module(),
# raise an error (some components were not found and we need all of them)
include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( COPT
	REQUIRED_VARS COPT_LIBRARY COPT_INCLUDE_DIR
)

# if we found the library, create a dedicated target with all needed information
if( COPT_FOUND )
	# do not show these variables as cached ones
	mark_as_advanced( COPT_LIBRARY COPT_INCLUDE_DIR )

	get_filename_component(COPT_INCLUDE_DIR_UP ${COPT_INCLUDE_DIR} DIRECTORY )

	get_filename_component(COPT_RPATHS ${COPT_LIBRARY} DIRECTORY )

	# Message(status " found COPT_LIB_DIR ${COPT_LIBRARY} and COPT_INCLUDE_DIR ${COPT_INCLUDE_DIR} ${COPT_RPATHS}")
    
	# create an imported target, i.e. a target NOT built internally, as from
	# https://cmake.org/cmake/help/latest/command/add_library.html#imported-libraries
	# this way, depending targets may link against libnuma with target_link_libraries(),
	# as if it was an internal target
	# UNKNOWN tells CMake to inspect the library type (static or shared)
	# e.g., if you compiled your own static libnuma and injected it via NUMA_ROOT
	# it will work out without changes
	add_library ( COPT::COPT UNKNOWN IMPORTED )

	# set its properties to the appropiate locations, for both headers and binaries
	set_target_properties( COPT::COPT			
		PROPERTIES		
        INTERFACE_COMPILE_DEFINITIONS COPT #def COPT definition
		IMPORTED_LOCATION ${COPT_LIBRARY}
		INTERFACE_INCLUDE_DIRECTORIES "${COPT_INCLUDE_DIR};${COPT_INCLUDE_DIR_UP}"
	)
endif()