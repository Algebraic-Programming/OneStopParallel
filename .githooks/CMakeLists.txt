find_package(Git)
if(NOT GIT_FOUND)
    message(WARNING "Git was not found -- skipping git hooks setup")
    return()
endif()

# Check if CMAKE_SOURCE_DIR is a git folder
if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
    # Initialize the hooks with the current folder containing pre-commit
    execute_process(
        COMMAND git config core.hooksPath ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE result
    )
    if(result)
        message(WARNING "Failed to set up git hooks")
    else()
        message(STATUS "Git hooks set up")
    endif()
endif()
