# Library: auxiliary
file(GLOB auxiliary_sources auxiliary/*.cpp)

# Library: simulation
file(GLOB simulation_sources simulation/*.cpp)

# Library: algorithms
file(GLOB algorithms_sources algorithms/*.cpp)


# Library: LKSchedulers
file(GLOB LocalSearchSchedulers_sources algorithms/LocalSearchSchedulers/*.cpp)

# Library: Partitioners
file(GLOB Partitioners_sources algorithms/Partitioners/*.cpp)

# Library: GreedySchedulers
file(GLOB GreedySchedulers_sources algorithms/GreedySchedulers/*.cpp)

# Library: HDaggSchedulers
file(GLOB HDagg_sources algorithms/HDagg/*.cpp)

# Library: WavefrontSchedulers
file(GLOB Wavefront_sources algorithms/Wavefront/*.cpp)

# Library: SerialSchedulers
file(GLOB SerialSchedulers_sources algorithms/Serial/*.cpp)

# Library: ContractRefineScheduler
file(GLOB ContractRefineScheduler_sources algorithms/ContractRefineScheduler/*.cpp)

# Library: Coarsers
file(GLOB Coarsers_sources algorithms/Coarsers/*.cpp)

# Library: Minimal_matching
file(GLOB MinimalMatching_sources algorithms/Minimal_matching/*.cpp)

# Library: file_interactions
file(GLOB file_interactions_sources file_interactions/*.cpp)

# Library: model
file(GLOB model_sources model/*.cpp)

# Library: refine
file(GLOB refine_sources refine/*.cpp)

# Library: schedulers
file(GLOB schedulers_sources schedulers/*.cpp)

# Library: structures
file(GLOB structures_sources structures/*.cpp)

# Library: coarsen
file(GLOB coarsen_sources coarsen/*.cpp)

# Library: SchedulePermutations
file(GLOB SchedulePermutations_sources algorithms/SchedulePermutations/*.cpp)

# Library: SubArchitectureSchedulers
file(GLOB SubArchitectureSchedulers_sources algorithms/SubArchitectureSchedulers/*.cpp)


# Declare the main library as an interface
add_library( BspSchedulingLibrary STATIC ${algorithms_sources}
                                  ${simulation_sources}
                                  ${LocalSearchSchedulers_sources}
                                  ${Partitioners_sources}
                                  ${GreedySchedulers_sources}
                                  ${ContractRefineScheduler_sources}
                                  ${Coarsers_sources}
                                  ${MinimalMatching_sources}
                                  ${file_interactions_sources}
                                  ${model_sources}
                                  ${refine_sources}
                                  ${schedulers_sources}
                                  ${structures_sources}
                                  ${auxiliary_sources}
                                  ${coarsen_sources}
                                  ${SchedulePermutations_sources}
                                  ${HDagg_sources}
                                  ${Wavefront_sources}
                                  ${SerialSchedulers_sources}
                                  ${SubArchitectureSchedulers_sources}
)
target_link_libraries( BspSchedulingLibrary PUBLIC
    Boost::graph
    Boost::log_setup
    Boost::log
    Threads::Threads
    OpenMP::OpenMP_CXX
)

