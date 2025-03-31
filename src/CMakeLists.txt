
# Library: model
file(GLOB model_bsp_sources model/bsp/*.cpp)

#Library: auxiliary
file(GLOB auxiliary_sources auxiliary/*.cpp)

# # Library: simulation
# file(GLOB simulation_sources simulation/*.cpp)

# # Library: scheduler
# file(GLOB scheduler_sources scheduler/*.cpp)

# # Library: scheduler
# file(GLOB dag_divider_sources dag_divider/*.cpp)

# # Library: IlpSchedulers
# file(GLOB IlpSchedulers_sources scheduler/IlpSchedulers/*.cpp)

# # Library: LKSchedulers
# file(GLOB LocalSearchSchedulers_sources scheduler/LocalSearchSchedulers/*.cpp)

# # Library: LKSchedulers
# file(GLOB KLLocalSearchSchedulers_sources scheduler/LocalSearchSchedulers/KernighanLin/*.cpp)

# # Library: MultiBspScheduler
# file(GLOB MultiBspScheduler_sources scheduler/MultiBspScheduler/*.cpp)

# # Library: Partitioners
# file(GLOB Partitioners_sources scheduler/Partitioners/*.cpp)

# # Library: GreedySchedulers
# file(GLOB GreedySchedulers_sources scheduler/GreedySchedulers/*.cpp)

# # Library: HDaggSchedulers
# file(GLOB HDagg_sources scheduler/HDagg/*.cpp)

# # Library: WavefrontSchedulers
# file(GLOB Wavefront_sources scheduler/Wavefront/*.cpp)

# # Library: SerialSchedulers
# file(GLOB SerialSchedulers_sources scheduler/Serial/*.cpp)

# # Library: PebllingILP
# file(GLOB PebblingILP_sources scheduler/PebblingILP/*.cpp)

# # Library: PebllingILP auxiliary
# file(GLOB PebblingILP_aux_sources scheduler/PebblingILP/AuxiliaryForPartialILP/*.cpp)

# # Library: ContractRefineScheduler
# file(GLOB ContractRefineScheduler_sources scheduler/ContractRefineScheduler/*.cpp)

# # Library: Coarsers
# file(GLOB Coarsers_sources scheduler/Coarsers/*.cpp)

# # Library: Minimal_matching
# file(GLOB MinimalMatching_sources scheduler/Minimal_matching/*.cpp)

# Library: io
file(GLOB io_sources io/*.cpp)

# # Library: model
# file(GLOB dag_algorithms model/dag_algorithms/*.cpp)

# # Library: refine
# file(GLOB refine_sources scheduler/ContractRefineScheduler/refine/*.cpp)

# # Library: structures
# file(GLOB structures_sources structures/*.cpp)

# # Library: coarsen
# file(GLOB coarsen_sources scheduler/ContractRefineScheduler/coarsen/*.cpp)

# # Library: SchedulePermutations
# file(GLOB SchedulePermutations_sources scheduler/SchedulePermutations/*.cpp)

# # Library: SubArchitectureSchedulers
# file(GLOB SubArchitectureSchedulers_sources scheduler/SubArchitectureSchedulers/*.cpp)

# # Library: advanced_partitioners
# file(GLOB dag_partitioners_sources dag_partitioners/*.cpp)

# # Library: coarser
# file(GLOB coarser_sources coarser/*.cpp)

# # Library: coarser
# file(GLOB top_order_coarsers_sources coarser/top_order/*.cpp)

# # Library: coarser
# file(GLOB hdagg_coarsers_sources coarser/hdagg/*.cpp)

# # Library: heavy_edge_coarser
# file(GLOB heavy_edge_coarser_sources coarser/heavy_edges/*.cpp)

#Declare the main library as an interface
add_library( BspSchedulingLibrary STATIC 
                                #   ${scheduler_sources}
                                #   ${IlpSchedulers_sources}
                                #   ${simulation_sources}
                                #   ${dag_divider_sources}
                                #   ${dag_algorithms}
                                #   ${KLLocalSearchSchedulers_sources}
                                #   ${LocalSearchSchedulers_sources}
                                #   ${PebblingILP_sources}
                                #   ${PebblingILP_aux_sources}
                                #   ${Partitioners_sources}
                                #   ${GreedySchedulers_sources}
                                #   ${MultiBspScheduler_sources}
                                #   ${ContractRefineScheduler_sources}
                                #   ${Coarsers_sources}
                                #   ${MinimalMatching_sources}
                                #   ${file_interactions_sources}
                                    ${model_bsp_sources}
                                    ${io_sources}
                                #   ${refine_sources}
                                #   ${structures_sources}
                                    ${auxiliary_sources}
                                #   ${coarsen_sources}
                                #   ${SchedulePermutations_sources}
                                #   ${HDagg_sources}
                                #   ${Wavefront_sources}
                                #   ${SerialSchedulers_sources}
                                #   ${SubArchitectureSchedulers_sources}
                                #   ${dag_partitioners_sources}
                                #   ${coarser_sources}
                                #   ${hdagg_coarsers_sources}
                                #   ${top_order_coarsers_sources}
                                #   ${heavy_edge_coarser_sources}
)
target_link_libraries( BspSchedulingLibrary PUBLIC
    # Boost::graph
    # Boost::log_setup
    # Boost::log
    Threads::Threads
    OpenMP::OpenMP_CXX
    # copt_cpp
    # copt
)

# # Modify the default LD_LIBRARY_PATH to include the "lib/copt" directory
# set_target_properties( BspSchedulingLibrary PROPERTIES
#     BUILD_RPATH "${CMAKE_SOURCE_DIR}/lib/copt"
# )