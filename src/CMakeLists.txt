# Library: auxiliary
file(GLOB auxiliary_sources auxiliary/*.cpp)

# Library: simulation
set(simulation_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/simulation/BspSptrsvCSR.cpp
)

if(Eigen3_FOUND)
    list(APPEND simulation_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/simulation/SmSptrsvCSR.cpp
    )
endif()





# Library: scheduler
file(GLOB scheduler_sources scheduler/*.cpp)

# Library: scheduler
file(GLOB dag_divider_sources dag_divider/*.cpp)

IF(COPT_FOUND)
# Library: IlpSchedulers
file(GLOB IlpSchedulers_sources scheduler/IlpSchedulers/*.cpp)
ENDIF(COPT_FOUND)

# Library: LKSchedulers
file(GLOB LocalSearchSchedulers_sources scheduler/LocalSearchSchedulers/*.cpp)

# Library: LKSchedulers
file(GLOB KLLocalSearchSchedulers_sources scheduler/LocalSearchSchedulers/KernighanLin/*.cpp)

# Library: MultiBspScheduler
file(GLOB MultiBspScheduler_sources scheduler/MultiBspScheduler/*.cpp)

# Library: Partitioners
#file(GLOB Partitioners_sources scheduler/Partitioners/*.cpp)
set(Partitioners_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler/Partitioners/binpacking_partitioner.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler/Partitioners/greedy_partitioner.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler/Partitioners/hill_climbing_partitioner.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler/Partitioners/kk_partitioner.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler/Partitioners/partitioners.cpp
)

if(COPT_FOUND)
    list(APPEND Partitioners_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/scheduler/Partitioners/ilp_partitioner.cpp
    )
endif()

# Library: GreedySchedulers
set(GreedySchedulers_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler/GreedySchedulers/ClassicSchedule.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler/GreedySchedulers/GreedyBspFillupScheduler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler/GreedySchedulers/GreedyBspGrowLocalAutoCores.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler/GreedySchedulers/GreedyBspGrowLocal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler/GreedySchedulers/GreedyBspGrowLocalParallel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler/GreedySchedulers/GreedyBspLocking.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler/GreedySchedulers/GreedyBspScheduler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler/GreedySchedulers/GreedyBspStoneAge.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler/GreedySchedulers/GreedyChildren.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler/GreedySchedulers/GreedyCilkScheduler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler/GreedySchedulers/GreedyEtfScheduler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler/GreedySchedulers/GreedyLayers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler/GreedySchedulers/GreedySspLocking.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler/GreedySchedulers/GreedyVarianceFillupScheduler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler/GreedySchedulers/GreedyVarianceScheduler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler/GreedySchedulers/GreedyVarianceSspScheduler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler/GreedySchedulers/MemConstListScheduler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler/GreedySchedulers/MetaGreedyScheduler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler/GreedySchedulers/RandomGreedy.cpp
)

if(Eigen3_FOUND)
    list(APPEND GreedySchedulers_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/scheduler/GreedySchedulers/SMFunGrowlv2.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/scheduler/GreedySchedulers/SMFunOriGrowlv2.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/scheduler/GreedySchedulers/SMGreedyBspGrowLocalAutoCoresParallel.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/scheduler/GreedySchedulers/SMGreedyVarianceFillupScheduler.cpp
    )
endif()


# Library: HDaggSchedulers
file(GLOB HDagg_sources scheduler/HDagg/*.cpp)

# Library: WavefrontSchedulers
file(GLOB Wavefront_sources scheduler/Wavefront/*.cpp)

# Library: SerialSchedulers
file(GLOB SerialSchedulers_sources scheduler/Serial/*.cpp)

IF(COPT_FOUND)
# Library: PebllingILP
file(GLOB PebblingILP_sources scheduler/PebblingILP/*.cpp)

# Library: PebllingILP auxiliary
file(GLOB PebblingILP_aux_sources scheduler/PebblingILP/AuxiliaryForPartialILP/*.cpp)
ENDIF(COPT_FOUND)

# Library: ContractRefineScheduler
file(GLOB ContractRefineScheduler_sources scheduler/ContractRefineScheduler/*.cpp)

# Library: Coarsers
file(GLOB Coarsers_sources scheduler/Coarsers/*.cpp)

# Library: Minimal_matching
file(GLOB MinimalMatching_sources scheduler/Minimal_matching/*.cpp)


# Library: Partitioners
#file(GLOB Partitioners_sources scheduler/Partitioners/*.cpp)
set(MinimalMatching_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler/Minimal_matching/Hungarian_algorithm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler/Minimal_matching/Hungarian_alg_process_permuter.cpp   
)

if(COPT_FOUND)
    list(APPEND MinimalMatching_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/scheduler/Minimal_matching/Numa_processor_reordering_heuristic.cpp
    )
endif()



# Library: file_interactions
file(GLOB file_interactions_sources file_interactions/*.cpp)

# Library: model
set(model_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/model/BspArchitecture.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/model/BspMemSchedule.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/model/BspSchedule.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/model/BspScheduleRecompute.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/model/ComputationalDag.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/model/DAGPartition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/model/SetSchedule.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/model/SspSchedule.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/model/VectorSchedule.cpp
)

if(Eigen3_FOUND)
    list(APPEND model_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/model/SmSchedule.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/model/SparseMatrix.cpp
    )
endif()


# Library: model
file(GLOB dag_algorithms model/dag_algorithms/*.cpp)

# Library: refine
file(GLOB refine_sources scheduler/ContractRefineScheduler/refine/*.cpp)

# Library: structures
file(GLOB structures_sources structures/*.cpp)

# Library: coarsen
file(GLOB coarsen_sources scheduler/ContractRefineScheduler/coarsen/*.cpp)

# Library: SchedulePermutations
file(GLOB SchedulePermutations_sources scheduler/SchedulePermutations/*.cpp)

# Library: SubArchitectureSchedulers
file(GLOB SubArchitectureSchedulers_sources scheduler/SubArchitectureSchedulers/*.cpp)

# Library: advanced_partitioners
file(GLOB dag_partitioners_sources dag_partitioners/*.cpp)

# Library: coarser
file(GLOB coarser_sources coarser/*.cpp)

# Library: coarser
file(GLOB top_order_coarsers_sources coarser/top_order/*.cpp)

# Library: coarser
file(GLOB hdagg_coarsers_sources coarser/hdagg/*.cpp)

# Library: heavy_edge_coarser
file(GLOB heavy_edge_coarser_sources coarser/heavy_edges/*.cpp)

# Declare the main library as an interface
set(BSP_SCHEDULER_SRCS
    ${scheduler_sources}
    ${simulation_sources}
    ${dag_divider_sources}
    ${dag_algorithms}
    ${KLLocalSearchSchedulers_sources}
    ${LocalSearchSchedulers_sources}
    ${Partitioners_sources}
    ${GreedySchedulers_sources}
    ${MultiBspScheduler_sources}
    ${ContractRefineScheduler_sources}
    ${Coarsers_sources}
    ${MinimalMatching_sources}
    ${file_interactions_sources}
    ${model_sources}
    ${refine_sources}
    ${structures_sources}
    ${auxiliary_sources}
    ${coarsen_sources}
    ${SchedulePermutations_sources}
    ${HDagg_sources}
    ${Wavefront_sources}
    ${SerialSchedulers_sources}
    ${SubArchitectureSchedulers_sources}
    ${dag_partitioners_sources}
    ${coarser_sources}
    ${hdagg_coarsers_sources}
    ${top_order_coarsers_sources}
    ${heavy_edge_coarser_sources}
)

if(COPT_FOUND)
    list(APPEND BSP_SCHEDULER_SRCS
        ${IlpSchedulers_sources}
        ${PebblingILP_sources}
        ${PebblingILP_aux_sources}
    )
endif()

add_library(BspSchedulingLibrary STATIC ${BSP_SCHEDULER_SRCS})

# Link libraries
target_link_libraries(BspSchedulingLibrary PUBLIC
    Boost::graph
    Boost::log_setup
    Boost::log
    Threads::Threads
    OpenMP::OpenMP_CXX
)

if (COPT_FOUND)
    target_link_libraries(BspSchedulingLibrary PUBLIC COPT::COPT)

    set_target_properties(BspSchedulingLibrary PROPERTIES
        BUILD_RPATH "${COPT_RPATHS}"
    )
endif()