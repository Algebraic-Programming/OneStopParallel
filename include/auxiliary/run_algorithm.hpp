#pragma once

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <filesystem>
#include <iostream>
#include <string>
#include <tuple>

#include "scheduler/Serial/Serial.hpp"

#include "scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "scheduler/GreedySchedulers/GreedyBspFillupScheduler.hpp"
#include "scheduler/GreedySchedulers/GreedyBspLocking.hpp"
#include "scheduler/GreedySchedulers/GreedyChildren.hpp"
#include "scheduler/GreedySchedulers/GreedyCilkScheduler.hpp"
#include "scheduler/GreedySchedulers/GreedyEtfScheduler.hpp"
#include "scheduler/GreedySchedulers/GreedyLayers.hpp"
#include "scheduler/GreedySchedulers/GreedyVarianceScheduler.hpp"
#include "scheduler/GreedySchedulers/GreedyVarianceFillupScheduler.hpp"
#include "scheduler/GreedySchedulers/MetaGreedyScheduler.hpp"
#include "scheduler/GreedySchedulers/RandomBadGreedy.hpp"
#include "scheduler/GreedySchedulers/RandomGreedy.hpp"

#include "scheduler/ContractRefineScheduler/BalDMixR.hpp"
#include "scheduler/ContractRefineScheduler/CoBalDMixR.hpp"
#include "scheduler/ContractRefineScheduler/MultiLevelHillClimbing.hpp"
#include "scheduler/Coarsers/TransitiveEdgeReductor.hpp"

#include "scheduler/Wavefront/Wavefront.hpp"
#include "scheduler/HDagg/HDagg_simple.hpp"

#include "scheduler/LocalSearchSchedulers/HillClimbingScheduler.hpp"
#include "scheduler/LocalSearchSchedulers/LKTotalCommScheduler.hpp"
#include "scheduler/Coarsers/SquashA.hpp"
#include "scheduler/Coarsers/HDaggCoarser.hpp"
#include "scheduler/Coarsers/WavefrontCoarser.hpp"
#include "scheduler/Coarsers/Funnel.hpp"
#include "scheduler/Coarsers/TreesUnited.hpp"

#include "scheduler/ImprovementScheduler.hpp"

#include "dag_partitioners/VariancePartitioner.hpp"
#include "dag_partitioners/LightEdgeVariancePartitioner.hpp"

#include "file_interactions/CommandLineParser.hpp"
#include "file_interactions/CommandLineParserPartition.hpp"
#include "file_interactions/FileReader.hpp"

#include "model/BspSchedule.hpp"
#include "model/DAGPartition.hpp"
#include "scheduler/Scheduler.hpp"


std::pair<RETURN_STATUS, BspSchedule> run_algorithm(const CommandLineParser &parser, const boost::property_tree::ptree &algorithm,
                                                    const BspInstance &bsp_instance, unsigned timeLimit, bool use_memory_constraint);

std::pair<RETURN_STATUS, DAGPartition> run_algorithm(const CommandLineParserPartition &parser, const boost::property_tree::ptree &algorithm,
                                                    const BspInstance &bsp_instance, unsigned timeLimit, bool use_memory_constraint);