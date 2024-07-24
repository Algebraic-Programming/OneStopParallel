/*
Copyright 2024 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author Toni Boehnlein, Benjamin Lozes, Pal Andras Papp, Raphael S. Steiner   
*/

#pragma once

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <filesystem>
#include <iostream>
#include <string>
#include <tuple>

#include "algorithms/Serial/Serial.hpp"

#include "algorithms/GreedySchedulers/GreedyBspScheduler.hpp"
#include "algorithms/GreedySchedulers/GreedyBspFillupScheduler.hpp"
#include "algorithms/GreedySchedulers/GreedyBspLocking.hpp"
#include "algorithms/GreedySchedulers/GreedyChildren.hpp"
#include "algorithms/GreedySchedulers/GreedyCilkScheduler.hpp"
#include "algorithms/GreedySchedulers/GreedyEtfScheduler.hpp"
#include "algorithms/GreedySchedulers/GreedyLayers.hpp"
#include "algorithms/GreedySchedulers/GreedyVarianceScheduler.hpp"
#include "algorithms/GreedySchedulers/GreedyVarianceFillupScheduler.hpp"
#include "algorithms/GreedySchedulers/MetaGreedyScheduler.hpp"
#include "algorithms/GreedySchedulers/RandomBadGreedy.hpp"
#include "algorithms/GreedySchedulers/RandomGreedy.hpp"

#include "algorithms/ContractRefineScheduler/BalDMixR.hpp"
#include "algorithms/ContractRefineScheduler/CoBalDMixR.hpp"
#include "algorithms/ContractRefineScheduler/MultiLevelHillClimbing.hpp"
#include "algorithms/Coarsers/TransitiveEdgeReductor.hpp"

#include "algorithms/Wavefront/Wavefront.hpp"
#include "algorithms/HDagg/HDagg_simple.hpp"

#include "algorithms/LocalSearchSchedulers/LKTotalCommScheduler.hpp"
#include "algorithms/LocalSearchSchedulers/HillClimbingScheduler.hpp"

#include "algorithms/Coarsers/SquashA.hpp"
#include "algorithms/Coarsers/HDaggCoarser.hpp"
#include "algorithms/Coarsers/WavefrontCoarser.hpp"
#include "algorithms/Coarsers/Funnel.hpp"
#include "algorithms/Coarsers/TreesUnited.hpp"

#include "algorithms/ImprovementScheduler.hpp"

#include "file_interactions/CommandLineParser.hpp"
#include "file_interactions/FileReader.hpp"

#include "model/BspSchedule.hpp"
#include "algorithms/Scheduler.hpp"


std::pair<RETURN_STATUS, BspSchedule> run_algorithm(const CommandLineParser &parser, const boost::property_tree::ptree &algorithm,
                                                    const BspInstance &bsp_instance, unsigned timeLimit);