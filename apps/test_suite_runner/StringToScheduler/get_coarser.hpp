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
#include <memory>
#include <string>

#include "../ConfigParser.hpp"
#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/coarser/BspScheduleCoarser.hpp"
#include "osp/coarser/Coarser.hpp"
#include "osp/coarser/MultilevelCoarser.hpp"
#include "osp/coarser/Sarkar/Sarkar.hpp"
#include "osp/coarser/Sarkar/SarkarMul.hpp"
#include "osp/coarser/SquashA/SquashA.hpp"
#include "osp/coarser/SquashA/SquashAMul.hpp"
#include "osp/coarser/funnel/FunnelBfs.hpp"
#include "osp/coarser/hdagg/hdagg_coarser.hpp"
#include "osp/coarser/top_order/top_order_coarser.hpp"
#include "osp/graph_algorithms/cuthill_mckee.hpp"

namespace osp {

template <typename GraphTIn, typename GraphTOut>
std::unique_ptr<Coarser<GraphTIn, GraphTOut>> GetCoarserByName(const ConfigParser &,
                                                               const boost::property_tree::ptree &coarserAlgorithm) {
    const std::string coarserName = coarserAlgorithm.get_child("name").get_value<std::string>();

    if (coarserName == "funnel") {
        typename FunnelBfs<GraphTIn, GraphTOut>::FunnelBfsParameters funnelParameters;
        if (auto paramsOpt = coarserAlgorithm.get_child_optional("parameters")) {
            const auto &paramsPt = paramsOpt.get();
            funnelParameters.funnelIncoming
                = paramsPt.get_optional<bool>("funnel_incoming").value_or(funnelParameters.funnelIncoming);
            funnelParameters.useApproxTransitiveReduction = paramsPt.get_optional<bool>("use_approx_transitive_reduction")
                                                                .value_or(funnelParameters.useApproxTransitiveReduction);
        }
        return std::make_unique<FunnelBfs<GraphTIn, GraphTOut>>(funnelParameters);

    } else if (coarserName == "hdagg") {
        auto coarser = std::make_unique<HdaggCoarser<GraphTIn, GraphTOut>>();
        if (auto paramsOpt = coarserAlgorithm.get_child_optional("parameters")) {
            const auto &paramsPt = paramsOpt.get();
            coarser->SetWorkThreshold(
                paramsPt.get_optional<VWorkwT<GraphTIn>>("max_work_weight").value_or(std::numeric_limits<VWorkwT<GraphTIn>>::max()));
            coarser->SetMemoryThreshold(
                paramsPt.get_optional<VMemwT<GraphTIn>>("max_memory_weight").value_or(std::numeric_limits<VMemwT<GraphTIn>>::max()));
            coarser->SetCommunicationThreshold(paramsPt.get_optional<VCommwT<GraphTIn>>("max_communication_weight")
                                                   .value_or(std::numeric_limits<VCommwT<GraphTIn>>::max()));
            coarser->SetSuperNodeSizeThreshold(
                paramsPt.get_optional<std::size_t>("max_super_node_size").value_or(std::numeric_limits<std::size_t>::max()));
        }
        return coarser;

    } else if (coarserName == "top_order") {
        std::string topOrderStrategy = "default";
        if (auto paramsOpt = coarserAlgorithm.get_child_optional("parameters")) {
            topOrderStrategy = paramsOpt.get().get<std::string>("strategy", "default");
        }

        auto setParams = [&](auto &coarserPtr) {
            if (auto paramsOpt = coarserAlgorithm.get_child_optional("parameters")) {
                const auto &paramsPt = paramsOpt.get();
                coarserPtr->SetWorkThreshold(paramsPt.get_optional<VWorkwT<GraphTIn>>("work_threshold")
                                                 .value_or(std::numeric_limits<VWorkwT<GraphTIn>>::max()));
                coarserPtr->SetMemoryThreshold(paramsPt.get_optional<VMemwT<GraphTIn>>("memory_threshold")
                                                   .value_or(std::numeric_limits<VMemwT<GraphTIn>>::max()));
                coarserPtr->SetCommunicationThreshold(paramsPt.get_optional<VCommwT<GraphTIn>>("communication_threshold")
                                                          .value_or(std::numeric_limits<VCommwT<GraphTIn>>::max()));
                coarserPtr->SetSuperNodeSizeThreshold(paramsPt.get_optional<std::size_t>("super_node_size_threshold").value_or(10));
                coarserPtr->SetNodeDistThreshold(paramsPt.get_optional<unsigned>("node_dist_threshold").value_or(10));
            }
        };

        if (topOrderStrategy == "bfs" || topOrderStrategy == "default") {
            auto coarser = std::make_unique<TopOrderCoarser<GraphTIn, GraphTOut, GetTopOrder>>();
            setParams(coarser);
            return coarser;
        } else if (topOrderStrategy == "dfs") {
            auto coarser = std::make_unique<TopOrderCoarser<GraphTIn, GraphTOut, DfsTopSort>>();
            setParams(coarser);
            return coarser;
        } else if (topOrderStrategy == "locality") {
            auto coarser = std::make_unique<TopOrderCoarser<GraphTIn, GraphTOut, GetTopOrderMinIndex>>();
            setParams(coarser);
            return coarser;
        } else if (topOrderStrategy == "max_children") {
            auto coarser = std::make_unique<TopOrderCoarser<GraphTIn, GraphTOut, GetTopOrderMaxChildren>>();
            setParams(coarser);
            return coarser;
        } else if (topOrderStrategy == "random") {
            auto coarser = std::make_unique<TopOrderCoarser<GraphTIn, GraphTOut, GetTopOrderRandom>>();
            setParams(coarser);
            return coarser;
        } else if (topOrderStrategy == "gorder") {
            auto coarser = std::make_unique<TopOrderCoarser<GraphTIn, GraphTOut, GetTopOrderGorder>>();
            setParams(coarser);
            return coarser;
        } else if (topOrderStrategy == "cuthill_mckee_wavefront") {
            auto coarser = std::make_unique<TopOrderCoarser<GraphTIn, GraphTOut, GetTopOrderCuthillMcKeeWavefront>>();
            setParams(coarser);
            return coarser;
        } else if (topOrderStrategy == "cuthill_mckee_undirected") {
            auto coarser = std::make_unique<TopOrderCoarser<GraphTIn, GraphTOut, GetTopOrderCuthillMcKeeUndirected>>();
            setParams(coarser);
            return coarser;
        } else {
            std::cerr << "Warning: Unknown top_order strategy '" << topOrderStrategy << "'. Falling back to default (bfs)."
                      << std::endl;
            auto coarser = std::make_unique<TopOrderCoarser<GraphTIn, GraphTOut, GetTopOrder>>();
            setParams(coarser);
            return coarser;
        }

    } else if (coarserName == "Sarkar") {
        SarkarParams::Parameters<VWorkwT<GraphTIn>> params;
        if (auto paramsOpt = coarserAlgorithm.get_child_optional("parameters")) {
            const auto &paramsPt = paramsOpt.get();
            params.commCost = paramsPt.get_optional<VWorkwT<GraphTIn>>("commCost").value_or(params.commCost);
            params.maxWeight = paramsPt.get_optional<VWorkwT<GraphTIn>>("maxWeight").value_or(params.maxWeight);
            params.smallWeightThreshold
                = paramsPt.get_optional<VWorkwT<GraphTIn>>("smallWeightThreshold").value_or(params.smallWeightThreshold);
            params.useTopPoset = paramsPt.get_optional<bool>("useTopPoset").value_or(params.useTopPoset);
            params.geomDecay = paramsPt.get_optional<double>("geomDecay").value_or(params.geomDecay);
            params.leniency = paramsPt.get_optional<double>("leniency").value_or(params.leniency);

            if (auto modeStrOpt = paramsPt.get_optional<std::string>("mode")) {
                const std::string &modeStr = modeStrOpt.get();
                if (modeStr == "LINES") {
                    params.mode = SarkarParams::Mode::LINES;
                } else if (modeStr == "FAN_IN_FULL") {
                    params.mode = SarkarParams::Mode::FAN_IN_FULL;
                } else if (modeStr == "FAN_IN_PARTIAL") {
                    params.mode = SarkarParams::Mode::FAN_IN_PARTIAL;
                } else if (modeStr == "FAN_OUT_FULL") {
                    params.mode = SarkarParams::Mode::FAN_OUT_FULL;
                } else if (modeStr == "FAN_OUT_PARTIAL") {
                    params.mode = SarkarParams::Mode::FAN_OUT_PARTIAL;
                } else if (modeStr == "LEVEL_EVEN") {
                    params.mode = SarkarParams::Mode::LEVEL_EVEN;
                } else if (modeStr == "LEVEL_ODD") {
                    params.mode = SarkarParams::Mode::LEVEL_ODD;
                } else if (modeStr == "FAN_IN_BUFFER") {
                    params.mode = SarkarParams::Mode::FAN_IN_BUFFER;
                } else if (modeStr == "FAN_OUT_BUFFER") {
                    params.mode = SarkarParams::Mode::FAN_OUT_BUFFER;
                } else if (modeStr == "HOMOGENEOUS_BUFFER") {
                    params.mode = SarkarParams::Mode::HOMOGENEOUS_BUFFER;
                } else {
                    throw std::invalid_argument(
                        "Invalid Sarkar mode: " + modeStr
                        + "!\nChoose from: LINES, FAN_IN_FULL, FAN_IN_PARTIAL, FAN_OUT_FULL, FAN_OUT_PARTIAL, LEVEL_EVEN, "
                          "LEVEL_ODD, FAN_IN_BUFFER, FAN_OUT_BUFFER, HOMOGENEOUS_BUFFER.");
                }
            }
        }
        return std::make_unique<Sarkar<GraphTIn, GraphTOut>>(params);

    } else if (coarserName == "SquashA") {
        SquashAParams::Parameters params;
        auto coarser = std::make_unique<SquashA<GraphTIn, GraphTOut>>(params);
        if (auto paramsOpt = coarserAlgorithm.get_child_optional("parameters")) {
            const auto &paramsPt = paramsOpt.get();
            params.useStructuredPoset = paramsPt.get_optional<bool>("use_structured_poset").value_or(params.useStructuredPoset);
            params.useTopPoset = paramsPt.get_optional<bool>("use_top_poset").value_or(params.useTopPoset);
            if (auto modeStrOpt = paramsPt.get_optional<std::string>("mode")) {
                if (modeStrOpt.get() == "EDGE_WEIGHT") {
                    params.mode = SquashAParams::Mode::EDGE_WEIGHT;
                } else if (modeStrOpt.get() == "TRIANGLES") {
                    params.mode = SquashAParams::Mode::TRIANGLES;
                } else {
                    throw std::invalid_argument("Invalid Squash mode: " + modeStrOpt.get()
                                                + "!\nChoose from: EDGE_WEIGHT, TRIANGLES.");
                }
            }
        }
        coarser->SetParams(params);
        return coarser;

    } else if (coarserName == "BspScheduleCoarser") {
        // This coarser requires an initial schedule and must be handled specially by the caller.
        return nullptr;
    }

    throw std::invalid_argument("Invalid coarser name: " + coarserName);
}

template <typename GraphTIn, typename GraphTOut>
std::unique_ptr<MultilevelCoarser<GraphTIn, GraphTOut>> GetMultilevelCoarserByName(
    const ConfigParser &, const boost::property_tree::ptree &coarserAlgorithm) {
    const std::string coarserName = coarserAlgorithm.get_child("name").get_value<std::string>();

    if (coarserName == "Sarkar") {
        auto coarser = std::make_unique<SarkarMul<GraphTIn, GraphTOut>>();
        SarkarParams::MulParameters<VWorkwT<GraphTIn>> mlParams;

        if (auto paramsOpt = coarserAlgorithm.get_child_optional("parameters")) {
            const auto &paramsPt = paramsOpt.get();
            mlParams.seed = paramsPt.get_optional<std::size_t>("seed").value_or(mlParams.seed);
            mlParams.geomDecay = paramsPt.get_optional<double>("geomDecay").value_or(mlParams.geomDecay);
            mlParams.leniency = paramsPt.get_optional<double>("leniency").value_or(mlParams.leniency);
            if (paramsPt.get_child_optional("commCostVec")) {
                mlParams.commCostVec.clear();
                for (const auto &item : paramsPt.get_child("commCostVec")) {
                    mlParams.commCostVec.push_back(item.second.get_value<VWorkwT<GraphTIn>>());
                }
                std::sort(mlParams.commCostVec.begin(), mlParams.commCostVec.end());
            }
            mlParams.maxWeight = paramsPt.get_optional<VWorkwT<GraphTIn>>("maxWeight").value_or(mlParams.maxWeight);
            mlParams.smallWeightThreshold
                = paramsPt.get_optional<VWorkwT<GraphTIn>>("smallWeightThreshold").value_or(mlParams.smallWeightThreshold);
            mlParams.maxNumIterationWithoutChanges = paramsPt.get_optional<unsigned>("max_num_iteration_without_changes")
                                                         .value_or(mlParams.maxNumIterationWithoutChanges);

            if (auto modeStrOpt = paramsPt.get_optional<std::string>("buffer_merge_mode")) {
                const std::string &modeStr = modeStrOpt.get();
                if (modeStr == "OFF") {
                    mlParams.bufferMergeMode = SarkarParams::BufferMergeMode::OFF;
                } else if (modeStr == "FAN_IN") {
                    mlParams.bufferMergeMode = SarkarParams::BufferMergeMode::FAN_IN;
                } else if (modeStr == "FAN_OUT") {
                    mlParams.bufferMergeMode = SarkarParams::BufferMergeMode::FAN_OUT;
                } else if (modeStr == "HOMOGENEOUS") {
                    mlParams.bufferMergeMode = SarkarParams::BufferMergeMode::HOMOGENEOUS;
                } else if (modeStr == "FULL") {
                    mlParams.bufferMergeMode = SarkarParams::BufferMergeMode::FULL;
                } else {
                    throw std::invalid_argument("Invalid Sarkar Buffer Merge mode: " + modeStr
                                                + "!\nChoose from: OFF, FAN_IN, FAN_OUT, HOMOGENEOUS, FULL.");
                }
            }
        }

        coarser->SetParameters(mlParams);
        return coarser;

    } else if (coarserName == "SquashA") {
        auto coarser = std::make_unique<SquashAMul<GraphTIn, GraphTOut>>();
        SquashAParams::Parameters params;

        if (auto paramsOpt = coarserAlgorithm.get_child_optional("parameters")) {
            const auto &paramsPt = paramsOpt.get();
            params.geomDecayNumNodes = paramsPt.get_optional<double>("geom_decay_num_nodes").value_or(params.geomDecayNumNodes);
            params.poissonPar = paramsPt.get_optional<double>("poisson_par").value_or(params.poissonPar);
            params.noise = paramsPt.get_optional<unsigned>("noise").value_or(params.noise);
            params.numRepWithoutNodeDecrease
                = paramsPt.get_optional<unsigned>("num_rep_without_node_decrease").value_or(params.numRepWithoutNodeDecrease);
            params.temperatureMultiplier
                = paramsPt.get_optional<double>("temperature_multiplier").value_or(params.temperatureMultiplier);
            params.numberOfTemperatureIncreases
                = paramsPt.get_optional<unsigned>("number_of_temperature_increases").value_or(params.numberOfTemperatureIncreases);

            if (auto modeStrOpt = paramsPt.get_optional<std::string>("mode")) {
                if (modeStrOpt.get() == "EDGE_WEIGHT") {
                    params.mode = SquashAParams::Mode::EDGE_WEIGHT;
                } else if (modeStrOpt.get() == "TRIANGLES") {
                    params.mode = SquashAParams::Mode::TRIANGLES;
                } else {
                    throw std::invalid_argument("Invalid Squash mode: " + modeStrOpt.get()
                                                + "!\nChoose from: EDGE_WEIGHT, TRIANGLES.");
                }
            }

            coarser->SetMinimumNumberVertices(paramsPt.get_optional<unsigned>("min_nodes").value_or(1));
        }

        coarser->SetParams(params);
        return coarser;
    }

    throw std::invalid_argument("Invalid multilevel coarser name: " + coarserName);
}

}    // namespace osp
