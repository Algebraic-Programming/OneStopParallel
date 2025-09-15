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
#include "osp/coarser/Coarser.hpp"
#include "osp/coarser/funnel/FunnelBfs.hpp"
#include "osp/coarser/BspScheduleCoarser.hpp"
#include "osp/coarser/hdagg/hdagg_coarser.hpp"
#include "osp/coarser/MultilevelCoarser.hpp"
#include "osp/coarser/Sarkar/Sarkar.hpp"
#include "osp/coarser/Sarkar/SarkarMul.hpp"
#include "osp/coarser/top_order/top_order_coarser.hpp"
#include "osp/graph_algorithms/cuthill_mckee.hpp"
#include "osp/coarser/SquashA/SquashA.hpp"
#include "osp/coarser/SquashA/SquashAMul.hpp"

namespace osp {

template<typename Graph_t_in, typename Graph_t_out>
std::unique_ptr<Coarser<Graph_t_in, Graph_t_out>>
get_coarser_by_name(const ConfigParser &, const boost::property_tree::ptree &coarser_algorithm) {

    const std::string coarser_name = coarser_algorithm.get_child("name").get_value<std::string>();

    if (coarser_name == "funnel") {
        typename FunnelBfs<Graph_t_in, Graph_t_out>::FunnelBfs_parameters funnel_parameters;
        if (auto params_opt = coarser_algorithm.get_child_optional("parameters")) {
            const auto &params_pt = params_opt.get();
            funnel_parameters.funnel_incoming =
                params_pt.get_optional<bool>("funnel_incoming").value_or(funnel_parameters.funnel_incoming);
            funnel_parameters.use_approx_transitive_reduction =
                params_pt.get_optional<bool>("use_approx_transitive_reduction")
                    .value_or(funnel_parameters.use_approx_transitive_reduction);
        }
        return std::make_unique<FunnelBfs<Graph_t_in, Graph_t_out>>(funnel_parameters);

    } else if (coarser_name == "hdagg") {
        auto coarser = std::make_unique<hdagg_coarser<Graph_t_in, Graph_t_out>>();
        if (auto params_opt = coarser_algorithm.get_child_optional("parameters")) {
            const auto &params_pt = params_opt.get();
            coarser->set_work_threshold(params_pt.get_optional<v_workw_t<Graph_t_in>>("max_work_weight")
                                            .value_or(std::numeric_limits<v_workw_t<Graph_t_in>>::max()));
            coarser->set_memory_threshold(params_pt.get_optional<v_memw_t<Graph_t_in>>("max_memory_weight")
                                              .value_or(std::numeric_limits<v_memw_t<Graph_t_in>>::max()));
            coarser->set_communication_threshold(
                params_pt.get_optional<v_commw_t<Graph_t_in>>("max_communication_weight")
                    .value_or(std::numeric_limits<v_commw_t<Graph_t_in>>::max()));
            coarser->set_super_node_size_threshold(params_pt.get_optional<std::size_t>("max_super_node_size")
                                                       .value_or(std::numeric_limits<std::size_t>::max()));
        }
        return coarser;

    } else if (coarser_name == "top_order") {
        std::string top_order_strategy = "default";
        if (auto params_opt = coarser_algorithm.get_child_optional("parameters")) {
            top_order_strategy = params_opt.get().get<std::string>("strategy", "default");
        }

        auto set_params = [&](auto &coarser_ptr) {
            if (auto params_opt = coarser_algorithm.get_child_optional("parameters")) {
                const auto &params_pt = params_opt.get();
                coarser_ptr->set_work_threshold(params_pt.get_optional<v_workw_t<Graph_t_in>>("work_threshold")
                                                    .value_or(std::numeric_limits<v_workw_t<Graph_t_in>>::max()));
                coarser_ptr->set_memory_threshold(params_pt.get_optional<v_memw_t<Graph_t_in>>("memory_threshold")
                                                      .value_or(std::numeric_limits<v_memw_t<Graph_t_in>>::max()));
                coarser_ptr->set_communication_threshold(
                    params_pt.get_optional<v_commw_t<Graph_t_in>>("communication_threshold")
                        .value_or(std::numeric_limits<v_commw_t<Graph_t_in>>::max()));
                coarser_ptr->set_super_node_size_threshold(
                    params_pt.get_optional<std::size_t>("super_node_size_threshold")
                        .value_or(10));
                coarser_ptr->set_node_dist_threshold(
                    params_pt.get_optional<unsigned>("node_dist_threshold").value_or(10));
                
            }
        };

        if (top_order_strategy == "bfs" || top_order_strategy == "default") {
            auto coarser = std::make_unique<top_order_coarser<Graph_t_in, Graph_t_out, GetTopOrder>>();
            set_params(coarser);
            return coarser;
        } else if (top_order_strategy == "dfs") {
            auto coarser = std::make_unique<top_order_coarser<Graph_t_in, Graph_t_out, dfs_top_sort>>();
            set_params(coarser);
            return coarser;
        } else if (top_order_strategy == "locality") {
            auto coarser = std::make_unique<top_order_coarser<Graph_t_in, Graph_t_out, GetTopOrderMinIndex>>();
            set_params(coarser);
            return coarser;
        } else if (top_order_strategy == "max_children") {
            auto coarser = std::make_unique<top_order_coarser<Graph_t_in, Graph_t_out, GetTopOrderMaxChildren>>();
            set_params(coarser);
            return coarser;
        } else if (top_order_strategy == "random") {
            auto coarser = std::make_unique<top_order_coarser<Graph_t_in, Graph_t_out, GetTopOrderRandom>>();
            set_params(coarser);
            return coarser;
        } else if (top_order_strategy == "gorder") {
            auto coarser = std::make_unique<top_order_coarser<Graph_t_in, Graph_t_out, GetTopOrderGorder>>();
            set_params(coarser);
            return coarser;
        } else if (top_order_strategy == "cuthill_mckee_wavefront") {
            auto coarser =
                std::make_unique<top_order_coarser<Graph_t_in, Graph_t_out, GetTopOrderCuthillMcKeeWavefront>>();
            set_params(coarser);
            return coarser;
        } else if (top_order_strategy == "cuthill_mckee_undirected") {
            auto coarser =
                std::make_unique<top_order_coarser<Graph_t_in, Graph_t_out, GetTopOrderCuthillMcKeeUndirected>>();
            set_params(coarser);
            return coarser;
        } else {
            std::cerr << "Warning: Unknown top_order strategy '" << top_order_strategy
                      << "'. Falling back to default (bfs)." << std::endl;
            auto coarser = std::make_unique<top_order_coarser<Graph_t_in, Graph_t_out, GetTopOrder>>();
            set_params(coarser);
            return coarser;
        }

    } else if (coarser_name == "Sarkar") {
        SarkarParams::Parameters<v_workw_t<Graph_t_in>> params;
        if (auto params_opt = coarser_algorithm.get_child_optional("parameters")) {
            const auto &params_pt = params_opt.get();
            params.commCost = params_pt.get_optional<v_workw_t<Graph_t_in>>("commCost").value_or(params.commCost);
            params.useTopPoset = params_pt.get_optional<bool>("useTopPoset").value_or(params.useTopPoset);

            if (auto mode_str_opt = params_pt.get_optional<std::string>("mode")) {
                const std::string &mode_str = mode_str_opt.get();
                if (mode_str == "LINES") params.mode = SarkarParams::Mode::LINES;
                else if (mode_str == "FAN_IN_FULL") params.mode = SarkarParams::Mode::FAN_IN_FULL;
                else if (mode_str == "FAN_IN_PARTIAL") params.mode = SarkarParams::Mode::FAN_IN_PARTIAL;
                else if (mode_str == "FAN_OUT_FULL") params.mode = SarkarParams::Mode::FAN_OUT_FULL;
                else if (mode_str == "FAN_OUT_PARTIAL") params.mode = SarkarParams::Mode::FAN_OUT_PARTIAL;
                else if (mode_str == "LEVEL_EVEN") params.mode = SarkarParams::Mode::LEVEL_EVEN;
                else if (mode_str == "LEVEL_ODD") params.mode = SarkarParams::Mode::LEVEL_ODD;
                else if (mode_str == "FAN_IN_BUFFER") params.mode = SarkarParams::Mode::FAN_IN_BUFFER;
                else if (mode_str == "FAN_OUT_BUFFER") params.mode = SarkarParams::Mode::FAN_OUT_BUFFER;
                else throw std::invalid_argument("Invalid Sarkar mode: " + mode_str
                    + "!\nChoose from: LINES, FAN_IN_FULL, FAN_IN_PARTIAL, FAN_OUT_FULL, FAN_OUT_PARTIAL, LEVEL_EVEN, LEVEL_ODD, FAN_IN_BUFFER, FAN_OUT_BUFFER.");
            }
        }
        return std::make_unique<Sarkar<Graph_t_in, Graph_t_out>>(params);

    } else if (coarser_name == "SquashA") {
        SquashAParams::Parameters params;
        auto coarser = std::make_unique<SquashA<Graph_t_in, Graph_t_out>>(params);
        if (auto params_opt = coarser_algorithm.get_child_optional("parameters")) {
            const auto &params_pt = params_opt.get();
            params.use_structured_poset =
                params_pt.get_optional<bool>("use_structured_poset").value_or(params.use_structured_poset);
            params.use_top_poset = params_pt.get_optional<bool>("use_top_poset").value_or(params.use_top_poset);
            if (auto mode_str_opt = params_pt.get_optional<std::string>("mode")) {
                if (mode_str_opt.get() == "EDGE_WEIGHT") params.mode = SquashAParams::Mode::EDGE_WEIGHT;
                else if (mode_str_opt.get() == "TRIANGLES") params.mode = SquashAParams::Mode::TRIANGLES;
                else throw std::invalid_argument("Invalid Squash mode: " + mode_str_opt.get()
                    + "!\nChoose from: EDGE_WEIGHT, TRIANGLES.");
            }
        }
        coarser->setParams(params);
        return coarser;

    } else if (coarser_name == "BspScheduleCoarser") {
        // This coarser requires an initial schedule and must be handled specially by the caller.
        return nullptr;
    }

    throw std::invalid_argument("Invalid coarser name: " + coarser_name);
}

template<typename Graph_t_in, typename Graph_t_out>
std::unique_ptr<MultilevelCoarser<Graph_t_in, Graph_t_out>>
get_multilevel_coarser_by_name(const ConfigParser &, const boost::property_tree::ptree &coarser_algorithm) {
    const std::string coarser_name = coarser_algorithm.get_child("name").get_value<std::string>();

    if (coarser_name == "Sarkar") {
        auto coarser = std::make_unique<SarkarMul<Graph_t_in, Graph_t_out>>();
        SarkarParams::MulParameters<v_workw_t<Graph_t_in>> ml_params;

        if (auto params_opt = coarser_algorithm.get_child_optional("parameters")) {
            const auto &params_pt = params_opt.get();
            ml_params.seed = params_pt.get_optional<std::size_t>("seed").value_or(ml_params.seed);
            ml_params.geomDecay = params_pt.get_optional<double>("geomDecay").value_or(ml_params.geomDecay);
            ml_params.leniency = params_pt.get_optional<double>("leniency").value_or(ml_params.leniency);
            if (params_pt.get_child_optional("commCostVec")) {
                ml_params.commCostVec.clear();
                for (const auto &item : params_pt.get_child("commCostVec")) {
                    ml_params.commCostVec.push_back(item.second.get_value<v_workw_t<Graph_t_in>>());
                }
                std::sort(ml_params.commCostVec.begin(), ml_params.commCostVec.end());
            }
            ml_params.maxWeight =
                params_pt.get_optional<v_workw_t<Graph_t_in>>("maxWeight").value_or(ml_params.maxWeight);
            ml_params.max_num_iteration_without_changes =
                params_pt.get_optional<unsigned>("max_num_iteration_without_changes")
                    .value_or(ml_params.max_num_iteration_without_changes);
            ml_params.use_buffer_merge =
                params_pt.get_optional<bool>("use_buffer_merge").value_or(ml_params.use_buffer_merge);
        }

        coarser->setParameters(ml_params);
        return coarser;

    } else if (coarser_name == "SquashA") {
        auto coarser = std::make_unique<SquashAMul<Graph_t_in, Graph_t_out>>();
        SquashAParams::Parameters params;

        if (auto params_opt = coarser_algorithm.get_child_optional("parameters")) {
            const auto &params_pt = params_opt.get();
            params.geom_decay_num_nodes =
                params_pt.get_optional<double>("geom_decay_num_nodes").value_or(params.geom_decay_num_nodes);
            params.poisson_par = params_pt.get_optional<double>("poisson_par").value_or(params.poisson_par);
            params.noise = params_pt.get_optional<unsigned>("noise").value_or(params.noise);
            params.num_rep_without_node_decrease =
                params_pt.get_optional<unsigned>("num_rep_without_node_decrease")
                    .value_or(params.num_rep_without_node_decrease);
            params.temperature_multiplier =
                params_pt.get_optional<double>("temperature_multiplier").value_or(params.temperature_multiplier);
            params.number_of_temperature_increases =
                params_pt.get_optional<unsigned>("number_of_temperature_increases")
                    .value_or(params.number_of_temperature_increases);

            if (auto mode_str_opt = params_pt.get_optional<std::string>("mode")) {
                if (mode_str_opt.get() == "EDGE_WEIGHT") {
                    params.mode = SquashAParams::Mode::EDGE_WEIGHT;
                } else if (mode_str_opt.get() == "TRIANGLES") {
                    params.mode = SquashAParams::Mode::TRIANGLES;
                } else {
                    throw std::invalid_argument("Invalid Squash mode: " + mode_str_opt.get()
                    + "!\nChoose from: EDGE_WEIGHT, TRIANGLES.");
                }
            }

            coarser->setMinimumNumberVertices(params_pt.get_optional<unsigned>("min_nodes").value_or(1));
        }

        coarser->setParams(params);
        return coarser;
    }

    throw std::invalid_argument("Invalid multilevel coarser name: " + coarser_name);
}

} // namespace osp