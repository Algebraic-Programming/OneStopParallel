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
#include <memory>

#include "auxiliary/test_suite_runner/ConfigParser.hpp"
#include "bsp/model/BspSchedule.hpp"
#include "bsp/scheduler/Scheduler.hpp"
#include "coarser/Coarser.hpp"
#include "coarser/hdagg/hdagg_coarser.hpp"
#include "coarser/funnel/FunnelBfs.hpp"
#include "coarser/top_order/top_order_coarser.hpp"

namespace osp {

template<typename Graph_t>
std::unique_ptr<Coarser<Graph_t, Graph_t>> get_coarser_by_name(const ConfigParser &,
                                                               const boost::property_tree::ptree &coarser_algorithm) {

    const std::string coarser_name = coarser_algorithm.get_child("name").get_value<std::string>();

    if (coarser_name == "funnel") { 

        typename FunnelBfs<Graph_t, Graph_t>::FunnelBfs_parameters funnel_parameters;

        funnel_parameters.funnel_incoming = coarser_algorithm.get_child("parameters").get_child("funnel_incoming").get_value_optional<bool>().value_or(true);
        funnel_parameters.use_approx_transitive_reduction = coarser_algorithm.get_child("parameters").get_child("use_approx_transitive_reduction").get_value_optional<bool>().value_or(true);
       // funnel_parameters.max_depth = coarser_algorithm.get_child("parameters").get_child("max_depth").get_value_optional<unsigned>().value_or(std::numeric_limits<unsigned>::max());

        return std::make_unique<FunnelBfs<Graph_t, Graph_t>>(funnel_parameters);

    } else if (coarser_name == "hdagg") {

        auto coarser = std::make_unique<hdagg_coarser<Graph_t, Graph_t>>();

        coarser->set_work_threshold(coarser_algorithm.get_child("parameters").get_child("max_work_weight").get_value_optional<v_workw_t<Graph_t>>().value_or(std::numeric_limits<v_workw_t<Graph_t>>::max()));
        coarser->set_memory_threshold(coarser_algorithm.get_value_optional<v_memw_t<Graph_t>>().value_or(std::numeric_limits<v_memw_t<Graph_t>>::max()));
        coarser->set_communication_threshold(coarser_algorithm.get_child("parameters").get_child("max_communication_weight").get_value_optional<v_commw_t<Graph_t>>().value_or(std::numeric_limits<v_commw_t<Graph_t>>::max()));
        coarser->set_super_node_size_threshold(coarser_algorithm.get_child("parameters").get_child("max_super_node_size").get_value_optional<std::size_t>().value_or(std::numeric_limits<std::size_t>::max()));

        return coarser;

    } else if (coarser_name == "top_order") {

        auto coarser = std::make_unique<top_order_coarser<Graph_t, Graph_t,  GetTopOrder>>();

        // const auto parameters = coarser_algorithm.get_child("parameters");

        //         auto scheduler_name = algorithm.get_child("parameters").get_child("scheduler");
        //         std::string top_order_name =
        //         algorithm.get_child("parameters").get_child("top_order").get_value<std::string>(); int work_threshold
        //         = algorithm.get_child("parameters")
        //                                  .get_child("work_threshold")
        //                                  .get_value_optional<int>()
        //                                  .value_or(std::numeric_limits<int>::max());
        //         int memory_threshold = algorithm.get_child("parameters")
        //                                    .get_child("memory_threshold")
        //                                    .get_value_optional<int>()
        //                                    .value_or(std::numeric_limits<int>::max());
        //         int communication_threshold = algorithm.get_child("parameters")
        //                                           .get_child("communication_threshold")
        //                                           .get_value_optional<int>()
        //                                           .value_or(std::numeric_limits<int>::max());
        //         unsigned degree_threshold = algorithm.get_child("parameters")
        //                                         .get_child("degree_threshold")
        //                                         .get_value_optional<unsigned>()
        //                                         .value_or(std::numeric_limits<unsigned>::max());
        //         unsigned super_node_size_threshold = algorithm.get_child("parameters")
        //                                                  .get_child("super_node_size_threshold")
        //                                                  .get_value_optional<unsigned>()
        //                                                  .value_or(std::numeric_limits<unsigned>::max());

        //         std::vector<VertexType> top_ordering;

        //         if (top_order_name == "locality") {
        //             top_ordering = dag_algorithms::top_sort_locality(bsp_instance.getComputationalDag());
        //         } else if (top_order_name == "bfs") {
        //             top_ordering = dag_algorithms::top_sort_bfs(bsp_instance.getComputationalDag());
        //         } else if (top_order_name == "dfs") {
        //             top_ordering = dag_algorithms::top_sort_dfs(bsp_instance.getComputationalDag());
        //         } else if (top_order_name == "max_children") {
        //             top_ordering = dag_algorithms::top_sort_max_children(bsp_instance.getComputationalDag());
        //         } else if (top_order_name == "random") {
        //             top_ordering = dag_algorithms::top_sort_random(bsp_instance.getComputationalDag());
        //         } else if (top_order_name == "heavy_edges") {
        //             top_ordering = dag_algorithms::top_sort_heavy_edges(bsp_instance.getComputationalDag());
        //         } else if (top_order_name == "cuthill_mckee_wavefront") {
        //             top_ordering = dag_algorithms::top_sort_priority_node_type(
        //                 bsp_instance.getComputationalDag(),
        //                 dag_algorithms::cuthill_mckee_wavefront(bsp_instance.getComputationalDag()));
        //         } else if (top_order_name == "cuthill_mckee_undirected") {
        //             top_ordering = dag_algorithms::top_sort_priority_node_type(
        //                 bsp_instance.getComputationalDag(),
        //                 dag_algorithms::cuthill_mckee_undirected(bsp_instance.getComputationalDag(), true, true));
        //         } else {
        //             throw std::invalid_argument("Invalid top order name: " + top_order_name);
        //         }

        // coarser->set_work_threshold(v_workw_t<Graph_t_in> work_threshold_);
        // coarser->set_memory_threshold(v_memw_t<Graph_t_in> memory_threshold_);
        // coarser->set_communication_threshold(v_commw_t<Graph_t_in> communication_threshold_);
        // coarser->set_super_node_size_threshold(std::size_t super_node_size_threshold_);

        return coarser;

    } else {
        throw std::invalid_argument("Invalid coarser name: " + coarser_name);
    }
}

} // namespace osp