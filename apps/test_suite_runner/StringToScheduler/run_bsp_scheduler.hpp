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

#include "../ConfigParser.hpp"
#include "bsp/model/BspSchedule.hpp"
#include "bsp/scheduler/CoarsenRefineSchedulers/MultiLevelHillClimbing.hpp"
#include "bsp/scheduler/GreedySchedulers/BspLocking.hpp"
#include "bsp/scheduler/GreedySchedulers/CilkScheduler.hpp"
#include "bsp/scheduler/GreedySchedulers/EtfScheduler.hpp"
#include "bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "bsp/scheduler/GreedySchedulers/GreedyChildren.hpp"
#include "bsp/scheduler/GreedySchedulers/GrowLocalAutoCores.hpp"
#include "bsp/scheduler/GreedySchedulers/RandomGreedy.hpp"
#include "bsp/scheduler/GreedySchedulers/VarianceFillup.hpp"
#include "bsp/scheduler/LocalSearch/HillClimbing/hill_climbing.hpp"
#include "bsp/scheduler/LocalSearch/KernighanLin/kl_total_comm.hpp"
#include "bsp/scheduler/LocalSearch/KernighanLin/kl_total_cut.hpp"
#include "bsp/scheduler/Scheduler.hpp"
#include "bsp/scheduler/Serial.hpp"
#include "coarser/coarser_util.hpp"
#include "get_coarser.hpp"
#include "graph_implementations/boost_graphs/boost_graph.hpp"

namespace osp {

const std::set<std::string> get_available_bsp_scheduler_names() {
    return {"Serial",       "GreedyBsp",      "GrowLocal", "BspLocking", "Cilk",        "Etf",
            "GreedyRandom", "GreedyChildren", "Variance",  "MultiHC",    "LocalSearch", "Coarser"};
}

template<typename Graph_t>
RETURN_STATUS run_bsp_improver(const ConfigParser &, const boost::property_tree::ptree &algorithm,
                               BspSchedule<Graph_t> &schedule) {

    const std::string improver_name = algorithm.get_child("name").get_value<std::string>();

    if (improver_name == "kl_total_comm") {

        kl_total_comm<Graph_t> improver;
        return improver.improveSchedule(schedule);

    } else if (improver_name == "kl_total_cut") {

        kl_total_cut<Graph_t> improver;
        return improver.improveSchedule(schedule);
    } else if (improver_name == "hill_climb") {

        HillClimbingScheduler<Graph_t> improver;
        return improver.improveSchedule(schedule);
    }

    throw std::invalid_argument("Invalid improver name: " + improver_name);
}

template<typename Graph_t>
RETURN_STATUS run_bsp_scheduler(const ConfigParser &parser, const boost::property_tree::ptree &algorithm,
                                BspSchedule<Graph_t> &schedule) {

    const unsigned timeLimit = parser.global_params.get_child("timeLimit").get_value<unsigned>();

    std::cout << "Running algorithm: " << algorithm.get_child("name").get_value<std::string>() << std::endl;

    if (algorithm.get_child("name").get_value<std::string>() == "Serial") {

        Serial<Graph_t> scheduler;
        scheduler.setTimeLimitSeconds(timeLimit);
        return scheduler.computeSchedule(schedule);

    } else if (algorithm.get_child("name").get_value<std::string>() == "GreedyBsp") {

        float max_percent_idle_processors =
            algorithm.get_child("parameters").get_child("max_percent_idle_processors").get_value<float>();
        bool increase_parallelism_in_new_superstep =
            algorithm.get_child("parameters").get_child("increase_parallelism_in_new_superstep").get_value<bool>();

        GreedyBspScheduler<Graph_t> scheduler(max_percent_idle_processors, increase_parallelism_in_new_superstep);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(schedule);

    } else if (algorithm.get_child("name").get_value<std::string>() == "GrowLocal") {

        GrowLocalAutoCores_Params<v_workw_t<Graph_t>> params;

        params.minSuperstepSize = algorithm.get_child("parameters").get_child("minSuperstepSize").get_value<unsigned>();
        params.syncCostMultiplierMinSuperstepWeight = algorithm.get_child("parameters")
                                                          .get_child("syncCostMultiplierMinSuperstepWeight")
                                                          .get_value<v_workw_t<Graph_t>>();
        params.syncCostMultiplierParallelCheck = algorithm.get_child("parameters")
                                                     .get_child("syncCostMultiplierParallelCheck")
                                                     .get_value<v_workw_t<Graph_t>>();

        GrowLocalAutoCores<Graph_t> scheduler(params);

        return scheduler.computeSchedule(schedule);
    } else if (algorithm.get_child("name").get_value<std::string>() == "BspLocking") {

        float max_percent_idle_processors =
            algorithm.get_child("parameters").get_child("max_percent_idle_processors").get_value<float>();
        bool increase_parallelism_in_new_superstep =
            algorithm.get_child("parameters").get_child("increase_parallelism_in_new_superstep").get_value<bool>();

        BspLocking<Graph_t> scheduler(max_percent_idle_processors, increase_parallelism_in_new_superstep);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(schedule);

    } else if (algorithm.get_child("name").get_value<std::string>() == "Cilk") {

        CilkScheduler<Graph_t> scheduler;
        scheduler.setTimeLimitSeconds(timeLimit);

        algorithm.get_child("parameters").get_child("mode").get_value<std::string>() == "SJF"
            ? scheduler.setMode(CilkMode::SJF)
            : scheduler.setMode(CilkMode::CILK);

        return scheduler.computeSchedule(schedule);

    } else if (algorithm.get_child("name").get_value<std::string>() == "Etf") {

        EtfScheduler<Graph_t> scheduler;
        scheduler.setTimeLimitSeconds(timeLimit);

        algorithm.get_child("parameters").get_child("mode").get_value<std::string>() == "BL_EST"
            ? scheduler.setMode(EtfMode::BL_EST)
            : scheduler.setMode(EtfMode::ETF);

        return scheduler.computeSchedule(schedule);

    } else if (algorithm.get_child("name").get_value<std::string>() == "GreedyRandom") {

        RandomGreedy<Graph_t> scheduler;
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(schedule);

    } else if (algorithm.get_child("name").get_value<std::string>() == "GreedyChildren") {

        GreedyChildren<Graph_t> scheduler;
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(schedule);

    } else if (algorithm.get_child("name").get_value<std::string>() == "Variance") {

        float max_percent_idle_processors =
            algorithm.get_child("parameters").get_child("max_percent_idle_processors").get_value<float>();
        bool increase_parallelism_in_new_superstep =
            algorithm.get_child("parameters").get_child("increase_parallelism_in_new_superstep").get_value<bool>();

        VarianceFillup<Graph_t> scheduler(max_percent_idle_processors, increase_parallelism_in_new_superstep);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(schedule);

    } else if (algorithm.get_child("name").get_value<std::string>() == "LocalSearch") {

        RETURN_STATUS status =
            run_bsp_scheduler(parser, algorithm.get_child("parameters").get_child("scheduler"), schedule);

        if (status == RETURN_STATUS::ERROR) {
            return RETURN_STATUS::ERROR;
        }
        return run_bsp_improver(parser, algorithm.get_child("parameters").get_child("improver"), schedule);

    } else if (algorithm.get_child("name").get_value<std::string>() == "Coarser") {

        using vertex_type_t_or_default =
            std::conditional_t<is_computational_dag_typed_vertices_v<Graph_t>, v_type_t<Graph_t>, unsigned>;
        using edge_commw_t_or_default =
            std::conditional_t<has_edge_weights_v<Graph_t>, e_commw_t<Graph_t>, v_commw_t<Graph_t>>;

        using boost_graph_t = boost_graph<v_workw_t<Graph_t>, v_commw_t<Graph_t>, v_memw_t<Graph_t>,
                                          vertex_type_t_or_default, edge_commw_t_or_default>;

        std::unique_ptr<Coarser<Graph_t, boost_graph_t>> coarser =
            get_coarser_by_name<Graph_t, boost_graph_t>(parser, algorithm.get_child("parameters").get_child("coarser"));

        const auto &instance = schedule.getInstance();

        BspInstance<boost_graph_t> instance_coarse;

        std::vector<vertex_idx_t<boost_graph_t>> reverse_vertex_map;

        bool status = coarser->coarsenDag(instance.getComputationalDag(), instance_coarse.getComputationalDag(),
                                          reverse_vertex_map);

        if (!status) {
            return RETURN_STATUS::ERROR;
        }

        instance_coarse.setArchitecture(instance.getArchitecture());
        instance_coarse.setNodeProcessorCompatibility(instance.getProcessorCompatibilityMatrix());

        BspSchedule<boost_graph_t> schedule_coarse(instance_coarse);

        const auto status_coarse =
            run_bsp_scheduler(parser, algorithm.get_child("parameters").get_child("scheduler"), schedule_coarse);

        if (status_coarse != RETURN_STATUS::OSP_SUCCESS and status_coarse != RETURN_STATUS::BEST_FOUND) {
            return status_coarse;
        }

        status = coarser_util::pull_back_schedule(schedule_coarse, reverse_vertex_map, schedule);

        if (!status) {
            return RETURN_STATUS::ERROR;
        }

        return RETURN_STATUS::OSP_SUCCESS;
    }

    if constexpr (is_constructable_cdag_v<Graph_t> || is_direct_constructable_cdag_v<Graph_t>) {

        if (algorithm.get_child("name").get_value<std::string>() == "MultiHC") {

            MultiLevelHillClimbingScheduler<Graph_t> scheduler;
            scheduler.setTimeLimitSeconds(timeLimit);

            unsigned step = algorithm.get_child("parameters").get_child("hill_climbing_steps").get_value<unsigned>();

            scheduler.setNumberOfHcSteps(step);

            return scheduler.computeSchedule(schedule);
        }        
    }

    throw std::invalid_argument("Parameter error: Unknown algorithm.\n");
};

} // namespace osp