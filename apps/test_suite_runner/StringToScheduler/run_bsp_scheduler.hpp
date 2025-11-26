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
#include "get_coarser.hpp"
#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/scheduler/CoarseAndSchedule.hpp"
#include "osp/bsp/scheduler/CoarsenRefineSchedulers/MultiLevelHillClimbing.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/BspLocking.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/CilkScheduler.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/EtfScheduler.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyChildren.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GrowLocalAutoCores.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/RandomGreedy.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/VarianceFillup.hpp"
#include "osp/bsp/scheduler/ImprovementScheduler.hpp"
#include "osp/bsp/scheduler/LocalSearch/HillClimbing/hill_climbing.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin_v2/kl_include.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin_v2/kl_include_mt.hpp"
#include "osp/bsp/scheduler/MultilevelCoarseAndSchedule.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/bsp/scheduler/Serial.hpp"
#include "osp/coarser/MultilevelCoarser.hpp"
#include "osp/coarser/coarser_util.hpp"
#include "osp/graph_implementations/boost_graphs/boost_graph.hpp"

#ifdef COPT
#include "osp/bsp/scheduler/IlpSchedulers/CoptFullScheduler.hpp"
// #include "osp/bsp/scheduler/IlpSchedulers/TotalCommunicationScheduler.hpp"
#endif

namespace osp {

const std::set<std::string> get_available_bsp_scheduler_names() {
    return {"Serial",         "GreedyBsp", "GrowLocal", "BspLocking",  "Cilk",    "Etf",     "GreedyRandom",
            "GreedyChildren", "Variance",  "MultiHC",   "LocalSearch", "Coarser", "FullILP", "MultiLevel"};
    ;
}

template<typename Graph_t>
std::unique_ptr<ImprovementScheduler<Graph_t>> get_bsp_improver_by_name(const ConfigParser &,
                                                                        const boost::property_tree::ptree &algorithm) {
    const std::string improver_name = algorithm.get_child("name").get_value<std::string>();

    if (improver_name == "kl_total_comm") {
        return std::make_unique<kl_total_comm_improver_mt<Graph_t>>();
    } else if (improver_name == "kl_total_lambda_comm") {
        return std::make_unique<kl_total_lambda_comm_improver_mt<Graph_t>>();
    } else if (improver_name == "hill_climb") {
        return std::make_unique<HillClimbingScheduler<Graph_t>>();
    }

    throw std::invalid_argument("Invalid improver name: " + improver_name);
}

template<typename Graph_t>
std::unique_ptr<Scheduler<Graph_t>> get_base_bsp_scheduler_by_name(const ConfigParser &parser,
                                                                   const boost::property_tree::ptree &algorithm) {

    const std::string id = algorithm.get_child("id").get_value<std::string>();

    if (id == "Serial") {
        auto scheduler = std::make_unique<Serial<Graph_t>>();
        return scheduler;

    } else if (id == "GreedyBsp") {
        float max_percent_idle_processors =
            algorithm.get_child("parameters").get_child("max_percent_idle_processors").get_value<float>();
        bool increase_parallelism_in_new_superstep =
            algorithm.get_child("parameters").get_child("increase_parallelism_in_new_superstep").get_value<bool>();
        auto scheduler = std::make_unique<GreedyBspScheduler<Graph_t>>(max_percent_idle_processors,
                                                                       increase_parallelism_in_new_superstep);

        return scheduler;

    } else if (id == "GrowLocal") {
        GrowLocalAutoCores_Params<v_workw_t<Graph_t>> params;
        params.minSuperstepSize = algorithm.get_child("parameters").get_child("minSuperstepSize").get_value<unsigned>();
        params.syncCostMultiplierMinSuperstepWeight = algorithm.get_child("parameters")
                                                          .get_child("syncCostMultiplierMinSuperstepWeight")
                                                          .get_value<v_workw_t<Graph_t>>();
        params.syncCostMultiplierParallelCheck = algorithm.get_child("parameters")
                                                     .get_child("syncCostMultiplierParallelCheck")
                                                     .get_value<v_workw_t<Graph_t>>();

        return std::make_unique<GrowLocalAutoCores<Graph_t>>(params);

    } else if (id == "BspLocking") {
        float max_percent_idle_processors =
            algorithm.get_child("parameters").get_child("max_percent_idle_processors").get_value<float>();
        bool increase_parallelism_in_new_superstep =
            algorithm.get_child("parameters").get_child("increase_parallelism_in_new_superstep").get_value<bool>();
        auto scheduler =
            std::make_unique<BspLocking<Graph_t>>(max_percent_idle_processors, increase_parallelism_in_new_superstep);

        return scheduler;

    } else if (id == "Cilk") {
        auto scheduler = std::make_unique<CilkScheduler<Graph_t>>();
        algorithm.get_child("parameters").get_child("mode").get_value<std::string>() == "SJF"
            ? scheduler->setMode(CilkMode::SJF)
            : scheduler->setMode(CilkMode::CILK);
        return scheduler;

    } else if (id == "Etf") {
        auto scheduler = std::make_unique<EtfScheduler<Graph_t>>();
        algorithm.get_child("parameters").get_child("mode").get_value<std::string>() == "BL_EST"
            ? scheduler->setMode(EtfMode::BL_EST)
            : scheduler->setMode(EtfMode::ETF);
        return scheduler;

    } else if (id == "GreedyRandom") {
        auto scheduler = std::make_unique<RandomGreedy<Graph_t>>();
        return scheduler;

    } else if (id == "GreedyChildren") {
        auto scheduler = std::make_unique<GreedyChildren<Graph_t>>();
        return scheduler;

    } else if (id == "Variance") {
        float max_percent_idle_processors =
            algorithm.get_child("parameters").get_child("max_percent_idle_processors").get_value<float>();
        bool increase_parallelism_in_new_superstep =
            algorithm.get_child("parameters").get_child("increase_parallelism_in_new_superstep").get_value<bool>();
        auto scheduler = std::make_unique<VarianceFillup<Graph_t>>(max_percent_idle_processors,
                                                                   increase_parallelism_in_new_superstep);

        return scheduler;
    }

    if constexpr (is_constructable_cdag_v<Graph_t> || is_direct_constructable_cdag_v<Graph_t>) {
        if (id == "MultiHC") {
            auto scheduler = std::make_unique<MultiLevelHillClimbingScheduler<Graph_t>>();
            const unsigned timeLimit = parser.global_params.get_child("timeLimit").get_value<unsigned>();
            // scheduler->setTimeLimitSeconds(timeLimit);

            unsigned step = algorithm.get_child("parameters").get_child("hill_climbing_steps").get_value<unsigned>();
            scheduler->setNumberOfHcSteps(step);

            const double contraction_rate =
                algorithm.get_child("parameters").get_child("contraction_rate").get_value<double>();
            scheduler->setContractionRate(contraction_rate);
            scheduler->useLinearRefinementSteps(20U);
            scheduler->setMinTargetNrOfNodes(100U);
            return scheduler;
        }
    }

    throw std::invalid_argument("Invalid base scheduler name: " + id);
}

template<typename Graph_t>
RETURN_STATUS run_bsp_scheduler(const ConfigParser &parser, const boost::property_tree::ptree &algorithm,
                                BspSchedule<Graph_t> &schedule) {

    using vertex_type_t_or_default =
        std::conditional_t<is_computational_dag_typed_vertices_v<Graph_t>, v_type_t<Graph_t>, unsigned>;
    using edge_commw_t_or_default =
        std::conditional_t<has_edge_weights_v<Graph_t>, e_commw_t<Graph_t>, v_commw_t<Graph_t>>;
    using boost_graph_t = boost_graph<v_workw_t<Graph_t>, v_commw_t<Graph_t>, v_memw_t<Graph_t>,
                                      vertex_type_t_or_default, edge_commw_t_or_default>;

    const std::string id = algorithm.get_child("id").get_value<std::string>();

    std::cout << "Running algorithm: " << id << std::endl;

    if (id == "LocalSearch") {
        RETURN_STATUS status =
            run_bsp_scheduler(parser, algorithm.get_child("parameters").get_child("scheduler"), schedule);
        if (status == RETURN_STATUS::ERROR)
            return RETURN_STATUS::ERROR;

        std::unique_ptr<ImprovementScheduler<Graph_t>> improver =
            get_bsp_improver_by_name<Graph_t>(parser, algorithm.get_child("parameters").get_child("improver"));
        return improver->improveSchedule(schedule);
#ifdef COPT
    } else if (id == "FullILP") {
        CoptFullScheduler<Graph_t> scheduler;
        const unsigned timeLimit = parser.global_params.get_child("timeLimit").get_value<unsigned>();

        // max supersteps
        scheduler.setMaxNumberOfSupersteps(
            algorithm.get_child("parameters").get_child("max_number_of_supersteps").get_value<unsigned>());

        // initial solution
        if (algorithm.get_child("parameters").get_child("use_initial_solution").get_value<bool>()) {
            std::string init_sched =
                algorithm.get_child("parameters").get_child("initial_solution_scheduler").get_value<std::string>();
            if (init_sched == "FullILP") {
                throw std::invalid_argument("Parameter error: Initial solution cannot be FullILP.\n");
            }

            BspSchedule<Graph_t> initial_schedule(schedule.getInstance());

            RETURN_STATUS status = run_bsp_scheduler(
                parser, algorithm.get_child("parameters").get_child("initial_solution_scheduler"), initial_schedule);

            if (status != RETURN_STATUS::OSP_SUCCESS && status != RETURN_STATUS::BEST_FOUND) {
                throw std::invalid_argument("Error while computing initial solution.\n");
            }
            BspScheduleCS<Graph_t> initial_schedule_cs(initial_schedule);
            scheduler.setInitialSolutionFromBspSchedule(initial_schedule_cs);
        }

        // intermediate solutions
        if (algorithm.get_child("parameters").get_child("write_intermediate_solutions").get_value<bool>()) {
            scheduler.enableWriteIntermediateSol(
                algorithm.get_child("parameters")
                    .get_child("intermediate_solutions_directory")
                    .get_value<std::string>(),
                algorithm.get_child("parameters").get_child("intermediate_solutions_prefix").get_value<std::string>());
        }

        return scheduler.computeScheduleWithTimeLimit(schedule, timeLimit);
#endif
    } else if (id == "Coarser") {
        std::unique_ptr<Coarser<Graph_t, boost_graph_t>> coarser =
            get_coarser_by_name<Graph_t, boost_graph_t>(parser, algorithm.get_child("parameters").get_child("coarser"));
        const auto &instance = schedule.getInstance();
        BspInstance<boost_graph_t> instance_coarse;
        std::vector<vertex_idx_t<boost_graph_t>> reverse_vertex_map;
        bool status = coarser->coarsenDag(instance.getComputationalDag(), instance_coarse.getComputationalDag(),
                                          reverse_vertex_map);
        if (!status)
            return RETURN_STATUS::ERROR;

        instance_coarse.setArchitecture(instance.getArchitecture());
        instance_coarse.setNodeProcessorCompatibility(instance.getProcessorCompatibilityMatrix());
        BspSchedule<boost_graph_t> schedule_coarse(instance_coarse);

        const auto status_coarse =
            run_bsp_scheduler(parser, algorithm.get_child("parameters").get_child("scheduler"), schedule_coarse);
        if (status_coarse != RETURN_STATUS::OSP_SUCCESS and status_coarse != RETURN_STATUS::BEST_FOUND)
            return status_coarse;

        status = coarser_util::pull_back_schedule(schedule_coarse, reverse_vertex_map, schedule);
        if (!status)
            return RETURN_STATUS::ERROR;

        return RETURN_STATUS::OSP_SUCCESS;

    } else if (id == "MultiLevel") {
        std::unique_ptr<MultilevelCoarser<Graph_t, boost_graph_t>> ml_coarser =
            get_multilevel_coarser_by_name<Graph_t, boost_graph_t>(
                parser, algorithm.get_child("parameters").get_child("coarser"));
        std::unique_ptr<ImprovementScheduler<boost_graph_t>> improver =
            get_bsp_improver_by_name<boost_graph_t>(parser, algorithm.get_child("parameters").get_child("improver"));
        std::unique_ptr<Scheduler<boost_graph_t>> scheduler = get_base_bsp_scheduler_by_name<boost_graph_t>(
            parser, algorithm.get_child("parameters").get_child("scheduler"));

        MultilevelCoarseAndSchedule<Graph_t, boost_graph_t> coarse_and_schedule(*scheduler, *improver, *ml_coarser);
        return coarse_and_schedule.computeSchedule(schedule);
    } else {
        auto scheduler = get_base_bsp_scheduler_by_name<Graph_t>(parser, algorithm);
        return scheduler->computeSchedule(schedule);
    }
}

} // namespace osp