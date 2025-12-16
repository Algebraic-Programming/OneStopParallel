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
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_include.hpp"
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_include_mt.hpp"
#include "osp/bsp/scheduler/MultilevelCoarseAndSchedule.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/bsp/scheduler/Serial.hpp"
#include "osp/coarser/MultilevelCoarser.hpp"
#include "osp/coarser/coarser_util.hpp"
#include "osp/graph_implementations/boost_graphs/boost_graph.hpp"

#ifdef COPT
#    include "osp/bsp/scheduler/IlpSchedulers/CoptFullScheduler.hpp"
// #include "osp/bsp/scheduler/IlpSchedulers/TotalCommunicationScheduler.hpp"
#endif

namespace osp {

const std::set<std::string> GetAvailableBspSchedulerNames() {
    return {"Serial",
            "GreedyBsp",
            "GrowLocal",
            "BspLocking",
            "Cilk",
            "Etf",
            "GreedyRandom",
            "GreedyChildren",
            "Variance",
            "MultiHC",
            "LocalSearch",
            "Coarser",
            "FullILP",
            "MultiLevel"};
}

template <typename GraphT>
std::unique_ptr<ImprovementScheduler<GraphT>> GetBspImproverByName(const ConfigParser &,
                                                                   const boost::property_tree::ptree &algorithm) {
    const std::string improverName = algorithm.get_child("name").get_value<std::string>();

    if (improverName == "kl_total_comm") {
        return std::make_unique<KlTotalCommImproverMt<GraphT>>();
    } else if (improverName == "kl_total_lambda_comm") {
        return std::make_unique<KlTotalLambdaCommImproverMt<GraphT>>();
    } else if (improverName == "hill_climb") {
        return std::make_unique<HillClimbingScheduler<GraphT>>();
    }

    throw std::invalid_argument("Invalid improver name: " + improverName);
}

template <typename GraphT>
std::unique_ptr<Scheduler<GraphT>> GetBaseBspSchedulerByName(const ConfigParser &parser,
                                                             const boost::property_tree::ptree &algorithm) {
    const std::string id = algorithm.get_child("id").get_value<std::string>();

    if (id == "Serial") {
        auto scheduler = std::make_unique<Serial<GraphT>>();
        return scheduler;

    } else if (id == "GreedyBsp") {
        float maxPercentIdleProcessors
            = algorithm.get_child("parameters").get_child("max_percent_idle_processors").get_value<float>();
        bool increaseParallelismInNewSuperstep
            = algorithm.get_child("parameters").get_child("increase_parallelism_in_new_superstep").get_value<bool>();
        auto scheduler = std::make_unique<GreedyBspScheduler<GraphT>>(maxPercentIdleProcessors, increaseParallelismInNewSuperstep);

        return scheduler;

    } else if (id == "GrowLocal") {
        GrowLocalAutoCoresParams<VWorkwT<GraphT>> params;
        params.minSuperstepSize_ = algorithm.get_child("parameters").get_child("minSuperstepSize").get_value<unsigned>();
        params.syncCostMultiplierMinSuperstepWeight_
            = algorithm.get_child("parameters").get_child("syncCostMultiplierMinSuperstepWeight").get_value<VWorkwT<GraphT>>();
        params.syncCostMultiplierParallelCheck_
            = algorithm.get_child("parameters").get_child("syncCostMultiplierParallelCheck").get_value<VWorkwT<GraphT>>();

        return std::make_unique<GrowLocalAutoCores<GraphT>>(params);

    } else if (id == "BspLocking") {
        float maxPercentIdleProcessors
            = algorithm.get_child("parameters").get_child("max_percent_idle_processors").get_value<float>();
        bool increaseParallelismInNewSuperstep
            = algorithm.get_child("parameters").get_child("increase_parallelism_in_new_superstep").get_value<bool>();
        auto scheduler = std::make_unique<BspLocking<GraphT>>(maxPercentIdleProcessors, increaseParallelismInNewSuperstep);

        return scheduler;

    } else if (id == "Cilk") {
        auto scheduler = std::make_unique<CilkScheduler<GraphT>>();
        algorithm.get_child("parameters").get_child("mode").get_value<std::string>() == "SJF" ? scheduler->SetMode(CilkMode::SJF)
                                                                                              : scheduler->SetMode(CilkMode::CILK);
        return scheduler;

    } else if (id == "Etf") {
        auto scheduler = std::make_unique<EtfScheduler<GraphT>>();
        algorithm.get_child("parameters").get_child("mode").get_value<std::string>() == "BL_EST"
            ? scheduler->SetMode(EtfMode::BL_EST)
            : scheduler->SetMode(EtfMode::ETF);
        return scheduler;

    } else if (id == "GreedyRandom") {
        auto scheduler = std::make_unique<RandomGreedy<GraphT>>();
        return scheduler;

    } else if (id == "GreedyChildren") {
        auto scheduler = std::make_unique<GreedyChildren<GraphT>>();
        return scheduler;

    } else if (id == "Variance") {
        float maxPercentIdleProcessors
            = algorithm.get_child("parameters").get_child("max_percent_idle_processors").get_value<float>();
        bool increaseParallelismInNewSuperstep
            = algorithm.get_child("parameters").get_child("increase_parallelism_in_new_superstep").get_value<bool>();
        auto scheduler = std::make_unique<VarianceFillup<GraphT>>(maxPercentIdleProcessors, increaseParallelismInNewSuperstep);

        return scheduler;
    }

    if constexpr (IsConstructableCdagV<GraphT> || IsDirectConstructableCdagV<GraphT>) {
        if (id == "MultiHC") {
            auto scheduler = std::make_unique<MultiLevelHillClimbingScheduler<GraphT>>();
            const unsigned timeLimit = parser.globalParams_.get_child("timeLimit").get_value<unsigned>();

            unsigned step = algorithm.get_child("parameters").get_child("hill_climbing_steps").get_value<unsigned>();
            scheduler->SetNumberOfHcSteps(step);

            const double contractionRate = algorithm.get_child("parameters").get_child("contraction_rate").get_value<double>();
            scheduler->SetContractionRate(contractionRate);
            scheduler->UseLinearRefinementSteps(20U);
            scheduler->SetMinTargetNrOfNodes(100U);
            return scheduler;
        }
    }

    throw std::invalid_argument("Invalid base scheduler name: " + id);
}

template <typename GraphT>
ReturnStatus RunBspScheduler(const ConfigParser &parser,
                             const boost::property_tree::ptree &algorithm,
                             BspSchedule<GraphT> &schedule) {
    using VertexTypeTOrDefault = std::conditional_t<IsComputationalDagTypedVerticesV<GraphT>, VTypeT<GraphT>, unsigned>;
    using EdgeCommwTOrDefault = std::conditional_t<HasEdgeWeightsV<GraphT>, ECommwT<GraphT>, VCommwT<GraphT>>;
    using boost_graph_t
        = BoostGraph<VWorkwT<GraphT>, VCommwT<GraphT>, VMemwT<GraphT>, VertexTypeTOrDefault, EdgeCommwTOrDefault>;

    const std::string id = algorithm.get_child("id").get_value<std::string>();

    std::cout << "Running algorithm: " << id << std::endl;

    if (id == "LocalSearch") {
        ReturnStatus status = RunBspScheduler(parser, algorithm.get_child("parameters").get_child("scheduler"), schedule);
        if (status == ReturnStatus::ERROR) {
            return ReturnStatus::ERROR;
        }

        std::unique_ptr<ImprovementScheduler<GraphT>> improver
            = GetBspImproverByName<GraphT>(parser, algorithm.get_child("parameters").get_child("improver"));
        return improver->ImproveSchedule(schedule);
#ifdef COPT
    } else if (id == "FullILP") {
        CoptFullScheduler<GraphT> scheduler;
        const unsigned timeLimit = parser.globalParams_.get_child("timeLimit").get_value<unsigned>();

        // max supersteps
        scheduler.SetMaxNumberOfSupersteps(
            algorithm.get_child("parameters").get_child("max_number_of_supersteps").get_value<unsigned>());

        // initial solution
        if (algorithm.get_child("parameters").get_child("use_initial_solution").get_value<bool>()) {
            std::string initSched
                = algorithm.get_child("parameters").get_child("initial_solution_scheduler").get_value<std::string>();
            if (initSched == "FullILP") {
                throw std::invalid_argument("Parameter error: Initial solution cannot be FullILP.\n");
            }

            BspSchedule<GraphT> initialSchedule(schedule.GetInstance());

            ReturnStatus status = RunBspScheduler(
                parser, algorithm.get_child("parameters").get_child("initial_solution_scheduler"), initialSchedule);

            if (status != ReturnStatus::OSP_SUCCESS && status != ReturnStatus::BEST_FOUND) {
                throw std::invalid_argument("Error while computing initial solution.\n");
            }
            BspScheduleCS<GraphT> initialScheduleCs(initialSchedule);
            scheduler.SetInitialSolutionFromBspSchedule(initialScheduleCs);
        }

        // intermediate solutions
        if (algorithm.get_child("parameters").get_child("write_intermediate_solutions").get_value<bool>()) {
            scheduler.EnableWriteIntermediateSol(
                algorithm.get_child("parameters").get_child("intermediate_solutions_directory").get_value<std::string>(),
                algorithm.get_child("parameters").get_child("intermediate_solutions_prefix").get_value<std::string>());
        }

        return scheduler.ComputeScheduleWithTimeLimit(schedule, timeLimit);
#endif
    } else if (id == "Coarser") {
        std::unique_ptr<Coarser<GraphT, boost_graph_t>> coarser
            = GetCoarserByName<GraphT, boost_graph_t>(parser, algorithm.get_child("parameters").get_child("coarser"));
        const auto &instance = schedule.GetInstance();
        BspInstance<boost_graph_t> instanceCoarse;
        std::vector<VertexIdxT<boost_graph_t>> reverseVertexMap;
        bool status = coarser->CoarsenDag(instance.GetComputationalDag(), instanceCoarse.GetComputationalDag(), reverseVertexMap);
        if (!status) {
            return ReturnStatus::ERROR;
        }

        instanceCoarse.GetArchitecture() = instance.GetArchitecture();
        instanceCoarse.SetNodeProcessorCompatibility(instance.GetProcessorCompatibilityMatrix());
        BspSchedule<boost_graph_t> scheduleCoarse(instanceCoarse);

        const auto statusCoarse = RunBspScheduler(parser, algorithm.get_child("parameters").get_child("scheduler"), scheduleCoarse);
        if (statusCoarse != ReturnStatus::OSP_SUCCESS and statusCoarse != ReturnStatus::BEST_FOUND) {
            return statusCoarse;
        }

        status = coarser_util::PullBackSchedule(scheduleCoarse, reverseVertexMap, schedule);
        if (!status) {
            return ReturnStatus::ERROR;
        }

        return ReturnStatus::OSP_SUCCESS;

    } else if (id == "MultiLevel") {
        std::unique_ptr<MultilevelCoarser<GraphT, boost_graph_t>> mlCoarser
            = GetMultilevelCoarserByName<GraphT, boost_graph_t>(parser, algorithm.get_child("parameters").get_child("coarser"));
        std::unique_ptr<ImprovementScheduler<boost_graph_t>> improver
            = GetBspImproverByName<boost_graph_t>(parser, algorithm.get_child("parameters").get_child("improver"));
        std::unique_ptr<Scheduler<boost_graph_t>> scheduler
            = GetBaseBspSchedulerByName<boost_graph_t>(parser, algorithm.get_child("parameters").get_child("scheduler"));

        MultilevelCoarseAndSchedule<GraphT, boost_graph_t> coarseAndSchedule(*scheduler, *improver, *mlCoarser);
        return coarseAndSchedule.ComputeSchedule(schedule);
    } else {
        auto scheduler = GetBaseBspSchedulerByName<GraphT>(parser, algorithm);
        return scheduler->ComputeSchedule(schedule);
    }
}

}    // namespace osp
