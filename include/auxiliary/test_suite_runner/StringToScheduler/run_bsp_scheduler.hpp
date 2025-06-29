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

#include "bsp/scheduler/GreedySchedulers/BspLocking.hpp"
#include "bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "bsp/scheduler/GreedySchedulers/GrowLocalAutoCores.hpp"
#include "bsp/scheduler/LocalSearch/HillClimbing/hill_climbing.hpp"
#include "bsp/scheduler/LocalSearch/KernighanLin/kl_total_comm.hpp"
#include "bsp/scheduler/LocalSearch/KernighanLin/kl_total_cut.hpp"
#include "bsp/scheduler/Serial.hpp"
#include "graph_implementations/boost_graphs/boost_graph.hpp"

// #include "scheduler/GreedySchedulers/GreedyChildren.hpp"
// #include "scheduler/GreedySchedulers/GreedyCilkScheduler.hpp"
// #include "scheduler/GreedySchedulers/GreedyEtfScheduler.hpp"
// #include "scheduler/GreedySchedulers/GreedyLayers.hpp"
// #include "scheduler/GreedySchedulers/GreedyVarianceScheduler.hpp"
// #include "scheduler/GreedySchedulers/GreedyVarianceFillupScheduler.hpp"
// #include "scheduler/GreedySchedulers/MetaGreedyScheduler.hpp"
// #include "scheduler/GreedySchedulers/RandomBadGreedy.hpp"
// #include "scheduler/GreedySchedulers/RandomGreedy.hpp"
// #include "scheduler/GreedySchedulers/GreedyBspStoneAge.hpp"
// #include "scheduler/LocalSearchSchedulers/HillClimbingScheduler.hpp"

#include "auxiliary/test_suite_runner/ConfigParser.hpp"
#include "bsp/model/BspSchedule.hpp"
#include "bsp/scheduler/Scheduler.hpp"
#include "coarser/coarser_util.hpp"
#include "get_coarser.hpp"

namespace osp {

const std::set<std::string> get_available_bsp_scheduler_names() {
    return {"Serial", "GreedyBsp", "GrowLocal", "BspLocking", "LocalSearch", "Coarser"};
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

    } else if (algorithm.get_child("name").get_value<std::string>() == "LocalSearch") {

        RETURN_STATUS status =
            run_bsp_scheduler(parser, algorithm.get_child("parameters").get_child("scheduler"), schedule);

        if (status == ERROR) {
            return ERROR;
        }
        return run_bsp_improver(parser, algorithm.get_child("parameters").get_child("improver"), schedule);

    } else if (algorithm.get_child("name").get_value<std::string>() == "Coarser") {

        using vertex_type_t_or_default = std::conditional_t<is_computational_dag_typed_vertices_v<Graph_t>, v_type_t<Graph_t>, unsigned>;
        using edge_commw_t_or_default = std::conditional_t<has_edge_weights_v<Graph_t>, e_commw_t<Graph_t>, v_commw_t<Graph_t>>;

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
            return ERROR;
        }

        instance_coarse.setArchitecture(instance.getArchitecture());
        instance_coarse.setNodeProcessorCompatibility(instance.getProcessorCompatibilityMatrix());

        BspSchedule<boost_graph_t> schedule_coarse(instance_coarse);

        const auto status_coarse =
            run_bsp_scheduler(parser, algorithm.get_child("parameters").get_child("scheduler"), schedule_coarse);

        if (status_coarse != SUCCESS and status_coarse != BEST_FOUND) {
            return status_coarse;
        }

        status = coarser_util::pull_back_schedule(schedule_coarse, reverse_vertex_map, schedule);

        if (!status) {
            return ERROR;
        }

        return SUCCESS;

    }

    //     } else if (algorithm.get_child("name").get_value<std::string>() == "GreedyCilk") {

    //         GreedyCilkScheduler scheduler;
    //         scheduler.setTimeLimitSeconds(timeLimit);

    //         algorithm.get_child("parameters").get_child("mode").get_value<std::string>() == "RANDOM"
    //             ? scheduler.setMode(CilkMode::RANDOM)
    //         : algorithm.get_child("parameters").get_child("mode").get_value<std::string>() == "SJF"
    //             ? scheduler.setMode(CilkMode::SJF)
    //             : scheduler.setMode(CilkMode::CILK);

    //         return scheduler.computeSchedule(bsp_instance);

    //     } else if (algorithm.get_child("name").get_value<std::string>() == "GreedyCilkLK") {

    //         GreedyCilkScheduler cilk_scheduler;

    //         algorithm.get_child("parameters").get_child("mode").get_value<std::string>() == "RANDOM"
    //             ? cilk_scheduler.setMode(CilkMode::RANDOM)
    //         : algorithm.get_child("parameters").get_child("mode").get_value<std::string>() == "SJF"
    //             ? cilk_scheduler.setMode(CilkMode::SJF)
    //             : cilk_scheduler.setMode(CilkMode::CILK);

    //         bool hyperedge = algorithm.get_child("parameters").get_child("hyperedge").get_value<bool>();

    //         kl_total_comm improver;

    //         ComboScheduler scheduler(cilk_scheduler, improver);

    //         scheduler.setTimeLimitSeconds(timeLimit);

    //         return scheduler.computeSchedule(bsp_instance);

    //     } else if (algorithm.get_child("name").get_value<std::string>() == "GreedyEtf") {

    //         GreedyEtfScheduler scheduler;
    //         scheduler.setTimeLimitSeconds(timeLimit);

    //         algorithm.get_child("parameters").get_child("mode").get_value<std::string>() == "BL_EST"
    //             ? scheduler.setMode(EtfMode::BL_EST)
    //             : scheduler.setMode(EtfMode::ETF);

    //         return scheduler.computeSchedule(bsp_instance);

    //     } else if (algorithm.get_child("name").get_value<std::string>() == "GreedyEtfLK") {

    //         GreedyEtfScheduler etf_scheduler;
    //         etf_scheduler.setTimeLimitSeconds(timeLimit);

    //         algorithm.get_child("parameters").get_child("mode").get_value<std::string>() == "BL_EST"
    //             ? etf_scheduler.setMode(EtfMode::BL_EST)
    //             : etf_scheduler.setMode(EtfMode::ETF);

    //         bool hyperedge = algorithm.get_child("parameters").get_child("hyperedge").get_value<bool>();

    //         kl_total_comm improver;

    //         ComboScheduler scheduler(etf_scheduler, improver);

    //         scheduler.setTimeLimitSeconds(timeLimit);

    //         return scheduler.computeSchedule(bsp_instance);

    //     } else if (algorithm.get_child("name").get_value<std::string>() == "GreedyLayers") {

    //         GreedyLayers scheduler;
    //         scheduler.setTimeLimitSeconds(timeLimit);

    //         return scheduler.computeSchedule(bsp_instance);

    //     } else if (algorithm.get_child("name").get_value<std::string>() == "GreedyRandom") {

    //         RandomGreedy scheduler;
    //         scheduler.setTimeLimitSeconds(timeLimit);

    //         return scheduler.computeSchedule(bsp_instance);

    //     } else if (algorithm.get_child("name").get_value<std::string>() == "GreedyBadRandom") {

    //         RandomBadGreedy scheduler;
    //         scheduler.setTimeLimitSeconds(timeLimit);

    //         return scheduler.computeSchedule(bsp_instance);

    //     } else if (algorithm.get_child("name").get_value<std::string>() == "GreedyChildren") {

    //         GreedyChildren scheduler;
    //         scheduler.setTimeLimitSeconds(timeLimit);

    //         return scheduler.computeSchedule(bsp_instance);

    //     } else if (algorithm.get_child("name").get_value<std::string>() == "GreedyMeta") {

    //         MetaGreedyScheduler scheduler;
    //         scheduler.setTimeLimitSeconds(timeLimit);

    //         return scheduler.computeSchedule(bsp_instance);

    //     } else if (algorithm.get_child("name").get_value<std::string>() == "MultiHC") {

    //         MultiLevelHillClimbingScheduler scheduler;
    //         scheduler.setTimeLimitSeconds(timeLimit);

    //         //        bool comp_contr_rate =
    //         // algorithm.get_child("parameters").get_child("compute_best_contraction_rate").get_value<bool>();

    //         double contraction_rate =
    //         algorithm.get_child("parameters").get_child("contraction_rate").get_value<double>(); unsigned step =
    //         algorithm.get_child("parameters").get_child("hill_climbing_steps").get_value<unsigned>(); bool
    //         fast_coarsification =
    //         algorithm.get_child("parameters").get_child("fast_coarsification").get_value<bool>();

    //         scheduler.setContractionFactor(contraction_rate);
    //         scheduler.setHcSteps(step);
    //         scheduler.setFastCoarsification(fast_coarsification);
    //         // scheduler.

    //         return scheduler.computeSchedule(bsp_instance);

    //     } else if (algorithm.get_child("name").get_value<std::string>() == "Wavefront") {

    //         unsigned hillclimb_balancer_iterations =
    //             algorithm.get_child("parameters").get_child("hillclimb_balancer_iterations").get_value<unsigned>();
    //         bool hungarian_alg = algorithm.get_child("parameters").get_child("hungarian_alg").get_value<bool>();

    //         Wavefront_parameters params(hillclimb_balancer_iterations, hungarian_alg);
    //         Wavefront scheduler(params);
    //         scheduler.setTimeLimitSeconds(timeLimit);

    //         return scheduler.computeSchedule(bsp_instance);
    //     } else if (algorithm.get_child("name").get_value<std::string>() == "CoarseWavefront") {

    //         unsigned hillclimb_balancer_iterations =
    //             algorithm.get_child("parameters").get_child("hillclimb_balancer_iterations").get_value<unsigned>();
    //         bool hungarian_alg = algorithm.get_child("parameters").get_child("hungarian_alg").get_value<bool>();

    //         Wavefront_parameters params(hillclimb_balancer_iterations, hungarian_alg);
    //         Wavefront wave_front_scheduler(params);
    //         HillClimbingScheduler wavefront_hillclimb;
    //         WavefrontCoarser wavefront_coarse_scheduler(&wave_front_scheduler);
    //         ComboScheduler scheduler(wavefront_coarse_scheduler, wavefront_hillclimb);
    //         scheduler.setTimeLimitSeconds(timeLimit);

    //         return scheduler.computeSchedule(bsp_instance);
    //     } else if (algorithm.get_child("name").get_value<std::string>() == "HDagg") {

    //         float balance_threshhold =
    //         algorithm.get_child("parameters").get_child("balance_threshhold").get_value<float>(); unsigned
    //         hillclimb_balancer_iterations =
    //             algorithm.get_child("parameters").get_child("hillclimb_balancer_iterations").get_value<unsigned>();
    //         bool hungarian_alg = algorithm.get_child("parameters").get_child("hungarian_alg").get_value<bool>();
    //         HDagg_parameters::BALANCE_FUNC balance_function =
    //             algorithm.get_child("parameters").get_child("balance_func").get_value<std::string>() == "xlogx"
    //                 ? HDagg_parameters::XLOGX
    //                 : HDagg_parameters::MAXIMUM;

    //         HDagg_parameters params(balance_threshhold, hillclimb_balancer_iterations, hungarian_alg,
    //         balance_function); HDagg_simple scheduler(params); scheduler.setTimeLimitSeconds(timeLimit);

    //         return scheduler.computeSchedule(bsp_instance);
    //     } else if (algorithm.get_child("name").get_value<std::string>() == "HDagg_original") {

    //         float balance_threshhold =
    //         algorithm.get_child("parameters").get_child("balance_threshhold").get_value<float>(); unsigned
    //         hillclimb_balancer_iterations =
    //             algorithm.get_child("parameters").get_child("hillclimb_balancer_iterations").get_value<unsigned>();
    //         bool hungarian_alg = algorithm.get_child("parameters").get_child("hungarian_alg").get_value<bool>();
    //         HDagg_parameters::BALANCE_FUNC balance_function =
    //             algorithm.get_child("parameters").get_child("balance_func").get_value<std::string>() == "xlogx"
    //                 ? HDagg_parameters::XLOGX
    //                 : HDagg_parameters::MAXIMUM;

    //         HDagg_parameters params(balance_threshhold, hillclimb_balancer_iterations, hungarian_alg,
    //         balance_function); HDagg_simple scheduler_inner(params); HDaggCoarser scheduler(&scheduler_inner);

    //         scheduler.setTimeLimitSeconds(timeLimit);

    //         return scheduler.computeSchedule(bsp_instance);
    //     } else if (algorithm.get_child("name").get_value<std::string>() == "HDagg_original_xlogx") {

    //         float balance_threshhold =
    //         algorithm.get_child("parameters").get_child("balance_threshhold").get_value<float>(); unsigned
    //         hillclimb_balancer_iterations =
    //             algorithm.get_child("parameters").get_child("hillclimb_balancer_iterations").get_value<unsigned>();
    //         bool hungarian_alg = algorithm.get_child("parameters").get_child("hungarian_alg").get_value<bool>();
    //         HDagg_parameters::BALANCE_FUNC balance_function =
    //             algorithm.get_child("parameters").get_child("balance_func").get_value<std::string>() == "xlogx"
    //                 ? HDagg_parameters::XLOGX
    //                 : HDagg_parameters::MAXIMUM;

    //         HDagg_parameters params(balance_threshhold, hillclimb_balancer_iterations, hungarian_alg,
    //         balance_function); HDagg_simple scheduler_inner(params); HDaggCoarser scheduler(&scheduler_inner);

    //         scheduler.setTimeLimitSeconds(timeLimit);

    //         return scheduler.computeSchedule(bsp_instance);
    //     } else if (algorithm.get_child("name").get_value<std::string>() == "BalDMixR") {

    //         unsigned number_of_partitions = bsp_instance.numberOfProcessors();
    //         float balance_threshhold =
    //         algorithm.get_child("parameters").get_child("balance_threshhold").get_value<float>();

    //         float nodes_per_clump =
    //         algorithm.get_child("parameters").get_child("nodes_per_clump").get_value<float>(); float
    //         nodes_per_partition =
    //             algorithm.get_child("parameters").get_child("nodes_per_partition").get_value<float>();
    //         float clumps_per_partition =
    //             algorithm.get_child("parameters").get_child("clumps_per_partition").get_value<float>();
    //         float max_weight_for_flag =
    //             algorithm.get_child("parameters").get_child("max_weight_for_flag").get_value<float>();
    //         float balanced_cut_ratio =
    //         algorithm.get_child("parameters").get_child("balanced_cut_ratio").get_value<float>(); float
    //         min_weight_for_split =
    //             algorithm.get_child("parameters").get_child("min_weight_for_split").get_value<float>();
    //         unsigned hill_climb_simple_improvement_attemps =
    //             algorithm.get_child("parameters").get_child("hill_climb_simple_improvement_attemps").get_value<unsigned>();
    //         int min_comp_generation_when_shaving =
    //             algorithm.get_child("parameters").get_child("min_comp_generation_when_shaving").get_value<int>();

    //         PartitionAlgorithm part_algo;
    //         if (algorithm.get_child("parameters").get_child("part_algo").get_value<std::string>() == "KarmarkarKarp")
    //         {
    //             part_algo = KarmarkarKarp;
    // #ifdef COPT
    //         } else if (algorithm.get_child("parameters").get_child("part_algo").get_value<std::string>() == "ILP") {
    //             part_algo = ILP;
    // #endif
    //         } else if (algorithm.get_child("parameters").get_child("part_algo").get_value<std::string>() == "Greedy")
    //         {
    //             part_algo = Greedy;
    //         } else {
    //             part_algo = Greedy;
    //         }

    //         CoinType coin_type;
    //         if (algorithm.get_child("parameters").get_child("coin_type").get_value<std::string>() == "Thue_Morse") {
    //             coin_type = Thue_Morse;
    //         } else if (algorithm.get_child("parameters").get_child("coin_type").get_value<std::string>() ==
    //                    "Biased_Randomly") {
    //             coin_type = Biased_Randomly;
    //         } else {
    //             coin_type = Thue_Morse;
    //         }

    //         Coarse_Scheduler_Params params(number_of_partitions, balance_threshhold, part_algo, coin_type,
    //                                        clumps_per_partition, nodes_per_clump, nodes_per_partition,
    //                                        max_weight_for_flag, balanced_cut_ratio, min_weight_for_split,
    //                                        hill_climb_simple_improvement_attemps, min_comp_generation_when_shaving);

    //         BalDMixR scheduler(params);
    //         scheduler.setTimeLimitSeconds(timeLimit);

    //         return scheduler.computeSchedule(bsp_instance);
    //     } else if (algorithm.get_child("name").get_value<std::string>() == "CoBalDMixR") {

    //         unsigned number_of_partitions = bsp_instance.numberOfProcessors();

    //         float geom_decay_num_nodes =
    //             algorithm.get_child("parameters").get_child("coarsen").get_child("geom_decay_num_nodes").get_value<float>();
    //         double poisson_par =
    //             algorithm.get_child("parameters").get_child("coarsen").get_child("poisson_par").get_value<double>();
    //         unsigned noise =
    //             algorithm.get_child("parameters").get_child("coarsen").get_child("noise").get_value<unsigned>();
    //         std::pair<unsigned, unsigned> edge_sort_ratio = std::make_pair(algorithm.get_child("parameters")
    //                                                                            .get_child("coarsen")
    //                                                                            .get_child("edge_sort_ratio_triangle")
    //                                                                            .get_value<unsigned>(),
    //                                                                        algorithm.get_child("parameters")
    //                                                                            .get_child("coarsen")
    //                                                                            .get_child("edge_sort_ratio_weight")
    //                                                                            .get_value<unsigned>());
    //         int num_rep_without_node_decrease = algorithm.get_child("parameters")
    //                                                 .get_child("coarsen")
    //                                                 .get_child("num_rep_without_node_decrease")
    //                                                 .get_value<int>();
    //         float temperature_multiplier = algorithm.get_child("parameters")
    //                                            .get_child("coarsen")
    //                                            .get_child("temperature_multiplier")
    //                                            .get_value<float>();
    //         float number_of_temperature_increases = algorithm.get_child("parameters")
    //                                                     .get_child("coarsen")
    //                                                     .get_child("number_of_temperature_increases")
    //                                                     .get_value<float>();

    //         CoarsenParams coarse_params(geom_decay_num_nodes, poisson_par, noise, edge_sort_ratio,
    //                                     num_rep_without_node_decrease, temperature_multiplier,
    //                                     number_of_temperature_increases);

    //         int min_nodes_after_coarsen_per_partition = algorithm.get_child("parameters")
    //                                                         .get_child("coarsen")
    //                                                         .get_child("min_nodes_after_coarsen_per_partition")
    //                                                         .get_value<int>();
    //         int number_of_final_no_change_reps = algorithm.get_child("parameters")
    //                                                  .get_child("coarsen")
    //                                                  .get_child("number_of_final_no_change_reps")
    //                                                  .get_value<int>();

    //         float initial_balance_threshhold =
    //             algorithm.get_child("parameters").get_child("initial").get_child("balance_threshhold").get_value<float>();

    //         float initial_nodes_per_clump =
    //             algorithm.get_child("parameters").get_child("initial").get_child("nodes_per_clump").get_value<float>();
    //         float initial_nodes_per_partition =
    //             algorithm.get_child("parameters").get_child("initial").get_child("nodes_per_partition").get_value<float>();
    //         float initial_clumps_per_partition =
    //             algorithm.get_child("parameters").get_child("initial").get_child("clumps_per_partition").get_value<float>();
    //         float initial_max_weight_for_flag =
    //             algorithm.get_child("parameters").get_child("initial").get_child("max_weight_for_flag").get_value<float>();
    //         float initial_balanced_cut_ratio =
    //             algorithm.get_child("parameters").get_child("initial").get_child("balanced_cut_ratio").get_value<float>();
    //         float initial_min_weight_for_split =
    //             algorithm.get_child("parameters").get_child("initial").get_child("min_weight_for_split").get_value<float>();
    //         unsigned initial_hill_climb_simple_improvement_attemps = algorithm.get_child("parameters")
    //                                                                      .get_child("initial")
    //                                                                      .get_child("hill_climb_simple_improvement_attemps")
    //                                                                      .get_value<unsigned>();
    //         int initial_min_comp_generation_when_shaving = algorithm.get_child("parameters")
    //                                                            .get_child("initial")
    //                                                            .get_child("min_comp_generation_when_shaving")
    //                                                            .get_value<int>();

    //         PartitionAlgorithm initial_part_algo;
    //         if
    //         (algorithm.get_child("parameters").get_child("initial").get_child("part_algo").get_value<std::string>()
    //         ==
    //             "KarmarkarKarp") {
    //             initial_part_algo = KarmarkarKarp;
    // #ifdef COPT
    //         } else if (algorithm.get_child("parameters")
    //                        .get_child("initial")
    //                        .get_child("part_algo")
    //                        .get_value<std::string>() == "ILP") {
    //             initial_part_algo = ILP;
    // #endif
    //         } else if (algorithm.get_child("parameters")
    //                        .get_child("initial")
    //                        .get_child("part_algo")
    //                        .get_value<std::string>() == "Greedy") {
    //             initial_part_algo = Greedy;
    //         } else {
    //             initial_part_algo = Greedy;
    //         }

    //         CoinType initial_coin_type;
    //         if
    //         (algorithm.get_child("parameters").get_child("initial").get_child("coin_type").get_value<std::string>()
    //         ==
    //             "Thue_Morse") {
    //             initial_coin_type = Thue_Morse;
    //         } else if (algorithm.get_child("parameters")
    //                        .get_child("initial")
    //                        .get_child("coin_type")
    //                        .get_value<std::string>() == "Biased_Randomly") {
    //             initial_coin_type = Biased_Randomly;
    //         } else {
    //             initial_coin_type = Thue_Morse;
    //         }

    //         Coarse_Scheduler_Params initial_params(
    //             number_of_partitions, initial_balance_threshhold, initial_part_algo, initial_coin_type,
    //             initial_clumps_per_partition, initial_nodes_per_clump, initial_nodes_per_partition,
    //             initial_max_weight_for_flag, initial_balanced_cut_ratio, initial_min_weight_for_split,
    //             initial_hill_climb_simple_improvement_attemps, initial_min_comp_generation_when_shaving);

    //         float final_balance_threshhold =
    //             algorithm.get_child("parameters").get_child("final").get_child("balance_threshhold").get_value<float>();

    //         float final_nodes_per_clump =
    //             algorithm.get_child("parameters").get_child("final").get_child("nodes_per_clump").get_value<float>();
    //         float final_nodes_per_partition =
    //             algorithm.get_child("parameters").get_child("final").get_child("nodes_per_partition").get_value<float>();
    //         float final_clumps_per_partition =
    //             algorithm.get_child("parameters").get_child("final").get_child("clumps_per_partition").get_value<float>();
    //         float final_max_weight_for_flag =
    //             algorithm.get_child("parameters").get_child("final").get_child("max_weight_for_flag").get_value<float>();
    //         float final_balanced_cut_ratio =
    //             algorithm.get_child("parameters").get_child("final").get_child("balanced_cut_ratio").get_value<float>();
    //         float final_min_weight_for_split =
    //             algorithm.get_child("parameters").get_child("final").get_child("min_weight_for_split").get_value<float>();
    //         unsigned final_hill_climb_simple_improvement_attemps = algorithm.get_child("parameters")
    //                                                                    .get_child("final")
    //                                                                    .get_child("hill_climb_simple_improvement_attemps")
    //                                                                    .get_value<unsigned>();
    //         int final_min_comp_generation_when_shaving = algorithm.get_child("parameters")
    //                                                          .get_child("final")
    //                                                          .get_child("min_comp_generation_when_shaving")
    //                                                          .get_value<int>();

    //         PartitionAlgorithm final_part_algo;
    //         if (algorithm.get_child("parameters").get_child("final").get_child("part_algo").get_value<std::string>()
    //         ==
    //             "KarmarkarKarp") {
    //             final_part_algo = KarmarkarKarp;
    // #ifdef COPT
    //         } else if (algorithm.get_child("parameters")
    //                        .get_child("final")
    //                        .get_child("part_algo")
    //                        .get_value<std::string>() == "ILP") {
    //             final_part_algo = ILP;
    // #endif
    //         } else if (algorithm.get_child("parameters")
    //                        .get_child("final")
    //                        .get_child("part_algo")
    //                        .get_value<std::string>() == "Greedy") {
    //             final_part_algo = Greedy;
    //         } else {
    //             final_part_algo = Greedy;
    //         }

    //         CoinType final_coin_type;
    //         if (algorithm.get_child("parameters").get_child("final").get_child("coin_type").get_value<std::string>()
    //         ==
    //             "Thue_Morse") {
    //             final_coin_type = Thue_Morse;
    //         } else if (algorithm.get_child("parameters")
    //                        .get_child("final")
    //                        .get_child("coin_type")
    //                        .get_value<std::string>() == "Biased_Randomly") {
    //             final_coin_type = Biased_Randomly;
    //         } else {
    //             final_coin_type = Thue_Morse;
    //         }

    //         Coarse_Scheduler_Params final_params(
    //             number_of_partitions, final_balance_threshhold, final_part_algo, final_coin_type,
    //             final_clumps_per_partition, final_nodes_per_clump, final_nodes_per_partition,
    //             final_max_weight_for_flag, final_balanced_cut_ratio, final_min_weight_for_split,
    //             final_hill_climb_simple_improvement_attemps, final_min_comp_generation_when_shaving);

    //         CoarseRefineScheduler_parameters params(initial_params, final_params, coarse_params,
    //                                                 min_nodes_after_coarsen_per_partition,
    //                                                 number_of_final_no_change_reps);

    //         CoBalDMixR scheduler(params);
    //         scheduler.setTimeLimitSeconds(timeLimit);

    //         return scheduler.computeSchedule(bsp_instance);

    //     } else if (algorithm.get_child("name").get_value<std::string>() == "BestGreedyLK") {

    //         MetaGreedyScheduler best_greedy;
    //         kl_total_comm improver;
    //         ComboScheduler scheduler(best_greedy, improver);
    //         scheduler.setTimeLimitSeconds(timeLimit);

    //         return scheduler.computeSchedule(bsp_instance);
    //     } else if (algorithm.get_child("name").get_value<std::string>() == "BestGreedyHC") {

    //         MetaGreedyScheduler best_greedy;
    //         HillClimbingScheduler hill_climbing;
    //         ComboScheduler scheduler(best_greedy, hill_climbing);
    //         scheduler.setTimeLimitSeconds(timeLimit);

    //         return scheduler.computeSchedule(bsp_instance);
    //     } else if (algorithm.get_child("name").get_value<std::string>() == "CoarseBestGreedyHC") {

    //         unsigned number_of_partitions = bsp_instance.numberOfProcessors();

    //         float geom_decay_num_nodes =
    //             algorithm.get_child("parameters").get_child("coarsen").get_child("geom_decay_num_nodes").get_value<float>();
    //         double poisson_par =
    //             algorithm.get_child("parameters").get_child("coarsen").get_child("poisson_par").get_value<double>();
    //         unsigned noise =
    //             algorithm.get_child("parameters").get_child("coarsen").get_child("noise").get_value<unsigned>();
    //         std::pair<unsigned, unsigned> edge_sort_ratio = std::make_pair(algorithm.get_child("parameters")
    //                                                                            .get_child("coarsen")
    //                                                                            .get_child("edge_sort_ratio_triangle")
    //                                                                            .get_value<unsigned>(),
    //                                                                        algorithm.get_child("parameters")
    //                                                                            .get_child("coarsen")
    //                                                                            .get_child("edge_sort_ratio_weight")
    //                                                                            .get_value<unsigned>());
    //         int num_rep_without_node_decrease = algorithm.get_child("parameters")
    //                                                 .get_child("coarsen")
    //                                                 .get_child("num_rep_without_node_decrease")
    //                                                 .get_value<int>();
    //         float temperature_multiplier = algorithm.get_child("parameters")
    //                                            .get_child("coarsen")
    //                                            .get_child("temperature_multiplier")
    //                                            .get_value<float>();
    //         float number_of_temperature_increases = algorithm.get_child("parameters")
    //                                                     .get_child("coarsen")
    //                                                     .get_child("number_of_temperature_increases")
    //                                                     .get_value<float>();

    //         CoarsenParams coarse_params(geom_decay_num_nodes, poisson_par, noise, edge_sort_ratio,
    //                                     num_rep_without_node_decrease, temperature_multiplier,
    //                                     number_of_temperature_increases);

    //         int min_nodes_after_coarsen_per_partition = algorithm.get_child("parameters")
    //                                                         .get_child("coarsen")
    //                                                         .get_child("min_nodes_after_coarsen_per_partition")
    //                                                         .get_value<int>();

    //         MetaGreedyScheduler best_greedy;
    //         HillClimbingScheduler hill_climbing;
    //         SquashA scheduler(&best_greedy, &hill_climbing, coarse_params,
    //                           min_nodes_after_coarsen_per_partition * number_of_partitions);
    //         scheduler.setTimeLimitSeconds(timeLimit);

    //         return scheduler.computeSchedule(bsp_instance);

    //     } else if (algorithm.get_child("name").get_value<std::string>() == "SquashBspGreedyHC") {

    //         unsigned number_of_partitions = bsp_instance.numberOfProcessors();

    //         float geom_decay_num_nodes =
    //             algorithm.get_child("parameters").get_child("coarsen").get_child("geom_decay_num_nodes").get_value<float>();
    //         double poisson_par =
    //             algorithm.get_child("parameters").get_child("coarsen").get_child("poisson_par").get_value<double>();
    //         unsigned noise =
    //             algorithm.get_child("parameters").get_child("coarsen").get_child("noise").get_value<unsigned>();
    //         std::pair<unsigned, unsigned> edge_sort_ratio = std::make_pair(algorithm.get_child("parameters")
    //                                                                            .get_child("coarsen")
    //                                                                            .get_child("edge_sort_ratio_triangle")
    //                                                                            .get_value<unsigned>(),
    //                                                                        algorithm.get_child("parameters")
    //                                                                            .get_child("coarsen")
    //                                                                            .get_child("edge_sort_ratio_weight")
    //                                                                            .get_value<unsigned>());
    //         int num_rep_without_node_decrease = algorithm.get_child("parameters")
    //                                                 .get_child("coarsen")
    //                                                 .get_child("num_rep_without_node_decrease")
    //                                                 .get_value<int>();
    //         float temperature_multiplier = algorithm.get_child("parameters")
    //                                            .get_child("coarsen")
    //                                            .get_child("temperature_multiplier")
    //                                            .get_value<float>();
    //         float number_of_temperature_increases = algorithm.get_child("parameters")
    //                                                     .get_child("coarsen")
    //                                                     .get_child("number_of_temperature_increases")
    //                                                     .get_value<float>();

    //         CoarsenParams coarse_params(geom_decay_num_nodes, poisson_par, noise, edge_sort_ratio,
    //                                     num_rep_without_node_decrease, temperature_multiplier,
    //                                     number_of_temperature_increases);

    //         int min_nodes_after_coarsen_per_partition = algorithm.get_child("parameters")
    //                                                         .get_child("coarsen")
    //                                                         .get_child("min_nodes_after_coarsen_per_partition")
    //                                                         .get_value<int>();

    //         GreedyBspScheduler greedy;
    //         HillClimbingScheduler hill_climbing;
    //         SquashA scheduler(&greedy, &hill_climbing, coarse_params,
    //                           min_nodes_after_coarsen_per_partition * number_of_partitions);
    //         scheduler.setTimeLimitSeconds(timeLimit);

    //         return scheduler.computeSchedule(bsp_instance);

    //     } else if (algorithm.get_child("name").get_value<std::string>() == "SquashBspGreedyLK") {

    //         unsigned number_of_partitions = bsp_instance.numberOfProcessors();

    //         float geom_decay_num_nodes =
    //             algorithm.get_child("parameters").get_child("coarsen").get_child("geom_decay_num_nodes").get_value<float>();
    //         double poisson_par =
    //             algorithm.get_child("parameters").get_child("coarsen").get_child("poisson_par").get_value<double>();
    //         unsigned noise =
    //             algorithm.get_child("parameters").get_child("coarsen").get_child("noise").get_value<unsigned>();
    //         std::pair<unsigned, unsigned> edge_sort_ratio = std::make_pair(algorithm.get_child("parameters")
    //                                                                            .get_child("coarsen")
    //                                                                            .get_child("edge_sort_ratio_triangle")
    //                                                                            .get_value<unsigned>(),
    //                                                                        algorithm.get_child("parameters")
    //                                                                            .get_child("coarsen")
    //                                                                            .get_child("edge_sort_ratio_weight")
    //                                                                            .get_value<unsigned>());
    //         int num_rep_without_node_decrease = algorithm.get_child("parameters")
    //                                                 .get_child("coarsen")
    //                                                 .get_child("num_rep_without_node_decrease")
    //                                                 .get_value<int>();
    //         float temperature_multiplier = algorithm.get_child("parameters")
    //                                            .get_child("coarsen")
    //                                            .get_child("temperature_multiplier")
    //                                            .get_value<float>();
    //         float number_of_temperature_increases = algorithm.get_child("parameters")
    //                                                     .get_child("coarsen")
    //                                                     .get_child("number_of_temperature_increases")
    //                                                     .get_value<float>();

    //         CoarsenParams coarse_params(geom_decay_num_nodes, poisson_par, noise, edge_sort_ratio,
    //                                     num_rep_without_node_decrease, temperature_multiplier,
    //                                     number_of_temperature_increases);

    //         int min_nodes_after_coarsen_per_partition = algorithm.get_child("parameters")
    //                                                         .get_child("coarsen")
    //                                                         .get_child("min_nodes_after_coarsen_per_partition")
    //                                                         .get_value<int>();

    //         GreedyBspScheduler greedy;
    //         kl_total_comm hill_climbing;
    //         SquashA scheduler(&greedy, &hill_climbing, coarse_params,
    //                           min_nodes_after_coarsen_per_partition * number_of_partitions);
    //         scheduler.setTimeLimitSeconds(timeLimit);

    //         return scheduler.computeSchedule(bsp_instance);

    //     } else if (algorithm.get_child("name").get_value<std::string>() == "SquashBspGreedy") {

    //         unsigned number_of_partitions = bsp_instance.numberOfProcessors();

    //         float geom_decay_num_nodes =
    //             algorithm.get_child("parameters").get_child("coarsen").get_child("geom_decay_num_nodes").get_value<float>();
    //         double poisson_par =
    //             algorithm.get_child("parameters").get_child("coarsen").get_child("poisson_par").get_value<double>();
    //         unsigned noise =
    //             algorithm.get_child("parameters").get_child("coarsen").get_child("noise").get_value<unsigned>();
    //         std::pair<unsigned, unsigned> edge_sort_ratio = std::make_pair(algorithm.get_child("parameters")
    //                                                                            .get_child("coarsen")
    //                                                                            .get_child("edge_sort_ratio_triangle")
    //                                                                            .get_value<unsigned>(),
    //                                                                        algorithm.get_child("parameters")
    //                                                                            .get_child("coarsen")
    //                                                                            .get_child("edge_sort_ratio_weight")
    //                                                                            .get_value<unsigned>());
    //         int num_rep_without_node_decrease = algorithm.get_child("parameters")
    //                                                 .get_child("coarsen")
    //                                                 .get_child("num_rep_without_node_decrease")
    //                                                 .get_value<int>();
    //         float temperature_multiplier = algorithm.get_child("parameters")
    //                                            .get_child("coarsen")
    //                                            .get_child("temperature_multiplier")
    //                                            .get_value<float>();
    //         float number_of_temperature_increases = algorithm.get_child("parameters")
    //                                                     .get_child("coarsen")
    //                                                     .get_child("number_of_temperature_increases")
    //                                                     .get_value<float>();

    //         CoarsenParams coarse_params(geom_decay_num_nodes, poisson_par, noise, edge_sort_ratio,
    //                                     num_rep_without_node_decrease, temperature_multiplier,
    //                                     number_of_temperature_increases);

    //         int min_nodes_after_coarsen_per_partition = algorithm.get_child("parameters")
    //                                                         .get_child("coarsen")
    //                                                         .get_child("min_nodes_after_coarsen_per_partition")
    //                                                         .get_value<int>();

    //         GreedyBspScheduler greedy;

    //         SquashA scheduler(&greedy, coarse_params, min_nodes_after_coarsen_per_partition * number_of_partitions);
    //         scheduler.setTimeLimitSeconds(timeLimit);

    //         return scheduler.computeSchedule(bsp_instance);

    //     } else if (algorithm.get_child("name").get_value<std::string>() == "CoarseBestGreedy") {

    //         unsigned number_of_partitions = bsp_instance.numberOfProcessors();

    //         float geom_decay_num_nodes =
    //             algorithm.get_child("parameters").get_child("coarsen").get_child("geom_decay_num_nodes").get_value<float>();
    //         double poisson_par =
    //             algorithm.get_child("parameters").get_child("coarsen").get_child("poisson_par").get_value<double>();
    //         unsigned noise =
    //             algorithm.get_child("parameters").get_child("coarsen").get_child("noise").get_value<unsigned>();
    //         std::pair<unsigned, unsigned> edge_sort_ratio = std::make_pair(algorithm.get_child("parameters")
    //                                                                            .get_child("coarsen")
    //                                                                            .get_child("edge_sort_ratio_triangle")
    //                                                                            .get_value<unsigned>(),
    //                                                                        algorithm.get_child("parameters")
    //                                                                            .get_child("coarsen")
    //                                                                            .get_child("edge_sort_ratio_weight")
    //                                                                            .get_value<unsigned>());
    //         int num_rep_without_node_decrease = algorithm.get_child("parameters")
    //                                                 .get_child("coarsen")
    //                                                 .get_child("num_rep_without_node_decrease")
    //                                                 .get_value<int>();
    //         float temperature_multiplier = algorithm.get_child("parameters")
    //                                            .get_child("coarsen")
    //                                            .get_child("temperature_multiplier")
    //                                            .get_value<float>();
    //         float number_of_temperature_increases = algorithm.get_child("parameters")
    //                                                     .get_child("coarsen")
    //                                                     .get_child("number_of_temperature_increases")
    //                                                     .get_value<float>();

    //         CoarsenParams coarse_params(geom_decay_num_nodes, poisson_par, noise, edge_sort_ratio,
    //                                     num_rep_without_node_decrease, temperature_multiplier,
    //                                     number_of_temperature_increases);

    //         int min_nodes_after_coarsen_per_partition = algorithm.get_child("parameters")
    //                                                         .get_child("coarsen")
    //                                                         .get_child("min_nodes_after_coarsen_per_partition")
    //                                                         .get_value<int>();

    //         MetaGreedyScheduler best_greedy;
    //         SquashA scheduler(&best_greedy, coarse_params, min_nodes_after_coarsen_per_partition *
    //         number_of_partitions); scheduler.setTimeLimitSeconds(timeLimit);

    //         return scheduler.computeSchedule(bsp_instance);
    //     } else if (algorithm.get_child("name").get_value<std::string>() == "SquashHDagg") {

    //         unsigned number_of_partitions = bsp_instance.numberOfProcessors();

    //         float geom_decay_num_nodes =
    //             algorithm.get_child("parameters").get_child("coarsen").get_child("geom_decay_num_nodes").get_value<float>();
    //         double poisson_par =
    //             algorithm.get_child("parameters").get_child("coarsen").get_child("poisson_par").get_value<double>();
    //         unsigned noise =
    //             algorithm.get_child("parameters").get_child("coarsen").get_child("noise").get_value<unsigned>();
    //         std::pair<unsigned, unsigned> edge_sort_ratio = std::make_pair(algorithm.get_child("parameters")
    //                                                                            .get_child("coarsen")
    //                                                                            .get_child("edge_sort_ratio_triangle")
    //                                                                            .get_value<unsigned>(),
    //                                                                        algorithm.get_child("parameters")
    //                                                                            .get_child("coarsen")
    //                                                                            .get_child("edge_sort_ratio_weight")
    //                                                                            .get_value<unsigned>());
    //         int num_rep_without_node_decrease = algorithm.get_child("parameters")
    //                                                 .get_child("coarsen")
    //                                                 .get_child("num_rep_without_node_decrease")
    //                                                 .get_value<int>();
    //         float temperature_multiplier = algorithm.get_child("parameters")
    //                                            .get_child("coarsen")
    //                                            .get_child("temperature_multiplier")
    //                                            .get_value<float>();
    //         float number_of_temperature_increases = algorithm.get_child("parameters")
    //                                                     .get_child("coarsen")
    //                                                     .get_child("number_of_temperature_increases")
    //                                                     .get_value<float>();

    //         CoarsenParams coarse_params(geom_decay_num_nodes, poisson_par, noise, edge_sort_ratio,
    //                                     num_rep_without_node_decrease, temperature_multiplier,
    //                                     number_of_temperature_increases);

    //         int min_nodes_after_coarsen_per_partition = algorithm.get_child("parameters")
    //                                                         .get_child("coarsen")
    //                                                         .get_child("min_nodes_after_coarsen_per_partition")
    //                                                         .get_value<int>();

    //         float balance_threshhold =
    //             algorithm.get_child("parameters").get_child("HDagg").get_child("balance_threshhold").get_value<float>();
    //         unsigned hillclimb_balancer_iterations = algorithm.get_child("parameters")
    //                                                      .get_child("HDagg")
    //                                                      .get_child("hillclimb_balancer_iterations")
    //                                                      .get_value<unsigned>();
    //         bool hungarian_alg =
    //             algorithm.get_child("parameters").get_child("HDagg").get_child("hungarian_alg").get_value<bool>();
    //         HDagg_parameters::BALANCE_FUNC balance_function =
    //             algorithm.get_child("parameters").get_child("HDagg").get_child("balance_func").get_value<std::string>()
    //             ==
    //                     "xlogx"
    //                 ? HDagg_parameters::XLOGX
    //                 : HDagg_parameters::MAXIMUM;

    //         HDagg_parameters params(balance_threshhold, hillclimb_balancer_iterations, hungarian_alg,
    //         balance_function);

    //         HDagg_simple hdagg_scheduler(params);
    //         SquashA scheduler(&hdagg_scheduler, coarse_params,
    //                           min_nodes_after_coarsen_per_partition * number_of_partitions);

    //         scheduler.setTimeLimitSeconds(timeLimit);

    //         return scheduler.computeSchedule(bsp_instance);
    //     }
    else {

        throw std::invalid_argument("Parameter error: Unknown algorithm.\n");
    }
};

} // namespace osp