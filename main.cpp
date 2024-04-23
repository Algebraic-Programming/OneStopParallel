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

#include <boost/log/utility/setup.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/log/trivial.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <filesystem>
#include <iostream>
#include <string>
#include <tuple>

#include "algorithms/GreedySchedulers/GreedyBspScheduler.hpp"
#include "algorithms/GreedySchedulers/GreedyChildren.hpp"
#include "algorithms/GreedySchedulers/GreedyCilkScheduler.hpp"
#include "algorithms/GreedySchedulers/GreedyEtfScheduler.hpp"
#include "algorithms/GreedySchedulers/GreedyLayers.hpp"
#include "algorithms/GreedySchedulers/GreedyVarianceScheduler.hpp"
#include "algorithms/GreedySchedulers/MetaGreedyScheduler.hpp"
#include "algorithms/GreedySchedulers/RandomBadGreedy.hpp"
#include "algorithms/GreedySchedulers/RandomGreedy.hpp"

#include "algorithms/ContractRefineScheduler/BalDMixR.hpp"
#include "algorithms/ContractRefineScheduler/CoBalDMixR.hpp"
#include "algorithms/ContractRefineScheduler/MultiLevelHillClimbing.hpp"

#include "algorithms/HDagg/HDagg_simple.hpp"
#include "file_interactions/BspScheduleWriter.hpp"
#include "algorithms/LocalSearchSchedulers/HillClimbingScheduler.hpp"

#include "algorithms/Coarsers/SquashA.hpp"

#include "file_interactions/CommandLineParser.hpp"
#include "file_interactions/FileReader.hpp"
#include "model/BspSchedule.hpp"

namespace pt = boost::property_tree;

std::pair<RETURN_STATUS, BspSchedule> run_algorithm(const CommandLineParser &parser, const pt::ptree &algorithm,
                                                    const BspInstance &bsp_instance, unsigned timeLimit) {

    std::cout << "Running algorithm: " << algorithm.get_child("name").get_value<std::string>() << std::endl;

    if (algorithm.get_child("name").get_value<std::string>() == "GreedyBsp") {

        GreedyBspScheduler scheduler;
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);

    } else if (algorithm.get_child("name").get_value<std::string>() == "GreedyVariance") {

        GreedyVarianceScheduler scheduler;
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);

    } else if (algorithm.get_child("name").get_value<std::string>() == "GreedyCilk") {

        GreedyCilkScheduler scheduler;
        scheduler.setTimeLimitSeconds(timeLimit);

        algorithm.get_child("parameters").get_child("mode").get_value<std::string>() == "RANDOM"
            ? scheduler.setMode(CilkMode::RANDOM)
        : algorithm.get_child("parameters").get_child("mode").get_value<std::string>() == "SJF"
            ? scheduler.setMode(CilkMode::SJF)
            : scheduler.setMode(CilkMode::CILK);

        return scheduler.computeSchedule(bsp_instance);

    } else if (algorithm.get_child("name").get_value<std::string>() == "GreedyEtf") {

        GreedyEtfScheduler scheduler;
        scheduler.setTimeLimitSeconds(timeLimit);

        algorithm.get_child("parameters").get_child("mode").get_value<std::string>() == "BL_EST"
            ? scheduler.setMode(EtfMode::BL_EST)
            : scheduler.setMode(EtfMode::ETF);

        return scheduler.computeSchedule(bsp_instance);

    } else if (algorithm.get_child("name").get_value<std::string>() == "GreedyLayers") {

        GreedyLayers scheduler;
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);

    } else if (algorithm.get_child("name").get_value<std::string>() == "GreedyRandom") {

        RandomGreedy scheduler;
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);

    } else if (algorithm.get_child("name").get_value<std::string>() == "GreedyBadRandom") {

        RandomBadGreedy scheduler;
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);

    } else if (algorithm.get_child("name").get_value<std::string>() == "GreedyChildren") {

        GreedyChildren scheduler;
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);

    } else if (algorithm.get_child("name").get_value<std::string>() == "GreedyMeta") {

        MetaGreedyScheduler scheduler;
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);

    } else if (algorithm.get_child("name").get_value<std::string>() == "MultiHC") {

        MultiLevelHillClimbingScheduler scheduler;
        scheduler.setTimeLimitSeconds(timeLimit);

        //        bool comp_contr_rate =
        // algorithm.get_child("parameters").get_child("compute_best_contraction_rate").get_value<bool>();

        double contraction_rate = algorithm.get_child("parameters").get_child("contraction_rate").get_value<double>();
        unsigned step = algorithm.get_child("parameters").get_child("hill_climbing_steps").get_value<unsigned>();
        bool fast_coarsification = algorithm.get_child("parameters").get_child("fast_coarsification").get_value<bool>();

        scheduler.setContractionFactor(contraction_rate);
        scheduler.setHcSteps(step);
        scheduler.setFastCoarsification(fast_coarsification);
        // scheduler.

        return scheduler.computeSchedule(bsp_instance);

    } else if (algorithm.get_child("name").get_value<std::string>() == "HDagg") {

        float balance_threshhold = algorithm.get_child("parameters").get_child("balance_threshhold").get_value<float>();
        unsigned hillclimb_balancer_iterations =
            algorithm.get_child("parameters").get_child("hillclimb_balancer_iterations").get_value<unsigned>();
        bool hungarian_alg = algorithm.get_child("parameters").get_child("hungarian_alg").get_value<bool>();

        HDagg_parameters params(balance_threshhold, hillclimb_balancer_iterations, hungarian_alg);
        HDagg_simple scheduler(params);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    } else if (algorithm.get_child("name").get_value<std::string>() == "BalDMixR") {

        unsigned number_of_partitions = bsp_instance.numberOfProcessors();
        float balance_threshhold = algorithm.get_child("parameters").get_child("balance_threshhold").get_value<float>();

        float nodes_per_clump = algorithm.get_child("parameters").get_child("nodes_per_clump").get_value<float>();
        float nodes_per_partition =
            algorithm.get_child("parameters").get_child("nodes_per_partition").get_value<float>();
        float clumps_per_partition =
            algorithm.get_child("parameters").get_child("clumps_per_partition").get_value<float>();
        float max_weight_for_flag =
            algorithm.get_child("parameters").get_child("max_weight_for_flag").get_value<float>();
        float balanced_cut_ratio = algorithm.get_child("parameters").get_child("balanced_cut_ratio").get_value<float>();
        float min_weight_for_split =
            algorithm.get_child("parameters").get_child("min_weight_for_split").get_value<float>();
        unsigned hill_climb_simple_improvement_attemps =
            algorithm.get_child("parameters").get_child("hill_climb_simple_improvement_attemps").get_value<unsigned>();
        int min_comp_generation_when_shaving =
            algorithm.get_child("parameters").get_child("min_comp_generation_when_shaving").get_value<int>();

        PartitionAlgorithm part_algo;
        if (algorithm.get_child("parameters").get_child("part_algo").get_value<std::string>() == "KarmarkarKarp") {
            part_algo = KarmarkarKarp;
        } else if (algorithm.get_child("parameters").get_child("part_algo").get_value<std::string>() == "ILP") {
            part_algo = ILP;
        } else if (algorithm.get_child("parameters").get_child("part_algo").get_value<std::string>() == "Greedy") {
            part_algo = Greedy;
        } else {
            part_algo = Greedy;
        }

        CoinType coin_type;
        if (algorithm.get_child("parameters").get_child("coin_type").get_value<std::string>() == "Thue_Morse") {
            coin_type = Thue_Morse;
        } else if (algorithm.get_child("parameters").get_child("coin_type").get_value<std::string>() ==
                   "Biased_Randomly") {
            coin_type = Biased_Randomly;
        } else {
            coin_type = Thue_Morse;
        }

        Coarse_Scheduler_Params params(number_of_partitions, balance_threshhold, part_algo, coin_type,
                                       clumps_per_partition, nodes_per_clump, nodes_per_partition, max_weight_for_flag,
                                       balanced_cut_ratio, min_weight_for_split, hill_climb_simple_improvement_attemps,
                                       min_comp_generation_when_shaving);

        BalDMixR scheduler(params);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    } else if (algorithm.get_child("name").get_value<std::string>() == "CoBalDMixR") {

        unsigned number_of_partitions = bsp_instance.numberOfProcessors();

        float geom_decay_num_nodes =
            algorithm.get_child("parameters").get_child("coarsen").get_child("geom_decay_num_nodes").get_value<float>();
        double poisson_par =
            algorithm.get_child("parameters").get_child("coarsen").get_child("poisson_par").get_value<double>();
        unsigned noise =
            algorithm.get_child("parameters").get_child("coarsen").get_child("noise").get_value<unsigned>();
        std::pair<unsigned, unsigned> edge_sort_ratio = std::make_pair(algorithm.get_child("parameters")
                                                                           .get_child("coarsen")
                                                                           .get_child("edge_sort_ratio_triangle")
                                                                           .get_value<unsigned>(),
                                                                       algorithm.get_child("parameters")
                                                                           .get_child("coarsen")
                                                                           .get_child("edge_sort_ratio_weight")
                                                                           .get_value<unsigned>());
        int num_rep_without_node_decrease = algorithm.get_child("parameters")
                                                .get_child("coarsen")
                                                .get_child("num_rep_without_node_decrease")
                                                .get_value<int>();
        float temperature_multiplier = algorithm.get_child("parameters")
                                           .get_child("coarsen")
                                           .get_child("temperature_multiplier")
                                           .get_value<float>();
        float number_of_temperature_increases = algorithm.get_child("parameters")
                                                    .get_child("coarsen")
                                                    .get_child("number_of_temperature_increases")
                                                    .get_value<float>();

        CoarsenParams coarse_params(geom_decay_num_nodes, poisson_par, noise, edge_sort_ratio,
                                    num_rep_without_node_decrease, temperature_multiplier,
                                    number_of_temperature_increases);

        int min_nodes_after_coarsen_per_partition = algorithm.get_child("parameters")
                                                        .get_child("coarsen")
                                                        .get_child("min_nodes_after_coarsen_per_partition")
                                                        .get_value<int>();
        int number_of_final_no_change_reps = algorithm.get_child("parameters")
                                                 .get_child("coarsen")
                                                 .get_child("number_of_final_no_change_reps")
                                                 .get_value<int>();

        float initial_balance_threshhold =
            algorithm.get_child("parameters").get_child("initial").get_child("balance_threshhold").get_value<float>();

        float initial_nodes_per_clump =
            algorithm.get_child("parameters").get_child("initial").get_child("nodes_per_clump").get_value<float>();
        float initial_nodes_per_partition =
            algorithm.get_child("parameters").get_child("initial").get_child("nodes_per_partition").get_value<float>();
        float initial_clumps_per_partition =
            algorithm.get_child("parameters").get_child("initial").get_child("clumps_per_partition").get_value<float>();
        float initial_max_weight_for_flag =
            algorithm.get_child("parameters").get_child("initial").get_child("max_weight_for_flag").get_value<float>();
        float initial_balanced_cut_ratio =
            algorithm.get_child("parameters").get_child("initial").get_child("balanced_cut_ratio").get_value<float>();
        float initial_min_weight_for_split =
            algorithm.get_child("parameters").get_child("initial").get_child("min_weight_for_split").get_value<float>();
        unsigned initial_hill_climb_simple_improvement_attemps = algorithm.get_child("parameters")
                                                                     .get_child("initial")
                                                                     .get_child("hill_climb_simple_improvement_attemps")
                                                                     .get_value<unsigned>();
        int initial_min_comp_generation_when_shaving = algorithm.get_child("parameters")
                                                           .get_child("initial")
                                                           .get_child("min_comp_generation_when_shaving")
                                                           .get_value<int>();

        PartitionAlgorithm initial_part_algo;
        if (algorithm.get_child("parameters").get_child("initial").get_child("part_algo").get_value<std::string>() ==
            "KarmarkarKarp") {
            initial_part_algo = KarmarkarKarp;
        } else if (algorithm.get_child("parameters")
                       .get_child("initial")
                       .get_child("part_algo")
                       .get_value<std::string>() == "ILP") {
            initial_part_algo = ILP;
        } else if (algorithm.get_child("parameters")
                       .get_child("initial")
                       .get_child("part_algo")
                       .get_value<std::string>() == "Greedy") {
            initial_part_algo = Greedy;
        } else {
            initial_part_algo = Greedy;
        }

        CoinType initial_coin_type;
        if (algorithm.get_child("parameters").get_child("initial").get_child("coin_type").get_value<std::string>() ==
            "Thue_Morse") {
            initial_coin_type = Thue_Morse;
        } else if (algorithm.get_child("parameters")
                       .get_child("initial")
                       .get_child("coin_type")
                       .get_value<std::string>() == "Biased_Randomly") {
            initial_coin_type = Biased_Randomly;
        } else {
            initial_coin_type = Thue_Morse;
        }

        Coarse_Scheduler_Params initial_params(
            number_of_partitions, initial_balance_threshhold, initial_part_algo, initial_coin_type,
            initial_clumps_per_partition, initial_nodes_per_clump, initial_nodes_per_partition,
            initial_max_weight_for_flag, initial_balanced_cut_ratio, initial_min_weight_for_split,
            initial_hill_climb_simple_improvement_attemps, initial_min_comp_generation_when_shaving);

        float final_balance_threshhold =
            algorithm.get_child("parameters").get_child("final").get_child("balance_threshhold").get_value<float>();

        float final_nodes_per_clump =
            algorithm.get_child("parameters").get_child("final").get_child("nodes_per_clump").get_value<float>();
        float final_nodes_per_partition =
            algorithm.get_child("parameters").get_child("final").get_child("nodes_per_partition").get_value<float>();
        float final_clumps_per_partition =
            algorithm.get_child("parameters").get_child("final").get_child("clumps_per_partition").get_value<float>();
        float final_max_weight_for_flag =
            algorithm.get_child("parameters").get_child("final").get_child("max_weight_for_flag").get_value<float>();
        float final_balanced_cut_ratio =
            algorithm.get_child("parameters").get_child("final").get_child("balanced_cut_ratio").get_value<float>();
        float final_min_weight_for_split =
            algorithm.get_child("parameters").get_child("final").get_child("min_weight_for_split").get_value<float>();
        unsigned final_hill_climb_simple_improvement_attemps = algorithm.get_child("parameters")
                                                                   .get_child("final")
                                                                   .get_child("hill_climb_simple_improvement_attemps")
                                                                   .get_value<unsigned>();
        int final_min_comp_generation_when_shaving = algorithm.get_child("parameters")
                                                         .get_child("final")
                                                         .get_child("min_comp_generation_when_shaving")
                                                         .get_value<int>();

        PartitionAlgorithm final_part_algo;
        if (algorithm.get_child("parameters").get_child("final").get_child("part_algo").get_value<std::string>() ==
            "KarmarkarKarp") {
            final_part_algo = KarmarkarKarp;
        } else if (algorithm.get_child("parameters")
                       .get_child("final")
                       .get_child("part_algo")
                       .get_value<std::string>() == "ILP") {
            final_part_algo = ILP;
        } else if (algorithm.get_child("parameters")
                       .get_child("final")
                       .get_child("part_algo")
                       .get_value<std::string>() == "Greedy") {
            final_part_algo = Greedy;
        } else {
            final_part_algo = Greedy;
        }

        CoinType final_coin_type;
        if (algorithm.get_child("parameters").get_child("final").get_child("coin_type").get_value<std::string>() ==
            "Thue_Morse") {
            final_coin_type = Thue_Morse;
        } else if (algorithm.get_child("parameters")
                       .get_child("final")
                       .get_child("coin_type")
                       .get_value<std::string>() == "Biased_Randomly") {
            final_coin_type = Biased_Randomly;
        } else {
            final_coin_type = Thue_Morse;
        }

        Coarse_Scheduler_Params final_params(
            number_of_partitions, final_balance_threshhold, final_part_algo, final_coin_type,
            final_clumps_per_partition, final_nodes_per_clump, final_nodes_per_partition, final_max_weight_for_flag,
            final_balanced_cut_ratio, final_min_weight_for_split, final_hill_climb_simple_improvement_attemps,
            final_min_comp_generation_when_shaving);

        CoarseRefineScheduler_parameters params(initial_params, final_params, coarse_params,
                                                min_nodes_after_coarsen_per_partition, number_of_final_no_change_reps);

        CoBalDMixR scheduler(params);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    } else if (algorithm.get_child("name").get_value<std::string>() == "CoarseBestGreedyHC") {

        unsigned number_of_partitions = bsp_instance.numberOfProcessors();

        float geom_decay_num_nodes =
            algorithm.get_child("parameters").get_child("coarsen").get_child("geom_decay_num_nodes").get_value<float>();
        double poisson_par =
            algorithm.get_child("parameters").get_child("coarsen").get_child("poisson_par").get_value<double>();
        unsigned noise =
            algorithm.get_child("parameters").get_child("coarsen").get_child("noise").get_value<unsigned>();
        std::pair<unsigned, unsigned> edge_sort_ratio = std::make_pair(algorithm.get_child("parameters")
                                                                           .get_child("coarsen")
                                                                           .get_child("edge_sort_ratio_triangle")
                                                                           .get_value<unsigned>(),
                                                                       algorithm.get_child("parameters")
                                                                           .get_child("coarsen")
                                                                           .get_child("edge_sort_ratio_weight")
                                                                           .get_value<unsigned>());
        int num_rep_without_node_decrease = algorithm.get_child("parameters")
                                                .get_child("coarsen")
                                                .get_child("num_rep_without_node_decrease")
                                                .get_value<int>();
        float temperature_multiplier = algorithm.get_child("parameters")
                                           .get_child("coarsen")
                                           .get_child("temperature_multiplier")
                                           .get_value<float>();
        float number_of_temperature_increases = algorithm.get_child("parameters")
                                                    .get_child("coarsen")
                                                    .get_child("number_of_temperature_increases")
                                                    .get_value<float>();

        CoarsenParams coarse_params(geom_decay_num_nodes, poisson_par, noise, edge_sort_ratio,
                                    num_rep_without_node_decrease, temperature_multiplier,
                                    number_of_temperature_increases);

        int min_nodes_after_coarsen_per_partition = algorithm.get_child("parameters")
                                                        .get_child("coarsen")
                                                        .get_child("min_nodes_after_coarsen_per_partition")
                                                        .get_value<int>();

        MetaGreedyScheduler best_greedy;
        HillClimbingScheduler hill_climbing;
        SquashA scheduler(&best_greedy, &hill_climbing, coarse_params,
                          min_nodes_after_coarsen_per_partition * number_of_partitions);
        scheduler.setTimeLimitSeconds(timeLimit);

        return scheduler.computeSchedule(bsp_instance);
    }

    else {

        throw std::invalid_argument("Parameter error: Unknown algorithm.\n");
    }
}

std::filesystem::path getExecutablePath() { return std::filesystem::canonical("/proc/self/exe"); }

// invoked upon program call
int main(int argc, char *argv[]) {

    std::string main_config_location = getExecutablePath().remove_filename().string();
    main_config_location += "main_config.json";

    try {
        const CommandLineParser parser(argc, argv, main_config_location);

        for (auto &instance : parser.instances) {
            try {
                std::string filename_graph = instance.second.get_child("graphFile").get_value<std::string>();
                std::string name_graph = filename_graph.substr(
                    filename_graph.rfind("/") + 1, filename_graph.rfind(".") - filename_graph.rfind("/") - 1);

                std::string filename_machine = instance.second.get_child("machineParamsFile").get_value<std::string>();
                std::string name_machine = filename_machine.substr(
                    filename_machine.rfind("/") + 1, filename_machine.rfind(".") - filename_machine.rfind("/") - 1);

                // std::cout << name_graph << " - " << name_machine << std::endl;

                std::pair<bool, ComputationalDag> read_graph(false, ComputationalDag());
                if (filename_graph.substr(filename_graph.rfind(".") + 1) == "txt") {
                    read_graph = FileReader::readComputationalDagHyperdagFormat(filename_graph);
                } else if (filename_graph.substr(filename_graph.rfind(".") + 1) == "mtx") {
                    read_graph = FileReader::readComputationalDagMartixMarketFormat(filename_graph);
                } else if (filename_graph.substr(filename_graph.rfind(".") + 1) == "dot") {
                    read_graph = FileReader::readComputationalDagDotFormat(filename_graph);
                } else {
                    std::cout << "Unknown file ending: ." << filename_graph.substr(filename_graph.rfind(".") + 1)
                              << " ...assuming hyperDag format." << std::endl;    
                    read_graph = FileReader::readComputationalDagHyperdagFormat(filename_graph);
                }

                bool status_graph = read_graph.first;
                ComputationalDag &graph = read_graph.second;

                auto [status_architecture, architecture] = FileReader::readBspArchitecture(filename_machine);

                if (!status_graph) {
                    throw std::invalid_argument("Reading graph file " + filename_graph + " failed.");
                }

                if (!status_architecture) {
                    throw std::invalid_argument("Reading architecture file " + filename_machine + " failed.");
                }

                BspInstance bsp_instance(graph, architecture);

                std::vector<std::string> schedulers_name(parser.algorithms.size(), "");
                std::vector<bool> schedulers_failed(parser.algorithms.size(), false);
                std::vector<unsigned> schedulers_costs(parser.algorithms.size(), 0);
                std::vector<unsigned> schedulers_work_costs(parser.algorithms.size(), 0);
                std::vector<unsigned> schedulers_supersteps(parser.algorithms.size(), 0);
                std::vector<long unsigned> schedulers_compute_time(parser.algorithms.size(), 0);

                size_t algorithm_counter = 0;
                for (auto &algorithm : parser.algorithms) {
                    schedulers_name[algorithm_counter] = algorithm.second.get_child("name").get_value<std::string>();

                    try {
                        const auto start_time = std::chrono::high_resolution_clock::now();

                        auto [return_status, schedule] =
                            run_algorithm(parser, algorithm.second, bsp_instance,
                                          parser.global_params.get_child("timeLimit").get_value<unsigned>());

                        const auto finish_time = std::chrono::high_resolution_clock::now();

                        schedulers_compute_time[algorithm_counter] =
                            std::chrono::duration_cast<std::chrono::milliseconds>(finish_time - start_time).count();

                        if (return_status != RETURN_STATUS::SUCCESS && return_status != RETURN_STATUS::BEST_FOUND) {
                            if (return_status == RETURN_STATUS::ERROR) {
                                throw std::runtime_error("Error while computing schedule " +
                                                         algorithm.second.get_child("name").get_value<std::string>() +
                                                         ".");
                            }
                            if (return_status == RETURN_STATUS::TIMEOUT) {
                                throw std::runtime_error("Scheduler " +
                                                         algorithm.second.get_child("name").get_value<std::string>() +
                                                         " timed out.");
                            }
                        }

                        schedulers_costs[algorithm_counter] = schedule.computeCosts();
                        schedulers_work_costs[algorithm_counter] = schedule.computeWorkCosts();
                        schedulers_supersteps[algorithm_counter] = schedule.numberOfSupersteps();

                        // unsigned total_costs = schedule.computeCosts();
                        // unsigned work_costs = schedule.computeWorkCosts();
                        // std::cout << "Computed schedule: total costs: " << total_costs << "\t work costs: " <<
                        // work_costs
                        //         << "\t comm costs: " << total_costs - work_costs
                        //         << "\t number of supersteps: " << schedule.numberOfSupersteps() << "\t compute time:
                        //         "
                        //         << std::chrono::duration_cast<std::chrono::milliseconds>(finish_time -
                        //         start_time).count()
                        //         << "ms"
                        //         << "\t scheduler: " << algorithm.second.get_child("name").get_value<std::string>()
                        //         << std::endl;

                        BspScheduleWriter sched_writer(schedule);
                        if (parser.global_params.get_child("outputSchedule").get_value<bool>()) {
                            try {
                                sched_writer.write_txt(name_graph + "_" + name_machine + "_" +
                                                       algorithm.second.get_child("name").get_value<std::string>() +
                                                       "_schedule.txt");
                            } catch (std::exception &e) {
                                std::cerr << "Writing schedule file for " + name_graph + ", " + name_machine + ", " +
                                                 schedulers_name[algorithm_counter] + " has failed."
                                          << std::endl;
                                std::cerr << e.what() << std::endl;
                            }
                        }

                        if (parser.global_params.get_child("outputSankeySchedule").get_value<bool>()) {
                            try {
                                sched_writer.write_sankey(name_graph + "_" + name_machine + "_" +
                                                          algorithm.second.get_child("name").get_value<std::string>() +
                                                          "_sankey.sankey");
                            } catch (std::exception &e) {
                                std::cerr << "Writing sankey file for " + name_graph + ", " + name_machine + ", " +
                                                 schedulers_name[algorithm_counter] + " has failed."
                                          << std::endl;
                                std::cerr << e.what() << std::endl;
                            }
                        }
                        if (parser.global_params.get_child("outputDotSchedule").get_value<bool>()) {
                            try {
                                sched_writer.write_dot(name_graph + "_" + name_machine + "_" +
                                                       algorithm.second.get_child("name").get_value<std::string>() +
                                                       "_schedule.dot");
                            } catch (std::exception &e) {
                                std::cerr << "Writing dot file for " + name_graph + ", " + name_machine + ", " +
                                                 schedulers_name[algorithm_counter] + " has failed."
                                          << std::endl;
                                std::cerr << e.what() << std::endl;
                            }
                        }

                    } catch (std::runtime_error &e) {
                        schedulers_failed[algorithm_counter] = true;
                        std::cerr << "Runtime error during execution of Scheduler " +
                                         algorithm.second.get_child("name").get_value<std::string>() + "."
                                  << std::endl;
                        std::cerr << e.what() << std::endl;
                    } catch (std::logic_error &e) {
                        schedulers_failed[algorithm_counter] = true;
                        std::cerr << "Logic error during execution of Scheduler " +
                                         algorithm.second.get_child("name").get_value<std::string>() + "."
                                  << std::endl;
                        std::cerr << e.what() << std::endl;
                    } catch (std::exception &e) {
                        schedulers_failed[algorithm_counter] = true;
                        std::cerr << "Error during execution of Scheduler " +
                                         algorithm.second.get_child("name").get_value<std::string>() + "."
                                  << std::endl;
                        std::cerr << e.what() << std::endl;
                    } catch (...) {
                        schedulers_failed[algorithm_counter] = true;
                        std::cerr << "Error during execution of Scheduler " +
                                         algorithm.second.get_child("name").get_value<std::string>() + "."
                                  << std::endl;
                    }
                    algorithm_counter++;
                }

                int tw = 1, ww = 1, cw = 1, nsw = 1, ct = 1;
                for (size_t i = 0; i < parser.algorithms.size(); i++) {
                    if (schedulers_failed[i])
                        continue;
                    tw = std::max(tw, 1 + int(std::log10(schedulers_costs[i])));
                    ww = std::max(ww, 1 + int(std::log10(schedulers_work_costs[i])));
                    cw = std::max(cw, 1 + int(std::log10(schedulers_costs[i] - schedulers_work_costs[i])));
                    nsw = std::max(nsw, 1 + int(std::log10(schedulers_supersteps[i])));
                    ct = std::max(ct, 1 + int(std::log10(schedulers_compute_time[i])));
                }

                bool sorted_by_total_costs = true;
                std::vector<size_t> ordering = sorting_arrangement(schedulers_costs);

                std::cout << std::endl << name_graph << " - " << name_machine << std::endl;
                std::cout << "Number of Vertices: " + std::to_string(graph.numberOfVertices()) +
                                 "  Number of Edges: " + std::to_string(graph.numberOfEdges())
                          << std::endl;
                for (size_t j = 0; j < parser.algorithms.size(); j++) {
                    size_t i = j;
                    if (sorted_by_total_costs)
                        i = ordering[j];
                    if (schedulers_failed[i]) {
                        std::cout << "scheduler " << schedulers_name[i] << " failed." << std::endl;
                    } else {
                        std::cout << "total costs:  " << std::right << std::setw(tw) << schedulers_costs[i]
                                  << "     work costs:  " << std::right << std::setw(ww) << schedulers_work_costs[i]
                                  << "     comm costs:  " << std::right << std::setw(cw)
                                  << schedulers_costs[i] - schedulers_work_costs[i]
                                  << "     number of supersteps:  " << std::right << std::setw(nsw)
                                  << schedulers_supersteps[i] << "     compute time:  " << std::right << std::setw(ct)
                                  << schedulers_compute_time[i] << "ms"
                                  << "     scheduler:  " << schedulers_name[i] << std::endl;
                    }
                }

            } catch (std::invalid_argument &e) {
                std::cerr << e.what() << std::endl;
            } catch (std::exception &e) {
                std::cerr << "Error during execution of Instance " +
                                 instance.second.get_child("graphFile").get_value<std::string>() + " " +
                                 instance.second.get_child("machineParamsFile").get_value<std::string>() + "."
                          << std::endl;
                std::cerr << e.what() << std::endl;
            } catch (...) {
                std::cerr << "Error during execution of Instance  " +
                                 instance.second.get_child("graphFile").get_value<std::string>() + " " +
                                 instance.second.get_child("machineParamsFile").get_value<std::string>() + "."
                          << std::endl;
            }
        }
    } catch (std::invalid_argument &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}
