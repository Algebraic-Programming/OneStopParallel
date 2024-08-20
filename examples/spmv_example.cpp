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

#include <chrono>
#include <omp.h>

#include "scheduler/Coarsers/HDaggCoarser.hpp"
#include "scheduler/Coarsers/SquashA.hpp"
#include "scheduler/ContractRefineScheduler/BalDMixR.hpp"
#include "scheduler/ContractRefineScheduler/CoBalDMixR.hpp"
#include "scheduler/ContractRefineScheduler/MultiLevelHillClimbing.hpp"
#include "scheduler/GreedySchedulers/MetaGreedyScheduler.hpp"
#include "scheduler/HDagg/HDagg_simple.hpp"
#include "scheduler/ImprovementScheduler.hpp"
#include "scheduler/LocalSearchSchedulers/HillClimbingScheduler.hpp"
#include "scheduler/LocalSearchSchedulers/LKTotalCommScheduler.hpp"
#include "scheduler/SchedulePermutations/ScheduleNodePermuter.hpp"
#include "scheduler/Serial/Serial.hpp"
#include "scheduler/SubArchitectureSchedulers/SubArchitectures.hpp"
#include "auxiliary/auxiliary.hpp"
#include "file_interactions/BspScheduleWriter.hpp"
#include "file_interactions/FileReader.hpp"
#include "file_interactions/FileWriter.hpp"

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <machine_file> optional: <bool_output>" << std::endl;
        return 1;
    }

    std::string filename_graph = argv[1];
    std::string name_graph =
        filename_graph.substr(filename_graph.rfind("/") + 1, filename_graph.rfind(".") - filename_graph.rfind("/") - 1);

    std::string filename_machine = argv[2];
    std::string name_machine = filename_machine.substr(filename_machine.rfind("/") + 1,
                                                       filename_machine.rfind(".") - filename_machine.rfind("/") - 1);

    bool make_schedule_and_perm_files = true;
    if (argc >= 4) {
        make_schedule_and_perm_files = std::stoi(argv[3]);
    }

    std::cout << name_graph << " - " << name_machine << std::endl;

    std::pair<bool, ComputationalDag> read_graph(false, ComputationalDag());
    if (filename_graph.substr(filename_graph.rfind(".") + 1) == "txt") {
        read_graph = FileReader::readComputationalDagHyperdagFormat(filename_graph);
    } else if (filename_graph.substr(filename_graph.rfind(".") + 1) == "mtx") {
        read_graph = FileReader::readComputationalDagMartixMarketFormat(filename_graph);
    }
    bool status_graph = read_graph.first;
    ComputationalDag &graph = read_graph.second;

    auto [status_architecture, architecture] = FileReader::readBspArchitecture(filename_machine);

    if (!status_graph || !status_architecture) {

        std::cout << "Reading files failed." << std::endl;
        return 1;
    }

    BspInstance instance(graph, architecture);
    std::cout << "Number of Vertices: " + std::to_string(graph.numberOfVertices()) +
                     "  Number of Edges: " + std::to_string(graph.numberOfEdges())
              << std::endl;

    std::vector<Scheduler *> schedulers;

    HillClimbingScheduler hill_climbing_scheduler1;
    HillClimbingScheduler hill_climbing_scheduler2;
    HillClimbingScheduler hill_climbing_scheduler3;
    HillClimbingScheduler hill_climbing_scheduler4;
    HillClimbingScheduler hill_climbing_scheduler5;
    HillClimbingScheduler hill_climbing_scheduler6;
    HillClimbingScheduler hill_climbing_scheduler7;
    HillClimbingScheduler hill_climbing_scheduler8;
    HillClimbingScheduler hill_climbing_scheduler9;
    HillClimbingScheduler hill_climbing_scheduler10;
    HillClimbingScheduler hill_climbing_scheduler11;
    HillClimbingScheduler hill_climbing_scheduler12;
    HillClimbingScheduler hill_climbing_scheduler13;
    HillClimbingScheduler hill_climbing_scheduler14;
    HillClimbingScheduler hill_climbing_scheduler15;
    HillClimbingScheduler hill_climbing_scheduler16;

    GreedyBspScheduler bsp_greedy_scheduler;
    GreedyBspScheduler bsp_greedy_scheduler_hc;
    GreedyVarianceScheduler variance_greedy_scheduler;
    GreedyVarianceScheduler variance_greedy_scheduler_hc;
    GreedyChildren chldrn_greedyS(true);
    GreedyChildren chldrn_greedy(false);
    GreedyCilkScheduler cilk_greedy;
    GreedyCilkScheduler cilk_greedy_hc;
    GreedyEtfScheduler etf_greedy;
    RandomGreedy rng_greedyS(true);
    RandomGreedy rng_greedy(false);
    GreedyLayers layer_greedy;
    MetaGreedyScheduler meta_greedy_scheduler;
    MetaGreedyScheduler meta_greedy_scheduler_lk;
    MetaGreedyScheduler meta_greedy_scheduler_lk_squash;
    MetaGreedyScheduler meta_greedy_scheduler_hc;
    MetaGreedyScheduler meta_greedy_scheduler_squash;
    MetaGreedyScheduler meta_greedy_scheduler_squash_hc;
    SquashA squash_meta_greedy_scheduler(&meta_greedy_scheduler_squash);
    SquashA squash_meta_greedy_scheduler_hc(&meta_greedy_scheduler_squash, &hill_climbing_scheduler13);

    ComboScheduler bsp_greedy_hc_scheduler(&bsp_greedy_scheduler_hc, &hill_climbing_scheduler1);
    ComboScheduler variance_greedy_hc_scheduler(&variance_greedy_scheduler_hc, &hill_climbing_scheduler12);
    ComboScheduler cilk_hc_scheduler(&cilk_greedy_hc, &hill_climbing_scheduler16);
    ComboScheduler bsp_meta_hc_scheduler(&meta_greedy_scheduler_hc, &hill_climbing_scheduler2);

    const unsigned bald_mixer_number_of_partitions_ = 4; // gets overriten by instance
    const float bald_mixer_balance_threshhold_5 = 1.05;
    const float bald_mixer_balance_threshhold_10 = 1.1;
    const float bald_mixer_balance_threshhold_20 = 1.2;
    const float bald_mixer_balance_threshhold_35 = 1.35;
    const PartitionAlgorithm bald_mixer_part_algo_ = Greedy;
    const CoinType bald_mixer_coin_type_ = Thue_Morse;
    const float bald_mixer_clumps_per_partition_ = 6;
    const float bald_mixer_nodes_per_clump_ = 4;
    const float bald_mixer_nodes_per_partition_5 = 60;   // approx 2 * (balance_thresh-1)^{-1}
    const float bald_mixer_nodes_per_partition_10 = 30;  // approx 2 * (balance_thresh-1)^{-1}
    const float bald_mixer_nodes_per_partition_20 = 15;  // approx 2 * (balance_thresh-1)^{-1}
    const float bald_mixer_nodes_per_partition_35 = 10;  // approx 2 * (balance_thresh-1)^{-1}
    const float bald_mixer_max_weight_for_flag_ = 1 / 3; // approx 2 / clumps_per_partition
    const float bald_mixer_balanced_cut_ratio_ = 1 / 3;
    const float bald_mixer_min_weight_for_split_ = 1 / 48;
    const int bald_mixer_hill_climb_simple_improvement_attemps_ = 10;

    Coarse_Scheduler_Params params5(
        bald_mixer_number_of_partitions_, bald_mixer_balance_threshhold_5, bald_mixer_part_algo_, bald_mixer_coin_type_,
        bald_mixer_clumps_per_partition_, bald_mixer_nodes_per_clump_, bald_mixer_nodes_per_partition_5,
        bald_mixer_max_weight_for_flag_, bald_mixer_balanced_cut_ratio_, bald_mixer_min_weight_for_split_,
        bald_mixer_hill_climb_simple_improvement_attemps_);
    Coarse_Scheduler_Params params10(
        bald_mixer_number_of_partitions_, bald_mixer_balance_threshhold_10, bald_mixer_part_algo_,
        bald_mixer_coin_type_, bald_mixer_clumps_per_partition_, bald_mixer_nodes_per_clump_,
        bald_mixer_nodes_per_partition_10, bald_mixer_max_weight_for_flag_, bald_mixer_balanced_cut_ratio_,
        bald_mixer_min_weight_for_split_, bald_mixer_hill_climb_simple_improvement_attemps_);
    Coarse_Scheduler_Params params20(
        bald_mixer_number_of_partitions_, bald_mixer_balance_threshhold_20, bald_mixer_part_algo_,
        bald_mixer_coin_type_, bald_mixer_clumps_per_partition_, bald_mixer_nodes_per_clump_,
        bald_mixer_nodes_per_partition_20, bald_mixer_max_weight_for_flag_, bald_mixer_balanced_cut_ratio_,
        bald_mixer_min_weight_for_split_, bald_mixer_hill_climb_simple_improvement_attemps_);
    Coarse_Scheduler_Params params35(
        bald_mixer_number_of_partitions_, bald_mixer_balance_threshhold_35, bald_mixer_part_algo_,
        bald_mixer_coin_type_, bald_mixer_clumps_per_partition_, bald_mixer_nodes_per_clump_,
        bald_mixer_nodes_per_partition_35, bald_mixer_max_weight_for_flag_, bald_mixer_balanced_cut_ratio_,
        bald_mixer_min_weight_for_split_, bald_mixer_hill_climb_simple_improvement_attemps_);

    BalDMixR bald_mixer_scheduler5(params5);
    BalDMixR bald_mixer_scheduler10(params10);
    BalDMixR bald_mixer_lk_scheduler10(params10);
    BalDMixR bald_mixer_scheduler20(params20);
    BalDMixR bald_mixer_scheduler35(params35);
    BalDMixR bald_mixer_scheduler_hc(params10);
    ComboScheduler bald_mixer_hc_scheduler(&bald_mixer_scheduler_hc, &hill_climbing_scheduler3);

    BalDMixR bald_mixer_squash_scheduler10(params10);
    BalDMixR bald_mixer_squash_scheduler20(params20);
    SquashA squash_bald_mixer_scheduler10_hc(&bald_mixer_squash_scheduler10, &hill_climbing_scheduler14);
    SquashA squash_bald_mixer_scheduler20_hc(&bald_mixer_squash_scheduler20, &hill_climbing_scheduler15);

    const unsigned number_of_partitions_ = 2;
    const float balance_threshhold_5 = 1.05;
    const float balance_threshhold_10 = 1.1;
    const float balance_threshhold_20 = 1.2;
    const float balance_threshhold_35 = 1.35;
    const PartitionAlgorithm part_algo_ = Greedy;
    const CoinType coin_type_ = Thue_Morse;
    const float clumps_per_partition_ = 6;
    const float nodes_per_clump_ = 8;
    const float nodes_per_partition_refine_5 = 60;  // approx 2 * (balance_thresh-1)^{-1}
    const float nodes_per_partition_final_5 = 30;   // approx 2 * (balance_thresh-1)^{-1}
    const float nodes_per_partition_refine_10 = 30; // approx 2 * (balance_thresh-1)^{-1}
    const float nodes_per_partition_final_10 = 20;  // approx 2 * (balance_thresh-1)^{-1}
    const float nodes_per_partition_refine_20 = 15; // approx 2 * (balance_thresh-1)^{-1}
    const float nodes_per_partition_final_20 = 8;   // approx 2 * (balance_thresh-1)^{-1}
    const float nodes_per_partition_refine_35 = 10; // approx 2 * (balance_thresh-1)^{-1}
    const float nodes_per_partition_final_35 = 6;   // approx 2 * (balance_thresh-1)^{-1}
    const float max_weight_for_flag_ = 1 / 3;       // approx 2 / clumps_per_partition
    const float balanced_cut_ratio_ = 1 / 3;
    const float min_weight_for_split_ = 1 / 48;
    const int hill_climb_simple_improvement_attemps_ = 5;
    const int min_comp_generation_when_shaving_ = 6;

    Coarse_Scheduler_Params params_init5(number_of_partitions_, balance_threshhold_5, part_algo_, coin_type_,
                                         clumps_per_partition_, nodes_per_clump_, nodes_per_partition_refine_5,
                                         max_weight_for_flag_, balanced_cut_ratio_, min_weight_for_split_,
                                         hill_climb_simple_improvement_attemps_, min_comp_generation_when_shaving_);
    Coarse_Scheduler_Params params_init10(number_of_partitions_, balance_threshhold_10, part_algo_, coin_type_,
                                          clumps_per_partition_, nodes_per_clump_, nodes_per_partition_refine_10,
                                          max_weight_for_flag_, balanced_cut_ratio_, min_weight_for_split_,
                                          hill_climb_simple_improvement_attemps_, min_comp_generation_when_shaving_);
    Coarse_Scheduler_Params params_init20(number_of_partitions_, balance_threshhold_20, part_algo_, coin_type_,
                                          clumps_per_partition_, nodes_per_clump_, nodes_per_partition_refine_20,
                                          max_weight_for_flag_, balanced_cut_ratio_, min_weight_for_split_,
                                          hill_climb_simple_improvement_attemps_, min_comp_generation_when_shaving_);
    Coarse_Scheduler_Params params_init35(number_of_partitions_, balance_threshhold_35, part_algo_, coin_type_,
                                          clumps_per_partition_, nodes_per_clump_, nodes_per_partition_refine_35,
                                          max_weight_for_flag_, balanced_cut_ratio_, min_weight_for_split_,
                                          hill_climb_simple_improvement_attemps_, min_comp_generation_when_shaving_);

    Coarse_Scheduler_Params params_final5(number_of_partitions_, balance_threshhold_5, part_algo_, coin_type_,
                                          clumps_per_partition_, nodes_per_clump_, nodes_per_partition_final_5,
                                          max_weight_for_flag_, balanced_cut_ratio_, min_weight_for_split_,
                                          hill_climb_simple_improvement_attemps_, min_comp_generation_when_shaving_);
    Coarse_Scheduler_Params params_final10(number_of_partitions_, balance_threshhold_10, part_algo_, coin_type_,
                                           clumps_per_partition_, nodes_per_clump_, nodes_per_partition_final_10,
                                           max_weight_for_flag_, balanced_cut_ratio_, min_weight_for_split_,
                                           hill_climb_simple_improvement_attemps_, min_comp_generation_when_shaving_);
    Coarse_Scheduler_Params params_final20(number_of_partitions_, balance_threshhold_20, part_algo_, coin_type_,
                                           clumps_per_partition_, nodes_per_clump_, nodes_per_partition_final_20,
                                           max_weight_for_flag_, balanced_cut_ratio_, min_weight_for_split_,
                                           hill_climb_simple_improvement_attemps_, min_comp_generation_when_shaving_);
    Coarse_Scheduler_Params params_final35(number_of_partitions_, balance_threshhold_35, part_algo_, coin_type_,
                                           clumps_per_partition_, nodes_per_clump_, nodes_per_partition_final_35,
                                           max_weight_for_flag_, balanced_cut_ratio_, min_weight_for_split_,
                                           hill_climb_simple_improvement_attemps_, min_comp_generation_when_shaving_);

    CoarseRefineScheduler_parameters cobald_params5(params_init5, params_final20);
    CoarseRefineScheduler_parameters cobald_params10(params_init10, params_final20);
    CoarseRefineScheduler_parameters cobald_params20(params_init20, params_final20);
    CoarseRefineScheduler_parameters cobald_params35(params_init35, params_final35);

    CoBalDMixR cobald_mixer_scheduler5(cobald_params5);
    CoBalDMixR cobald_mixer_scheduler10(cobald_params10);
    CoBalDMixR cobald_mixer_scheduler20(cobald_params20);
    CoBalDMixR cobald_mixer_lk_scheduler20(cobald_params20);
    CoBalDMixR cobald_mixer_scheduler35(cobald_params35);
    CoBalDMixR cobald_mixer_scheduler_hc(cobald_params20);
    ComboScheduler cobald_mixer_hc_scheduler(&cobald_mixer_scheduler_hc, &hill_climbing_scheduler4);

    HDagg_parameters hdagg_param_orig5(1.05, 0, false);
    HDagg_parameters hdagg_param_orig10(1.1, 0, false);
    HDagg_parameters hdagg_param_orig20(1.2, 0, false);
    HDagg_parameters hdagg_param_orig35(1.35, 0, false);
    HDagg_parameters hdagg_param5(1.05);
    HDagg_parameters hdagg_param10(1.1);
    HDagg_parameters hdagg_param20(1.2);
    HDagg_parameters hdagg_param35(1.35);
    HDagg_simple hdagg_scheduler5(hdagg_param5);
    HDagg_simple hdagg_scheduler10(hdagg_param10);
    HDagg_simple hdagg_scheduler20(hdagg_param20);
    HDagg_simple hdagg_scheduler35(hdagg_param35);
    HDagg_simple hdagg_scheduler_original5(hdagg_param_orig5);
    HDagg_simple hdagg_scheduler_original10(hdagg_param_orig10);
    HDagg_simple hdagg_scheduler_original20(hdagg_param_orig20);
    HDagg_simple hdagg_scheduler_original35(hdagg_param_orig35);
    HDagg_simple hdagg_scheduler_squash5(hdagg_param5);
    HDagg_simple hdagg_scheduler_squash10(hdagg_param10);
    HDagg_simple hdagg_scheduler_squash20(hdagg_param20);
    HDagg_simple hdagg_scheduler_squash35(hdagg_param35);
    HDagg_simple hdagg_scheduler_hc;

    HDaggCoarser hdagg_original5(&hdagg_scheduler_original5);
    HDaggCoarser hdagg_original10(&hdagg_scheduler_original10);
    HDaggCoarser hdagg_original20(&hdagg_scheduler_original20);
    HDaggCoarser hdagg_original35(&hdagg_scheduler_original35);
    SquashA squash_a_hdag_hc_scheduler5(&hdagg_scheduler_squash5, &hill_climbing_scheduler5);
    SquashA squash_a_hdag_hc_scheduler10(&hdagg_scheduler_squash10, &hill_climbing_scheduler6);
    SquashA squash_a_hdag_hc_scheduler20(&hdagg_scheduler_squash20, &hill_climbing_scheduler7);
    SquashA squash_a_hdag_hc_scheduler35(&hdagg_scheduler_squash35, &hill_climbing_scheduler8);
    ComboScheduler hdag_hc_scheduler(&hdagg_scheduler_hc, &hill_climbing_scheduler9);

    MultiLevelHillClimbingScheduler multi_level_hill_climbing_scheduler_15(2000, 0.15);
    MultiLevelHillClimbingScheduler multi_level_hill_climbing_scheduler_30(2000, 0.3);
    SquashA squasha_multi_level_hill_climbing_scheduler_15(&multi_level_hill_climbing_scheduler_15,
                                                           &hill_climbing_scheduler10);
    SquashA squasha_multi_level_hill_climbing_scheduler_30(&multi_level_hill_climbing_scheduler_30,
                                                           &hill_climbing_scheduler11);

    LKTotalCommScheduler lk_scheduler1;
    LKTotalCommScheduler lk_scheduler2;
    LKTotalCommScheduler lk_scheduler3;
    LKTotalCommScheduler lk_scheduler4;
    ComboScheduler lk_hc_scheduler(&meta_greedy_scheduler_lk, &lk_scheduler1);
    ComboScheduler lk_hc_bald_scheduler(&bald_mixer_lk_scheduler10, &lk_scheduler2);
    ComboScheduler lk_hc_cobald_scheduler(&cobald_mixer_lk_scheduler20, &lk_scheduler3);
    ComboScheduler lk_hc_max_greed_squash(&meta_greedy_scheduler_lk_squash, &lk_scheduler4);
    SquashA squasha_greedy_meta_lk_scheduler(&lk_hc_max_greed_squash);

    Serial serial_scheduler;

    unsigned large_graph_vertex_barrier = 10000;
    unsigned large_graph_edge_barrier = 25000;

    // Algorithms that take long earlier
    if (instance.getComputationalDag().numberOfVertices() <= large_graph_vertex_barrier &&
        instance.getComputationalDag().numberOfEdges() <= large_graph_edge_barrier) {
        // schedulers.push_back(&multi_level_hill_climbing_scheduler_15);
        // schedulers.push_back(&multi_level_hill_climbing_scheduler_30);

    } else {
        // schedulers.push_back(&squasha_multi_level_hill_climbing_scheduler_15);
        // schedulers.push_back(&squasha_multi_level_hill_climbing_scheduler_30);
    }

    if (instance.getComputationalDag().numberOfVertices() <= 2 * large_graph_vertex_barrier &&
        instance.getComputationalDag().numberOfEdges() <= 2 * large_graph_edge_barrier) {
        // schedulers.push_back(&bald_mixer_scheduler5);
        schedulers.push_back(&bald_mixer_scheduler10);
        // schedulers.push_back(&bald_mixer_scheduler20);
        // schedulers.push_back(&bald_mixer_scheduler35);
        // schedulers.push_back(&bald_mixer_hc_scheduler);

        // schedulers.push_back(&lk_hc_bald_scheduler);
    } else {
        // schedulers.push_back(&squash_bald_mixer_scheduler10_hc);
        // schedulers.push_back(&squash_bald_mixer_scheduler20_hc);
    }

    // schedulers.push_back(&lk_hc_scheduler);
    // schedulers.push_back(&lk_hc_cobald_scheduler);
    schedulers.push_back(&lk_hc_scheduler);
    schedulers.push_back(&lk_hc_cobald_scheduler);
    schedulers.push_back(&squasha_greedy_meta_lk_scheduler);

    // schedulers.push_back(&cobald_mixer_scheduler5);
    // schedulers.push_back(&cobald_mixer_scheduler10);
    schedulers.push_back(&cobald_mixer_scheduler20);
    // schedulers.push_back(&cobald_mixer_scheduler35);
    // schedulers.push_back(&cobald_mixer_hc_scheduler);

    // schedulers.push_back(&meta_greedy_scheduler);
    // schedulers.push_back(&squash_meta_greedy_scheduler);
    schedulers.push_back(&squash_meta_greedy_scheduler_hc);
    // schedulers.push_back(&bsp_meta_hc_scheduler);
    // schedulers.push_back(&etf_greedy);
    // schedulers.push_back(&chldrn_greedy);
    // schedulers.push_back(&chldrn_greedyS);
    // schedulers.push_back(&cilk_greedy);
    schedulers.push_back(&cilk_hc_scheduler);
    // schedulers.push_back(&rng_greedy);
    // schedulers.push_back(&rng_greedyS);
    // schedulers.push_back(&layer_greedy);
    // schedulers.push_back(&bsp_greedy_scheduler);
    schedulers.push_back(&bsp_greedy_hc_scheduler);
    // schedulers.push_back(&variance_greedy_scheduler);
    schedulers.push_back(&variance_greedy_hc_scheduler);

    // schedulers.push_back(&hdagg_scheduler5);
    schedulers.push_back(&hdagg_scheduler10);
    // schedulers.push_back(&hdagg_scheduler20);
    // schedulers.push_back(&hdagg_scheduler35);
    // schedulers.push_back(&hdagg_original5);
    // schedulers.push_back(&hdagg_original10);
    // schedulers.push_back(&hdagg_original20);
    // schedulers.push_back(&hdagg_original35);
    // schedulers.push_back(&hdag_hc_scheduler);
    // schedulers.push_back(&squash_a_hdag_hc_scheduler5);
    schedulers.push_back(&squash_a_hdag_hc_scheduler10);
    // schedulers.push_back(&squash_a_hdag_hc_scheduler20);
    // schedulers.push_back(&squash_a_hdag_hc_scheduler35);

    schedulers.push_back(&serial_scheduler);

    std::vector<Scheduler *> subarch_schedulers;
    for (size_t j = 0; j < schedulers.size(); j++) {
        subarch_schedulers.push_back(new SubArchitectureScheduler(schedulers[j]));
    }

    std::vector<Scheduler *> alternate_schedulers;
    for (size_t j = 0; j < schedulers.size(); j++) {
        alternate_schedulers.push_back(schedulers[j]);
        // alternate_schedulers.push_back(subarch_schedulers[j]);
    }

    std::vector<std::string> schedulers_name(alternate_schedulers.size());
    std::vector<unsigned> schedulers_costs(alternate_schedulers.size());
    std::vector<unsigned> schedulers_work_costs(alternate_schedulers.size());
    std::vector<unsigned> schedulers_supersteps(alternate_schedulers.size());
    std::vector<long unsigned> schedulers_compute_time(alternate_schedulers.size());

    // #pragma omp parallel default(shared)
    {
        // #pragma omp for schedule(dynamic, 1)
        for (size_t i = 0; i < alternate_schedulers.size(); i += 1) {
            std::cout << "Start of Scheduler " + alternate_schedulers[i]->getScheduleName() + "\n";

            const auto start_time = std::chrono::high_resolution_clock::now();
            auto [return_status, return_schedule] = alternate_schedulers[i]->computeSchedule(instance);
            const auto finish_time = std::chrono::high_resolution_clock::now();

            std::cout << "End of Scheduler " + alternate_schedulers[i]->getScheduleName() + "\n";

            schedulers_compute_time[i] =
                std::chrono::duration_cast<std::chrono::milliseconds>(finish_time - start_time).count();

            if (return_status == SUCCESS || return_status == BEST_FOUND) {

                if ((!return_schedule.satisfiesPrecedenceConstraints()) || (!return_schedule.hasValidCommSchedule())) {
                    throw std::logic_error(alternate_schedulers[i]->getScheduleName() +
                                           " did not return a valid Schedule!\n");
                }

                if (make_schedule_and_perm_files) {
                    BspScheduleWriter sched_writer(return_schedule);

                    sched_writer.write_txt(name_graph + "_" + name_machine + "_" +
                                           alternate_schedulers[i]->getScheduleName() + "_schedule.txt");
                    sched_writer.write_sankey(name_graph + "_" + name_machine + "_" +
                                              alternate_schedulers[i]->getScheduleName() + "_sankey.sankey");

                    std::vector<size_t> perm = schedule_node_permuter(return_schedule, 8);
                    FileWriter::writePermutationFile(perm, name_graph + "_" + name_machine + "_" +
                                                               alternate_schedulers[i]->getScheduleName() +
                                                               "_perm.perm");
                }

                schedulers_name[i] = alternate_schedulers[i]->getScheduleName();
                schedulers_costs[i] = return_schedule.computeCosts();
                schedulers_work_costs[i] = return_schedule.computeWorkCosts();
                schedulers_supersteps[i] = return_schedule.numberOfSupersteps();

            } else {
                std::cout << "Computing schedule failed." << std::endl;
            }
        }

        // #pragma omp for schedule(dynamic, 1)
        // for (size_t i = 1; i < alternate_schedulers.size(); i+=2) {
        //     std::cout << "Start of Scheduler " << alternate_schedulers[i]->getScheduleName() << std::endl;

        //     const auto start_time = std::chrono::high_resolution_clock::now();
        //     auto [return_status, return_schedule] = alternate_schedulers[i]->computeSchedule(instance);
        //     const auto finish_time = std::chrono::high_resolution_clock::now();

        //     schedulers_compute_time[i] =
        //     std::chrono::duration_cast<std::chrono::milliseconds>(finish_time-start_time).count();

        //     if (return_status == SUCCESS || return_status == BEST_FOUND) {

        //         assert(return_schedule.satisfiesPrecedenceConstraints());

        //         if (make_schedule_and_perm_files) {
        //             BspScheduleWriter sched_writer(return_schedule);

        //             sched_writer.write_txt(name_graph + "_" + name_machine + "_" +
        //             alternate_schedulers[i]->getScheduleName() +
        //                                 "_schedule.txt");

        //             std::vector<size_t> perm = schedule_node_permuter(return_schedule, 8);
        //             FileWriter::writePermutationFile(perm, name_graph + "_" + name_machine + "_" +
        //                                                     alternate_schedulers[i]->getScheduleName() +
        //                                                     "_perm.perm");
        //         }

        //         schedulers_name[i] = alternate_schedulers[i]->getScheduleName();
        //         schedulers_costs[i] = return_schedule.computeCosts();
        //         schedulers_work_costs[i] = return_schedule.computeWorkCosts();
        //         schedulers_supersteps[i] = return_schedule.numberOfSupersteps();

        //     } else {
        //         std::cout << "Computing schedule failed." << std::endl;
        //     }
        // }
    }

    int tw = 1, ww = 1, cw = 1, nsw = 1, ct = 1;
    for (size_t i = 0; i < alternate_schedulers.size(); i++) {
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
    for (size_t j = 0; j < alternate_schedulers.size(); j++) {
        size_t i = j;
        if (sorted_by_total_costs)
            i = ordering[j];
        std::cout << "total costs:  " << std::right << std::setw(tw) << schedulers_costs[i]
                  << "     work costs:  " << std::right << std::setw(ww) << schedulers_work_costs[i]
                  << "     comm costs:  " << std::right << std::setw(cw)
                  << schedulers_costs[i] - schedulers_work_costs[i] << "     number of supersteps:  " << std::right
                  << std::setw(nsw) << schedulers_supersteps[i] << "     compute time:  " << std::right << std::setw(ct)
                  << schedulers_compute_time[i] << "ms" << "     scheduler:  " << schedulers_name[i] << std::endl;
    }

    for (size_t j = 0; j < subarch_schedulers.size(); j++) {
        delete subarch_schedulers[j];
    }

    return 0;
}
