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

#include "boost/log/utility/setup.hpp"
#include <boost/graph/graphviz.hpp>
#include <boost/log/trivial.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <tuple>
#include <unistd.h>

#include "scheduler/Coarsers/Funnel.hpp"
#include "scheduler/Coarsers/HDaggCoarser.hpp"
#include "scheduler/Coarsers/SquashA.hpp"
#include "scheduler/Coarsers/WavefrontCoarser.hpp"
#include "scheduler/ContractRefineScheduler/BalDMixR.hpp"
#include "scheduler/ContractRefineScheduler/CoBalDMixR.hpp"
#include "scheduler/ContractRefineScheduler/MultiLevelHillClimbing.hpp"
#include "scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "scheduler/GreedySchedulers/GreedyChildren.hpp"
#include "scheduler/GreedySchedulers/GreedyCilkScheduler.hpp"
#include "scheduler/GreedySchedulers/GreedyEtfScheduler.hpp"
#include "scheduler/GreedySchedulers/GreedyLayers.hpp"
#include "scheduler/GreedySchedulers/GreedyVarianceScheduler.hpp"
#include "scheduler/GreedySchedulers/MetaGreedyScheduler.hpp"
#include "scheduler/GreedySchedulers/RandomBadGreedy.hpp"
#include "scheduler/GreedySchedulers/RandomGreedy.hpp"
#include "scheduler/HDagg/HDagg_simple.hpp"
#include "scheduler/ImprovementScheduler.hpp"
#include "scheduler/LocalSearchSchedulers/HillClimbingScheduler.hpp"
#include "scheduler/Serial/Serial.hpp"
#include "scheduler/Wavefront/Wavefront.hpp"

#include "file_interactions/CommandLineParser.hpp"
#include "file_interactions/FileReader.hpp"
#include "file_interactions/BspScheduleWriter.hpp"
#include "model/BspSchedule.hpp"

#include "scheduler/SchedulePermutations/ScheduleNodePermuter.hpp"
#include "simulation/BspSptrsvCSR.hpp"

#include "auxiliary/run_algorithm.hpp"

#define NO_PERMUTE

namespace pt = boost::property_tree;

bool check_vectors_equal(const std::vector<double> &v1, const std::vector<double> &v2) {
    if (v1.size() != v2.size()) {
        std::cout << "Vectors have different sizes!" << std::endl;
        return false;
    }

    bool ret = true;
    for (unsigned i = 0; i < v1.size(); i++) {
        if (std::abs(v1[i] - v2[i]) / (std::abs(v1[i]) + std::abs(v2[i]) + 0.00001) > 0.00001) {
            std::cout << "Difference at index " << i << ": " << v1[i] << " != " << v2[i] << std::endl;
            ret = false;
        }
    }
    return ret;
}

void flush_cache(const unsigned proc_num) {
    std::vector<long> flash(proc_num * 2 * 1024 * 1024);

#pragma omp parallel for schedule(auto)
    for (size_t i = 0; i < flash.size(); i++) {
        flash[i] = rand();
    }
}

std::filesystem::path getExecutablePath() { return std::filesystem::canonical("/proc/self/exe"); }

// invoked upon program call
int main(int argc, char *argv[]) {

    std::string executable_dir = getExecutablePath().remove_filename().string();
    std::string main_config_location = executable_dir + "main_config.json";

    try {
        const CommandLineParser parser(argc, argv, main_config_location);

        std::string graph_dir = parser.global_params.get_child("graphDirectory").get_value<std::string>();
        if (graph_dir.substr(0, 1) != "/") {
            // graph_dir = graph_dir.substr(2);

            graph_dir = executable_dir + graph_dir;
        }

        std::string machine_dir = parser.global_params.get_child("machineDirectory").get_value<std::string>();
        if (machine_dir.substr(0, 1) != "/") {
            // machine_dir = machine_dir.substr(2);
            machine_dir = executable_dir + machine_dir;
        }

        std::string schedule_dir = parser.global_params.get_child("scheduleDirectory").get_value<std::string>();
        if (schedule_dir.substr(0, 1) != "/") {
            // schedule_dir = schedule_dir.substr(2);

            schedule_dir = executable_dir + schedule_dir;
        }

        std::string stats_file = parser.global_params.get_child("outputStatsFile").get_value<std::string>();
        if (stats_file.substr(0, 1) != "/") {
            // stats_file = stats_file.substr(2);
            stats_file = executable_dir + stats_file;
        }

        std::string log_file = parser.global_params.get_child("outputLogFile").get_value<std::string>();
        if (log_file.substr(0, 1) != "/") {
            // log_file = log_file.substr(2);

            log_file = executable_dir + log_file;
        }

        std::ofstream log;

        std::string header = "Graph,Machine,Algorithm,Permutation,SpTrSV_Runtime,Work_Cost,Base_Comm_Cost,Supersteps,_"
                             "Base_Buffered_Sending,Base_CostsTotalCommunication,Schedule_Compute_time";
        std::ofstream stats;

        bool stats_has_header = false;
        std::string first_line;
        std::ifstream header_read(stats_file);
        getline(header_read, first_line);
        if (first_line == header) {
            stats_has_header = true;
        }
        header_read.close();
        if (!stats_has_header) {
            stats.open(stats_file, std::ios_base::app);
            stats << header << "\n";
            stats.close();
        }

        for (const auto &machine_entry : std::filesystem::recursive_directory_iterator(machine_dir)) {
            if (std::filesystem::is_directory(machine_entry))
                continue;
            std::string machine_path_str = machine_entry.path();

            log.open(log_file, std::ios_base::app);
            log << "Start Machine: " + machine_path_str + "\n";
            log.close();

            for (const auto &graph_entry : std::filesystem::recursive_directory_iterator(graph_dir)) {
                if (std::filesystem::is_directory(graph_entry))
                    continue;
                std::string graph_path_str = graph_entry.path();

                log.open(log_file, std::ios_base::app);
                log << "Start Graph: " + graph_path_str + "\n";
                log.close();

                try {
                    std::string filename_graph = graph_path_str;
                    std::string name_graph = filename_graph.substr(
                        filename_graph.rfind("/") + 1, filename_graph.rfind(".") - filename_graph.rfind("/") - 1);

                    std::string filename_machine = machine_path_str;
                    std::string name_machine = filename_machine.substr(
                        filename_machine.rfind("/") + 1, filename_machine.rfind(".") - filename_machine.rfind("/") - 1);

                    // std::cout << name_graph << " - " << name_machine << std::endl;

                    std::cout << std::endl << "Running graph: " << name_graph << " - " << name_machine << std::endl;

                    BspInstance bsp_instance;

                    bool mtx_format = false;
                    bool graph_status = false;

                    if (filename_graph.substr(filename_graph.rfind(".") + 1) == "txt") {
                        std::tie(graph_status, bsp_instance.getComputationalDag()) = FileReader::readComputationalDagHyperdagFormat(filename_graph);

                    } else if (filename_graph.substr(filename_graph.rfind(".") + 1) == "mtx") {
                        mtx_format = true;
                        std::tie(graph_status, bsp_instance.getComputationalDag()) = FileReader::readComputationalDagMartixMarketFormat(filename_graph);
 
                    } else if (filename_graph.substr(filename_graph.rfind(".") + 1) == "dot") {
                        std::tie(graph_status, bsp_instance.getComputationalDag()) = FileReader::readComputationalDagDotFormat(filename_graph);

                    } else {
                        log.open(log_file, std::ios_base::app);
                        log << "Unknown file ending: ." << filename_graph.substr(filename_graph.rfind(".") + 1)
                            << " ...assuming hyperDag format." << std::endl;
                        log.close();
                        std::tie(graph_status, bsp_instance.getComputationalDag()) = FileReader::readComputationalDagHyperdagFormat(filename_graph);
                    }

                    bool arch_status = false;
                    std::tie(arch_status, bsp_instance.getArchitecture()) = FileReader::readBspArchitecture(filename_machine);

                    ComputationalDag &graph = bsp_instance.getComputationalDag();

                    if (!graph_status) {
                        throw std::invalid_argument("Reading graph file " + filename_graph + " failed.");
                    }

                    if (!arch_status) {
                        throw std::invalid_argument("Reading architecture file " + filename_machine + " failed.");
                    }

                    BspSptrsvCSR simulator(bsp_instance, not mtx_format);
                    
                    const auto ref_solution = simulator.compute_sptrsv();

                    std::vector<std::string> schedulers_name(parser.scheduler.size(), "");
                    std::vector<bool> schedulers_failed(parser.scheduler.size(), false);
                    std::vector<unsigned> schedulers_costs(parser.scheduler.size(), 0);
                    std::vector<unsigned> schedulers_work_costs(parser.scheduler.size(), 0);
                    std::vector<unsigned> schedulers_base_comm_costs(parser.scheduler.size(), 0);
                    std::vector<unsigned> schedulers_supersteps(parser.scheduler.size(), 0);
                    std::vector<unsigned> schedulers_base_buffered_sending(parser.scheduler.size(), 0);
                    std::vector<double> schedulers_base_costsTotalCommunication(parser.scheduler.size(), 0);
                    std::vector<long unsigned> schedulers_compute_time(parser.scheduler.size(), 0);
                    std::vector<double> schedulers_sptrsv_simulation_time(parser.scheduler.size(), 0);

                    size_t algorithm_counter = 0;
                    for (auto &algorithm : parser.scheduler) {
                        schedulers_name[algorithm_counter] =
                            algorithm.second.get_child("name").get_value<std::string>();

                        log.open(log_file, std::ios_base::app);
                        log << "Start Algorithm " + schedulers_name[algorithm_counter] + "\n";
                        log.close();

                        try {
                            const auto start_time = std::chrono::high_resolution_clock::now();

                            auto [return_status, schedule] =
                                run_algorithm(parser, algorithm.second, bsp_instance,
                                              parser.global_params.get_child("timeLimit").get_value<unsigned>(), false);

                            const auto finish_time = std::chrono::high_resolution_clock::now();

                            schedulers_compute_time[algorithm_counter] =
                                std::chrono::duration_cast<std::chrono::milliseconds>(finish_time - start_time).count();

                            if (return_status != RETURN_STATUS::SUCCESS && return_status != RETURN_STATUS::BEST_FOUND) {
                                if (return_status == RETURN_STATUS::ERROR) {
                                    throw std::runtime_error(
                                        "Error while computing schedule " +
                                        algorithm.second.get_child("name").get_value<std::string>() + ".");
                                }
                                if (return_status == RETURN_STATUS::TIMEOUT) {
                                    throw std::runtime_error(
                                        "Scheduler " + algorithm.second.get_child("name").get_value<std::string>() +
                                        " timed out.");
                                }
                            }

                            schedulers_costs[algorithm_counter] = schedule.computeCosts();
                            schedulers_work_costs[algorithm_counter] = schedule.computeWorkCosts();
                            schedulers_base_comm_costs[algorithm_counter] = schedule.computeBaseCommCost();
                            schedulers_supersteps[algorithm_counter] = schedule.numberOfSupersteps();
                            schedulers_base_buffered_sending[algorithm_counter] =
                                schedule.computeBaseCommCostsBufferedSending();
                            schedulers_base_costsTotalCommunication[algorithm_counter] =
                                schedule.computeBaseCommCostsTotalCommunication();

                            BspScheduleWriter sched_writer(schedule);
                            if (parser.global_params.get_child("outputSchedule").get_value<bool>()) {
                                try {
                                    sched_writer.write_txt(schedule_dir + name_graph + "_" + name_machine + "_" +
                                                           algorithm.second.get_child("name").get_value<std::string>() +
                                                           "_schedule.txt");
                                } catch (std::exception &e) {
                                    log.open(log_file, std::ios_base::app);
                                    log << "Writing schedule file for " + name_graph + ", " + name_machine + ", " +
                                               schedulers_name[algorithm_counter] + " has failed."
                                        << std::endl;
                                    log << e.what() << std::endl;
                                    log.close();
                                }
                            }

                            if (parser.global_params.get_child("outputSankeySchedule").get_value<bool>()) {
                                try {
                                    sched_writer.write_sankey(
                                        schedule_dir + name_graph + "_" + name_machine + "_" +
                                        algorithm.second.get_child("name").get_value<std::string>() + "_sankey.sankey");
                                } catch (std::exception &e) {
                                    log.open(log_file, std::ios_base::app);
                                    log << "Writing sankey file for " + name_graph + ", " + name_machine + ", " +
                                               schedulers_name[algorithm_counter] + " has failed."
                                        << std::endl;
                                    log << e.what() << std::endl;
                                    log.close();
                                }
                            }
                            if (parser.global_params.get_child("outputDotSchedule").get_value<bool>()) {
                                try {
                                    sched_writer.write_dot(schedule_dir + name_graph + "_" + name_machine + "_" +
                                                           algorithm.second.get_child("name").get_value<std::string>() +
                                                           "_schedule.dot");
                                } catch (std::exception &e) {
                                    log.open(log_file, std::ios_base::app);
                                    log << "Writing dot file for " + name_graph + ", " + name_machine + ", " +
                                               schedulers_name[algorithm_counter] + " has failed."
                                        << std::endl;
                                    log << e.what() << std::endl;
                                    log.close();
                                }
                            }

                            const auto start_perm_time = std::chrono::high_resolution_clock::now();
                            std::vector<size_t> perm;
                            if (schedulers_name[algorithm_counter] == "Serial") {
                                perm = std::vector<size_t>(bsp_instance.numberOfVertices());
                                std::iota(perm.begin(), perm.end(), 0);

                            } else {

                                perm = schedule_node_permuter_basic(schedule, LOOP_PROCESSORS);
                            }
                            const auto fin_perm_time = std::chrono::high_resolution_clock::now();

#ifdef NO_PERMUTE
                            simulator.setup_csr_no_permutation(schedule);
                       
#else
                            simulator.setup_csr(schedule, perm);
#endif

                            const auto fin_reorder_time = std::chrono::high_resolution_clock::now();

                            std::cout << "Permutation time: "
                                      << (double)std::chrono::duration_cast<std::chrono::nanoseconds>(fin_perm_time -
                                                                                                      start_perm_time)
                                                 .count() /
                                             1000000000
                                      << ", Reordering time: "
                                      << (double)std::chrono::duration_cast<std::chrono::nanoseconds>(fin_reorder_time -
                                                                                                      fin_perm_time)
                                                 .count() /
                                             1000000000
                                      << ", Time to compute schedule: "
                                      << (double)schedulers_compute_time[algorithm_counter] / 1000 << std::endl;

                            if (schedulers_name[algorithm_counter] == "Serial") {
#ifdef NO_PERMUTE
                                throw std::invalid_argument("Serial not supported with NO_PERMUTE");
#else
                                simulator.simulate_sptrsv_serial();
#endif
                            } else {
#ifdef NO_PERMUTE
                                simulator.simulate_sptrsv_no_permutation();
                                //simulator.simulate_sptrsv_graph_mtx();
#else
                                simulator.simulate_sptrsv();
#endif
                            }

                            simulator.reset_x();

                            std::cout << "Starting actual simulation" << std::endl;

                            for (unsigned record = 0; record < 100; record++) {

                                double inner_record;

                                if (schedulers_name[algorithm_counter] == "Serial") {

                                    const auto start_time = std::chrono::high_resolution_clock::now();

                                    simulator.simulate_sptrsv_serial();

                                    const auto finish_time = std::chrono::high_resolution_clock::now();

                                    simulator.reset_x();

                                    inner_record = ((double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                        finish_time - start_time)
                                                        .count() /
                                                    1000000000);

                                } else {

                                    const auto start_time = std::chrono::high_resolution_clock::now();
#ifdef NO_PERMUTE
                                    
                                    simulator.simulate_sptrsv_no_permutation();
                                    //simulator.simulate_sptrsv_graph_mtx();
#else
                                    simulator.simulate_sptrsv();
#endif

                                    const auto finish_time = std::chrono::high_resolution_clock::now();

                                    auto result = simulator.get_result();

#ifndef NO_PERMUTE
                                    simulator.permute_vector(result, perm);
#endif
                                    if (not check_vectors_equal(result, ref_solution)) {

                                        std::cout << "Solution not equal to reference solution!" << std::endl;
                                    } 

                                    simulator.reset_x();

                                    inner_record = ((double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                        finish_time - start_time)
                                                        .count() /
                                                    1000000000);
                                }

                                schedulers_sptrsv_simulation_time[algorithm_counter] = inner_record;

                                // std::cout << schedulers_sptrsv_simulation_time[algorithm_counter] << std::endl;

                                stats.open(stats_file, std::ios_base::app);
                                stats << name_graph << "," << name_machine << "," << schedulers_name[algorithm_counter]
#ifdef NO_PERMUTE
                                      << "," << "NO_PERMUTE" << ","
#else
                                      << "," << "LOOP_PROCESSORS" << ","
#endif
                                      << schedulers_sptrsv_simulation_time[algorithm_counter] << ","
                                      << schedulers_work_costs[algorithm_counter] << ","
                                      << schedulers_base_comm_costs[algorithm_counter] << ","
                                      << schedulers_supersteps[algorithm_counter] << ","
                                      << schedulers_base_buffered_sending[algorithm_counter] << ","
                                      << schedulers_base_costsTotalCommunication[algorithm_counter] << ","
                                      << schedulers_compute_time[algorithm_counter] << "\n";
                                stats.close();
                            }
                        } catch (std::runtime_error &e) {
                            schedulers_failed[algorithm_counter] = true;
                            log.open(log_file, std::ios_base::app);
                            log << "Runtime error during execution of Scheduler " +
                                       algorithm.second.get_child("name").get_value<std::string>() + "."
                                << std::endl;
                            log << e.what() << std::endl;
                            log.close();
                        } catch (std::logic_error &e) {
                            schedulers_failed[algorithm_counter] = true;
                            log.open(log_file, std::ios_base::app);
                            log << "Logic error during execution of Scheduler " +
                                       algorithm.second.get_child("name").get_value<std::string>() + "."
                                << std::endl;
                            log << e.what() << std::endl;
                            log.close();
                        } catch (std::exception &e) {
                            schedulers_failed[algorithm_counter] = true;
                            log.open(log_file, std::ios_base::app);
                            log << "Error during execution of Scheduler " +
                                       algorithm.second.get_child("name").get_value<std::string>() + "."
                                << std::endl;
                            log << e.what() << std::endl;
                            log.close();
                        } catch (...) {
                            schedulers_failed[algorithm_counter] = true;
                            log.open(log_file, std::ios_base::app);
                            log << "Unkown error during execution of Scheduler " +
                                       algorithm.second.get_child("name").get_value<std::string>() + "."
                                << std::endl;
                            log.close();
                        }
                        algorithm_counter++;
                    }

                    int tw = 1, ww = 1, cw = 1, nsw = 1, ct = 1;
                    for (size_t i = 0; i < parser.scheduler.size(); i++) {
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
                    for (size_t j = 0; j < parser.scheduler.size(); j++) {
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
                                      << schedulers_supersteps[i] << "     compute time:  " << std::right
                                      << std::setw(ct) << schedulers_compute_time[i] << "ms"
                                      << "     scheduler:  " << schedulers_name[i] << std::endl;
                        }
                    }
                } catch (std::invalid_argument &e) {
                    log.open(log_file, std::ios_base::app);
                    log << e.what() << std::endl;
                    log.close();
                } catch (std::exception &e) {
                    log.open(log_file, std::ios_base::app);
                    log << "Error during execution of Instance " + graph_path_str + " " + machine_path_str + "."
                        << std::endl;
                    log << e.what() << std::endl;
                    log.close();
                } catch (...) {
                    log.open(log_file, std::ios_base::app);
                    log << "Unknown error during execution of Instance  " + graph_path_str + " " + machine_path_str +
                               "."
                        << std::endl;
                    log.close();
                }
            }
        }
    } catch (std::invalid_argument &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}
