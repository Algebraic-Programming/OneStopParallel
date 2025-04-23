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

#include "file_interactions/CommandLineParser.hpp"
#include "file_interactions/FileReader.hpp"
#include "model/BspSchedule.hpp"

#include "auxiliary/run_algorithm.hpp"

namespace pt = boost::property_tree;

std::filesystem::path getExecutablePath() { return std::filesystem::canonical("/proc/self/exe"); }

void write_stats_header(const std::string &stats_file, const std::string &header, std::ofstream &stats) {

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
}

// invoked upon program call
int main(int argc, char *argv[]) {

    std::string executable_dir = getExecutablePath().remove_filename().string();
    std::string main_config_location = executable_dir + "main_config.json";

    const CommandLineParser parser(argc, argv, main_config_location);

    std::string graph_dir, machine_dir, schedule_dir, stats_file, log_file;
    bool write_schedule = false;
    unsigned time_limit = 0;
    bool use_memory_constraint = false; 

    try {

        
        time_limit = parser.global_params.get_child("timeLimit").get_value<unsigned>();
        write_schedule = parser.global_params.get_child("outputSchedule").get_value_optional<bool>().value_or(false);
        use_memory_constraint = parser.global_params.get_child("useMemoryConstraint").get_value_optional<bool>().value_or(false);

        graph_dir = parser.global_params.get_child("graphDirectory").get_value<std::string>();
        if (graph_dir.substr(0, 1) != "/") {
            graph_dir = executable_dir + graph_dir;
        }

        machine_dir = parser.global_params.get_child("machineDirectory").get_value<std::string>();
        if (machine_dir.substr(0, 1) != "/") {
            machine_dir = executable_dir + machine_dir;
        }

        schedule_dir = parser.global_params.get_child("scheduleDirectory").get_value<std::string>();
        if (schedule_dir.substr(0, 1) != "/") {
            schedule_dir = executable_dir + schedule_dir;
        }

        stats_file = parser.global_params.get_child("outputStatsFile").get_value<std::string>();
        if (stats_file.substr(0, 1) != "/") {
            stats_file = executable_dir + stats_file;
        }

        log_file = parser.global_params.get_child("outputLogFile").get_value<std::string>();
        if (log_file.substr(0, 1) != "/") {
            log_file = executable_dir + log_file;
        }

    } catch (const std::exception &e) {
        std::cerr << "Error, invalid config file: " << e.what() << std::endl;
        return 1;
    }

    std::ofstream log;
    std::ofstream stats;

    std::string header = "Graph,Machine,Algorithm,BspCost,TotalCommCost,BufferedSendingCosts,WorkCosts,Supersteps,TimeToCompute";

    write_stats_header(stats_file, header, stats);

    log.open(log_file, std::ios_base::app);

    for (const auto &machine_entry : std::filesystem::recursive_directory_iterator(machine_dir)) {

        if (std::filesystem::is_directory(machine_entry)) {
            log << "Skipping directory " << machine_entry.path() << std::endl;
            continue;
        }

        std::string machine_path_str = machine_entry.path();
        std::string filename_machine = machine_path_str;
        std::string name_machine = filename_machine.substr(
            filename_machine.rfind("/") + 1, filename_machine.rfind(".") - filename_machine.rfind("/") - 1);

        BspArchitecture arch;

        bool arch_status = false;
        std::tie(arch_status, arch) = FileReader::readBspArchitecture(filename_machine);

        if (not arch_status) {
            log << "Reading architecture file " << filename_machine << " failed." << std::endl;
            continue;
        }

        log << "Start Machine: " + machine_path_str + "\n";

        for (const auto &graph_entry : std::filesystem::recursive_directory_iterator(graph_dir)) {

            if (std::filesystem::is_directory(graph_entry)) {
                log << "Skipping directory " << graph_entry.path() << std::endl;
                continue;
            }

            std::string graph_path_str = graph_entry.path();
            log << "Start Graph: " + graph_path_str + "\n";

            std::string filename_graph = graph_path_str;
            std::string name_graph = filename_graph.substr(filename_graph.rfind("/") + 1,
                                                           filename_graph.rfind(".") - filename_graph.rfind("/") - 1);

            BspInstance bsp_instance;
            bsp_instance.setArchitecture(arch);
            bool graph_status = false;

            if (filename_graph.substr(filename_graph.rfind(".") + 1) == "txt") {
                std::tie(graph_status, bsp_instance.getComputationalDag()) =
                    FileReader::readComputationalDagHyperdagFormat(filename_graph);

            } else if (filename_graph.substr(filename_graph.rfind(".") + 1) == "mtx") {
                std::tie(graph_status, bsp_instance.getComputationalDag()) =
                    FileReader::readComputationalDagMartixMarketFormat(filename_graph);

            } else if (filename_graph.substr(filename_graph.rfind(".") + 1) == "dot") {
                std::tie(graph_status, bsp_instance.getComputationalDag()) =
                    FileReader::readComputationalDagDotFormat(filename_graph);

            } else {
                log << "Unknown file ending: ." << filename_graph.substr(filename_graph.rfind(".") + 1)
                    << " ...assuming hyperDag format." << std::endl;
                std::tie(graph_status, bsp_instance.getComputationalDag()) =
                    FileReader::readComputationalDagHyperdagFormat(filename_graph);
            }

            if (!graph_status) {
                log << "Reading graph file " << filename_graph << " failed." << std::endl;
                continue;
            }

            bsp_instance.getComputationalDag().mergeMultipleEdges();
            bsp_instance.setAllOnesCompatibilityMatrix();

            // uncomment the two lines below if testing for AST
            // bsp_instance.getArchitecture().setProcessorType(0, 1);
            // bsp_instance.setDiagonalCompatibilityMatrix(2);

            // const std::vector<unsigned> min_mem_per_node_type = BspMemSchedule::minimumMemoryRequiredPerNodeType(bsp_instance);
            // std::vector<unsigned> mem_bound_per_proc_type(bsp_instance.getArchitecture().getNumberOfProcessorTypes(), 0);

            // for (unsigned node_type = 0; node_type < bsp_instance.getComputationalDag().getNumberOfNodeTypes(); ++node_type)
            //     for (unsigned proc_type = 0; proc_type < bsp_instance.getArchitecture().getNumberOfProcessorTypes(); ++proc_type)
            //         if(bsp_instance.isCompatibleType(node_type, proc_type))
            //         {
            //             unsigned bound_for_this_node_type = memory_bound_factor * (double) min_mem_per_node_type[node_type];
            //             mem_bound_per_proc_type[proc_type]=std::max(bound_for_this_node_type, mem_bound_per_proc_type[proc_type]);
            //         }

            // for (unsigned proc = 0; proc < bsp_instance.getArchitecture().numberOfProcessors(); ++proc) {
            //     unsigned bound = mem_bound_per_proc_type[bsp_instance.getArchitecture().processorType(proc)];
            //     bsp_instance.getArchitecture().setMemoryBound(bound, proc);
            //     log << "Setting memory bound to " << bound << " for each processor of type " << bsp_instance.getArchitecture().processorType(proc) << "." << std::endl;
            // }


            const unsigned num_algs = parser.scheduler.size();
            std::vector<std::string> schedulers_name(num_algs, "");
            std::vector<unsigned> schedulers_bsp_costs(num_algs, 0);
            std::vector<double> schedulers_total_costs(num_algs, 0);
            std::vector<unsigned> schedulers_buffered_sending_costs(num_algs, 0);
            std::vector<unsigned> schedulers_work_costs(num_algs, 0);
            std::vector<unsigned> schedulers_supersteps(num_algs, 0);
            std::vector<long unsigned> schedulers_compute_time(num_algs, 0);


            size_t algorithm_counter = 0;
            for (auto &algorithm : parser.scheduler) {

                std::string name_suffix = algorithm.second.get_child("name_suffix").get_value_optional<std::string>().value_or("");
                
                if (name_suffix != "") {
                    name_suffix = "_" + name_suffix;
                }

                schedulers_name[algorithm_counter] = algorithm.second.get_child("name").get_value<std::string>() + name_suffix;

                log << "Start Algorithm " + schedulers_name[algorithm_counter] + "\n";

                const auto start_time = std::chrono::high_resolution_clock::now();

                auto [return_status, schedule] = run_algorithm(parser, algorithm.second, bsp_instance, time_limit, use_memory_constraint);

                const auto finish_time = std::chrono::high_resolution_clock::now();

                schedulers_compute_time[algorithm_counter] =
                    std::chrono::duration_cast<std::chrono::milliseconds>(finish_time - start_time).count();

                if (return_status != RETURN_STATUS::SUCCESS && return_status != RETURN_STATUS::BEST_FOUND) {
                    if (return_status == RETURN_STATUS::ERROR) {
                        log << "Error while computing schedule " +
                                   algorithm.second.get_child("name").get_value<std::string>() + "."
                            << std::endl;
                        continue;
                    }
                    if (return_status == RETURN_STATUS::TIMEOUT) {
                        log << "Scheduler " + algorithm.second.get_child("name").get_value<std::string>() +
                                   " timed out."
                            << std::endl;
                        continue;
                    }
                }

                schedulers_bsp_costs[algorithm_counter] = schedule.computeCosts();
                schedulers_total_costs[algorithm_counter] = schedule.computeCostsTotalCommunication();
                schedulers_buffered_sending_costs[algorithm_counter] = schedule.computeCostsBufferedSending();
                schedulers_work_costs[algorithm_counter] = schedule.computeWorkCosts();
                schedulers_supersteps[algorithm_counter] = schedule.numberOfSupersteps();
 

                // if (write_schedule) {
                //     BspScheduleWriter sched_writer(schedule);
                //     try {
                //         sched_writer.write_txt(schedule_dir + name_graph + "_" + name_machine + "_" +
                //                                algorithm.second.get_child("name").get_value<std::string>() +
                //                                "_schedule.txt");
                //     } catch (std::exception &e) {
                //         log << "Writing schedule file for " + name_graph + ", " + name_machine + ", " +
                //                    schedulers_name[algorithm_counter] + " has failed."
                //             << std::endl;
                //         log << e.what() << std::endl;
                //     }
                // }

                stats.open(stats_file, std::ios_base::app);
                stats << name_graph << "," 
                      << name_machine << "," 
                      << schedulers_name[algorithm_counter] << ","
                      << schedulers_bsp_costs[algorithm_counter] << ","
                      << schedulers_total_costs[algorithm_counter] << ","
                      << schedulers_buffered_sending_costs[algorithm_counter] << ","
                      << schedulers_work_costs[algorithm_counter] << ","
                      << schedulers_supersteps[algorithm_counter] << ","
                      << schedulers_compute_time[algorithm_counter]
                      << std::endl;
                    
 
                stats.close();
            }

            algorithm_counter++;
        }
    }

    log.close();

    return 0;
}
