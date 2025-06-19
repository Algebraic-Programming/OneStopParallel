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
#include <memory> // For std::unique_ptr
#include <set>    // For std::set to collect unique headers



#include "bsp/model/BspSchedule.hpp" 
#include "graph_implementations/boost_graphs/boost_graph.hpp" // For graph_t definition

#include "io/arch_file_reader.hpp"
#include "io/dot_graph_file_reader.hpp"
#include "io/hdag_graph_file_reader.hpp"
#include "io/mtx_graph_file_reader.hpp"
#include "io/bsp_schedule_file_writer.hpp"
#include "util/CommandLineParser.hpp"
#include "auxiliary/run_algorithm.hpp"
#include "util/BspScheduleStatistics/IStatisticModule.hpp"
#include "util/BspScheduleStatistics/BasicBspStatsModule.hpp"
#include "util/BspScheduleStatistics/CommStatsModule.hpp"


using namespace osp;
namespace pt = boost::property_tree;
using graph_t = boost_graph_int_t;

std::filesystem::path getExecutablePath() { return std::filesystem::canonical("/proc/self/exe"); }


// invoked upon program call
int main(int argc, char *argv[]) {

    std::string executable_dir = getExecutablePath().remove_filename().string();
    std::string main_config_location = executable_dir + "main_config.json";

    const CommandLineParser parser(argc, argv, main_config_location);

    std::string graph_dir, machine_dir, schedule_dir, log_file, statistics_output_file;
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

        log_file = parser.global_params.get_child("outputLogFile").get_value<std::string>();
        if (log_file.substr(0, 1) != "/") {
            log_file = executable_dir + log_file;
        }

        statistics_output_file = parser.global_params.get_child("outputStatisticsFile").get_value<std::string>();
        if (statistics_output_file.substr(0, 1) != "/") {
            statistics_output_file = executable_dir + statistics_output_file;
        }

    } catch (const std::exception &e) {
        std::cerr << "Error, invalid config file: " << e.what() << std::endl;
        return 1;
    }

    std::ofstream log;
    log.open(log_file, std::ios_base::app);
    if (!log.is_open()) {
        std::cerr << "Error: Could not open log file: " << log_file << std::endl;
        return 1;
    }

    // --- Statistics Module Setup ---
    std::vector<std::unique_ptr<IStatisticModule<graph_t>>> active_stats_modules;
    std::vector<std::string> active_module_names_from_config; // = parser.getActiveStatisticModuleNames();
    try {
        for (const auto& item : parser.global_params.get_child("activeStatisticModules")) {
            active_module_names_from_config.push_back(item.second.get_value<std::string>());
        }
    } catch (const pt::ptree_bad_path& e) {
        log << "Warning: 'activeStatisticModules' not found in globalParameters. No statistics modules will be run. " << e.what() << std::endl;
    }

    for (const std::string& module_name : active_module_names_from_config) {
        pt::ptree module_specific_config;
        try {
            // module_specific_config = parser.getStatisticModuleSettings(module_name); // Assumed method
             module_specific_config = parser.global_params.get_child_optional("statisticModuleSettings." + module_name).value_or(pt::ptree());
        } catch (const std::exception& e) {
            log << "Warning: Could not get settings for statistic module " << module_name << ". Using defaults. Error: " << e.what() << std::endl;
        }

        if (module_name == "BasicBspStats") {
            auto module = std::make_unique<BasicBspStatsModule<graph_t>>();
            // Modules no longer initialized here with file paths
            active_stats_modules.push_back(std::move(module)); // Constructor might take module_specific_config if needed
        } else if (module_name == "DetailedCommStats") { // Name from CommStatsModule::get_name()
            auto module = std::make_unique<CommStatsModule<graph_t>>();
            active_stats_modules.push_back(std::move(module));
        }
    }

    if (active_stats_modules.empty()) {
        log << "No active statistic modules configured or loaded." << std::endl;
    }

    // --- Setup Statistics Output File and Header ---
    std::vector<std::string> all_csv_headers = {"Graph", "Machine", "Algorithm", "TimeToCompute"};
    std::set<std::string> unique_module_metric_headers;
    for (const auto& mod : active_stats_modules) {
        for (const auto& header : mod->get_metric_headers()) {
            unique_module_metric_headers.insert(header);
        }
    }
    all_csv_headers.insert(all_csv_headers.end(), unique_module_metric_headers.begin(), unique_module_metric_headers.end());

    std::ofstream stats_out_stream;
    // Ensure directory for stats file exists
    std::filesystem::path stats_p(statistics_output_file);
    if (stats_p.has_parent_path()) {
        std::filesystem::create_directories(stats_p.parent_path());
    }

    bool stats_file_exists_and_has_header = false;
    std::ifstream stats_file_check(statistics_output_file);
    if (stats_file_check.is_open()) {
        std::string first_line_in_file;
        getline(stats_file_check, first_line_in_file);
        std::string expected_header_line;
        for (size_t i = 0; i < all_csv_headers.size(); ++i) {
            expected_header_line += all_csv_headers[i] + (i == all_csv_headers.size() - 1 ? "" : ",");
        }
        if (first_line_in_file == expected_header_line) {
            stats_file_exists_and_has_header = true;
        }
        stats_file_check.close();
    }

    stats_out_stream.open(statistics_output_file, std::ios_base::app);
    if (!stats_out_stream.is_open()) {
        log << "CRITICAL ERROR: Could not open statistics output file: " << statistics_output_file << std::endl;
        std::cerr << "CRITICAL ERROR: Could not open statistics output file: " << statistics_output_file << std::endl;
        // active_stats_modules.clear(); // Prevent attempts to write if file can't be opened
    } else if (!stats_file_exists_and_has_header) {
        for (size_t i = 0; i < all_csv_headers.size(); ++i) {
            stats_out_stream << all_csv_headers[i] << (i == all_csv_headers.size() - 1 ? "" : ",");
        }
        stats_out_stream << "\n";
        log << "Initialized statistics file " << statistics_output_file << " with header." << std::endl;
    }

    for (const auto &machine_entry : std::filesystem::recursive_directory_iterator(machine_dir)) {

        if (std::filesystem::is_directory(machine_entry)) {
            log << "Skipping directory " << machine_entry.path() << std::endl;
            continue;
        }

        std::string machine_path_str = machine_entry.path();
        std::string filename_machine = machine_path_str;
        std::string name_machine = filename_machine.substr(
            filename_machine.rfind("/") + 1, filename_machine.rfind(".") - filename_machine.rfind("/") - 1);

        BspArchitecture<graph_t> arch;

        bool arch_status = file_reader::readBspArchitecture(filename_machine, arch);

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

            BspInstance<graph_t> bsp_instance;
            bsp_instance.setArchitecture(arch);
            bool graph_status = false;

            if (filename_graph.substr(filename_graph.rfind(".") + 1) == "hdag") {
                graph_status = file_reader::readComputationalDagHyperdagFormat(filename_graph, bsp_instance.getComputationalDag());

            } else if (filename_graph.substr(filename_graph.rfind(".") + 1) == "mtx") {
                graph_status = file_reader::readComputationalDagMartixMarketFormat(filename_graph, bsp_instance.getComputationalDag());

            } else if (filename_graph.substr(filename_graph.rfind(".") + 1) == "dot") {
                graph_status = file_reader::readComputationalDagDotFormat(filename_graph, bsp_instance.getComputationalDag());

            } else {
                log << "Unknown file ending: ." << filename_graph.substr(filename_graph.rfind(".") + 1)
                    << " ...assuming hyperDag format." << std::endl;
                graph_status = file_reader::readComputationalDagHyperdagFormat(filename_graph, bsp_instance.getComputationalDag());
            }

            if (!graph_status) {
                log << "Reading graph file " << filename_graph << " failed." << std::endl;
                continue;
            }


            size_t algorithm_counter = 0;
            for (auto &algorithm : parser.scheduler) {

                std::string name_suffix = algorithm.second.get_child("name_suffix").get_value_optional<std::string>().value_or("");
                
                if (name_suffix != "") {
                    name_suffix = "_" + name_suffix;
                }

                std::string current_scheduler_name = algorithm.second.get_child("name").get_value<std::string>() + name_suffix;

                BspSchedule<graph_t> schedule(bsp_instance);

                log << "Start Algorithm " + current_scheduler_name + "\n";

                const auto start_time = std::chrono::high_resolution_clock::now();

                auto return_status = run_algorithm<graph_t>(parser, algorithm.second, schedule);

                const auto finish_time = std::chrono::high_resolution_clock::now();

                long long compute_time_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(finish_time - start_time).count();

                if (return_status != RETURN_STATUS::SUCCESS && return_status != RETURN_STATUS::BEST_FOUND) {
                    if (return_status == RETURN_STATUS::ERROR) {
                        log << "Error while computing schedule " +
                                   algorithm.second.get_child("name").get_value<std::string>() + "."
                            << std::endl;
                        algorithm_counter++; // Ensure counter increments even on error before continue
                        continue;
                    }
                    if (return_status == RETURN_STATUS::TIMEOUT) {
                        log << "Scheduler " + algorithm.second.get_child("name").get_value<std::string>() +
                                   " timed out."
                            << std::endl;
                        continue;
                    }
                }

                // if (write_schedule) {
                //     try {
                //         file_writer::write_txt(schedule_dir + name_graph + "_" + name_machine + "_" +
                //                                        current_scheduler_name + 
                //                                        "_schedule.txt",
                //                                    schedule);
                //     } catch (const std::exception &e) {
                //         log << "Writing schedule file for " + name_graph + ", " + name_machine + ", " +
                //                    current_scheduler_name + " has failed."
                //             << std::endl;
                //         log << e.what() << std::endl;
                //     }
                // }

                // --- Record Statistics ---
                if (stats_out_stream.is_open()) {
                    std::map<std::string, std::string> current_row_values;
                    current_row_values["Graph"] = name_graph;
                    current_row_values["Machine"] = machine_name;
                    current_row_values["Algorithm"] = current_scheduler_name;
                    current_row_values["TimeToCompute"] = std::to_string(compute_time_ms);

                    for (auto& stat_module : active_stats_modules) {
                        auto module_metrics = stat_module->record_statistics(schedule, log);
                        current_row_values.insert(module_metrics.begin(), module_metrics.end());
                    }

                    for (size_t i = 0; i < all_csv_headers.size(); ++i) {
                        stats_out_stream << current_row_values[all_csv_headers[i]] << (i == all_csv_headers.size() - 1 ? "" : ",");
                    }
                    stats_out_stream << "\n";
                }
                algorithm_counter++;
            }
        }
    }

    if (stats_out_stream.is_open()) stats_out_stream.close();
    log.close();

    return 0;
}
