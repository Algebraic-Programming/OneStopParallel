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

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include "ConfigParser.hpp"
#include "StatsModules/IStatsModule.hpp"
#include "bsp/model/BspInstance.hpp"
#include "io/arch_file_reader.hpp"
#include "io/dot_graph_file_reader.hpp"
#include "io/hdag_graph_file_reader.hpp"
#include "io/mtx_graph_file_reader.hpp"

namespace osp {

namespace pt = boost::property_tree;

template<typename TargetObjectType, typename GraphType>
class AbstractTestSuiteRunner {
  protected:
    std::string executable_dir;
    ConfigParser parser;
    std::ofstream log_stream;
    std::ofstream stats_out_stream;
    std::vector<std::string> all_csv_headers;
    std::vector<std::unique_ptr<IStatisticModule<TargetObjectType>>> active_stats_modules;

    std::string graph_dir_path, machine_dir_path, output_target_object_dir_path, log_file_path,
        statistics_output_file_path;
    bool write_target_object_to_file = false;
    unsigned time_limit_seconds = 0;

    virtual std::filesystem::path getExecutablePath() const { return std::filesystem::canonical("/proc/self/exe"); }

    virtual bool parse_common_config() {
        try {
            executable_dir = getExecutablePath().remove_filename().string();
            time_limit_seconds = parser.global_params.get_child("timeLimit").get_value<unsigned>();
            write_target_object_to_file =
                parser.global_params.get_child("outputSchedule").get_value_optional<bool>().value_or(false);

            graph_dir_path = parser.global_params.get_child("graphDirectory").get_value<std::string>();
            if (graph_dir_path.substr(0, 1) != "/")
                graph_dir_path = executable_dir + graph_dir_path;

            machine_dir_path = parser.global_params.get_child("archDirectory").get_value<std::string>();
            if (machine_dir_path.substr(0, 1) != "/")
                machine_dir_path = executable_dir + machine_dir_path;

            if (write_target_object_to_file) {
                output_target_object_dir_path = parser.global_params.get_child("scheduleDirectory")
                                                    .get_value<std::string>(); 
                if (output_target_object_dir_path.substr(0, 1) != "/")
                    output_target_object_dir_path = executable_dir + output_target_object_dir_path;
                if (!output_target_object_dir_path.empty() && !std::filesystem::exists(output_target_object_dir_path)) {
                    std::filesystem::create_directories(output_target_object_dir_path);
                }
            }

            log_file_path = parser.global_params.get_child("outputLogFile").get_value<std::string>();
            if (log_file_path.substr(0, 1) != "/")
                log_file_path = executable_dir + log_file_path;

            statistics_output_file_path =
                parser.global_params.get_child("outputStatsFile").get_value<std::string>();
            if (statistics_output_file_path.substr(0, 1) != "/")
                statistics_output_file_path = executable_dir + statistics_output_file_path;

            return true;
        } catch (const std::exception &e) {
            std::cerr << "Error, invalid common config: " << e.what() << std::endl;
            return false;
        }
    }

    virtual void setup_log_file() {
        log_stream.open(log_file_path, std::ios_base::app);
        if (!log_stream.is_open()) {
            std::cerr << "Error: Could not open log file: " << log_file_path << std::endl;
        }
    }

    virtual void setup_statistics_file() {
        all_csv_headers = {"Graph", "Machine", "Algorithm", "TimeToCompute"};

        std::set<std::string> unique_module_metric_headers;
        for (const auto &mod : active_stats_modules) {
            for (const auto &header : mod->get_metric_headers()) {
                auto pair = unique_module_metric_headers.insert(header);

                if (!pair.second) {
                    log_stream << "Warning: Duplicate metric header '" << header
                               << "' found across statistic modules. Using the first one encountered." << std::endl;
                }
            }
        }

        all_csv_headers.insert(all_csv_headers.end(), unique_module_metric_headers.begin(),
                               unique_module_metric_headers.end());

        std::filesystem::path stats_p(statistics_output_file_path);
        if (stats_p.has_parent_path() && !std::filesystem::exists(stats_p.parent_path())) {
            std::filesystem::create_directories(stats_p.parent_path());
        }

        bool file_exists_and_has_header = false;
        std::ifstream stats_file_check(statistics_output_file_path);
        if (stats_file_check.is_open()) {
            std::string first_line_in_file;
            getline(stats_file_check, first_line_in_file);
            std::string expected_header_line;
            for (size_t i = 0; i < all_csv_headers.size(); ++i) {
                expected_header_line += all_csv_headers[i] + (i == all_csv_headers.size() - 1 ? "" : ",");
            }
            if (first_line_in_file == expected_header_line) {
                file_exists_and_has_header = true;
            }
            stats_file_check.close();
        }

        stats_out_stream.open(statistics_output_file_path, std::ios_base::app);
        if (!stats_out_stream.is_open()) {
            log_stream << "CRITICAL ERROR: Could not open statistics output file: " << statistics_output_file_path
                       << std::endl;
            std::cerr << "CRITICAL ERROR: Could not open statistics output file: " << statistics_output_file_path
                      << std::endl;
        } else if (!file_exists_and_has_header) {
            for (size_t i = 0; i < all_csv_headers.size(); ++i) {
                stats_out_stream << all_csv_headers[i] << (i == all_csv_headers.size() - 1 ? "" : ",");
            }
            stats_out_stream << "\n";
            log_stream << "Initialized statistics file " << statistics_output_file_path << " with header." << std::endl;
        }
    }

    virtual RETURN_STATUS compute_target_object_impl(const BspInstance<GraphType> &instance, TargetObjectType* target_object,
                                                        const pt::ptree &algo_config,  
                                                        long long &computation_time_ms) = 0;

    virtual void create_and_register_statistic_modules(const std::string &module_name) = 0;

    virtual void write_target_object_hook(const TargetObjectType &, const std::string &, const std::string &,
                                          const std::string &) {
    } // default in case TargetObjectType cannot be written to file

  public:
    AbstractTestSuiteRunner() {}

    virtual ~AbstractTestSuiteRunner() {
        if (log_stream.is_open())
            log_stream.close();
        if (stats_out_stream.is_open())
            stats_out_stream.close();
    }

    int run(int argc, char *argv[]) {

        try {
            parser.parse_args(argc, argv);
        } catch (const std::exception &e) {
            std::cerr << "Error parsing command line arguments: " << e.what() << std::endl;
            return 1;
        }

        if (!parse_common_config())
            return 1;

        setup_log_file();

        std::vector<std::string> active_module_names_from_config;
        try {
            for (const auto &item : parser.global_params.get_child("activeStatisticModules")) {
                active_module_names_from_config.push_back(item.second.get_value<std::string>());
            }
        } catch (const pt::ptree_bad_path &e) {
            log_stream << "Warning: 'activeStatisticModules' not found. No statistics modules will be run. " << e.what()
                       << std::endl;
        }

        for (const std::string &module_name : active_module_names_from_config) {
            create_and_register_statistic_modules(module_name);
        }

        if (active_stats_modules.empty()) {
            log_stream << "No active statistic modules configured or loaded." << std::endl;
        }

        setup_statistics_file();

        for (const auto &machine_entry : std::filesystem::recursive_directory_iterator(machine_dir_path)) {
            if (std::filesystem::is_directory(machine_entry)) {
                log_stream << "Skipping directory " << machine_entry.path().string() << std::endl;
                continue;
            }
            std::string filename_machine = machine_entry.path().string();
            std::string name_machine = filename_machine.substr(filename_machine.rfind('/') + 1);
            if (name_machine.rfind('.') != std::string::npos)
                name_machine = name_machine.substr(0, name_machine.rfind('.'));

            BspArchitecture<GraphType> arch;
            if (!file_reader::readBspArchitecture(filename_machine, arch)) {
                log_stream << "Reading architecture file " << filename_machine << " failed." << std::endl;
                continue;
            }
            log_stream << "Start Machine: " + filename_machine + "\n";

            for (const auto &graph_entry : std::filesystem::recursive_directory_iterator(graph_dir_path)) {
                if (std::filesystem::is_directory(graph_entry)) {
                    log_stream << "Skipping directory " << graph_entry.path().string() << std::endl;
                    continue;
                }
                std::string filename_graph = graph_entry.path().string();
                std::string name_graph = filename_graph.substr(filename_graph.rfind('/') + 1);
                if (name_graph.rfind('.') != std::string::npos)
                    name_graph = name_graph.substr(0, name_graph.rfind('.'));
                log_stream << "Start Graph: " + filename_graph + "\n";

                BspInstance<GraphType> bsp_instance;
                bsp_instance.setArchitecture(arch);
                bool graph_status = false;
                std::string ext;
                if (filename_graph.rfind('.') != std::string::npos)
                    ext = filename_graph.substr(filename_graph.rfind('.') + 1);

                if (ext == "hdag")
                    graph_status = file_reader::readComputationalDagHyperdagFormat(filename_graph,
                                                                                   bsp_instance.getComputationalDag());
                else if (ext == "mtx")
                    graph_status = file_reader::readComputationalDagMartixMarketFormat(
                        filename_graph, bsp_instance.getComputationalDag());
                else if (ext == "dot")
                    graph_status =
                        file_reader::readComputationalDagDotFormat(filename_graph, bsp_instance.getComputationalDag());
                else {
                    log_stream << "Unknown file ending: ." << ext << " ...assuming hyperDag format." << std::endl;
                    graph_status = file_reader::readComputationalDagHyperdagFormat(filename_graph,
                                                                                   bsp_instance.getComputationalDag());
                }

                if (!graph_status) {
                    log_stream << "Reading graph file " << filename_graph << " failed." << std::endl;
                    continue;
                }

                for (auto &algorithm_config_pair : parser.scheduler) {
                    const pt::ptree &algo_config = algorithm_config_pair.second;
                    
                    std::string name_suffix = "";
                    try {
                        name_suffix = algo_config.get_child("name_suffix").get_value<std::string>();
                    } catch (const pt::ptree_bad_path &e) {
                    }
                    
                    if (!name_suffix.empty())
                        name_suffix = "_" + name_suffix;


                    std::string current_algo_name =
                        algo_config.get_child("name").get_value<std::string>() + name_suffix;
                    log_stream << "Start Algorithm " + current_algo_name + "\n";

                    
                    long long computation_time_ms;
                    TargetObjectType *target_object = nullptr; 
                    
                    RETURN_STATUS exec_status = compute_target_object_impl(bsp_instance, target_object, algo_config, computation_time_ms);

                    if (exec_status != RETURN_STATUS::SUCCESS && exec_status != RETURN_STATUS::BEST_FOUND) {
                        if (exec_status == RETURN_STATUS::ERROR)
                            log_stream << "Error computing with " << current_algo_name << "." << std::endl;
                        else if (exec_status == RETURN_STATUS::TIMEOUT)
                            log_stream << "Scheduler " << current_algo_name << " timed out." << std::endl;

                        delete target_object;
                            
                        continue;
                    }

                    if (write_target_object_to_file) {
                        try {
                            write_target_object_hook(*target_object, name_graph, name_machine, current_algo_name);
                        } catch (const std::exception &e) {
                            log_stream << "Writing target object file for " << name_graph << ", " << name_machine
                                       << ", " << current_algo_name << " has failed: " << e.what() << std::endl;
                        }
                    }

                    if (stats_out_stream.is_open()) {
                        std::map<std::string, std::string> current_row_values;
                        current_row_values["Graph"] = name_graph;
                        current_row_values["Machine"] = name_machine;
                        current_row_values["Algorithm"] = current_algo_name;
                        current_row_values["TimeToCompute"] = std::to_string(computation_time_ms);

                        for (auto &stat_module : active_stats_modules) {
                            auto module_metrics = stat_module->record_statistics(*target_object, log_stream);
                            current_row_values.insert(module_metrics.begin(), module_metrics.end());
                        }

                        for (size_t i = 0; i < all_csv_headers.size(); ++i) {
                            stats_out_stream << current_row_values[all_csv_headers[i]]
                                             << (i == all_csv_headers.size() - 1 ? "" : ",");
                        }
                        stats_out_stream << "\n";
                    }

                    delete target_object;
                }
            }
        }
        return 0;
    }
};

} // namespace osp
