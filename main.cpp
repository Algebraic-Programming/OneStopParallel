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
#include <filesystem>
#include <iostream>
#include <string>
#include <tuple>

#include "algorithms/ImprovementScheduler.hpp"

#include "file_interactions/CommandLineParser.hpp"
#include "file_interactions/FileReader.hpp"
#include "file_interactions/BspScheduleWriter.hpp"
#include "model/BspSchedule.hpp"

#include "auxiliary/run_algorithm.hpp"

namespace pt = boost::property_tree;

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

                BspArchitecture architecture;
                std::string name_machine;
                if (!filename_machine.empty() &&
                        std::find_if(filename_machine.begin(), filename_machine.end(),
                                     [](unsigned char c) { return !std::isdigit(c); }) == filename_machine.end()) {
                    std::cout << "Number of processors: " << filename_machine << std::endl;
                    architecture.setNumberOfProcessors(std::atol(filename_machine.c_str()));
                    name_machine = "p" + filename_machine;
                } else {

                    name_machine = filename_machine.substr(
                        filename_machine.rfind("/") + 1, filename_machine.rfind(".") - filename_machine.rfind("/") - 1);
                    auto [status_architecture, architecture_] = FileReader::readBspArchitecture(filename_machine);
                    
                    architecture = architecture_;

                    if (!status_architecture) {
                        throw std::invalid_argument("Reading architecture file " + filename_machine + " failed.");
                    }
                }

                bool mtx_format = false;
                bool status_graph;
                ComputationalDag graph;
                std::unordered_map<std::pair<VertexType, VertexType>, double, pair_hash> matrix_entries;

                if (filename_graph.substr(filename_graph.rfind(".") + 1) == "txt") {
                    std::pair<bool, ComputationalDag> read_graph = FileReader::readComputationalDagHyperdagFormat(filename_graph);
                    status_graph = read_graph.first;
                    graph = read_graph.second;
                } else if (filename_graph.substr(filename_graph.rfind(".") + 1) == "mtx") {
                    mtx_format = true;
                    std::pair<bool, ComputationalDag> read_graph = FileReader::readComputationalDagMartixMarketFormat(filename_graph);
                    status_graph = read_graph.first;
                    graph = read_graph.second;
                } else if (filename_graph.substr(filename_graph.rfind(".") + 1) == "dot") {
                    std::pair<bool, ComputationalDag> read_graph = FileReader::readComputationalDagDotFormat(filename_graph);
                    status_graph = read_graph.first;
                    graph = read_graph.second;
                } else {
                    std::cout << "Unknown file ending: ." << filename_graph.substr(filename_graph.rfind(".") + 1)
                        << " ...assuming hyperDag format." << std::endl;
                    std::pair<bool, ComputationalDag> read_graph = FileReader::readComputationalDagHyperdagFormat(filename_graph);
                    status_graph = read_graph.first;
                    graph = read_graph.second;
                }

                if (!status_graph) {
                    throw std::invalid_argument("Reading graph file " + filename_graph + " failed.");
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
                                  << schedulers_compute_time[i] << "ms" << "     scheduler:  " << schedulers_name[i]
                                  << std::endl;
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
