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

#include "file_interactions/CommandLineParserPartition.hpp"
#include "file_interactions/FileReader.hpp"
#include "file_interactions/DAGPartitionWriter.hpp"
#include "model/BspSchedule.hpp"
#include "model/DAGPartition.hpp"

#include "auxiliary/run_algorithm.hpp"

namespace pt = boost::property_tree;

std::filesystem::path getExecutablePath() { return std::filesystem::canonical("/proc/self/exe"); }

// invoked upon program call
int main(int argc, char *argv[]) {

    std::string main_config_location = getExecutablePath().remove_filename().string();
   
    main_config_location += "main_partition_config.json";

    try {
        const CommandLineParserPartition parser(argc, argv, main_config_location);

        for (auto &instance : parser.instances) {
            try {
                std::string filename_graph = instance.second.get_child("graphFile").get_value<std::string>();
                std::string name_graph = filename_graph.substr(
                    filename_graph.rfind("/") + 1, filename_graph.rfind(".") - filename_graph.rfind("/") - 1);

                std::string filename_machine = instance.second.get_child("machineParamsFile").get_value<std::string>();

                BspInstance bsp_instance;
                
                std::string name_machine;
                if (!filename_machine.empty() &&
                        std::find_if(filename_machine.begin(), filename_machine.end(),
                                     [](unsigned char c) { return !std::isdigit(c); }) == filename_machine.end()) {
                    std::cout << "Number of processors: " << filename_machine << std::endl;
                    bsp_instance.getArchitecture().setNumberOfProcessors(std::atol(filename_machine.c_str()));
                    name_machine = "p" + filename_machine;
                } else {

                    name_machine = filename_machine.substr(
                        filename_machine.rfind("/") + 1, filename_machine.rfind(".") - filename_machine.rfind("/") - 1);
                    
                    bool status_architecture = false;
                    std::tie(status_architecture, bsp_instance.getArchitecture()) = FileReader::readBspArchitecture(filename_machine);
                    
                    if (!status_architecture) {
                        throw std::invalid_argument("Reading architecture file " + filename_machine + " failed.");
                    }
                }

                bool mtx_format = false;
                bool status_graph = false;
            
                if (filename_graph.substr(filename_graph.rfind(".") + 1) == "txt") {
                    std::tie(status_graph, bsp_instance.getComputationalDag()) = FileReader::readComputationalDagHyperdagFormat(filename_graph);

                } else if (filename_graph.substr(filename_graph.rfind(".") + 1) == "mtx") {
                    mtx_format = true;
                    std::tie(status_graph, bsp_instance.getComputationalDag()) = FileReader::readComputationalDagMartixMarketFormat(filename_graph);

                } else if (filename_graph.substr(filename_graph.rfind(".") + 1) == "dot") {
                    std::tie(status_graph, bsp_instance.getComputationalDag()) = FileReader::readComputationalDagDotFormat(filename_graph);

                } else {
                    std::cout << "Unknown file ending: ." << filename_graph.substr(filename_graph.rfind(".") + 1)
                        << " ...assuming hyperDag format." << std::endl;
                    std::tie(status_graph, bsp_instance.getComputationalDag()) = FileReader::readComputationalDagHyperdagFormat(filename_graph);
 
                }

                if (!status_graph) {
                    throw std::invalid_argument("Reading graph file " + filename_graph + " failed.");
                }
  
                std::vector<std::string> partitioners_name(parser.scheduler.size(), "");
                std::vector<bool> partitioners_failed(parser.scheduler.size(), false);
                std::vector<bool> partitioners_memory_satisfied(parser.scheduler.size(), true);
                std::vector<float> partitioners_imbalance(parser.scheduler.size(), 0);
                std::vector<unsigned> partitioners_max_memory(parser.scheduler.size(), 0);
                std::vector<long unsigned> partitioners_compute_time(parser.scheduler.size(), 0);

                size_t algorithm_counter = 0;
                for (auto &algorithm : parser.scheduler) {
                    partitioners_name[algorithm_counter] = algorithm.second.get_child("name").get_value<std::string>();

                    try {
                        const auto start_time = std::chrono::high_resolution_clock::now();

                        auto [return_status, partition] =
                            run_algorithm(parser, algorithm.second, bsp_instance,
                                          parser.global_params.get_child("timeLimit").get_value<unsigned>(),
                                          parser.global_params.get_child("use_memory_constraints").get_value<bool>());

                        const auto finish_time = std::chrono::high_resolution_clock::now();

                        partitioners_compute_time[algorithm_counter] =
                            std::chrono::duration_cast<std::chrono::milliseconds>(finish_time - start_time).count();

                        if (return_status != RETURN_STATUS::SUCCESS && return_status != RETURN_STATUS::BEST_FOUND) {
                            if (return_status == RETURN_STATUS::ERROR) {
                                throw std::runtime_error("Error while computing partition " +
                                                         algorithm.second.get_child("name").get_value<std::string>() +
                                                         ".");
                            }
                            if (return_status == RETURN_STATUS::TIMEOUT) {
                                throw std::runtime_error("Partition " +
                                                         algorithm.second.get_child("name").get_value<std::string>() +
                                                         " timed out.");
                            }
                        }

                        partitioners_imbalance[algorithm_counter] = partition.computeWorkImbalance();
                        partitioners_max_memory[algorithm_counter] = partition.computeMaxMemoryCosts();
                        partitioners_memory_satisfied[algorithm_counter] = partition.satisfiesMemoryConstraints();

                        DAGPartitionWriter part_writer(partition);
                        if (parser.global_params.get_child("outputPartition").get_value<bool>()) {
                            try {
                                part_writer.write_txt(name_graph + "_" + name_machine + "_" +
                                                       algorithm.second.get_child("name").get_value<std::string>() +
                                                       "_partition.txt");
                            } catch (std::exception &e) {
                                std::cerr << "Writing partition file for " + name_graph + ", " + name_machine + ", " +
                                                 partitioners_name[algorithm_counter] + " has failed."
                                          << std::endl;
                                std::cerr << e.what() << std::endl;
                            }
                        }

                        if (parser.global_params.get_child("outputDotPartition").get_value<bool>()) {
                            try {
                                part_writer.write_dot(name_graph + "_" + name_machine + "_" +
                                                       algorithm.second.get_child("name").get_value<std::string>() +
                                                       "_partition.dot");
                            } catch (std::exception &e) {
                                std::cerr << "Writing dot file for " + name_graph + ", " + name_machine + ", " +
                                                 partitioners_name[algorithm_counter] + " has failed."
                                          << std::endl;
                                std::cerr << e.what() << std::endl;
                            }
                        }

                    } catch (std::runtime_error &e) {
                        partitioners_failed[algorithm_counter] = true;
                        std::cerr << "Runtime error during execution of Partitioner " +
                                         algorithm.second.get_child("name").get_value<std::string>() + "."
                                  << std::endl;
                        std::cerr << e.what() << std::endl;
                    } catch (std::logic_error &e) {
                        partitioners_failed[algorithm_counter] = true;
                        std::cerr << "Logic error during execution of Partitioner " +
                                         algorithm.second.get_child("name").get_value<std::string>() + "."
                                  << std::endl;
                        std::cerr << e.what() << std::endl;
                    } catch (std::exception &e) {
                        partitioners_failed[algorithm_counter] = true;
                        std::cerr << "Error during execution of Partitioner " +
                                         algorithm.second.get_child("name").get_value<std::string>() + "."
                                  << std::endl;
                        std::cerr << e.what() << std::endl;
                    } catch (...) {
                        partitioners_failed[algorithm_counter] = true;
                        std::cerr << "Error during execution of Partitioner " +
                                         algorithm.second.get_child("name").get_value<std::string>() + "."
                                  << std::endl;
                    }
                    algorithm_counter++;
                }

                int imb = 1, mm = 1, ct = 1;
                for (size_t i = 0; i < parser.scheduler.size(); i++) {
                    if (partitioners_failed[i])
                        continue;
                    imb = std::max(imb, 1 + int(std::log10(partitioners_imbalance[i])));
                    mm = std::max(mm, 1 + int(std::log10(partitioners_max_memory[i])));
                    ct = std::max(ct, 1 + int(std::log10(partitioners_compute_time[i])));
                }

                bool sorted_by_total_costs = true;
                std::vector<size_t> ordering = sorting_arrangement(partitioners_imbalance);

                std::cout << std::endl << name_graph << " - " << name_machine << std::endl;
                std::cout << "Number of Vertices: " + std::to_string(bsp_instance.getComputationalDag().numberOfVertices()) +
                                 "  Number of Edges: " + std::to_string(bsp_instance.getComputationalDag().numberOfEdges())
                          << std::endl;
                for (size_t j = 0; j < parser.scheduler.size(); j++) {
                    size_t i = j;
                    if (sorted_by_total_costs)
                        i = ordering[j];
                    if (partitioners_failed[i]) {
                        std::cout << "scheduler " << partitioners_name[i] << " failed." << std::endl;
                    } else {
                        std::string memory_ok = "Y";
                        if (parser.global_params.get_child("use_memory_constraints").get_value<bool>() && !partitioners_memory_satisfied[i]) memory_ok = "N";
                        std::cout << "imbalance:  " << std::right << std::setw(imb) << partitioners_imbalance[i]
                                  << "     max memory:  " << std::right << std::setw(mm) << partitioners_max_memory[i]
                                  << "     memory ok:  " << memory_ok
                                  << "     compute time:  " << std::right << std::setw(ct)
                                  << partitioners_compute_time[i] << "ms" << "     scheduler:  " << partitioners_name[i]
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
