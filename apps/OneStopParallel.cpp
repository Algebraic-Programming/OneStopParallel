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

#include "auxiliary/misc.hpp"
#include "bsp/model/BspSchedule.hpp"
#include "graph_implementations/boost_graphs/boost_graph.hpp"
#include "io/DotFileWriter.hpp"
#include "io/arch_file_reader.hpp"
#include "io/dot_graph_file_reader.hpp"
#include "io/hdag_graph_file_reader.hpp"
#include "io/mtx_graph_file_reader.hpp"
#include "util/CommandLineParser.hpp"
#include "util/run_bsp_scheduler.hpp"

namespace pt = boost::property_tree;
using namespace osp;

using graph_t = boost_graph_int_t;

std::filesystem::path getExecutablePath() { return std::filesystem::canonical("/proc/self/exe"); }

// invoked upon program call
int main(int argc, char *argv[]) {

    CommandLineParser parser(getExecutablePath().remove_filename().string() += "osp_config.json");

    try {
        parser.parse_args(argc, argv);
    } catch (std::invalid_argument &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    for (auto &instance : parser.instances) {

        BspInstance<graph_t> bsp_instance;

        std::string filename_graph = instance.second.get_child("graphFile").get_value<std::string>();
        std::string name_graph = filename_graph.substr(filename_graph.rfind("/") + 1,
                                                       filename_graph.rfind(".") - filename_graph.rfind("/") - 1);

        std::string filename_machine = instance.second.get_child("machineParamsFile").get_value<std::string>();

        std::string name_machine = filename_machine.substr(
            filename_machine.rfind("/") + 1, filename_machine.rfind(".") - filename_machine.rfind("/") - 1);

        bool status_architecture = file_reader::readBspArchitecture(filename_machine, bsp_instance.getArchitecture());

        if (!status_architecture) {
            std::cerr << "Reading architecture files " + filename_machine << " failed." << std::endl;
            continue;
        }

        bool status_graph = false;

        if (filename_graph.substr(filename_graph.rfind(".") + 1) == "hdag") {
            status_graph =
                file_reader::readComputationalDagHyperdagFormat(filename_graph, bsp_instance.getComputationalDag());

        } else if (filename_graph.substr(filename_graph.rfind(".") + 1) == "mtx") {

            status_graph =
                file_reader::readComputationalDagMartixMarketFormat(filename_graph, bsp_instance.getComputationalDag());

        } else if (filename_graph.substr(filename_graph.rfind(".") + 1) == "dot") {
            status_graph =
                file_reader::readComputationalDagDotFormat(filename_graph, bsp_instance.getComputationalDag());

        } else {
            std::cout << "Unknown file ending: ." << filename_graph.substr(filename_graph.rfind(".") + 1)
                      << " ...assuming hyperDag format." << std::endl;
            status_graph =
                file_reader::readComputationalDagHyperdagFormat(filename_graph, bsp_instance.getComputationalDag());
        }

        if (!status_graph) {
            std::cerr << "Reading graph files " + filename_graph << " failed." << std::endl;
            continue;
        }

        std::cout << "Warning: assuming all node types can be scheduled on all processor types!\n";
        bsp_instance.setAllOnesCompatibilityMatrix();

        std::vector<std::string> schedulers_name(parser.scheduler.size(), "");
        std::vector<bool> schedulers_failed(parser.scheduler.size(), false);
        std::vector<v_workw_t<graph_t>> schedulers_costs(parser.scheduler.size(), 0);
        std::vector<v_workw_t<graph_t>> schedulers_work_costs(parser.scheduler.size(), 0);
        std::vector<unsigned> schedulers_supersteps(parser.scheduler.size(), 0);
        std::vector<long> schedulers_compute_time(parser.scheduler.size(), 0);

        size_t algorithm_counter = 0;
        for (auto &algorithm : parser.scheduler) {

            schedulers_name[algorithm_counter] = algorithm.second.get_child("name").get_value<std::string>();

            const auto start_time = std::chrono::high_resolution_clock::now();

            RETURN_STATUS return_status;
            BspSchedule<graph_t> schedule(bsp_instance);

            try {
                return_status = run_bsp_scheduler(parser, algorithm.second, schedule);
            } catch (...) {
                schedulers_failed[algorithm_counter] = true;
                std::cerr << "Error during execution of Scheduler " +
                                 algorithm.second.get_child("name").get_value<std::string>() + "."
                          << std::endl;
                continue;
            }

            const auto finish_time = std::chrono::high_resolution_clock::now();

            schedulers_compute_time[algorithm_counter] =
                std::chrono::duration_cast<std::chrono::milliseconds>(finish_time - start_time).count();

            if (return_status != RETURN_STATUS::SUCCESS && return_status != RETURN_STATUS::BEST_FOUND) {

                schedulers_failed[algorithm_counter] = true;
                if (return_status == RETURN_STATUS::ERROR) {
                    std::cerr << "Error while computing schedule " +
                                     algorithm.second.get_child("name").get_value<std::string>() + "."
                              << std::endl;
                } else if (return_status == RETURN_STATUS::TIMEOUT) {
                    std::cerr << "Timeout while computing schedule " +
                                     algorithm.second.get_child("name").get_value<std::string>() + "."
                              << std::endl;
                } else {
                    std::cerr << "Unknown return status while computing schedule " +
                                     algorithm.second.get_child("name").get_value<std::string>() + "."
                              << std::endl;
                }
            } else {

                schedulers_costs[algorithm_counter] = schedule.computeCosts();
                schedulers_work_costs[algorithm_counter] = schedule.computeWorkCosts();
                schedulers_supersteps[algorithm_counter] = schedule.numberOfSupersteps();

                // BspScheduleWriter sched_writer(schedule);
                // if (parser.global_params.get_child("outputSchedule").get_value<bool>()) {
                //     try {
                //         sched_writer.write_txt(name_graph + "_" + name_machine + "_" +
                //                                algorithm.second.get_child("name").get_value<std::string>() +
                //                                "_schedule.txt");
                //     } catch (std::exception &e) {
                //         std::cerr << "Writing schedule file for " + name_graph + ", " + name_machine + ", " +
                //                          schedulers_name[algorithm_counter] + " has failed."
                //                   << std::endl;
                //         std::cerr << e.what() << std::endl;
                //     }
                // }

                // if (parser.global_params.get_child("outputSankeySchedule").get_value<bool>()) {
                //     try {
                //         sched_writer.write_sankey(name_graph + "_" + name_machine + "_" +
                //                                   algorithm.second.get_child("name").get_value<std::string>() +
                //                                   "_sankey.sankey");
                //     } catch (std::exception &e) {
                //         std::cerr << "Writing sankey file for " + name_graph + ", " + name_machine + ", " +
                //                          schedulers_name[algorithm_counter] + " has failed."
                //                   << std::endl;
                //         std::cerr << e.what() << std::endl;
                //     }
                // }

                if (parser.global_params.get_child("outputDotSchedule").get_value<bool>()) {
                    try {

                        DotFileWriter sched_writer;
                        sched_writer.write_schedule(name_graph + "_" + name_machine + "_" +
                                                        algorithm.second.get_child("name").get_value<std::string>() +
                                                        "_schedule.dot",
                                                    schedule);

                    } catch (std::exception &e) {
                        std::cerr << "Writing dot file for " + name_graph + ", " + name_machine + ", " +
                                         schedulers_name[algorithm_counter] + " has failed."
                                  << std::endl;
                        std::cerr << e.what() << std::endl;
                    }
                }
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

        std::vector<size_t> ordering = sorting_arrangement(schedulers_costs);

        std::cout << std::endl << name_graph << " - " << name_machine << std::endl;
        std::cout << "Number of Vertices: " + std::to_string(bsp_instance.getComputationalDag().num_vertices()) +
                         "  Number of Edges: " + std::to_string(bsp_instance.getComputationalDag().num_edges())
                  << std::endl;
        for (size_t j = 0; j < parser.scheduler.size(); j++) {
            size_t i = j;

            i = ordering[j];

            if (schedulers_failed[i]) {
                std::cout << "scheduler " << schedulers_name[i] << " failed." << std::endl;
            } else {
                std::cout << "total costs:  " << std::right << std::setw(tw) << schedulers_costs[i]
                          << "     work costs:  " << std::right << std::setw(ww) << schedulers_work_costs[i]
                          << "     comm costs:  " << std::right << std::setw(cw)
                          << schedulers_costs[i] - schedulers_work_costs[i]
                          << "     number of supersteps:  " << std::right << std::setw(nsw) << schedulers_supersteps[i]
                          << "     compute time:  " << std::right << std::setw(ct) << schedulers_compute_time[i] << "ms"
                          << "     scheduler:  " << schedulers_name[i] << std::endl;
            }
        }
    }

    return 0;
}
