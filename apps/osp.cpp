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

#include <boost/graph/graphviz.hpp>
#include <boost/log/trivial.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <filesystem>
#include <iostream>
#include <string>
#include <tuple>

#include "boost/log/utility/setup.hpp"
#include "osp/auxiliary/io/DotFileWriter.hpp"
#include "osp/auxiliary/io/arch_file_reader.hpp"
#include "osp/auxiliary/io/bsp_schedule_file_writer.hpp"
#include "osp/auxiliary/io/general_file_reader.hpp"
#include "osp/auxiliary/misc.hpp"
#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "test_suite_runner/ConfigParser.hpp"
#include "test_suite_runner/StringToScheduler/run_bsp_scheduler.hpp"

namespace pt = boost::property_tree;
using namespace osp;

using GraphT = computational_dag_edge_idx_vector_impl_def_int_t;

std::filesystem::path GetExecutablePath() { return std::filesystem::canonical("/proc/self/exe"); }

// invoked upon program call
int main(int argc, char *argv[]) {
    ConfigParser parser(GetExecutablePath().remove_filename().string() += "osp_config.json");

    try {
        parser.ParseArgs(argc, argv);
    } catch (std::invalid_argument &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    for (auto &instance : parser.instances_) {
        BspInstance<GraphT> bspInstance;

        std::string filenameGraph = instance.second.get_child("graphFile").get_value<std::string>();
        std::string nameGraph
            = filenameGraph.substr(filenameGraph.rfind("/") + 1, filenameGraph.rfind(".") - filenameGraph.rfind("/") - 1);

        std::string filenameMachine = instance.second.get_child("machineParamsFile").get_value<std::string>();

        std::string nameMachine
            = filenameMachine.substr(filenameMachine.rfind("/") + 1, filenameMachine.rfind(".") - filenameMachine.rfind("/") - 1);

        bool statusArchitecture = file_reader::readBspArchitecture(filenameMachine, bspInstance.GetArchitecture());

        if (!statusArchitecture) {
            std::cerr << "Reading architecture files " + filenameMachine << " failed." << std::endl;
            continue;
        }

        bool statusGraph = file_reader::readGraph(filenameGraph, bspInstance.GetComputationalDag());
        if (!statusGraph) {
            std::cerr << "Reading graph files " + filenameGraph << " failed." << std::endl;
            continue;
        }

        std::cout << "Warning: assuming all node types can be scheduled on all processor types!\n";
        bspInstance.setAllOnesCompatibilityMatrix();

        std::vector<std::string> schedulersName(parser.scheduler_.size(), "");
        std::vector<bool> schedulersFailed(parser.scheduler_.size(), false);
        std::vector<VWorkwT<GraphT>> schedulersCosts(parser.scheduler_.size(), 0);
        std::vector<VWorkwT<GraphT>> schedulersWorkCosts(parser.scheduler_.size(), 0);
        std::vector<unsigned> schedulersSupersteps(parser.scheduler_.size(), 0);
        std::vector<long> schedulersComputeTime(parser.scheduler_.size(), 0);

        size_t algorithmCounter = 0;
        for (auto &algorithm : parser.scheduler_) {
            schedulersName[algorithmCounter] = algorithm.second.get_child("name").get_value<std::string>();

            const auto startTime = std::chrono::high_resolution_clock::now();

            RETURN_STATUS returnStatus;
            BspSchedule<GraphT> schedule(bspInstance);

            try {
                returnStatus = run_bsp_scheduler(parser, algorithm.second, schedule);
            } catch (...) {
                schedulersFailed[algorithmCounter] = true;
                std::cerr << "Error during execution of Scheduler " + algorithm.second.get_child("name").get_value<std::string>()
                                 + "."
                          << std::endl;
                continue;
            }

            const auto finishTime = std::chrono::high_resolution_clock::now();

            schedulersComputeTime[algorithmCounter]
                = std::chrono::duration_cast<std::chrono::milliseconds>(finishTime - startTime).count();

            if (returnStatus != RETURN_STATUS::OSP_SUCCESS && returnStatus != RETURN_STATUS::BEST_FOUND) {
                schedulersFailed[algorithmCounter] = true;
                if (returnStatus == RETURN_STATUS::ERROR) {
                    std::cerr << "Error while computing schedule " + algorithm.second.get_child("name").get_value<std::string>()
                                     + "."
                              << std::endl;
                } else if (returnStatus == RETURN_STATUS::TIMEOUT) {
                    std::cerr << "Timeout while computing schedule " + algorithm.second.get_child("name").get_value<std::string>()
                                     + "."
                              << std::endl;
                } else {
                    std::cerr << "Unknown return status while computing schedule "
                                     + algorithm.second.get_child("name").get_value<std::string>() + "."
                              << std::endl;
                }
            } else {
                schedulersCosts[algorithmCounter] = BspScheduleCS<GraphT>(schedule).computeCosts();
                schedulersWorkCosts[algorithmCounter] = schedule.computeWorkCosts();
                schedulersSupersteps[algorithmCounter] = schedule.NumberOfSupersteps();

                if (parser.globalParams_.get_child("outputSchedule").get_value<bool>()) {
                    try {
                        file_writer::write_txt(nameGraph + "_" + nameMachine + "_"
                                                   + algorithm.second.get_child("name").get_value<std::string>() + "_schedule.txt",
                                               schedule);
                    } catch (std::exception &e) {
                        std::cerr << "Writing schedule file for " + nameGraph + ", " + nameMachine + ", "
                                         + schedulersName[algorithmCounter] + " has failed."
                                  << std::endl;
                        std::cerr << e.what() << std::endl;
                    }
                }

                if (parser.globalParams_.get_child("outputSankeySchedule").get_value<bool>()) {
                    try {
                        file_writer::write_sankey(nameGraph + "_" + nameMachine + "_"
                                                      + algorithm.second.get_child("name").get_value<std::string>()
                                                      + "_sankey.sankey",
                                                  BspScheduleCS<GraphT>(schedule));
                    } catch (std::exception &e) {
                        std::cerr << "Writing sankey file for " + nameGraph + ", " + nameMachine + ", "
                                         + schedulersName[algorithmCounter] + " has failed."
                                  << std::endl;
                        std::cerr << e.what() << std::endl;
                    }
                }

                if (parser.globalParams_.get_child("outputDotSchedule").get_value<bool>()) {
                    try {
                        DotFileWriter schedWriter;
                        schedWriter.write_schedule(nameGraph + "_" + nameMachine + "_"
                                                       + algorithm.second.get_child("name").get_value<std::string>()
                                                       + "_schedule.dot",
                                                   schedule);

                    } catch (std::exception &e) {
                        std::cerr << "Writing dot file for " + nameGraph + ", " + nameMachine + ", "
                                         + schedulersName[algorithmCounter] + " has failed."
                                  << std::endl;
                        std::cerr << e.what() << std::endl;
                    }
                }
            }

            algorithmCounter++;
        }

        int tw = 1, ww = 1, cw = 1, nsw = 1, ct = 1;
        for (size_t i = 0; i < parser.scheduler_.size(); i++) {
            if (schedulersFailed[i]) {
                continue;
            }
            tw = std::max(tw, 1 + int(std::log10(schedulersCosts[i])));
            ww = std::max(ww, 1 + int(std::log10(schedulersWorkCosts[i])));
            cw = std::max(cw, 1 + int(std::log10(schedulersCosts[i] - schedulersWorkCosts[i])));
            nsw = std::max(nsw, 1 + int(std::log10(schedulersSupersteps[i])));
            ct = std::max(ct, 1 + int(std::log10(schedulersComputeTime[i])));
        }

        std::vector<size_t> ordering = sorting_arrangement(schedulersCosts);

        std::cout << std::endl << nameGraph << " - " << nameMachine << std::endl;
        std::cout << "Number of Vertices: " + std::to_string(bspInstance.GetComputationalDag().NumVertices())
                         + "  Number of Edges: " + std::to_string(bspInstance.GetComputationalDag().NumEdges())
                  << std::endl;
        for (size_t j = 0; j < parser.scheduler_.size(); j++) {
            size_t i = j;

            i = ordering[j];

            if (schedulersFailed[i]) {
                std::cout << "scheduler " << schedulersName[i] << " failed." << std::endl;
            } else {
                std::cout << "total costs:  " << std::right << std::setw(tw) << schedulersCosts[i]
                          << "     work costs:  " << std::right << std::setw(ww) << schedulersWorkCosts[i]
                          << "     comm costs:  " << std::right << std::setw(cw) << schedulersCosts[i] - schedulersWorkCosts[i]
                          << "     number of supersteps:  " << std::right << std::setw(nsw) << schedulersSupersteps[i]
                          << "     compute time:  " << std::right << std::setw(ct) << schedulersComputeTime[i] << "ms"
                          << "     scheduler:  " << schedulersName[i] << std::endl;
            }
        }
    }

    return 0;
}
