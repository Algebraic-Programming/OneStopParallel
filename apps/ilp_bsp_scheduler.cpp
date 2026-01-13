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

#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "osp/auxiliary/io/DotFileWriter.hpp"
#include "osp/auxiliary/io/arch_file_reader.hpp"
#include "osp/auxiliary/io/bsp_schedule_file_writer.hpp"
#include "osp/auxiliary/io/general_file_reader.hpp"
#include "osp/auxiliary/misc.hpp"
#include "osp/bsp/scheduler/IlpSchedulers/CoptFullScheduler.hpp"
#include "osp/graph_algorithms/directed_graph_path_util.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"

using namespace osp;

using ComputationalDag = ComputationalDagEdgeIdxVectorImplDefIntT;

int main(int argc, char *argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <machine_file> <max_number_step> [time_limit_seconds] [recomp]"
                  << std::endl;
        return 1;
    }

    std::string filenameGraph = argv[1];
    std::string nameGraph = filenameGraph.substr(0, filenameGraph.rfind("."));

    std::cout << nameGraph << std::endl;

    std::string filenameMachine = argv[2];
    std::string nameMachine = filenameMachine.substr(filenameMachine.find_last_of("/\\") + 1);
    nameMachine = nameMachine.substr(0, nameMachine.rfind("."));

    int stepInt = std::stoi(argv[3]);
    if (stepInt < 1) {
        std::cerr << "Argument max_number_step must be a positive integer: " << stepInt << std::endl;
        return 1;
    }

    unsigned steps = static_cast<unsigned>(stepInt);

    // Default time limit: 3600 seconds (1 hour)
    unsigned timeLimitSeconds = 3600;
    bool recomp = false;

    // Parse optional arguments
    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "recomp") {
            recomp = true;
        } else {
            // Try to parse as time limit
            try {
                int timeLimitInt = std::stoi(arg);
                if (timeLimitInt < 1) {
                    std::cerr << "Argument time_limit_seconds must be a positive integer: " << timeLimitInt << std::endl;
                    return 1;
                }
                timeLimitSeconds = static_cast<unsigned>(timeLimitInt);
            } catch (const std::exception &) {
                std::cerr << "Unknown argument: " << arg << ". Expected a time limit (integer) or 'recomp'." << std::endl;
                return 1;
            }
        }
    }

    BspInstance<ComputationalDag> instance;
    ComputationalDag &graph = instance.GetComputationalDag();

    bool statusGraph = file_reader::ReadGraph(filenameGraph, graph);
    bool statusArch = file_reader::ReadBspArchitecture(filenameMachine, instance.GetArchitecture());
    // instance.SetDiagonalCompatibilityMatrix(graph.NumVertexTypes());
    // instance.GetArchitecture().SetProcessorsWithTypes({0,0,1,1,1,1});

    if (!statusGraph || !statusArch) {
        std::cout << "Reading files failed." << std::endl;
        return 1;
    }

    // for (const auto &vertex : graph.Vertices()) {

    //     graph.SetVertexWorkWeight(vertex, graph.VertexWorkWeight(vertex) * 80);
    // }

    CoptFullScheduler<ComputationalDag> scheduler;
    scheduler.SetMaxNumberOfSupersteps(steps);
    scheduler.SetTimeLimitSeconds(timeLimitSeconds);

    std::cout << "Time limit set to " << timeLimitSeconds << " seconds." << std::endl;

    if (recomp) {
        BspScheduleRecomp<ComputationalDag> schedule(instance);

        auto statusSchedule = scheduler.ComputeScheduleRecomp(schedule);

        if (statusSchedule == ReturnStatus::OSP_SUCCESS || statusSchedule == ReturnStatus::BEST_FOUND) {
            DotFileWriter dotWriter;
            dotWriter.WriteScheduleRecomp(nameGraph + "_" + nameMachine + "_maxS_" + std::to_string(steps) + "_"
                                              + scheduler.GetScheduleName() + "_recomp_schedule.dot",
                                          schedule);

            dotWriter.WriteScheduleRecompDuplicate(nameGraph + "_" + nameMachine + "_maxS_" + std::to_string(steps) + "_"
                                                       + scheduler.GetScheduleName() + "_duplicate_recomp_schedule.dot",
                                                   schedule);

            std::cout << "Recomp Schedule computed with costs: " << schedule.ComputeCosts() << std::endl;

        } else {
            std::cout << "Computing schedule failed." << std::endl;
            return 1;
        }

    } else {
        BspSchedule<ComputationalDag> schedule(instance);

        auto statusSchedule = scheduler.ComputeSchedule(schedule);

        if (statusSchedule == ReturnStatus::OSP_SUCCESS || statusSchedule == ReturnStatus::BEST_FOUND) {
            DotFileWriter dotWriter;
            dotWriter.WriteSchedule(nameGraph + "_" + nameMachine + "_maxS_" + std::to_string(steps) + "_"
                                        + scheduler.GetScheduleName() + "_schedule.dot",
                                    schedule);

            std::cout << "Schedule computed with costs: " << schedule.ComputeCosts() << std::endl;

        } else {
            std::cout << "Computing schedule failed." << std::endl;
            return 1;
        }
    }
    return 0;
}
