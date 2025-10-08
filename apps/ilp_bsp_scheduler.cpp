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

#include "osp/auxiliary/misc.hpp"
#include "osp/bsp/scheduler/IlpSchedulers/CoptFullScheduler.hpp"
#include "osp/graph_algorithms/directed_graph_path_util.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"
#include "osp/auxiliary/io/DotFileWriter.hpp"
#include "osp/auxiliary/io/arch_file_reader.hpp"
#include "osp/auxiliary/io/bsp_schedule_file_writer.hpp"
#include "osp/auxiliary/io/general_file_reader.hpp"

using namespace osp;

using ComputationalDag = computational_dag_edge_idx_vector_impl_def_int_t;

int main(int argc, char *argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <machine_file> <max_number_step> <optional:recomp>"
                  << std::endl;
        return 1;
    }

    std::string filename_graph = argv[1];
    std::string name_graph = filename_graph.substr(0, filename_graph.rfind("."));

    std::cout << name_graph << std::endl;

    std::string filename_machine = argv[2];
    std::string name_machine = filename_machine.substr(filename_machine.find_last_of("/\\") + 1);
    name_machine = name_machine.substr(0, name_machine.rfind("."));

    int step_int = std::stoi(argv[3]);
    if (step_int < 1) {
        std::cerr << "Argument max_number_step must be a positive integer: " << step_int << std::endl;
        return 1;
    }

    bool recomp = false;

    if (argc > 4 && std::string(argv[4]) == "recomp") {
        recomp = true;
    } else if (argc > 4) {
        std::cerr << "Unknown argument: " << argv[4] << ". Expected 'recomp' for recomputation." << std::endl;
        return 1;
    }

    unsigned steps = static_cast<unsigned>(step_int);

    BspInstance<ComputationalDag> instance;
    ComputationalDag &graph = instance.getComputationalDag();

    bool status_graph = file_reader::readGraph(filename_graph, graph);
    bool status_arch = file_reader::readBspArchitecture(filename_machine, instance.getArchitecture());
    // instance.setDiagonalCompatibilityMatrix(graph.num_vertex_types());
    // instance.getArchitecture().setProcessorsWithTypes({0,0,1,1,1,1});

    if (!status_graph || !status_arch) {

        std::cout << "Reading files failed." << std::endl;
        return 1;
    }

    // for (const auto &vertex : graph.vertices()) {

    //     graph.set_vertex_work_weight(vertex, graph.vertex_work_weight(vertex) * 80);
    // }

    CoptFullScheduler<ComputationalDag> scheduler;
    scheduler.setMaxNumberOfSupersteps(steps);
    scheduler.setTimeLimitHours(48);

    if (recomp) {

        BspScheduleRecomp<ComputationalDag> schedule(instance);

        auto status_schedule = scheduler.computeScheduleRecomp(schedule);

        if (status_schedule == RETURN_STATUS::OSP_SUCCESS || status_schedule == RETURN_STATUS::BEST_FOUND) {

            DotFileWriter dot_writer;
            dot_writer.write_schedule_recomp(name_graph + "_" + name_machine + "_maxS_" + std::to_string(steps) + "_" +
                                                 scheduler.getScheduleName() + "_recomp_schedule.dot",
                                             schedule);

            dot_writer.write_schedule_recomp_duplicate(name_graph + "_" + name_machine + "_maxS_" +
                                                           std::to_string(steps) + "_" + scheduler.getScheduleName() +
                                                           "_duplicate_recomp_schedule.dot",
                                                       schedule);

            std::cout << "Recomp Schedule computed with costs: " << schedule.computeCosts() << std::endl;

        } else {
            std::cout << "Computing schedule failed." << std::endl;
            return 1;
        }

    } else {

        BspSchedule<ComputationalDag> schedule(instance);

        auto status_schedule = scheduler.computeSchedule(schedule);

        if (status_schedule == RETURN_STATUS::OSP_SUCCESS || status_schedule == RETURN_STATUS::BEST_FOUND) {

            DotFileWriter dot_writer;
            dot_writer.write_schedule(name_graph + "_" + name_machine + "_maxS_" + std::to_string(steps) + "_" +
                                          scheduler.getScheduleName() + "_schedule.dot",
                                      schedule);

            std::cout << "Schedule computed with costs: " << schedule.computeCosts() << std::endl;

        } else {
            std::cout << "Computing schedule failed." << std::endl;
            return 1;
        }
    }
    return 0;
}
