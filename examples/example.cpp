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

#include "algorithms/GreedySchedulers/MetaGreedyScheduler.hpp"
#include "algorithms/LocalSearchSchedulers/HillClimbingScheduler.hpp"
#include "algorithms/LocalSearchSchedulers/HillClimbingScheduler.hpp"
#include "file_interactions/FileReader.hpp"


int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <machine_file>" << std::endl;
        return 1;
    }

    std::string filename_graph = argv[1];
    std::string name_graph =
        filename_graph.substr(filename_graph.rfind("/") + 1, filename_graph.rfind(".") - filename_graph.rfind("/") - 1);

    std::string filename_machine = argv[2];
    std::string name_machine = filename_machine.substr(filename_machine.rfind("/") + 1,
                                                       filename_machine.rfind(".") - filename_machine.rfind("/") - 1);

    std::cout << "Input graph file: " << name_graph << " machine file: " << name_machine << std::endl;

    std::pair<bool, ComputationalDag> read_graph(false, ComputationalDag());
    if (filename_graph.substr(filename_graph.rfind(".") + 1) == "txt") {
        read_graph = FileReader::readComputationalDagHyperdagFormat(filename_graph);
    } else if (filename_graph.substr(filename_graph.rfind(".") + 1) == "mtx") {
        read_graph = FileReader::readComputationalDagMartixMarketFormat(filename_graph);
    }

    bool status_graph = read_graph.first;
    ComputationalDag &graph = read_graph.second;

    auto [status_architecture, architecture] = FileReader::readBspArchitecture(filename_machine);

    if (!status_graph || !status_architecture) {

        std::cout << "Reading files failed." << std::endl;
        return 1;
    }

    BspInstance instance(graph, architecture);
    std::cout << "Instance read, number of vertices: " + std::to_string(graph.numberOfVertices()) +
                     "  number of edges: " + std::to_string(graph.numberOfEdges())
              << std::endl;

    std::cout << "Computing greedy schedule." << std::endl;
    GreedyBspScheduler greedyBsp;

    auto start_time = std::chrono::high_resolution_clock::now();
    auto [return_status_greedy, greedy_sched] = greedyBsp.computeSchedule(instance);
    auto finish_time = std::chrono::high_resolution_clock::now();

    std::cout << "Greedy schedule computed! costs: " << greedy_sched.computeCosts() << " number of supersteps: " << greedy_sched.numberOfSupersteps()
              << " time: " << std::chrono::duration_cast<std::chrono::milliseconds>(finish_time - start_time).count() << " ms"
              << std::endl;

    std::cout << "Improving the greedy schedule with hill climbing heuristic." << std::endl;
    HillClimbingScheduler hill_climbing_scheduler;

    start_time = std::chrono::high_resolution_clock::now();
    hill_climbing_scheduler.improveSchedule(greedy_sched);
    finish_time = std::chrono::high_resolution_clock::now();

    std::cout << "Hill climbing complete! Improved cost: " << greedy_sched.computeCosts()
              << " supersteps: " << greedy_sched.numberOfSupersteps()
              << " time: " << std::chrono::duration_cast<std::chrono::milliseconds>(finish_time - start_time).count() << " ms"
              << std::endl;
    
    return 0;
}
