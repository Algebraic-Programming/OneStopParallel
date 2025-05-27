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

#include <boost/algorithm/string.hpp>
#include <iostream>
#include <string>

#include "graph_implementations/boost_graphs/boost_graph.hpp"
#include "io/dot_graph_file_reader.hpp"
#include "io/hdag_graph_file_reader.hpp"
#include "io/mtx_graph_file_reader.hpp"
#include "io/bsp_schedule_file_writer.hpp"

#include "bsp/scheduler/GreedySchedulers/EtfScheduler.hpp"
#include "bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "bsp/scheduler/LoadBalanceScheduler/LightEdgeVariancePartitioner.hpp"

using namespace osp;

using graph_t = boost_graph_int_t;
using mem_constr = persistent_transient_memory_constraint<graph_t>;

// invoked upon program call
int main(int argc, char *argv[]) {

    if (argc != 5) {
        std::cout << "Usage: " << argv[0] << " <input_file_dag> <num_proc> <memory_bound> <algorithm>" << std::endl;
        std::cout << "Available algorithms: bsp, etf, variance" << std::endl;
        return 1;
    }

    BspInstance<graph_t> bsp_instance;

    bsp_instance.getArchitecture().setNumberOfProcessors(static_cast<unsigned>(std::stoul(argv[2])));
    bsp_instance.getArchitecture().setMemoryBound(std::atoi(argv[3]));
    bsp_instance.getArchitecture().setMemoryConstraintType(PERSISTENT_AND_TRANSIENT);

    std::string algorithm_name = argv[4];

    std::string filename_graph = argv[1];

    bool status_graph = false;

    if (filename_graph.substr(filename_graph.rfind(".") + 1) == "hdag") {
        status_graph =
            file_reader::readComputationalDagHyperdagFormat(filename_graph, bsp_instance.getComputationalDag());

    } else if (filename_graph.substr(filename_graph.rfind(".") + 1) == "mtx") {

        status_graph = file_reader::readComputationalDagMartixMarketFormat(filename_graph, bsp_instance.getComputationalDag());

    } else if (filename_graph.substr(filename_graph.rfind(".") + 1) == "dot") {
        status_graph = file_reader::readComputationalDagDotFormat(filename_graph, bsp_instance.getComputationalDag());

    } else {
        std::cout << "Unknown file ending: ." << filename_graph.substr(filename_graph.rfind(".") + 1)
                  << " ...assuming hyperDag format." << std::endl;
        status_graph =
            file_reader::readComputationalDagHyperdagFormat(filename_graph, bsp_instance.getComputationalDag());
    }

    if (!status_graph) {
        std::cout << "Error while reading the graph from file: " << filename_graph << std::endl;
        return 1;
    }

    if (bsp_instance.getComputationalDag().num_vertex_types() > 1) {
        std::cout << "The graph has more than one vertex type, which is not supported by this scheduler." << std::endl;
        return 1;
    }

    boost::algorithm::to_lower(algorithm_name); // modifies str

    BspSchedule<graph_t> bsp_schedule(bsp_instance);
    Scheduler<graph_t> *scheduler = nullptr;

    if (algorithm_name == "bsp") {

        float max_percent_idle_processors = 0.2f;
        bool increase_parallelism_in_new_superstep = true;

        scheduler = new GreedyBspScheduler<graph_t, mem_constr>(
            max_percent_idle_processors, increase_parallelism_in_new_superstep);

    } else if (algorithm_name == "etf") {

        scheduler = new EtfScheduler<graph_t, mem_constr>(BL_EST);

    } else if (algorithm_name == "variance") {

        const double max_percent_idle_processors = 0.0;
        const bool increase_parallelism_in_new_superstep = true;
        const double variance_power = 6.0;
        const float max_priority_difference_percent = 0.34f;
        const double heavy_is_x_times_median = 3.0;
        const double min_percent_components_retained = 0.25;
        const float bound_component_weight_percent = 4.0f;
        const float slack = 0.0f;

        scheduler = new LightEdgeVariancePartitioner<graph_t, flat_spline_interpolation,mem_constr>(
            max_percent_idle_processors, variance_power, heavy_is_x_times_median, min_percent_components_retained,
            bound_component_weight_percent, increase_parallelism_in_new_superstep, 
            max_priority_difference_percent, slack);

    } else {
        std::cout << "Unknown algorithm: " << algorithm_name << std::endl;
        return 1;
    }

    auto scheduler_status = scheduler->computeSchedule(bsp_schedule);

    if (scheduler_status == ERROR) {
        std::cout << "Error while scheduling!" << std::endl;
        delete scheduler;
        return 1;
    }

    delete scheduler;

    file_writer::write_txt(filename_graph + "_" + algorithm_name + "_schedule.shed", bsp_schedule);

    std::cout << "OSP Success" << std::endl;
    return 0;
}
