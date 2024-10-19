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

#include "file_interactions/BspScheduleWriter.hpp"
#include "file_interactions/DAGPartitionWriter.hpp"
#include "file_interactions/FileReader.hpp"

#include "scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "dag_partitioners/LightEdgeVariancePartitioner.hpp"
#include "scheduler/GreedySchedulers/GreedyEtfScheduler.hpp"


// invoked upon program call
int main(int argc, char *argv[]) {

    if (argc != 5) {
        std::cout << "Usage: " << argv[0] << " <input_file_dag> <num_proc> <memory_bound> <algorithm>" << std::endl;
        std::cout << "Available algorithms: bsp, etf, variance" << std::endl;
        return 1;
    }

    BspInstance bsp_instance;

    bsp_instance.getArchitecture().setNumberOfProcessors(std::atol(argv[2]));
    bsp_instance.getArchitecture().setMemoryBound(std::atol(argv[3]));
    bsp_instance.getArchitecture().setMemoryConstraintType(PERSISTENT_AND_TRANSIENT);

    std::string algorithm_name = argv[4];

    std::string filename_graph = argv[1];

    bool status_graph = false;

    if (filename_graph.substr(filename_graph.rfind(".") + 1) == "txt") {
        std::tie(status_graph, bsp_instance.getComputationalDag()) =
            FileReader::readComputationalDagHyperdagFormat(filename_graph);

    } else if (filename_graph.substr(filename_graph.rfind(".") + 1) == "mtx") {

        std::tie(status_graph, bsp_instance.getComputationalDag()) =
            FileReader::readComputationalDagMartixMarketFormat(filename_graph);

    } else if (filename_graph.substr(filename_graph.rfind(".") + 1) == "dot") {
        std::tie(status_graph, bsp_instance.getComputationalDag()) =
            FileReader::readComputationalDagDotFormat(filename_graph);

    } else {
        std::cout << "Unknown file ending: ." << filename_graph.substr(filename_graph.rfind(".") + 1)
                  << " ...assuming hyperDag format." << std::endl;
        std::tie(status_graph, bsp_instance.getComputationalDag()) =
            FileReader::readComputationalDagHyperdagFormat(filename_graph);
    }

    if (!status_graph) {
        throw std::invalid_argument("Reading graph file " + filename_graph + " failed.");
    }

    boost::algorithm::to_lower(algorithm_name); // modifies str

    if (algorithm_name == "bsp") {

        float max_percent_idle_processors = 0.2;
        bool increase_parallelism_in_new_superstep = true;

        GreedyBspScheduler scheduler(max_percent_idle_processors, increase_parallelism_in_new_superstep);
        scheduler.setUseMemoryConstraint(true);

        auto [scheduler_status, schedule] = scheduler.computeSchedule(bsp_instance);

        if (scheduler_status == ERROR) {
            std::cout << "Error in scheduler: " << scheduler_status << std::endl;
            return 1;
        }

        BspScheduleWriter sched_writer(schedule);
        sched_writer.write_txt(filename_graph + "_" + algorithm_name + "_schedule.txt");

    } else if (algorithm_name == "etf") {

        GreedyEtfScheduler scheduler;
        scheduler.setUseMemoryConstraint(true);
        scheduler.setMode(BL_EST);

        auto [scheduler_status, schedule] = scheduler.computeSchedule(bsp_instance);

        if (scheduler_status == ERROR) {
            std::cout << "Error in scheduler: " << scheduler_status << std::endl;
            return 1;
        }

        BspScheduleWriter sched_writer(schedule);
        sched_writer.write_txt(filename_graph + "_" + algorithm_name + "_schedule.txt");

    } else if (algorithm_name == "variance") {

        IListPartitioner::ProcessorPriorityMethod proc_priority_method = IListPartitioner::FLATSPLINE;

        const float max_percent_idle_processors = 0.0;
        const bool increase_parallelism_in_new_superstep = true;
        const double variance_power = 6.0;
        const double memory_capacity_increase = 1.1;
        const float max_priority_difference_percent = 0.34;
        const float heavy_is_x_times_median = 3.0;
        const float min_percent_components_retained = 0.25;
        const float bound_component_weight_percent = 4.0;
        const float slack = 0.0;

        LightEdgeVariancePartitioner partitioner(
            proc_priority_method, true, max_percent_idle_processors, variance_power, heavy_is_x_times_median,
            min_percent_components_retained, bound_component_weight_percent, increase_parallelism_in_new_superstep,
            memory_capacity_increase, max_priority_difference_percent, slack);

        auto [partitio_status, partition] = partitioner.computePartition(bsp_instance);

        if (partitio_status == ERROR) {
            std::cout << "Error in partitioner: " << partitio_status << std::endl;
            return 1;
        }

        DAGPartitionWriter part_writer(partition);
        part_writer.write_txt(filename_graph + "_" + algorithm_name + "_schedule.txt");

    } else {
        std::cout << "Unknown algorithm: " << algorithm_name << std::endl;
        return 1;
    }

    std::cout << "OSP Success" << std::endl;
    return 0;
}
