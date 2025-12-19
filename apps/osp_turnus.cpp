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

#include "osp/auxiliary/io/bsp_schedule_file_writer.hpp"
#include "osp/auxiliary/io/general_file_reader.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/EtfScheduler.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "osp/bsp/scheduler/LoadBalanceScheduler/LightEdgeVariancePartitioner.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"

using namespace osp;

using GraphT = ComputationalDagEdgeIdxVectorImplDefIntT;
using MemConstr = PersistentTransientMemoryConstraint<GraphT>;

// invoked upon program call
int main(int argc, char *argv[]) {
    if (argc != 5) {
        std::cout << "Usage: " << argv[0] << " <input_file_dag> <num_proc> <memory_bound> <algorithm>" << std::endl;
        std::cout << "Available algorithms: bsp, etf, variance" << std::endl;
        return 1;
    }

    BspInstance<GraphT> bspInstance;

    bspInstance.GetArchitecture().SetNumberOfProcessors(static_cast<unsigned>(std::stoul(argv[2])));
    bspInstance.GetArchitecture().SetMemoryBound(std::atoi(argv[3]));
    bspInstance.GetArchitecture().SetMemoryConstraintType(MemoryConstraintType::PERSISTENT_AND_TRANSIENT);

    std::string algorithmName = argv[4];

    std::string filenameGraph = argv[1];

    bool statusGraph = file_reader::ReadGraph(filenameGraph, bspInstance.GetComputationalDag());

    if (!statusGraph) {
        std::cout << "Error while reading the graph from file: " << filenameGraph << std::endl;
        return 1;
    }

    if (bspInstance.GetComputationalDag().NumVertexTypes() > 1) {
        std::cout << "The graph has more than one vertex type, which is not supported by this scheduler." << std::endl;
        return 1;
    }

    boost::algorithm::to_lower(algorithmName);    // modifies str

    BspSchedule<GraphT> bspSchedule(bspInstance);
    Scheduler<GraphT> *scheduler = nullptr;

    if (algorithmName == "bsp") {
        float maxPercentIdleProcessors = 0.2f;
        bool increaseParallelismInNewSuperstep = true;

        scheduler = new GreedyBspScheduler<GraphT, MemConstr>(maxPercentIdleProcessors, increaseParallelismInNewSuperstep);

    } else if (algorithmName == "etf") {
        scheduler = new EtfScheduler<GraphT, MemConstr>(BL_EST);

    } else if (algorithmName == "variance") {
        const double maxPercentIdleProcessors = 0.0;
        const bool increaseParallelismInNewSuperstep = true;
        const double variancePower = 6.0;
        const float maxPriorityDifferencePercent = 0.34f;
        const double heavyIsXTimesMedian = 3.0;
        const double minPercentComponentsRetained = 0.25;
        const float boundComponentWeightPercent = 4.0f;
        const float slack = 0.0f;

        scheduler = new LightEdgeVariancePartitioner<GraphT, FlatSplineInterpolation, MemConstr>(maxPercentIdleProcessors,
                                                                                                 variancePower,
                                                                                                 heavyIsXTimesMedian,
                                                                                                 minPercentComponentsRetained,
                                                                                                 boundComponentWeightPercent,
                                                                                                 increaseParallelismInNewSuperstep,
                                                                                                 maxPriorityDifferencePercent,
                                                                                                 slack);

    } else {
        std::cout << "Unknown algorithm: " << algorithmName << std::endl;
        return 1;
    }

    auto schedulerStatus = scheduler->ComputeSchedule(bspSchedule);

    if (schedulerStatus == ReturnStatus::ERROR) {
        std::cout << "Error while scheduling!" << std::endl;
        delete scheduler;
        return 1;
    }

    delete scheduler;

    file_writer::WriteTxt(filenameGraph + "_" + algorithmName + "_schedule.shed", bspSchedule);

    std::cout << "OSP Success" << std::endl;
    return 0;
}
