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

#pragma once

#include "ConnectedComponentDivider.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/graph_algorithms/computational_dag_util.hpp"

namespace osp {

template <typename GraphT, typename ConstrGraphT>
class ConnectedComponentScheduler : public Scheduler<GraphT> {
    Scheduler<ConstrGraphT> *scheduler_;

  public:
    ConnectedComponentScheduler(Scheduler<ConstrGraphT> &scheduler) : scheduler_(&scheduler) {}

    std::string GetScheduleName() const override { return "SubDagScheduler"; }

    ReturnStatus ComputeSchedule(BspSchedule<GraphT> &schedule) override {
        const auto &instance = schedule.GetInstance();

        const GraphT &dag = instance.GetComputationalDag();
        ConnectedComponentDivider<GraphT, ConstrGraphT> partitioner;

        partitioner.Divide(dag);

        VWorkwT<GraphT> totalWorkWeight = SumOfVerticesWorkWeights(dag);

        unsigned numProcessorsOffset = 0;

        for (std::size_t i = 0; i < partitioner.GetSubDags().size(); i++) {
            const auto &subDag = partitioner.GetSubDags()[i];
            const auto &mapping = partitioner.GetVertexMapping()[i];

            VWorkwT<ConstrGraphT> subDagWorkWeight = SumOfVerticesWorkWeights(subDag);

            BspInstance<ConstrGraphT> subInstance(subDag, instance.GetArchitecture());
            BspArchitecture<ConstrGraphT> &subArchitecture = subInstance.GetArchitecture();

            const double subDagWorkWeightPercent = static_cast<double>(subDagWorkWeight) / static_cast<double>(totalWorkWeight);
            const unsigned subDagProcessors = static_cast<unsigned>(subDagWorkWeightPercent * subArchitecture.NumberOfProcessors());

            subArchitecture.SetNumberOfProcessors(subDagProcessors);

            BspSchedule<ConstrGraphT> subSchedule(subInstance);
            auto status = scheduler_->ComputeSchedule(subSchedule);

            if (status != ReturnStatus::OSP_SUCCESS && status != ReturnStatus::BEST_FOUND) {
                return status;
            }

            for (const auto &v : subInstance.Vertices()) {
                schedule.SetAssignedProcessor(mapping.at(v), subSchedule.AssignedProcessor(v) + numProcessorsOffset);
                schedule.SetAssignedSuperstep(mapping.at(v), subSchedule.AssignedSuperstep(v));
            }

            numProcessorsOffset += subArchitecture.NumberOfProcessors();
        }

        return ReturnStatus::OSP_SUCCESS;
    }
};

}    // namespace osp
