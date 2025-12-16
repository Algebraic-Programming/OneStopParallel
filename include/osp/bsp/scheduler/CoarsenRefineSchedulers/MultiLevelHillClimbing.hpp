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

#include "osp/bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "osp/bsp/scheduler/LocalSearch/HillClimbing/hill_climbing.hpp"
#include "osp/coarser/StepByStep/StepByStepCoarser.hpp"

namespace osp {

template <typename GraphT>
class MultiLevelHillClimbingScheduler : public Scheduler<GraphT> {
    using VertexIdx = VertexIdxT<GraphT>;

    using VertexTypeTOrDefault = std::conditional_t<isComputationalDagTypedVerticesV<GraphT>, VTypeT<GraphT>, unsigned>;
    using EdgeCommwTOrDefault = std::conditional_t<hasEdgeWeightsV<GraphT>, ECommwT<GraphT>, VCommwT<GraphT>>;

  private:
    typename StepByStepCoarser<GraphT>::CoarseningStrategy coarseningStrategy_
        = StepByStepCoarser<GraphT>::CoarseningStrategy::EDGE_BY_EDGE;
    unsigned numberHcSteps_;
    unsigned targetNrOfNodes_ = 0;
    unsigned minTargetNrOfNodes_ = 1U;
    double contractionRate_ = 0.5;

    unsigned linearRefinementStepSize_ = 20;
    bool useLinearRefinement_ = true;

    double exponentialRefinementStepRatio_ = 1.1;
    bool useExponentialRefinement_ = false;

    std::deque<VertexIdx> refinementPoints_;

    BspSchedule<GraphT> Refine(const BspInstance<GraphT> &instance,
                               const StepByStepCoarser<GraphT> &coarser,
                               const BspSchedule<GraphT> &coarseSchedule) const;

    BspSchedule<GraphT> ComputeUncontractedSchedule(const StepByStepCoarser<GraphT> &coarser,
                                                    const BspInstance<GraphT> &fullInstance,
                                                    const BspSchedule<GraphT> &coarseSchedule,
                                                    VertexIdx indexUntil) const;

    void SetLinearRefinementPoints(VertexIdx originalNrOfNodes, unsigned stepSize);
    void SetExponentialRefinementPoints(VertexIdx originalNrOfNodes, double stepRatio);

    void SetParameter(const size_t numVertices) {
        targetNrOfNodes_ = std::max(minTargetNrOfNodes_, static_cast<unsigned>(static_cast<float>(numVertices) * contractionRate_));
        targetNrOfNodes_ = std::min(targetNrOfNodes_, static_cast<unsigned>(numVertices));

        if (useLinearRefinement_) {
            SetLinearRefinementPoints(numVertices, linearRefinementStepSize_);
        } else if (useExponentialRefinement_) {
            SetExponentialRefinementPoints(numVertices, exponentialRefinementStepRatio_);
        }
    }

  public:
    virtual ~MultiLevelHillClimbingScheduler() = default;

    virtual ReturnStatus ComputeSchedule(BspSchedule<GraphT> &schedule) override;

    virtual std::string GetScheduleName() const override { return "MultiLevelHillClimbing"; }

    void SetCoarseningStrategy(typename StepByStepCoarser<GraphT>::CoarseningStrategy strategy) {
        coarseningStrategy_ = strategy;
    }

    void SetContractionRate(double rate) { contractionRate_ = rate; }

    void SetNumberOfHcSteps(unsigned steps) { numberHcSteps_ = steps; }

    void SetMinTargetNrOfNodes(unsigned minTargetNrOfNodes) { minTargetNrOfNodes_ = minTargetNrOfNodes; }

    void UseLinearRefinementSteps(unsigned steps) {
        useLinearRefinement_ = true;
        useExponentialRefinement_ = false;
        linearRefinementStepSize_ = steps;
    }

    void UseExponentialRefinementPoints(double ratio) {
        useExponentialRefinement_ = true;
        useLinearRefinement_ = false;
        exponentialRefinementStepRatio_ = ratio;
    }
};

template <typename GraphT>
ReturnStatus MultiLevelHillClimbingScheduler<GraphT>::ComputeSchedule(BspSchedule<GraphT> &schedule) {
    StepByStepCoarser<GraphT> coarser;
    GraphT coarseDAG;
    std::vector<VertexIdx> newVertexId;

    const auto numVertices = schedule.GetInstance().NumberOfVertices();
    SetParameter(numVertices);

    newVertexId = coarser.GenerateVertexContractionMap(schedule.GetInstance().GetComputationalDag());
    coarseDAG = coarser.Contract(newVertexId);

    BspInstance<GraphT> coarseInstance(coarseDAG, schedule.GetInstance().GetArchitecture());

    GreedyBspScheduler<GraphT> greedy;
    BspSchedule<GraphT> coarseSchedule(coarseInstance);
    greedy.ComputeSchedule(coarseSchedule);

    HillClimbingScheduler<GraphT> coarseHc;
    coarseHc.ImproveSchedule(coarseSchedule);

    if (refinementPoints_.empty()) {
        SetExponentialRefinementPoints(numVertices, 1.1);
    }
    while (!refinementPoints_.empty() && refinementPoints_.front() <= coarseDAG.NumVertices()) {
        refinementPoints_.pop_front();
    }

    schedule = Refine(schedule.GetInstance(), coarser, coarseSchedule);

    return ReturnStatus::OSP_SUCCESS;
}

// run refinement: uncoarsify the DAG in small batches, and apply some steps of hill climbing after each iteration
template <typename GraphT>
BspSchedule<GraphT> MultiLevelHillClimbingScheduler<GraphT>::Refine(const BspInstance<GraphT> &fullInstance,
                                                                    const StepByStepCoarser<GraphT> &coarser,
                                                                    const BspSchedule<GraphT> &coarseSchedule) const {
    BspSchedule<GraphT> scheduleOnFullGraph
        = ComputeUncontractedSchedule(coarser, fullInstance, coarseSchedule, coarser.GetContractionHistory().size());

    for (VertexIdx nextSize : refinementPoints_) {
        const VertexIdx contractSteps = coarser.GetOriginalDag().NumVertices() - nextSize;
        std::vector<VertexIdx> newIds = coarser.GetIntermediateIDs(contractSteps);
        GraphT dag = coarser.Contract(newIds);

        BspInstance<GraphT> instance(dag, fullInstance.GetArchitecture());
        BspSchedule<GraphT> schedule(instance);

        // Project full schedule to current graph
        for (VertexIdx node = 0; node < fullInstance.NumberOfVertices(); ++node) {
            schedule.SetAssignedProcessor(newIds[node], scheduleOnFullGraph.AssignedProcessor(node));
            schedule.SetAssignedSuperstep(newIds[node], scheduleOnFullGraph.AssignedSuperstep(node));
        }

        HillClimbingScheduler<GraphT> hc;
        hc.ImproveScheduleWithStepLimit(schedule, numberHcSteps_);

        scheduleOnFullGraph = ComputeUncontractedSchedule(coarser, fullInstance, schedule, contractSteps);
    }

    std::cout << "Refined cost: " << scheduleOnFullGraph.ComputeCosts() << std::endl;
    return scheduleOnFullGraph;
}

// given an original DAG G, a schedule on the coarsified G and the contraction steps, project the coarse schedule to the entire G
template <typename GraphT>
BspSchedule<GraphT> MultiLevelHillClimbingScheduler<GraphT>::ComputeUncontractedSchedule(const StepByStepCoarser<GraphT> &coarser,
                                                                                         const BspInstance<GraphT> &fullInstance,
                                                                                         const BspSchedule<GraphT> &coarseSchedule,
                                                                                         VertexIdx indexUntil) const {
    std::vector<VertexIdx> newIds = coarser.GetIntermediateIDs(indexUntil);

    BspSchedule<GraphT> schedule(fullInstance);

    for (VertexIdx node = 0; node < fullInstance.NumberOfVertices(); ++node) {
        schedule.SetAssignedProcessor(node, coarseSchedule.AssignedProcessor(newIds[node]));
        schedule.SetAssignedSuperstep(node, coarseSchedule.AssignedSuperstep(newIds[node]));
    }
    return schedule;
}

template <typename GraphT>
void MultiLevelHillClimbingScheduler<GraphT>::SetLinearRefinementPoints(VertexIdx originalNrOfNodes, unsigned stepSize) {
    refinementPoints_.clear();
    if (stepSize < 5) {
        stepSize = 5;
    }

    for (VertexIdx nextN = targetNrOfNodes_ + stepSize; nextN < originalNrOfNodes; nextN += stepSize) {
        refinementPoints_.push_back(nextN);
    }

    if (!refinementPoints_.empty()) {
        refinementPoints_.pop_back();
    }
    refinementPoints_.push_back(originalNrOfNodes);
}

template <typename GraphT>
void MultiLevelHillClimbingScheduler<GraphT>::SetExponentialRefinementPoints(VertexIdx originalNrOfNodes, double stepRatio) {
    refinementPoints_.clear();
    if (stepRatio < 1.01) {
        stepRatio = 1.01;
    }

    for (VertexIdx nextN = std::max(static_cast<unsigned>(std::round(targetNrOfNodes_ * stepRatio)), targetNrOfNodes_ + 5);
         nextN < originalNrOfNodes;
         nextN
         = std::max(static_cast<VertexIdx>(std::round(static_cast<double>(nextN) * stepRatio)), refinementPoints_.back() + 5)) {
        refinementPoints_.push_back(nextN);
    }

    refinementPoints_.push_back(originalNrOfNodes);
}

}    // namespace osp
