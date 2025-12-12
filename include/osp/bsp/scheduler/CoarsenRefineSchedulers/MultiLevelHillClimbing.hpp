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
    using vertex_idx = VertexIdxT<GraphT>;

    using vertex_type_t_or_default = std::conditional_t<IsComputationalDagTypedVerticesV<GraphT>, VTypeT<GraphT>, unsigned>;
    using edge_commw_t_or_default = std::conditional_t<HasEdgeWeightsV<GraphT>, ECommwT<GraphT>, VCommwT<GraphT>>;

  private:
    typename StepByStepCoarser<GraphT>::COARSENING_STRATEGY coarseningStrategy_
        = StepByStepCoarser<GraphT>::COARSENING_STRATEGY::EDGE_BY_EDGE;
    unsigned numberHcSteps_;
    unsigned targetNrOfNodes_ = 0;
    unsigned minTargetNrOfNodes_ = 1U;
    double contractionRate_ = 0.5;

    unsigned linearRefinementStepSize_ = 20;
    bool useLinearRefinement_ = true;

    double exponentialRefinementStepRatio_ = 1.1;
    bool useExponentialRefinement_ = false;

    std::deque<vertex_idx> refinementPoints_;

    BspSchedule<GraphT> Refine(const BspInstance<GraphT> &instance,
                               const StepByStepCoarser<GraphT> &coarser,
                               const BspSchedule<GraphT> &coarseSchedule) const;

    BspSchedule<GraphT> ComputeUncontractedSchedule(const StepByStepCoarser<GraphT> &coarser,
                                                    const BspInstance<GraphT> &fullInstance,
                                                    const BspSchedule<GraphT> &coarseSchedule,
                                                    vertex_idx indexUntil) const;

    void SetLinearRefinementPoints(vertex_idx originalNrOfNodes, unsigned stepSize);
    void SetExponentialRefinementPoints(vertex_idx originalNrOfNodes, double stepRatio);

    void SetParameter(const size_t numVertices) {
        targetNrOfNodes_ = std::max(minTargetNrOfNodes_, static_cast<unsigned>(static_cast<float>(numVertices) * contractionRate_));
        targetNrOfNodes_ = std::min(targetNrOfNodes_, static_cast<unsigned>(numVertices));

        if (useLinearRefinement_) {
            setLinearRefinementPoints(numVertices, linear_refinement_step_size_);
        } else if (useExponentialRefinement_) {
            setExponentialRefinementPoints(numVertices, exponential_refinement_step_ratio_);
        }
    }

  public:
    virtual ~MultiLevelHillClimbingScheduler() = default;

    virtual ReturnStatus ComputeSchedule(BspSchedule<GraphT> &schedule) override;

    virtual std::string getScheduleName() const override { return "MultiLevelHillClimbing"; }

    void SetCoarseningStrategy(typename StepByStepCoarser<GraphT>::COARSENING_STRATEGY strategy) {
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
    std::vector<vertex_idx> newVertexId;

    const auto numVerices = schedule.GetInstance().NumberOfVertices();
    SetParameter(numVerices);

    coarser.coarsenDag(schedule.GetInstance().GetComputationalDag(), coarseDAG, new_vertex_id);

    BspInstance<GraphT> coarseInstance(coarseDAG, schedule.GetInstance().GetArchitecture());

    GreedyBspScheduler<GraphT> greedy;
    BspSchedule<GraphT> coarseSchedule(coarseInstance);
    greedy.ComputeSchedule(coarseSchedule);

    HillClimbingScheduler<GraphT> coarseHc;
    coarseHc.improveSchedule(coarseSchedule);

    if (refinement_points.empty()) {
        setExponentialRefinementPoints(num_verices, 1.1);
    }
    while (!refinement_points.empty() && refinement_points.front() <= coarseDAG.NumVertices()) {
        refinement_points.pop_front();
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
        = ComputeUncontractedSchedule(coarser, full_instance, coarse_schedule, coarser.getContractionHistory().size());

    for (vertex_idx next_size : refinement_points) {
        const vertex_idx contract_steps = coarser.getOriginalDag().NumVertices() - next_size;
        std::vector<vertex_idx> new_ids = coarser.GetIntermediateIDs(contract_steps);
        Graph_t dag = coarser.Contract(new_ids);

        BspInstance<GraphT> instance(dag, full_instance.GetArchitecture());
        BspSchedule<GraphT> schedule(instance);

        // Project full schedule to current graph
        for (vertex_idx node = 0; node < full_instance.NumberOfVertices(); ++node) {
            schedule.SetAssignedProcessor(new_ids[node], schedule_on_full_graph.AssignedProcessor(node));
            schedule.SetAssignedSuperstep(new_ids[node], schedule_on_full_graph.AssignedSuperstep(node));
        }

        HillClimbingScheduler<GraphT> hc;
        hc.improveScheduleWithStepLimit(schedule, number_hc_steps);

        schedule_on_full_graph = ComputeUncontractedSchedule(coarser, full_instance, schedule, contract_steps);
    }

    std::cout << "Refined cost: " << scheduleOnFullGraph.computeCosts() << std::endl;
    return scheduleOnFullGraph;
}

// given an original DAG G, a schedule on the coarsified G and the contraction steps, project the coarse schedule to the entire G
template <typename GraphT>
BspSchedule<GraphT> MultiLevelHillClimbingScheduler<GraphT>::ComputeUncontractedSchedule(const StepByStepCoarser<GraphT> &coarser,
                                                                                         const BspInstance<GraphT> &fullInstance,
                                                                                         const BspSchedule<GraphT> &coarseSchedule,
                                                                                         vertex_idx indexUntil) const {
    std::vector<vertex_idx> newIds = coarser.GetIntermediateIDs(index_until);

    BspSchedule<GraphT> schedule(fullInstance);

    for (vertex_idx node = 0; node < fullInstance.NumberOfVertices(); ++node) {
        schedule.SetAssignedProcessor(node, coarseSchedule.AssignedProcessor(new_ids[node]));
        schedule.SetAssignedSuperstep(node, coarseSchedule.AssignedSuperstep(new_ids[node]));
    }
    return schedule;
}

template <typename GraphT>
void MultiLevelHillClimbingScheduler<GraphT>::SetLinearRefinementPoints(vertex_idx originalNrOfNodes, unsigned stepSize) {
    refinement_points.clear();
    if (stepSize < 5) {
        stepSize = 5;
    }

    for (vertex_idx nextN = targetNrOfNodes_ + stepSize; nextN < OriginalNrOfNodes; nextN += stepSize) {
        refinement_points.push_back(nextN);
    }

    if (!refinement_points.empty()) {
        refinement_points.pop_back();
    }
    refinement_points.push_back(OriginalNrOfNodes);
}

template <typename GraphT>
void MultiLevelHillClimbingScheduler<GraphT>::SetExponentialRefinementPoints(vertex_idx originalNrOfNodes, double stepRatio) {
    refinement_points.clear();
    if (stepRatio < 1.01) {
        stepRatio = 1.01;
    }

    for (vertex_idx nextN = std::max(static_cast<unsigned>(std::round(targetNrOfNodes_ * stepRatio)), targetNrOfNodes_ + 5);
         nextN < OriginalNrOfNodes;
         nextN
         = std::max(static_cast<vertex_idx>(std::round(static_cast<double>(nextN) * stepRatio)), refinement_points.back() + 5)) {
        refinement_points.push_back(nextN);
    }

    refinement_points.push_back(OriginalNrOfNodes);
}

}    // namespace osp
