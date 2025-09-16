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
#include "osp/coarser/StepByStep/StepByStepCoarser.hpp"
#include "osp/bsp/scheduler/LocalSearch/HillClimbing/hill_climbing.hpp"

namespace osp{

template<typename Graph_t>
class MultiLevelHillClimbingScheduler : public Scheduler<Graph_t> {

    using vertex_idx = vertex_idx_t<Graph_t>;

    using vertex_type_t_or_default = std::conditional_t<is_computational_dag_typed_vertices_v<Graph_t>, v_type_t<Graph_t>, unsigned>;
    using edge_commw_t_or_default = std::conditional_t<has_edge_weights_v<Graph_t>, e_commw_t<Graph_t>, v_commw_t<Graph_t>>;

    private:

    typename StepByStepCoarser<Graph_t>::COARSENING_STRATEGY coarsening_strategy = StepByStepCoarser<Graph_t>::COARSENING_STRATEGY::EDGE_BY_EDGE;
    unsigned number_hc_steps;
    unsigned target_nr_of_nodes = 0;
    unsigned min_target_nr_of_nodes_ = 1U;
    double contraction_rate_ = 0.5;

    unsigned linear_refinement_step_size_ = 20;
    bool use_linear_refinement_ = true;

    double exponential_refinement_step_ratio_ = 1.1;
    bool use_exponential_refinement_ = false;

    std::deque<vertex_idx> refinement_points;

    BspSchedule<Graph_t> Refine(const BspInstance<Graph_t>& instance, const StepByStepCoarser<Graph_t>& coarser,
        const BspSchedule<Graph_t> &coarse_schedule) const;

    BspSchedule<Graph_t> ComputeUncontractedSchedule(const StepByStepCoarser<Graph_t>& coarser,
                                                const BspInstance<Graph_t>& full_instance,
                                                const BspSchedule<Graph_t> &coarse_schedule, vertex_idx index_until) const;

    void setLinearRefinementPoints(vertex_idx OriginalNrOfNodes, unsigned stepSize);
    void setExponentialRefinementPoints(vertex_idx OriginalNrOfNodes, double stepRatio);

    void set_parameter(const size_t num_vertices) {
        target_nr_of_nodes = std::max(min_target_nr_of_nodes_, static_cast<unsigned>(static_cast<float>(num_vertices) * contraction_rate_));
        target_nr_of_nodes = std::min(target_nr_of_nodes, static_cast<unsigned>(num_vertices));

        if(use_linear_refinement_) {
            setLinearRefinementPoints(num_vertices, linear_refinement_step_size_);
        } else if (use_exponential_refinement_)  {
            setExponentialRefinementPoints(num_vertices, exponential_refinement_step_ratio_);
        }
    }

  public:

    virtual ~MultiLevelHillClimbingScheduler() = default;

    virtual RETURN_STATUS computeSchedule(BspSchedule<Graph_t> &schedule) override;

    virtual std::string getScheduleName() const override { return "MultiLevelHillClimbing"; }

    void setCoarseningStrategy(typename StepByStepCoarser<Graph_t>::COARSENING_STRATEGY strategy_){ coarsening_strategy = strategy_;}
    void setContractionRate(double rate_){ contraction_rate_ = rate_;}
    void setNumberOfHcSteps(unsigned steps_) { number_hc_steps = steps_; }
    void setMinTargetNrOfNodes(unsigned min_target_nr_of_nodes) { min_target_nr_of_nodes_ = min_target_nr_of_nodes; }

    void useLinearRefinementSteps(unsigned steps) { 
        use_linear_refinement_ = true;
        use_exponential_refinement_ = false;
        linear_refinement_step_size_ = steps;
    }

    void useExponentialRefinementPoints(double ratio) { 
        use_exponential_refinement_ = true;
        use_linear_refinement_ = false;
        exponential_refinement_step_ratio_ = ratio;
    }

};

template<typename Graph_t>
RETURN_STATUS MultiLevelHillClimbingScheduler<Graph_t>::computeSchedule(BspSchedule<Graph_t> &schedule) {

    StepByStepCoarser<Graph_t> coarser;
    Graph_t coarseDAG;
    std::vector<vertex_idx> new_vertex_id;

    const auto num_verices = schedule.getInstance().numberOfVertices();
    set_parameter(num_verices);

    coarser.coarsenDag(schedule.getInstance().getComputationalDag(), coarseDAG, new_vertex_id);

    BspInstance<Graph_t> coarse_instance(coarseDAG, schedule.getInstance().getArchitecture());

    GreedyBspScheduler<Graph_t> greedy;
    BspSchedule<Graph_t> coarse_schedule(coarse_instance);
    greedy.computeSchedule(coarse_schedule);

    HillClimbingScheduler<Graph_t> coarse_hc;
    coarse_hc.improveSchedule(coarse_schedule);

    if(refinement_points.empty())
        setExponentialRefinementPoints(num_verices, 1.1);
    while(!refinement_points.empty() && refinement_points.front() <= coarseDAG.num_vertices())
        refinement_points.pop_front();

    schedule = Refine(schedule.getInstance(), coarser, coarse_schedule);

    return RETURN_STATUS::OSP_SUCCESS;
}

// run refinement: uncoarsify the DAG in small batches, and apply some steps of hill climbing after each iteration
template<typename Graph_t>
BspSchedule<Graph_t> MultiLevelHillClimbingScheduler<Graph_t>::Refine(const BspInstance<Graph_t>& full_instance, const StepByStepCoarser<Graph_t>& coarser, const BspSchedule<Graph_t> &coarse_schedule) const {

    BspSchedule<Graph_t> schedule_on_full_graph = ComputeUncontractedSchedule(coarser, full_instance, coarse_schedule, coarser.getContractionHistory().size());

    for (vertex_idx next_size : refinement_points)
    {
        const vertex_idx contract_steps = coarser.getOriginalDag().num_vertices() - next_size;
        std::vector<vertex_idx> new_ids = coarser.GetIntermediateIDs(contract_steps);
        Graph_t dag = coarser.Contract(new_ids);

        BspInstance<Graph_t> instance(dag, full_instance.getArchitecture());
        BspSchedule<Graph_t> schedule(instance);

        // Project full schedule to current graph
        for (vertex_idx node = 0; node < full_instance.numberOfVertices(); ++node) {
            schedule.setAssignedProcessor(new_ids[node], schedule_on_full_graph.assignedProcessor(node));
            schedule.setAssignedSuperstep(new_ids[node], schedule_on_full_graph.assignedSuperstep(node));
        }

        HillClimbingScheduler<Graph_t> hc;
        hc.improveScheduleWithStepLimit(schedule, number_hc_steps);

        schedule_on_full_graph = ComputeUncontractedSchedule(coarser, full_instance, schedule, contract_steps);
    }

    std::cout << "Refined cost: " << schedule_on_full_graph.computeCosts() << std::endl;
    return schedule_on_full_graph;
};

// given an original DAG G, a schedule on the coarsified G and the contraction steps, project the coarse schedule to the entire G
template<typename Graph_t>
BspSchedule<Graph_t> MultiLevelHillClimbingScheduler<Graph_t>::ComputeUncontractedSchedule(const StepByStepCoarser<Graph_t>& coarser,
                                                const BspInstance<Graph_t>& full_instance,
                                                const BspSchedule<Graph_t> &coarse_schedule, vertex_idx index_until) const {
                                                    
    std::vector<vertex_idx> new_ids = coarser.GetIntermediateIDs(index_until);

    BspSchedule<Graph_t> schedule(full_instance);

    for (vertex_idx node = 0; node < full_instance.numberOfVertices(); ++node)
    {
        schedule.setAssignedProcessor(node, coarse_schedule.assignedProcessor(new_ids[node]));
        schedule.setAssignedSuperstep(node, coarse_schedule.assignedSuperstep(new_ids[node]));
    }
    return schedule;
};

template<typename Graph_t>
void MultiLevelHillClimbingScheduler<Graph_t>::setLinearRefinementPoints(vertex_idx OriginalNrOfNodes, unsigned stepSize)
{
    refinement_points.clear();
    if(stepSize<5)
        stepSize = 5;

    for (vertex_idx nextN = target_nr_of_nodes + stepSize; nextN < OriginalNrOfNodes; nextN += stepSize)
        refinement_points.push_back(nextN);

    refinement_points.pop_back();
    refinement_points.push_back(OriginalNrOfNodes);
}

template<typename Graph_t>
void MultiLevelHillClimbingScheduler<Graph_t>::setExponentialRefinementPoints(vertex_idx OriginalNrOfNodes, double stepRatio)
{
    refinement_points.clear();
    if(stepRatio<1.01)
        stepRatio = 1.01;

    for (vertex_idx nextN = std::max(static_cast<unsigned>(std::round(target_nr_of_nodes * stepRatio)), target_nr_of_nodes+5);
                        nextN < OriginalNrOfNodes;
                        nextN = std::max(static_cast<vertex_idx>(std::round(static_cast<double>(nextN) * stepRatio)), refinement_points.back()+5))
        refinement_points.push_back(nextN);

    refinement_points.push_back(OriginalNrOfNodes);
}


} // namespace osp