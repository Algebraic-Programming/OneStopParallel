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

#include "IBspScheduleEval.hpp"
#include "osp/bsp/model/BspScheduleCS.hpp"
#include "osp/concepts/computational_dag_concept.hpp"

namespace osp {

template <typename GraphT>
class BspScheduleRecomp : public IBspScheduleEval<GraphT> {
  public:
    using vertex_idx = vertex_idx_t<Graph_t>;
    using cost_type = v_workw_t<Graph_t>;

    using KeyTriple = std::tuple<vertex_idx_t<Graph_t>, unsigned int, unsigned int>;

    static_assert(is_computational_dag_v<Graph_t>, "BspScheduleRecomp can only be used with computational DAGs.");
    static_assert(std::is_same_v<v_workw_t<Graph_t>, v_commw_t<Graph_t>>,
                  "BspScheduleRecomp requires work and comm. weights to have the same type.");

  private:
    const BspInstance<GraphT> *instance_;

    unsigned int numberOfSupersteps_ = 0;

    std::vector<std::vector<std::pair<unsigned, unsigned>>> nodeToProcessorAndSupertepAssignment_;

    std::map<KeyTriple, unsigned> commSchedule_;

  public:
    BspScheduleRecomp() = default;

    BspScheduleRecomp(const BspInstance<GraphT> &inst) : instance_(&inst) {
        nodeToProcessorAndSupertepAssignment_.resize(inst.numberOfVertices());
    }

    BspScheduleRecomp(const BspScheduleCS<GraphT> &schedule);

    BspScheduleRecomp(const BspSchedule<GraphT> &schedule) : BspScheduleRecomp<GraphT>(BspScheduleCS<GraphT>(schedule)) {}

    virtual ~BspScheduleRecomp() = default;

    const BspInstance<GraphT> &GetInstance() const { return *instance_; }

    /**
     * @brief Returns the number of supersteps in the schedule.
     *
     * @return The number of supersteps in the schedule.
     */
    virtual unsigned numberOfSupersteps() const override { return numberOfSupersteps_; }

    void SetNumberOfSupersteps(unsigned numberOfSupersteps) { numberOfSupersteps_ = numberOfSupersteps; }

    std::vector<std::pair<unsigned, unsigned>> &Assignments(vertex_idx node) {
        return nodeToProcessorAndSupertepAssignment_[node];
    }

    const std::vector<std::pair<unsigned, unsigned>> &Assignments(vertex_idx node) const {
        return nodeToProcessorAndSupertepAssignment_[node];
    }

    /**
     * @brief Sets the communication schedule for the schedule.
     *
     * @param cs The communication schedule to set.
     */
    void SetCommunicationSchedule(const std::map<KeyTriple, unsigned int> &cs);

    /**
     * @brief Adds an entry to the communication schedule.
     *
     * @param key The key for the communication schedule entry.
     * @param step The superstep for the communication schedule entry.
     */
    void AddCommunicationScheduleEntry(KeyTriple key, unsigned step);

    /**
     * @brief Adds an entry to the communication schedule.
     *
     * @param node The node resp. its data which is sent.
     * @param from_proc The processor from which the data is sent.
     * @param to_proc The processor to which the data is sent.
     * @param step The superstep in which the data is sent.
     */
    void AddCommunicationScheduleEntry(unsigned node, unsigned fromProc, unsigned toProc, unsigned step);

    /**
     * @brief Returns the communication schedule for the schedule.
     *
     * @return The communication schedule for the schedule.
     */
    const std::map<KeyTriple, unsigned int> &GetCommunicationSchedule() const { return commSchedule; }

    std::map<KeyTriple, unsigned int> &GetCommunicationSchedule() { return commSchedule; }

    virtual cost_type computeWorkCosts() const override;

    virtual cost_type computeCosts() const override;

    /**
     * @brief Returns true if the schedule is valid, i.e. if every time we compute a node, all its parents are already available
     * on the given processor, and every time we send a value, it is also already available on the given source processor.
     *
     * @return True if the schedule is valid, false otherwise.
     */
    bool SatisfiesConstraints() const;

    vertex_idx GetTotalAssignments() const;

    void MergeSupersteps();
};

template <typename GraphT>
BspScheduleRecomp<GraphT>::BspScheduleRecomp(const BspScheduleCS<GraphT> &schedule) : instance_(&schedule.getInstance()) {
    nodeToProcessorAndSupertepAssignment_.clear();
    nodeToProcessorAndSupertepAssignment_.resize(instance_->numberOfVertices());
    numberOfSupersteps_ = schedule.numberOfSupersteps();

    for (vertex_idx node = 0; node < instance_->numberOfVertices(); ++node) {
        nodeToProcessorAndSupertepAssignment_[node].emplace_back(schedule.assignedProcessor(node),
                                                                 schedule.assignedSuperstep(node));
    }

    commSchedule = schedule.getCommunicationSchedule();
}

template <typename GraphT>
void BspScheduleRecomp<GraphT>::AddCommunicationScheduleEntry(unsigned node, unsigned fromProc, unsigned toProc, unsigned step) {
    AddCommunicationScheduleEntry(std::make_tuple(node, fromProc, toProc), step);
}

template <typename GraphT>
void BspScheduleRecomp<GraphT>::AddCommunicationScheduleEntry(KeyTriple key, unsigned step) {
    if (step >= numberOfSupersteps_) {
        throw std::invalid_argument("Invalid Argument while adding communication schedule entry: step out of range.");
    }

    if (std::get<0>(key) >= instance_->numberOfVertices()) {
        throw std::invalid_argument("Invalid Argument while adding communication schedule entry: node out of range.");
    }

    if (std::get<1>(key) >= instance_->numberOfProcessors()) {
        throw std::invalid_argument("Invalid Argument while adding communication schedule entry: from processor out of range.");
    }

    if (std::get<2>(key) >= instance_->numberOfProcessors()) {
        throw std::invalid_argument("Invalid Argument while adding communication schedule entry: to processor out of range.");
    }

    commSchedule[key] = step;
}

template <typename GraphT>
bool BspScheduleRecomp<GraphT>::SatisfiesConstraints() const {
    // find first availability

    std::vector<std::vector<unsigned>> nodeFirstAvailableOnProc(
        instance_->numberOfVertices(),
        std::vector<unsigned>(instance_->numberOfProcessors(), std::numeric_limits<unsigned>::max()));

    for (vertex_idx node = 0; node < instance_->numberOfVertices(); ++node) {
        for (const std::pair<unsigned, unsigned> &compute_step : node_to_processor_and_supertep_assignment[node]) {
            node_first_available_on_proc[node][compute_step.first]
                = std::min(node_first_available_on_proc[node][compute_step.first], compute_step.second);
        }
    }

    for (auto const &[key, val] : commSchedule) {
        const vertex_idx &node = std::get<0>(key);
        const unsigned &to_proc = std::get<2>(key);

        node_first_available_on_proc[node][to_proc] = std::min(node_first_available_on_proc[node][to_proc], val + 1);
    }

    // check validity

    for (vertex_idx node = 0; node < instance_->numberOfVertices(); ++node) {
        for (vertex_idx pred : instance->getComputationalDag().parents(node)) {
            for (const std::pair<unsigned, unsigned> &compute_step : node_to_processor_and_supertep_assignment[node]) {
                if (node_first_available_on_proc[pred][compute_step.first] > compute_step.second) {
                    // std::cout << "Not a valid schedule: parent " << pred << " of node "<< node <<
                    //" not yet available on processor " << compute_step.first << " in superstep "<< compute_step.second <<"." << std::endl;
                    return false;
                }
            }
        }
    }

    for (auto const &[key, val] : commSchedule) {
        const vertex_idx &node = std::get<0>(key);
        const unsigned &from_proc = std::get<1>(key);

        if (node_first_available_on_proc[node][from_proc] > val) {
            // std::cout << "Not a valid schedule: node " << node << " not yet available for sending from processor "
            // << from_proc << " in superstep "<< val <<"." << std::endl;
            return false;
        }
    }

    return true;
}

template <typename GraphT>
v_workw_t<Graph_t> BspScheduleRecomp<GraphT>::ComputeWorkCosts() const {
    assert(SatisfiesConstraints());

    std::vector<std::vector<cost_type>> stepProcWork(number_of_supersteps,
                                                     std::vector<cost_type>(instance->numberOfProcessors(), 0));

    for (vertex_idx node = 0; node < instance_->numberOfVertices(); node++) {
        for (const std::pair<unsigned, unsigned> &processor_superstep : node_to_processor_and_supertep_assignment[node]) {
            step_proc_work[processor_superstep.second][processor_superstep.first]
                += instance->getComputationalDag().vertex_work_weight(node);
        }
    }

    cost_type totalCosts = 0;
    for (unsigned step = 0; step < numberOfSupersteps_; step++) {
        cost_type maxWork = 0;

        for (unsigned proc = 0; proc < instance_->numberOfProcessors(); proc++) {
            if (max_work < step_proc_work[step][proc]) {
                maxWork = step_proc_work[step][proc];
            }
        }

        totalCosts += max_work;
    }

    return total_costs;
}

template <typename GraphT>
v_workw_t<Graph_t> BspScheduleRecomp<GraphT>::ComputeCosts() const {
    assert(SatisfiesConstraints());

    std::vector<std::vector<cost_type>> rec(number_of_supersteps, std::vector<cost_type>(instance->numberOfProcessors(), 0));
    std::vector<std::vector<cost_type>> send(number_of_supersteps, std::vector<cost_type>(instance->numberOfProcessors(), 0));

    for (auto const &[key, val] : commSchedule) {
        send[val][std::get<1>(key)] += instance->sendCosts(std::get<1>(key), std::get<2>(key))
                                       * instance->getComputationalDag().vertex_comm_weight(std::get<0>(key));
        rec[val][std::get<2>(key)] += instance->sendCosts(std::get<1>(key), std::get<2>(key))
                                      * instance->getComputationalDag().vertex_comm_weight(std::get<0>(key));
    }

    cost_type totalCosts = 0;
    for (unsigned step = 0; step < numberOfSupersteps_; step++) {
        cost_type maxComm = 0;

        for (unsigned proc = 0; proc < instance_->numberOfProcessors(); proc++) {
            if (max_comm < send[step][proc]) {
                maxComm = send[step][proc];
            }
            if (max_comm < rec[step][proc]) {
                maxComm = rec[step][proc];
            }
        }

        if (maxComm > 0) {
            totalCosts += instance_->synchronisationCosts() + max_comm * instance_->communicationCosts();
        }
    }

    total_costs += computeWorkCosts();

    return total_costs;
}

template <typename GraphT>
vertex_idx_t<Graph_t> BspScheduleRecomp<GraphT>::GetTotalAssignments() const {
    vertex_idx total = 0;
    for (vertex_idx node = 0; node < instance_->numberOfVertices(); ++node) {
        total += nodeToProcessorAndSupertepAssignment_[node].size();
    }
    return total;
}

template <typename GraphT>
void BspScheduleRecomp<GraphT>::MergeSupersteps() {
    std::vector<unsigned> newStepIdx(numberOfSupersteps_);
    std::vector<bool> commPhaseEmpty(numberOfSupersteps_, true);

    for (auto const &[key, val] : commSchedule) {
        comm_phase_empty[val] = false;
    }

    unsigned currentStepIdx = 0;
    for (unsigned step = 0; step < numberOfSupersteps_; ++step) {
        newStepIdx[step] = currentStepIdx;
        if (!commPhaseEmpty[step] || step == numberOfSupersteps_ - 1) {
            ++currentStepIdx;
        }
    }
    for (vertex_idx node = 0; node < instance_->numberOfVertices(); ++node) {
        std::vector<std::pair<unsigned, unsigned>> newAssignment;
        for (const std::pair<unsigned, unsigned> &entry : node_to_processor_and_supertep_assignment[node]) {
            new_assignment.emplace_back(entry.first, new_step_idx[entry.second]);
        }
        nodeToProcessorAndSupertepAssignment_[node] = newAssignment;
    }
    for (auto &key_step_pair : commSchedule) {
        auto &step = key_step_pair.second;
        step = new_step_idx[step];
    }

    numberOfSupersteps_ = currentStepIdx;
}

}    // namespace osp
