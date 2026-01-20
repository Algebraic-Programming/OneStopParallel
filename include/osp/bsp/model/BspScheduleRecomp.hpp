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
    using VertexIdx = VertexIdxT<GraphT>;
    using CostType = VWorkwT<GraphT>;

    using KeyTriple = std::tuple<VertexIdx, unsigned int, unsigned int>;

    static_assert(isComputationalDagV<GraphT>, "BspScheduleRecomp can only be used with computational DAGs.");
    static_assert(std::is_same_v<VWorkwT<GraphT>, VCommwT<GraphT>>,
                  "BspScheduleRecomp requires work and comm. weights to have the same type.");

  private:
    const BspInstance<GraphT> *instance_;

    unsigned int numberOfSupersteps_ = 0;

    std::vector<std::vector<std::pair<unsigned, unsigned>>> nodeToProcessorAndSupertepAssignment_;

    std::map<KeyTriple, unsigned> commSchedule_;

  public:
    BspScheduleRecomp() = default;

    BspScheduleRecomp(const BspInstance<GraphT> &inst) : instance_(&inst) {
        nodeToProcessorAndSupertepAssignment_.resize(inst.NumberOfVertices());
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
    virtual unsigned NumberOfSupersteps() const override { return numberOfSupersteps_; }

    void SetNumberOfSupersteps(unsigned numberOfSupersteps) { numberOfSupersteps_ = numberOfSupersteps; }

    std::vector<std::pair<unsigned, unsigned>> &Assignments(VertexIdx node) {
        return nodeToProcessorAndSupertepAssignment_[node];
    }

    const std::vector<std::pair<unsigned, unsigned>> &Assignments(VertexIdx node) const {
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
    const std::map<KeyTriple, unsigned int> &GetCommunicationSchedule() const { return commSchedule_; }

    std::map<KeyTriple, unsigned int> &GetCommunicationSchedule() { return commSchedule_; }

    virtual CostType ComputeWorkCosts() const override;

    virtual CostType ComputeCosts() const override;

    /**
     * @brief Returns true if the schedule is valid, i.e. if every time we compute a node, all its parents are already available
     * on the given processor, and every time we send a value, it is also already available on the given source processor.
     *
     * @return True if the schedule is valid, false otherwise.
     */
    bool SatisfiesConstraints() const;

    VertexIdx GetTotalAssignments() const;

    void MergeSupersteps();
    void CleanSchedule();
};

template <typename GraphT>
BspScheduleRecomp<GraphT>::BspScheduleRecomp(const BspScheduleCS<GraphT> &schedule) : instance_(&schedule.GetInstance()) {
    nodeToProcessorAndSupertepAssignment_.clear();
    nodeToProcessorAndSupertepAssignment_.resize(instance_->NumberOfVertices());
    numberOfSupersteps_ = schedule.NumberOfSupersteps();

    for (VertexIdx node = 0; node < instance_->NumberOfVertices(); ++node) {
        nodeToProcessorAndSupertepAssignment_[node].emplace_back(schedule.AssignedProcessor(node),
                                                                 schedule.AssignedSuperstep(node));
    }

    commSchedule_ = schedule.GetCommunicationSchedule();
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

    if (std::get<0>(key) >= instance_->NumberOfVertices()) {
        throw std::invalid_argument("Invalid Argument while adding communication schedule entry: node out of range.");
    }

    if (std::get<1>(key) >= instance_->NumberOfProcessors()) {
        throw std::invalid_argument("Invalid Argument while adding communication schedule entry: from processor out of range.");
    }

    if (std::get<2>(key) >= instance_->NumberOfProcessors()) {
        throw std::invalid_argument("Invalid Argument while adding communication schedule entry: to processor out of range.");
    }

    commSchedule_[key] = step;
}

template <typename GraphT>
bool BspScheduleRecomp<GraphT>::SatisfiesConstraints() const {
    // find first availability

    std::vector<std::vector<unsigned>> nodeFirstAvailableOnProc(
        instance_->NumberOfVertices(),
        std::vector<unsigned>(instance_->NumberOfProcessors(), std::numeric_limits<unsigned>::max()));

    for (VertexIdx node = 0; node < instance_->NumberOfVertices(); ++node) {
        for (const std::pair<unsigned, unsigned> &computeStep : nodeToProcessorAndSupertepAssignment_[node]) {
            nodeFirstAvailableOnProc[node][computeStep.first]
                = std::min(nodeFirstAvailableOnProc[node][computeStep.first], computeStep.second);
        }
    }

    for (auto const &[key, val] : commSchedule_) {
        const VertexIdx &node = std::get<0>(key);
        const unsigned &toProc = std::get<2>(key);

        nodeFirstAvailableOnProc[node][toProc] = std::min(nodeFirstAvailableOnProc[node][toProc], val + 1);
    }

    // check validity

    for (VertexIdx node = 0; node < instance_->NumberOfVertices(); ++node) {
        for (VertexIdx pred : instance_->GetComputationalDag().Parents(node)) {
            for (const std::pair<unsigned, unsigned> &computeStep : nodeToProcessorAndSupertepAssignment_[node]) {
                if (nodeFirstAvailableOnProc[pred][computeStep.first] > computeStep.second) {
                    // std::cout << "Not a valid schedule: parent " << pred << " of node "<< node <<
                    //" not yet available on processor " << computeStep.first << " in superstep "<< computeStep.second <<"." << std::endl;
                    return false;
                }
            }
        }
    }

    for (auto const &[key, val] : commSchedule_) {
        const VertexIdx &node = std::get<0>(key);
        const unsigned &fromProc = std::get<1>(key);

        if (nodeFirstAvailableOnProc[node][fromProc] > val) {
            // std::cout << "Not a valid schedule: node " << node << " not yet available for sending from processor "
            // << from_proc << " in superstep "<< val <<"." << std::endl;
            return false;
        }
    }

    return true;
}

template <typename GraphT>
VWorkwT<GraphT> BspScheduleRecomp<GraphT>::ComputeWorkCosts() const {
    assert(SatisfiesConstraints());

    std::vector<std::vector<CostType>> stepProcWork(numberOfSupersteps_, std::vector<CostType>(instance_->NumberOfProcessors(), 0));

    for (VertexIdx node = 0; node < instance_->NumberOfVertices(); node++) {
        for (const std::pair<unsigned, unsigned> &processorSuperstep : nodeToProcessorAndSupertepAssignment_[node]) {
            stepProcWork[processorSuperstep.second][processorSuperstep.first]
                += instance_->GetComputationalDag().VertexWorkWeight(node);
        }
    }

    CostType totalCosts = 0;
    for (unsigned step = 0; step < numberOfSupersteps_; step++) {
        CostType maxWork = 0;

        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
            if (maxWork < stepProcWork[step][proc]) {
                maxWork = stepProcWork[step][proc];
            }
        }

        totalCosts += maxWork;
    }

    return totalCosts;
}

template <typename GraphT>
VWorkwT<GraphT> BspScheduleRecomp<GraphT>::ComputeCosts() const {
    assert(SatisfiesConstraints());

    std::vector<std::vector<CostType>> rec(numberOfSupersteps_, std::vector<CostType>(instance_->NumberOfProcessors(), 0));
    std::vector<std::vector<CostType>> send(numberOfSupersteps_, std::vector<CostType>(instance_->NumberOfProcessors(), 0));

    for (auto const &[key, val] : commSchedule_) {
        send[val][std::get<1>(key)] += instance_->SendCosts(std::get<1>(key), std::get<2>(key))
                                       * instance_->GetComputationalDag().VertexCommWeight(std::get<0>(key));
        rec[val][std::get<2>(key)] += instance_->SendCosts(std::get<1>(key), std::get<2>(key))
                                      * instance_->GetComputationalDag().VertexCommWeight(std::get<0>(key));
    }

    CostType totalCosts = 0;
    for (unsigned step = 0; step < numberOfSupersteps_; step++) {
        CostType maxComm = 0;

        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
            if (maxComm < send[step][proc]) {
                maxComm = send[step][proc];
            }
            if (maxComm < rec[step][proc]) {
                maxComm = rec[step][proc];
            }
        }

        if (maxComm > 0) {
            totalCosts += instance_->SynchronisationCosts() + maxComm * instance_->CommunicationCosts();
        }
    }

    totalCosts += ComputeWorkCosts();

    return totalCosts;
}

template <typename GraphT>
VertexIdxT<GraphT> BspScheduleRecomp<GraphT>::GetTotalAssignments() const {
    VertexIdx total = 0;
    for (VertexIdx node = 0; node < instance_->NumberOfVertices(); ++node) {
        total += nodeToProcessorAndSupertepAssignment_[node].size();
    }
    return total;
}

template <typename GraphT>
void BspScheduleRecomp<GraphT>::MergeSupersteps() {
    std::vector<unsigned> newStepIdx(numberOfSupersteps_);
    std::vector<bool> commPhaseEmpty(numberOfSupersteps_, true);

    for (auto const &[key, val] : commSchedule_) {
        commPhaseEmpty[val] = false;
    }

    unsigned currentStepIdx = 0;
    for (unsigned step = 0; step < numberOfSupersteps_; ++step) {
        newStepIdx[step] = currentStepIdx;
        if (!commPhaseEmpty[step] || step == numberOfSupersteps_ - 1) {
            ++currentStepIdx;
        }
    }
    for (VertexIdx node = 0; node < instance_->NumberOfVertices(); ++node) {
        std::vector<std::pair<unsigned, unsigned>> newAssignment;
        for (const std::pair<unsigned, unsigned> &entry : nodeToProcessorAndSupertepAssignment_[node]) {
            newAssignment.emplace_back(entry.first, newStepIdx[entry.second]);
        }
        nodeToProcessorAndSupertepAssignment_[node] = newAssignment;
    }
    for (auto &keyStepPair : commSchedule_) {
        auto &step = keyStepPair.second;
        step = newStepIdx[step];
    }

    numberOfSupersteps_ = currentStepIdx;
}

// remove unneeded comm. schedule entries - these can happen in several algorithms
template<typename Graph_t>
void BspScheduleRecomp<Graph_t>::CleanSchedule()
{
    // I. Data that is already present before it arrives
    std::vector<std::vector<std::multiset<unsigned>>> arrivesAt(instance_->NumberOfVertices(),
                                                                    std::vector<std::multiset<unsigned>>(instance_->NumberOfProcessors()));
    for (const auto &node : instance_->GetComputationalDag().Vertices()) {
        for (const auto &procAndStep : nodeToProcessorAndSupertepAssignment_[node]) {
            arrivesAt[node][procAndStep.first].insert(procAndStep.second);
        }
    }

    for (auto const &[key, val] : commSchedule_) {
        arrivesAt[std::get<0>(key)][std::get<2>(key)].insert(val);
    }

    // - computation steps
    for (const auto &node : instance_->GetComputationalDag().Vertices()) {
        for (unsigned index = 0; index < nodeToProcessorAndSupertepAssignment_[node].size(); ) {
            const auto &procAndStep = nodeToProcessorAndSupertepAssignment_[node][index];
            if(*arrivesAt[node][procAndStep.first].begin() < procAndStep.second) {
                nodeToProcessorAndSupertepAssignment_[node][index] = nodeToProcessorAndSupertepAssignment_[node].back();
                nodeToProcessorAndSupertepAssignment_[node].pop_back();
            } else {
                ++index;
            }
        }
    }

    // - communication steps
    std::vector<KeyTriple> toErase;
    for (auto const &[key, val] : commSchedule_) {
        auto itr = arrivesAt[std::get<0>(key)][std::get<2>(key)].begin();
        if (*itr < val) {
            toErase.push_back(key);
        } else if (*itr == val && ++itr != arrivesAt[std::get<0>(key)][std::get<2>(key)].end() && *itr == val) {
            toErase.push_back(key);
            arrivesAt[std::get<0>(key)][std::get<2>(key)].erase(itr);
        }
    }

    for (const KeyTriple &key : toErase) {
        commSchedule_.erase(key);
    }

    // II. Data that is not used after being computed/sent
    std::vector<std::vector<std::multiset<unsigned>>> usedAt(instance_->NumberOfVertices(),
                                                                std::vector<std::multiset<unsigned>>(instance_->NumberOfProcessors()));
    for (const auto &node : instance_->GetComputationalDag().Vertices()) {
        for (const auto &child : instance_->GetComputationalDag().Children(node)) {
            for (const auto &procAndStep : nodeToProcessorAndSupertepAssignment_[child]) {
                usedAt[node][procAndStep.first].insert(procAndStep.second);
            }
        }
    }

    for (auto const &[key, val] : commSchedule_) {
        usedAt[std::get<0>(key)][std::get<1>(key)].insert(val);
    }

    // - computation steps
    for (const auto &node : instance_->GetComputationalDag().Vertices()) {
        for (unsigned index = 0; index < nodeToProcessorAndSupertepAssignment_[node].size(); ) {
            const auto &procAndStep = nodeToProcessorAndSupertepAssignment_[node][index];
            if ((usedAt[node][procAndStep.first].empty() || *usedAt[node][procAndStep.first].rbegin() < procAndStep.second)
                && index > 0)
            {
                nodeToProcessorAndSupertepAssignment_[node][index] = nodeToProcessorAndSupertepAssignment_[node].back();
                nodeToProcessorAndSupertepAssignment_[node].pop_back();
            } else {
                ++index;
            }
        }
    }

    // - communication steps (need to visit cs entries in reverse superstep order here)
    std::vector<std::vector<KeyTriple>> entries(numberOfSupersteps_);
    for (auto const &[key, val] : commSchedule_) {
        entries[val].push_back(key);
    }

    toErase.clear();
    for (unsigned step = numberOfSupersteps_ - 1; step < numberOfSupersteps_; --step) {
        for (const KeyTriple &key : entries[step]) {
            if (usedAt[std::get<0>(key)][std::get<2>(key)].empty()
                || *usedAt[std::get<0>(key)][std::get<2>(key)].rbegin() <= step) {
                toErase.push_back(key);
                auto itr = usedAt[std::get<0>(key)][std::get<1>(key)].find(step);
                usedAt[std::get<0>(key)][std::get<1>(key)].erase(itr);
            }
        }
    }

    for (const KeyTriple &key : toErase) {
        commSchedule_.erase(key);
    }
}

}    // namespace osp
