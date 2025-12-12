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

#include <algorithm>
#include <iostream>
#include <list>
#include <map>
#include <stdexcept>
#include <vector>

#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/concepts/computational_dag_concept.hpp"

namespace osp {

typedef std::tuple<unsigned int, unsigned int, unsigned int> KeyTriple;

/**
 * @class PebblingSchedule
 * @brief Represents a schedule for 2-level MultiBSP model with memory constraints.
 *
 * Alternatively, can be understood as the generalization of multiprocessor red-blue pebble game with node weights.
 * The synchronous interpretation is essentially a 2-level Multi-BSP, while the asynchronous interpretation is
 * closer to makespan metrics in classical schedules.
 *
 * Besides basic utility such as validity check, cost computation and conversion from a Bsp Schedule, it also allows
 * conversions to/from several MultiProcessorPebbling ILP methods that address this problem.
 *
 * Works with a `BspInstance` object, which represents the instance of the scheduling problem being solved.
 *
 * @see BspInstance
 */
template <typename GraphT>
class PebblingSchedule {
    static_assert(IsComputationalDagV<Graph_t>, "PebblingSchedule can only be used with computational DAGs.");

  private:
    using vertex_idx = vertex_idx_t<Graph_t>;
    using cost_type = VWorkwT<Graph_t>;
    using memweight_type = VMemwT<Graph_t>;

    static_assert(std::is_same_v<VWorkwT<Graph_t>, VCommwT<Graph_t>>,
                  "PebblingSchedule requires work and comm. weights to have the same type.");

    const BspInstance<GraphT> *instance_;

    unsigned int numberOfSupersteps_;

    bool needToLoadInputs_ = true;

    struct ComputeStep {
        vertex_idx node_;
        std::vector<vertex_idx> nodesEvictedAfter_;

        ComputeStep() {}

        ComputeStep(vertex_idx node) : node(node_) {}

        ComputeStep(vertex_idx node, const std::vector<vertex_idx> &evicted) : node(node_), nodes_evicted_after(evicted_) {}
    };

    // executed nodes in order in a computation phase, for processor p and superstep s
    std::vector<std::vector<std::vector<ComputeStep>>> computeStepsForProcSuperstep_;

    // nodes evicted from cache in a given superstep's comm phase
    std::vector<std::vector<std::vector<vertex_idx>>> nodesEvictedInComm_;

    // nodes sent down to processor p in superstep s
    std::vector<std::vector<std::vector<vertex_idx>>> nodesSentDown_;

    // nodes sent up from processor p in superstep s
    std::vector<std::vector<std::vector<vertex_idx>>> nodesSentUp_;

    // set of nodes that need to have blue pebble at end, sinks by default, and
    // set of nodes on each processor that begin with red pebble, nothing by default
    // (TODO: maybe move to problem definition classes instead?)
    std::set<vertex_idx> needsBlueAtEnd_;
    std::vector<std::set<vertex_idx>> hasRedInBeginning_;

    // nodes that are from a previous part of a larger DAG, handled differently in conversion
    std::set<vertex_idx> externalSources_;

  public:
    enum CacheEvictionStrategy { FORESIGHT, LEAST_RECENTLY_USED, LARGEST_ID };

    /**
     * @brief Default constructor for the PebblingSchedule class.
     */
    PebblingSchedule() : instance_(nullptr), numberOfSupersteps_(0) {}

    PebblingSchedule(const BspInstance<GraphT> &inst) : instance_(&inst) {
        BspSchedule<GraphT> schedule(
            inst, std::vector<unsigned int>(inst.NumberOfVertices(), 0), std::vector<unsigned int>(inst.NumberOfVertices(), 0));
        ConvertFromBsp(schedule);
    }

    PebblingSchedule(const BspInstance<GraphT> &inst,
                     const std::vector<unsigned> &processorAssignment,
                     const std::vector<unsigned> &superstepAssignment)
        : instance_(&inst) {
        BspSchedule<GraphT> schedule(inst, processorAssignment, superstepAssignment);
        ConvertFromBsp(schedule);
    }

    PebblingSchedule(const BspInstance<GraphT> &inst,
                     const std::vector<std::vector<std::vector<vertex_idx>>> &computeSteps,
                     const std::vector<std::vector<std::vector<std::vector<vertex_idx>>>> &nodesEvictedAfterCompute,
                     const std::vector<std::vector<std::vector<vertex_idx>>> &nodesSentUp,
                     const std::vector<std::vector<std::vector<vertex_idx>>> &nodesSentDown,
                     const std::vector<std::vector<std::vector<vertex_idx>>> &nodesEvictedInComm,
                     const std::set<vertex_idx> &needsBlueAtEnd = std::set<vertex_idx>(),
                     const std::vector<std::set<vertex_idx>> &hasRedInBeginning = std::vector<std::set<vertex_idx>>(),
                     const bool needToLoadInputs = false)
        : instance_(&inst),
          numberOfSupersteps_(0),
          needToLoadInputs_(needToLoadInputs),
          nodes_evicted_in_comm(nodes_evicted_in_comm_),
          nodes_sent_down(nodes_sent_down_),
          nodes_sent_up(nodes_sent_up_),
          needs_blue_at_end(needs_blue_at_end_),
          has_red_in_beginning(has_red_in_beginning_) {
        computeStepsForProcSuperstep_.resize(compute_steps.size(), std::vector<std::vector<ComputeStep>>(compute_steps[0].size()));
        for (unsigned proc = 0; proc < computeSteps.size(); ++proc) {
            numberOfSupersteps_ = std::max(numberOfSupersteps_, static_cast<unsigned>(compute_steps[proc].size()));
            for (unsigned supstep = 0; supstep < static_cast<unsigned>(compute_steps[proc].size()); ++supstep) {
                for (unsigned stepIndex = 0; stepIndex < static_cast<unsigned>(compute_steps[proc][supstep].size()); ++stepIndex) {
                    computeStepsForProcSuperstep_[proc][supstep].emplace_back(
                        compute_steps[proc][supstep][stepIndex], nodes_evicted_after_compute[proc][supstep][stepIndex]);
                }
            }
        }
    }

    PebblingSchedule(const BspSchedule<GraphT> &schedule, CacheEvictionStrategy evictRule = LARGEST_ID)
        : instance_(&schedule.GetInstance()) {
        ConvertFromBsp(schedule, evictRule);
    }

    virtual ~PebblingSchedule() = default;

    // cost computation
    cost_type ComputeCost() const;
    cost_type ComputeAsynchronousCost() const;

    // remove unnecessary steps (e.g. from ILP solution)
    void CleanSchedule();

    // convert from unconstrained schedule
    void ConvertFromBsp(const BspSchedule<GraphT> &schedule, CacheEvictionStrategy evictRule = LARGEST_ID);

    // auxiliary for conversion
    std::vector<std::vector<std::vector<vertex_idx>>> ComputeTopOrdersDfs(const BspSchedule<GraphT> &schedule) const;
    static bool HasValidSolution(const BspInstance<GraphT> &instance,
                                 const std::set<vertex_idx> &externalSources = std::set<vertex_idx>());
    void SplitSupersteps(const BspSchedule<GraphT> &schedule);
    void SetMemoryMovement(CacheEvictionStrategy evictRule = LARGEST_ID);

    // delete current communication schedule, and switch to foresight policy instead
    void ResetToForesight();

    // other basic operations
    bool IsValid() const;
    static std::vector<memweight_type> MinimumMemoryRequiredPerNodeType(const BspInstance<GraphT> &instance,
                                                                        const std::set<vertex_idx> &externalSources
                                                                        = std::set<vertex_idx>());

    // expand a MemSchedule from a coarsened DAG to the original DAG
    PebblingSchedule<GraphT> ExpandMemSchedule(const BspInstance<GraphT> &originalInstance,
                                               const std::vector<vertex_idx> mappingToCoarse) const;

    // convert to BSP (ignores vertical I/O and recomputation)
    BspSchedule<GraphT> ConvertToBsp() const;

    /**
     * @brief Returns a reference to the BspInstance for the schedule.
     *
     * @return A reference to the BspInstance for the schedule.
     */
    const BspInstance<GraphT> &GetInstance() const { return *instance_; }

    /**
     * @brief Returns the number of supersteps in the schedule.
     *
     * @return The number of supersteps in the schedule.
     */
    unsigned NumberOfSupersteps() const { return numberOfSupersteps_; }

    void UpdateNumberOfSupersteps(unsigned newNumberOfSupersteps);

    inline bool NeedsToLoadInputs() const { return needToLoadInputs_; }

    inline void SetNeedToLoadInputs(const bool loadInputs) { needToLoadInputs_ = loadInputs; }

    void GetDataForMultiprocessorPebbling(std::vector<std::vector<std::vector<vertex_idx>>> &computeSteps,
                                          std::vector<std::vector<std::vector<vertex_idx>>> &sendUpSteps,
                                          std::vector<std::vector<std::vector<vertex_idx>>> &sendDownSteps,
                                          std::vector<std::vector<std::vector<vertex_idx>>> &nodesEvictedAfterStep) const;

    // utility for partial ILPs
    std::vector<std::set<vertex_idx>> GetMemContentAtEnd() const;
    void RemoveEvictStepsFromEnd();

    void CreateFromPartialPebblings(const BspInstance<GraphT> &bspInstance,
                                    const std::vector<PebblingSchedule<GraphT>> &pebblings,
                                    const std::vector<std::set<unsigned>> &processorsToParts,
                                    const std::vector<std::map<vertex_idx, vertex_idx>> &originalNodeId,
                                    const std::vector<std::map<unsigned, unsigned>> &originalProcId,
                                    const std::vector<std::vector<std::set<vertex_idx>>> &hasRedsInBeginning);

    // auxiliary function to remove some unnecessary communications after assembling from partial pebblings
    void FixForceEvicts(const std::vector<std::tuple<vertex_idx, unsigned, unsigned>> forceEvictNodeProcStep);

    // auxiliary after partial pebblings: try to merge supersteps
    void TryToMergeSupersteps();

    const std::vector<ComputeStep> &GetComputeStepsForProcSuperstep(unsigned proc, unsigned supstep) const {
        return computeStepsForProcSuperstep_[proc][supstep];
    }

    const std::vector<vertex_idx> &GetNodesEvictedInComm(unsigned proc, unsigned supstep) const {
        return nodes_evicted_in_comm[proc][supstep];
    }

    const std::vector<vertex_idx> &GetNodesSentDown(unsigned proc, unsigned supstep) const {
        return nodes_sent_down[proc][supstep];
    }

    const std::vector<vertex_idx> &GetNodesSentUp(unsigned proc, unsigned supstep) const { return nodes_sent_up[proc][supstep]; }

    void SetNeedsBlueAtEnd(const std::set<vertex_idx> &nodes) { needs_blue_at_end = nodes_; }

    void SetExternalSources(const std::set<vertex_idx> &nodes) { external_sources = nodes_; }

    void SetHasRedInBeginning(const std::vector<std::set<vertex_idx>> &nodes) { has_red_in_beginning = nodes_; }
};

template <typename GraphT>
void PebblingSchedule<GraphT>::UpdateNumberOfSupersteps(unsigned newNumberOfSupersteps) {
    numberOfSupersteps_ = newNumberOfSupersteps;

    computeStepsForProcSuperstep_.clear();
    computeStepsForProcSuperstep_.resize(instance_->NumberOfProcessors(),
                                         std::vector<std::vector<ComputeStep>>(numberOfSupersteps_));

    nodes_evicted_in_comm.clear();
    nodes_evicted_in_comm.resize(instance->NumberOfProcessors(), std::vector<std::vector<vertex_idx>>(number_of_supersteps));

    nodes_sent_down.clear();
    nodes_sent_down.resize(instance->NumberOfProcessors(), std::vector<std::vector<vertex_idx>>(number_of_supersteps));

    nodes_sent_up.clear();
    nodes_sent_up.resize(instance->NumberOfProcessors(), std::vector<std::vector<vertex_idx>>(number_of_supersteps));
}

template <typename GraphT>
VWorkwT<Graph_t> PebblingSchedule<GraphT>::ComputeCost() const {
    cost_type totalCosts = 0;
    for (unsigned step = 0; step < numberOfSupersteps_; ++step) {
        // compute phase
        cost_type maxWork = std::numeric_limits<cost_type>::min();
        for (unsigned proc = 0; proc < instance_->GetArchitecture().NumberOfProcessors(); ++proc) {
            cost_type work = 0;
            for (const auto &computeStep : computeStepsForProcSuperstep_[proc][step]) {
                work += instance_->GetComputationalDag().VertexWorkWeight(computeStep.node);
            }

            if (work > max_work) {
                maxWork = work;
            }
        }
        totalCosts += max_work;

        // communication phase
        cost_type maxSendUp = std::numeric_limits<cost_type>::min();
        for (unsigned proc = 0; proc < instance_->GetArchitecture().NumberOfProcessors(); ++proc) {
            cost_type sendUp = 0;
            for (vertex_idx node : nodes_sent_up[proc][step]) {
                send_up
                    += instance->GetComputationalDag().VertexCommWeight(node) * instance->GetArchitecture().CommunicationCosts();
            }

            if (sendUp > max_send_up) {
                maxSendUp = send_up;
            }
        }
        totalCosts += max_send_up;

        totalCosts += static_cast<cost_type>(instance_->GetArchitecture().SynchronisationCosts());

        cost_type maxSendDown = std::numeric_limits<cost_type>::min();
        for (unsigned proc = 0; proc < instance_->GetArchitecture().NumberOfProcessors(); ++proc) {
            cost_type sendDown = 0;
            for (vertex_idx node : nodes_sent_down[proc][step]) {
                send_down
                    += instance->GetComputationalDag().VertexCommWeight(node) * instance->GetArchitecture().CommunicationCosts();
            }

            if (sendDown > max_send_down) {
                maxSendDown = send_down;
            }
        }
        totalCosts += max_send_down;
    }

    return total_costs;
}

template <typename GraphT>
VWorkwT<Graph_t> PebblingSchedule<GraphT>::ComputeAsynchronousCost() const {
    std::vector<cost_type> currentTimeAtProcessor(instance_->GetArchitecture().NumberOfProcessors(), 0);
    std::vector<cost_type> timeWhenNodeGetsBlue(instance->GetComputationalDag().NumVertices(),
                                                std::numeric_limits<cost_type>::max());
    if (needToLoadInputs_) {
        for (vertex_idx node = 0; node < instance_->NumberOfVertices(); ++node) {
            if (instance_->GetComputationalDag().in_degree(node) == 0) {
                timeWhenNodeGetsBlue[node] = 0;
            }
        }
    }

    for (unsigned step = 0; step < numberOfSupersteps_; ++step) {
        // compute phase
        for (unsigned proc = 0; proc < instance_->GetArchitecture().NumberOfProcessors(); ++proc) {
            for (const auto &computeStep : computeStepsForProcSuperstep_[proc][step]) {
                currentTimeAtProcessor[proc] += instance_->GetComputationalDag().VertexWorkWeight(computeStep.node);
            }
        }

        // communication phase - send up
        for (unsigned proc = 0; proc < instance_->GetArchitecture().NumberOfProcessors(); ++proc) {
            for (vertex_idx node : nodes_sent_up[proc][step]) {
                current_time_at_processor[proc]
                    += instance->GetComputationalDag().VertexCommWeight(node) * instance->GetArchitecture().CommunicationCosts();
                if (time_when_node_gets_blue[node] > current_time_at_processor[proc]) {
                    time_when_node_gets_blue[node] = current_time_at_processor[proc];
                }
            }
        }

        // communication phase - send down
        for (unsigned proc = 0; proc < instance_->GetArchitecture().NumberOfProcessors(); ++proc) {
            for (vertex_idx node : nodes_sent_down[proc][step]) {
                if (current_time_at_processor[proc] < time_when_node_gets_blue[node]) {
                    current_time_at_processor[proc] = time_when_node_gets_blue[node];
                }
                current_time_at_processor[proc]
                    += instance->GetComputationalDag().VertexCommWeight(node) * instance->GetArchitecture().CommunicationCosts();
            }
        }
    }

    cost_type makespan = 0;
    for (unsigned proc = 0; proc < instance_->GetArchitecture().NumberOfProcessors(); ++proc) {
        if (currentTimeAtProcessor[proc] > makespan) {
            makespan = current_time_at_processor[proc];
        }
    }

    return makespan;
}

template <typename GraphT>
void PebblingSchedule<GraphT>::CleanSchedule() {
    if (!IsValid()) {
        return;
    }

    // NOTE - this function removes unnecessary steps in most cases, but not all (some require e.g. multiple iterations)

    std::vector<std::vector<std::deque<bool>>> needed(instance_->NumberOfVertices(),
                                                      std::vector<std::deque<bool>>(instance_->NumberOfProcessors()));
    std::vector<std::vector<bool>> keepFalse(instance_->NumberOfVertices(),
                                             std::vector<bool>(instance_->NumberOfProcessors(), false));
    std::vector<std::vector<bool>> hasRedAfterCleaning(instance_->NumberOfVertices(),
                                                       std::vector<bool>(instance_->NumberOfProcessors(), false));

    std::vector<bool> everNeededAsBlue(instance_->NumberOfVertices(), false);
    if (needs_blue_at_end.empty()) {
        for (vertex_idx node = 0; node < instance_->NumberOfVertices(); ++node) {
            if (instance_->GetComputationalDag().OutDegree(node) == 0) {
                everNeededAsBlue[node] = true;
            }
        }
    } else {
        for (vertex_idx node : needs_blue_at_end) {
            ever_needed_as_blue[node] = true;
        }
    }

    for (unsigned step = 0; step < numberOfSupersteps_; ++step) {
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
            for (vertex_idx node : nodes_sent_down[proc][step]) {
                ever_needed_as_blue[node] = true;
            }
        }
    }

    if (!has_red_in_beginning.empty()) {
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
            for (vertex_idx node : has_red_in_beginning[proc]) {
                has_red_after_cleaning[node][proc] = true;
            }
        }
    }

    for (unsigned step = 0; step < numberOfSupersteps_; ++step) {
        // compute phase
        for (unsigned proc = 0; proc < instance_->GetArchitecture().NumberOfProcessors(); ++proc) {
            for (const auto &computeStep : computeStepsForProcSuperstep_[proc][step]) {
                vertex_idx node = computeStep.node;
                needed[node][proc].emplace_back(false);
                keepFalse[node][proc] = hasRedAfterCleaning[node][proc];
                for (vertex_idx pred : instance->GetComputationalDag().Parents(node)) {
                    has_red_after_cleaning[pred][proc] = true;
                    if (!keep_false[pred][proc]) {
                        needed[pred][proc].back() = true;
                    }
                }
                for (vertex_idx to_evict : computeStep.nodes_evicted_after) {
                    has_red_after_cleaning[to_evict][proc] = false;
                }
            }
        }

        // send up phase
        for (unsigned proc = 0; proc < instance_->GetArchitecture().NumberOfProcessors(); ++proc) {
            for (vertex_idx node : nodes_sent_up[proc][step]) {
                if (ever_needed_as_blue[node]) {
                    has_red_after_cleaning[node][proc] = true;
                    if (!keep_false[node][proc]) {
                        needed[node][proc].back() = true;
                    }
                }
            }
        }

        for (unsigned proc = 0; proc < instance_->GetArchitecture().NumberOfProcessors(); ++proc) {
            for (vertex_idx node : nodes_evicted_in_comm[proc][step]) {
                has_red_after_cleaning[node][proc] = false;
            }
        }

        // send down phase
        for (unsigned proc = 0; proc < instance_->GetArchitecture().NumberOfProcessors(); ++proc) {
            for (vertex_idx node : nodes_sent_down[proc][step]) {
                needed[node][proc].emplace_back(false);
                keep_false[node][proc] = has_red_after_cleaning[node][proc];
            }
        }
    }

    std::vector<std::vector<std::vector<ComputeStep>>> newComputeStepsForProcSuperstep(
        instance_->NumberOfProcessors(), std::vector<std::vector<ComputeStep>>(numberOfSupersteps_));
    std::vector<std::vector<std::vector<vertex_idx>>> newNodesEvictedInComm(
        instance->NumberOfProcessors(), std::vector<std::vector<vertex_idx>>(number_of_supersteps));
    std::vector<std::vector<std::vector<vertex_idx>>> newNodesSentDown(
        instance->NumberOfProcessors(), std::vector<std::vector<vertex_idx>>(number_of_supersteps));
    std::vector<std::vector<std::vector<vertex_idx>>> newNodesSentUp(instance->NumberOfProcessors(),
                                                                     std::vector<std::vector<vertex_idx>>(number_of_supersteps));

    std::vector<std::vector<bool>> hasRed(instance_->NumberOfVertices(), std::vector<bool>(instance_->NumberOfProcessors(), false));
    if (!has_red_in_beginning.empty()) {
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
            for (vertex_idx node : has_red_in_beginning[proc]) {
                has_red[node][proc] = true;
            }
        }
    }

    std::vector<bool> hasBlue(instance_->NumberOfVertices());
    std::vector<cost_type> timeWhenNodeGetsBlue(instance->GetComputationalDag().NumVertices(),
                                                std::numeric_limits<cost_type>::max());
    if (needToLoadInputs_) {
        for (vertex_idx node = 0; node < instance_->NumberOfVertices(); ++node) {
            if (instance_->GetComputationalDag().in_degree(node) == 0) {
                hasBlue[node] = true;
                timeWhenNodeGetsBlue[node] = 0;
            }
        }
    }

    std::vector<cost_type> currentTimeAtProcessor(instance_->GetArchitecture().NumberOfProcessors(), 0);

    for (unsigned superstep = 0; superstep < numberOfSupersteps_; ++superstep) {
        // compute phase
        for (unsigned proc = 0; proc < instance_->GetArchitecture().NumberOfProcessors(); ++proc) {
            std::vector<bool> stepRemains(computeStepsForProcSuperstep_[proc][superstep].size(), false);
            std::vector<std::vector<vertex_idx>> newEvictAfter(computeStepsForProcSuperstep_[proc][superstep].size());

            unsigned newStepIndex = 0;
            for (unsigned stepIndex = 0; stepIndex < computeStepsForProcSuperstep_[proc][superstep].size(); ++stepIndex) {
                vertex_idx node = computeStepsForProcSuperstep_[proc][superstep][stepIndex].node;

                if (needed[node][proc].front()) {
                    new_compute_steps_for_proc_superstep[proc][superstep].emplace_back(node, std::vector<vertex_idx>());
                    stepRemains[stepIndex] = true;
                    hasRed[node][proc] = true;
                    ++newStepIndex;
                    currentTimeAtProcessor[proc] += instance_->GetComputationalDag().VertexWorkWeight(node);
                }

                needed[node][proc].pop_front();

                for (vertex_idx to_evict : compute_steps_for_proc_superstep[proc][superstep][stepIndex].nodes_evicted_after) {
                    if (has_red[to_evict][proc]) {
                        new_evict_after[stepIndex].push_back(to_evict);
                    }
                    has_red[to_evict][proc] = false;
                }
            }

            // go backwards to fix cache eviction steps
            std::vector<vertex_idx> toEvict;
            for (size_t stepIndex = computeStepsForProcSuperstep_[proc][superstep].size() - 1;
                 stepIndex < computeStepsForProcSuperstep_[proc][superstep].size();
                 --stepIndex) {
                for (vertex_idx node : new_evict_after[stepIndex]) {
                    to_evict.push_back(node);
                }

                if (stepRemains[stepIndex]) {
                    newComputeStepsForProcSuperstep[proc][superstep][newStepIndex - 1].nodes_evicted_after = to_evict;
                    toEvict.clear();
                    --newStepIndex;
                }
            }
            if (!to_evict.empty() && superstep >= 1) {
                for (vertex_idx node : to_evict) {
                    auto itr = std::find(
                        new_nodes_sent_down[proc][superstep - 1].begin(), new_nodes_sent_down[proc][superstep - 1].end(), node);
                    if (itr == new_nodes_sent_down[proc][superstep - 1].end()) {
                        new_nodes_evicted_in_comm[proc][superstep - 1].push_back(node);
                    } else {
                        new_nodes_sent_down[proc][superstep - 1].erase(itr);
                    }
                }
            }
        }
        for (unsigned proc = 0; proc < instance_->GetArchitecture().NumberOfProcessors(); ++proc) {
            // send up phase
            for (vertex_idx node : nodes_sent_up[proc][superstep]) {
                if (!ever_needed_as_blue[node]) {
                    continue;
                }

                cost_type new_time_at_processor
                    = current_time_at_processor[proc]
                      + instance->GetComputationalDag().VertexCommWeight(node) * instance->GetArchitecture().CommunicationCosts();

                // only copy send up step if it is not obsolete in at least one of the two cases (sync or async schedule)
                if (!has_blue[node] || new_time_at_processor < time_when_node_gets_blue[node]) {
                    new_nodes_sent_up[proc][superstep].push_back(node);
                    has_blue[node] = true;
                    current_time_at_processor[proc] = new_time_at_processor;
                    if (time_when_node_gets_blue[node] > new_time_at_processor) {
                        time_when_node_gets_blue[node] = new_time_at_processor;
                    }
                }
            }
        }

        // comm phase evict
        for (unsigned proc = 0; proc < instance_->GetArchitecture().NumberOfProcessors(); ++proc) {
            for (vertex_idx node : nodes_evicted_in_comm[proc][superstep]) {
                if (has_red[node][proc]) {
                    new_nodes_evicted_in_comm[proc][superstep].push_back(node);
                    has_red[node][proc] = false;
                }
            }
        }

        for (unsigned proc = 0; proc < instance_->GetArchitecture().NumberOfProcessors(); ++proc) {
            // send down phase
            for (vertex_idx node : nodes_sent_down[proc][superstep]) {
                if (needed[node][proc].front()) {
                    new_nodes_sent_down[proc][superstep].push_back(node);
                    has_red[node][proc] = true;
                    if (current_time_at_processor[proc] < time_when_node_gets_blue[node]) {
                        current_time_at_processor[proc] = time_when_node_gets_blue[node];
                    }
                    current_time_at_processor[proc] += instance->GetComputationalDag().VertexCommWeight(node)
                                                       * instance->GetArchitecture().CommunicationCosts();
                }
                needed[node][proc].pop_front();
            }
        }
    }

    computeStepsForProcSuperstep_ = newComputeStepsForProcSuperstep;
    nodes_evicted_in_comm = new_nodes_evicted_in_comm;
    nodes_sent_down = new_nodes_sent_down;
    nodes_sent_up = new_nodes_sent_up;
}

template <typename GraphT>
void PebblingSchedule<GraphT>::ConvertFromBsp(const BspSchedule<GraphT> &schedule, CacheEvictionStrategy evictRule) {
    instance_ = &schedule.GetInstance();

    // check if conversion possible at all
    if (!hasValidSolution(schedule.GetInstance(), external_sources)) {
        std::cout << "Conversion failed." << std::endl;
        return;
    }

    // split supersteps
    SplitSupersteps(schedule);

    // track memory
    SetMemoryMovement(evictRule);
}

template <typename GraphT>
bool PebblingSchedule<GraphT>::HasValidSolution(const BspInstance<GraphT> &instance, const std::set<vertex_idx> &externalSources) {
    std::vector<memweight_type> memoryRequired = minimumMemoryRequiredPerNodeType(instance);
    std::vector<bool> hasEnoughMemory(instance.GetComputationalDag().NumVertexTypes(), true);
    for (vertex_idx node = 0; node < instance.NumberOfVertices(); ++node) {
        if (externalSources.find(node) == external_sources.end()) {
            hasEnoughMemory[instance.GetComputationalDag().VertexType(node)] = false;
        }
    }

    for (v_type_t<Graph_t> nodeType = 0; node_type < instance.GetComputationalDag().NumVertexTypes(); ++node_type) {
        for (unsigned proc = 0; proc < instance.NumberOfProcessors(); ++proc) {
            if (instance.isCompatibleType(node_type, instance.GetArchitecture().processorType(proc))
                && instance.GetArchitecture().memoryBound(proc) >= memory_required[node_type]) {
                hasEnoughMemory[node_type] = true;
                break;
            }
        }
    }

    for (v_type_t<Graph_t> nodeType = 0; node_type < instance.GetComputationalDag().NumVertexTypes(); ++node_type) {
        if (!hasEnoughMemory[node_type]) {
            std::cout << "No valid solution exists. Minimum memory required for node type " << node_type << " is "
                      << memory_required[node_type] << std::endl;
            return false;
        }
    }
    return true;
}

template <typename GraphT>
void PebblingSchedule<GraphT>::SplitSupersteps(const BspSchedule<GraphT> &schedule) {
    // get DFS topological order in each superstep
    std::vector<std::vector<std::vector<vertex_idx>>> topOrders = computeTopOrdersDFS(schedule);

    std::vector<unsigned> topOrderIdx(instance_->GetComputationalDag().NumVertices(), 0);
    for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
        for (unsigned step = 0; step < schedule.NumberOfSupersteps(); ++step) {
            for (unsigned idx = 0; idx < topOrders[proc][step].size(); ++idx) {
                topOrderIdx[top_orders[proc][step][idx]] = idx;
            }
        }
    }

    // split supersteps as needed
    std::vector<unsigned> newSuperstepId(instance_->GetComputationalDag().NumVertices());
    unsigned superstepIndex = 0;
    for (unsigned step = 0; step < schedule.NumberOfSupersteps(); ++step) {
        unsigned maxSegmentsInSuperstep = 0;
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
            if (topOrders[proc][step].empty()) {
                continue;
            }

            // the superstep will be split into smaller segments
            std::vector<std::pair<unsigned, unsigned>> segments;
            unsigned startIdx = 0;
            while (startIdx < topOrders[proc][step].size()) {
                // binary search for largest segment that still statisfies mem constraint
                bool doublingPhase = true;
                unsigned endLowerBound = startIdx, endUpperBound = static_cast<unsigned>(top_orders[proc][step].size() - 1);
                while (endLowerBound < endUpperBound) {
                    unsigned endCurrent;

                    if (doublingPhase) {
                        if (endLowerBound == startIdx) {
                            endCurrent = startIdx + 1;
                        } else {
                            endCurrent = std::min(startIdx + 2 * (endLowerBound - startIdx),
                                                  static_cast<unsigned>(top_orders[proc][step].size()) - 1);
                        }
                    } else {
                        endCurrent = endLowerBound + (endUpperBound - endLowerBound + 1) / 2;
                    }

                    // check if this segment is valid
                    bool valid = true;

                    std::map<vertex_idx, bool> neededAfter;
                    for (unsigned idx = startIdx; idx <= endCurrent; ++idx) {
                        vertex_idx node = top_orders[proc][step][idx];
                        neededAfter[node] = false;
                        if (needs_blue_at_end.empty()) {
                            neededAfter[node] = (instance_->GetComputationalDag().OutDegree(node) == 0);
                        } else {
                            neededAfter[node] = (needs_blue_at_end.find(node) != needs_blue_at_end.end());
                        }
                        for (vertex_idx succ : instance->GetComputationalDag().Children(node)) {
                            if (schedule.assignedSuperstep(succ) > step) {
                                neededAfter[node] = true;
                            }
                            if (schedule.assignedSuperstep(succ) == step && top_order_idx[succ] <= end_current) {
                                neededAfter[node] = true;
                            }
                        }
                    }

                    std::map<vertex_idx, vertex_idx> lastUsedBy;
                    std::set<vertex_idx> valuesNeeded;
                    for (unsigned idx = startIdx; idx <= endCurrent; ++idx) {
                        vertex_idx node = top_orders[proc][step][idx];
                        for (vertex_idx pred : instance->GetComputationalDag().Parents(node)) {
                            if (schedule.assignedSuperstep(pred) < step
                                || (schedule.assignedSuperstep(pred) == step && !neededAfter[pred])) {
                                lastUsedBy[pred] = node;
                            }
                            if (schedule.assignedSuperstep(pred) < step
                                || (schedule.assignedSuperstep(pred) == step && top_order_idx[pred] < start_idx)
                                || (need_to_load_inputs && instance->GetComputationalDag().in_degree(pred) == 0)
                                || external_sources.find(pred) != external_sources.end()) {
                                values_needed.insert(pred);
                            }
                        }
                    }

                    memweight_type memNeeded = 0;
                    for (vertex_idx node : values_needed) {
                        mem_needed += instance->GetComputationalDag().VertexMemWeight(node);
                    }

                    for (unsigned idx = startIdx; idx <= endCurrent; ++idx) {
                        vertex_idx node = top_orders[proc][step][idx];

                        if (needToLoadInputs_ && instance_->GetComputationalDag().in_degree(node) == 0) {
                            continue;
                        }

                        memNeeded += instance_->GetComputationalDag().VertexMemWeight(node);
                        if (memNeeded > instance_->GetArchitecture().memoryBound(proc)) {
                            valid = false;
                            break;
                        }

                        for (vertex_idx pred : instance->GetComputationalDag().Parents(node)) {
                            if (lastUsedBy[pred] == node) {
                                mem_needed -= instance->GetComputationalDag().VertexMemWeight(pred);
                            }
                        }
                    }

                    if (valid) {
                        endLowerBound = endCurrent;
                        if (endCurrent == top_orders[proc][step].size() - 1) {
                            doublingPhase = false;
                            endUpperBound = endCurrent;
                        }
                    } else {
                        doublingPhase = false;
                        endUpperBound = endCurrent - 1;
                    }
                }
                segments.emplace_back(startIdx, endLowerBound);
                startIdx = endLowerBound + 1;
            }

            unsigned stepIdx = 0;
            for (auto segment : segments) {
                for (unsigned idx = segment.first; idx <= segment.second; ++idx) {
                    newSuperstepId[top_orders[proc][step][idx]] = superstepIndex + stepIdx;
                }

                ++stepIdx;
            }

            if (stepIdx > maxSegmentsInSuperstep) {
                maxSegmentsInSuperstep = stepIdx;
            }
        }
        superstepIndex += maxSegmentsInSuperstep;
    }

    std::vector<unsigned> reindexToShrink(superstepIndex);
    std::vector<bool> hasCompute(superstepIndex, false);
    for (vertex_idx node = 0; node < instance_->NumberOfVertices(); ++node) {
        if (!needToLoadInputs_ || instance_->GetComputationalDag().in_degree(node) > 0) {
            hasCompute[newSuperstepId[node]] = true;
        }
    }

    unsigned currentIndex = 0;
    for (unsigned superstep = 0; superstep < superstepIndex; ++superstep) {
        if (hasCompute[superstep]) {
            reindexToShrink[superstep] = currentIndex;
            ++currentIndex;
        }
    }

    unsigned offset = needToLoadInputs_ ? 1 : 0;
    UpdateNumberOfSupersteps(currentIndex + offset);
    std::cout << schedule.NumberOfSupersteps() << " -> " << numberOfSupersteps_ << std::endl;

    // TODO: might not need offset for first step when beginning with red pebbles

    for (unsigned step = 0; step < schedule.NumberOfSupersteps(); ++step) {
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
            for (vertex_idx node : top_orders[proc][step]) {
                if (!need_to_load_inputs || instance->GetComputationalDag().in_degree(node) > 0) {
                    compute_steps_for_proc_superstep[proc][reindex_to_shrink[new_superstep_ID[node]] + offset].emplace_back(node);
                }
            }
        }
    }
}

template <typename GraphT>
void PebblingSchedule<GraphT>::SetMemoryMovement(CacheEvictionStrategy evictRule) {
    const size_t n = instance_->GetComputationalDag().NumVertices();

    std::vector<memweight_type> memUsed(instance_->NumberOfProcessors(), 0);
    std::vector<std::set<vertex_idx>> inMem(instance_->NumberOfProcessors());

    std::vector<bool> inSlowMem(n, false);
    if (needToLoadInputs_) {
        for (vertex_idx node = 0; node < n; ++node) {
            if (instance_->GetComputationalDag().in_degree(node) == 0) {
                inSlowMem[node] = true;
            }
        }
    }

    std::vector<std::set<std::pair<std::pair<unsigned, unsigned>, vertex_idx>>> evictable(instance_->NumberOfProcessors());
    std::vector<std::set<vertex_idx>> nonEvictable(instance_->NumberOfProcessors());

    // iterator to its position in "evictable" - for efficient delete
    std::vector<std::vector<decltype(evictable[0].begin())>> placeInEvictable(
        N, std::vector<decltype(evictable[0].begin())>(instance->NumberOfProcessors()));
    for (vertex_idx node = 0; node < n; ++node) {
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
            placeInEvictable[node][proc] = evictable[proc].end();
        }
    }

    // utility for LRU eviction strategy
    std::vector<std::vector<unsigned>> nodeLastUsedOnProc;
    if (evictRule == CacheEvictionStrategy::LEAST_RECENTLY_USED) {
        nodeLastUsedOnProc.resize(n, std::vector<unsigned>(instance_->NumberOfProcessors(), 0));
    }
    std::vector<unsigned> totalStepCountOnProc(instance_->NumberOfProcessors(), 0);

    // select a representative compute step for each node, in case of being computed multiple times
    // (NOTE - the conversion assumes that there is enough fast memory to keep each value until the end of
    // its representative step, if the value in question is ever needed on another processor/superster
    // without being recomputed there - otherwise, it would be even hard to decide whether a solution exists)
    std::vector<unsigned> selectedProcessor(n);
    std::vector<std::pair<unsigned, unsigned>> selectedStep(n, std::make_pair(numberOfSupersteps_, 0));
    for (unsigned superstep = 0; superstep < numberOfSupersteps_; ++superstep) {
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
            for (unsigned stepIndex = 0; stepIndex < computeStepsForProcSuperstep_[proc][superstep].size(); ++stepIndex) {
                vertex_idx node = computeStepsForProcSuperstep_[proc][superstep][stepIndex].node;
                if (selectedStep[node].first > superstep
                    || (selectedStep[node].first == superstep && selectedStep[node].second < stepIndex)) {
                    selectedProcessor[node] = proc;
                    selectedStep[node] = std::make_pair(superstep, stepIndex);
                }
            }
        }
    }

    // check if the node needs to be kept until the end of its representative superstep
    std::vector<bool> mustBePreserved(n, false);
    std::vector<bool> computedInCurrentSuperstep(n, false);
    for (unsigned superstep = 0; superstep < numberOfSupersteps_; ++superstep) {
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
            for (unsigned stepIndex = 0; stepIndex < computeStepsForProcSuperstep_[proc][superstep].size(); ++stepIndex) {
                vertex_idx node = computeStepsForProcSuperstep_[proc][superstep][stepIndex].node;
                computedInCurrentSuperstep[node] = true;
                for (vertex_idx pred : instance->GetComputationalDag().Parents(node)) {
                    if (!computed_in_current_superstep[pred]) {
                        must_be_preserved[pred] = true;
                    }
                }
            }
            for (unsigned stepIndex = 0; stepIndex < computeStepsForProcSuperstep_[proc][superstep].size(); ++stepIndex) {
                computedInCurrentSuperstep[computeStepsForProcSuperstep_[proc][superstep][stepIndex].node] = false;
            }
        }
    }
    if (needs_blue_at_end.empty()) {
        for (vertex_idx node = 0; node < n; ++node) {
            if (instance_->GetComputationalDag().OutDegree(node) == 0) {
                mustBePreserved[node] = true;
            }
        }
    } else {
        for (vertex_idx node : needs_blue_at_end) {
            must_be_preserved[node] = true;
        }
    }

    // superstep-step pairs where a node is required (on a given proc) - opening a separate queue after each time it's recomputed
    std::vector<std::vector<std::deque<std::deque<std::pair<unsigned, unsigned>>>>> nodeUsedAtProcLists(
        n,
        std::vector<std::deque<std::deque<std::pair<unsigned, unsigned>>>>(
            instance_->NumberOfProcessors(), std::deque<std::deque<std::pair<unsigned, unsigned>>>(1)));
    for (unsigned superstep = 0; superstep < numberOfSupersteps_; ++superstep) {
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
            for (unsigned stepIndex = 0; stepIndex < computeStepsForProcSuperstep_[proc][superstep].size(); ++stepIndex) {
                vertex_idx node = computeStepsForProcSuperstep_[proc][superstep][stepIndex].node;
                for (vertex_idx pred : instance->GetComputationalDag().Parents(node)) {
                    node_used_at_proc_lists[pred][proc].back().emplace_back(superstep, stepIndex);
                }

                nodeUsedAtProcLists[node][proc].emplace_back();
            }
        }
    }

    // set up initial content of fast memories
    if (!has_red_in_beginning.empty()) {
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
            in_mem = has_red_in_beginning;
            for (vertex_idx node : in_mem[proc]) {
                mem_used[proc] += instance->GetComputationalDag().VertexMemWeight(node);

                std::pair<unsigned, unsigned> prio;
                if (evict_rule == CACHE_EVICTION_STRATEGY::FORESIGHT) {
                    prio = node_used_at_proc_lists[node][proc].front().front();
                } else if (evict_rule == CACHE_EVICTION_STRATEGY::LEAST_RECENTLY_USED) {
                    prio = std::make_pair(UINT_MAX - node_last_used_on_proc[node][proc], static_cast<unsigned>(node));
                } else if (evict_rule == CACHE_EVICTION_STRATEGY::LARGEST_ID) {
                    prio = std::make_pair(static_cast<unsigned>(node), 0);
                }

                place_in_evictable[node][proc] = evictable[proc].emplace(prio, node).first;
            }
        }
    }

    // iterate through schedule
    for (unsigned superstep = 0; superstep < numberOfSupersteps_; ++superstep) {
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
            if (computeStepsForProcSuperstep_[proc][superstep].empty()) {
                continue;
            }

            // before compute phase, evict data in comm phase of previous superstep
            std::set<vertex_idx> newValuesNeeded;
            for (unsigned stepIndex = 0; stepIndex < computeStepsForProcSuperstep_[proc][superstep].size(); ++stepIndex) {
                vertex_idx node = computeStepsForProcSuperstep_[proc][superstep][stepIndex].node;
                computedInCurrentSuperstep[node] = true;
                for (vertex_idx pred : instance->GetComputationalDag().Parents(node)) {
                    if (!computed_in_current_superstep[pred]) {
                        non_evictable[proc].insert(pred);

                        if (place_in_evictable[pred][proc] != evictable[proc].end()) {
                            evictable[proc].erase(place_in_evictable[pred][proc]);
                            place_in_evictable[pred][proc] = evictable[proc].end();
                        }

                        if (in_mem[proc].find(pred) == in_mem[proc].end()) {
                            new_values_needed.insert(pred);
                        }
                    }
                }
            }
            for (unsigned stepIndex = 0; stepIndex < computeStepsForProcSuperstep_[proc][superstep].size(); ++stepIndex) {
                computedInCurrentSuperstep[computeStepsForProcSuperstep_[proc][superstep][stepIndex].node] = false;
            }

            for (vertex_idx node : new_values_needed) {
                in_mem[proc].insert(node);
                mem_used[proc] += instance->GetComputationalDag().VertexMemWeight(node);
                nodes_sent_down[proc][superstep - 1].push_back(node);
                if (!in_slow_mem[node]) {
                    in_slow_mem[node] = true;
                    nodes_sent_up[selected_processor[node]][selected_step[node].first].push_back(node);
                }
            }

            memweight_type firstNodeWeight
                = instance_->GetComputationalDag().VertexMemWeight(computeStepsForProcSuperstep_[proc][superstep][0].node);

            while (memUsed[proc] + first_node_weight
                   > instance_->GetArchitecture().memoryBound(proc))    // no sliding pebbles for now
            {
                if (evictable[proc].empty()) {
                    std::cout << "ERROR: Cannot create valid memory movement for these superstep lists." << std::endl;
                    return;
                }
                vertex_idx evicted = (--evictable[proc].end())->second;
                evictable[proc].erase(--evictable[proc].end());
                placeInEvictable[evicted][proc] = evictable[proc].end();

                memUsed[proc] -= instance_->GetComputationalDag().VertexMemWeight(evicted);
                inMem[proc].erase(evicted);

                nodes_evicted_in_comm[proc][superstep - 1].push_back(evicted);
            }

            // indicates if the node will be needed after (and thus cannot be deleted during) this compute phase
            std::map<vertex_idx, bool> neededAfter;

            // during compute phase
            for (unsigned stepIndex = 0; stepIndex < computeStepsForProcSuperstep_[proc][superstep].size(); ++stepIndex) {
                vertex_idx node = computeStepsForProcSuperstep_[proc][superstep][stepIndex].node;
                memweight_type nodeWeight = instance_->GetComputationalDag().VertexMemWeight(node);

                if (stepIndex > 0) {
                    // evict nodes to make space
                    while (memUsed[proc] + node_weight > instance_->GetArchitecture().memoryBound(proc)) {
                        if (evictable[proc].empty()) {
                            std::cout << "ERROR: Cannot create valid memory movement for these superstep lists." << std::endl;
                            return;
                        }
                        vertex_idx evicted = (--evictable[proc].end())->second;
                        evictable[proc].erase(--evictable[proc].end());
                        placeInEvictable[evicted][proc] = evictable[proc].end();

                        memUsed[proc] -= instance_->GetComputationalDag().VertexMemWeight(evicted);
                        inMem[proc].erase(evicted);

                        computeStepsForProcSuperstep_[proc][superstep][stepIndex - 1].nodes_evicted_after.push_back(evicted);
                    }
                }

                inMem[proc].insert(node);
                memUsed[proc] += node_weight;

                nonEvictable[proc].insert(node);

                if (evictRule == CacheEvictionStrategy::LEAST_RECENTLY_USED)    // update usage times for LRU strategy
                {
                    ++totalStepCountOnProc[proc];
                    nodeLastUsedOnProc[node][proc] = totalStepCountOnProc[proc];
                    for (vertex_idx pred : instance->GetComputationalDag().Parents(node)) {
                        node_last_used_on_proc[pred][proc] = total_step_count_on_proc[proc];
                    }
                }

                if (selectedProcessor[node] == proc && selectedStep[node] == std::make_pair(superstep, stepIndex)
                    && mustBePreserved[node]) {
                    neededAfter[node] = true;
                } else {
                    neededAfter[node] = false;
                }

                nodeUsedAtProcLists[node][proc].pop_front();

                for (vertex_idx pred : instance->GetComputationalDag().Parents(node)) {
                    node_used_at_proc_lists[pred][proc].front().pop_front();

                    if (needed_after[pred]) {
                        continue;
                    }

                    // autoevict
                    if (node_used_at_proc_lists[pred][proc].front().empty()) {
                        in_mem[proc].erase(pred);
                        non_evictable[proc].erase(pred);
                        mem_used[proc] -= instance->GetComputationalDag().VertexMemWeight(pred);
                        compute_steps_for_proc_superstep[proc][superstep][stepIndex].nodes_evicted_after.push_back(pred);
                    } else if (node_used_at_proc_lists[pred][proc].front().front().first > superstep) {
                        non_evictable[proc].erase(pred);

                        std::pair<unsigned, unsigned> prio;
                        if (evict_rule == CACHE_EVICTION_STRATEGY::FORESIGHT) {
                            prio = node_used_at_proc_lists[pred][proc].front().front();
                        } else if (evict_rule == CACHE_EVICTION_STRATEGY::LEAST_RECENTLY_USED) {
                            prio = std::make_pair(UINT_MAX - node_last_used_on_proc[pred][proc], static_cast<unsigned>(pred));
                        } else if (evict_rule == CACHE_EVICTION_STRATEGY::LARGEST_ID) {
                            prio = std::make_pair(static_cast<unsigned>(pred), 0);
                        }

                        place_in_evictable[pred][proc] = evictable[proc].emplace(prio, pred).first;
                    }
                }
            }

            // after compute phase
            for (vertex_idx node : non_evictable[proc]) {
                if (node_used_at_proc_lists[node][proc].front().empty()) {
                    mem_used[proc] -= instance->GetComputationalDag().VertexMemWeight(node);
                    in_mem[proc].erase(node);
                    nodes_evicted_in_comm[proc][superstep].push_back(node);
                    if ((instance->GetComputationalDag().OutDegree(node) == 0
                         || needs_blue_at_end.find(node) != needs_blue_at_end.end())
                        && !in_slow_mem[node]) {
                        in_slow_mem[node] = true;
                        nodes_sent_up[proc][superstep].push_back(node);
                    }
                } else {
                    std::pair<unsigned, unsigned> prio;
                    if (evict_rule == CACHE_EVICTION_STRATEGY::FORESIGHT) {
                        prio = node_used_at_proc_lists[node][proc].front().front();
                    } else if (evict_rule == CACHE_EVICTION_STRATEGY::LEAST_RECENTLY_USED) {
                        prio = std::make_pair(UINT_MAX - node_last_used_on_proc[node][proc], static_cast<unsigned>(node));
                    } else if (evict_rule == CACHE_EVICTION_STRATEGY::LARGEST_ID) {
                        prio = std::make_pair(static_cast<unsigned>(node), 0);
                    }

                    place_in_evictable[node][proc] = evictable[proc].emplace(prio, node).first;

                    if (needs_blue_at_end.find(node) != needs_blue_at_end.end() && !in_slow_mem[node]) {
                        in_slow_mem[node] = true;
                        nodes_sent_up[proc][superstep].push_back(node);
                    }
                }
            }
            nonEvictable[proc].clear();
        }
    }
}

template <typename GraphT>
void PebblingSchedule<GraphT>::ResetToForesight() {
    nodes_evicted_in_comm.clear();
    nodes_evicted_in_comm.resize(instance->NumberOfProcessors(), std::vector<std::vector<vertex_idx>>(number_of_supersteps));

    nodes_sent_down.clear();
    nodes_sent_down.resize(instance->NumberOfProcessors(), std::vector<std::vector<vertex_idx>>(number_of_supersteps));

    nodes_sent_up.clear();
    nodes_sent_up.resize(instance->NumberOfProcessors(), std::vector<std::vector<vertex_idx>>(number_of_supersteps));

    SetMemoryMovement(CacheEvictionStrategy::FORESIGHT);
}

template <typename GraphT>
bool PebblingSchedule<GraphT>::IsValid() const {
    std::vector<memweight_type> memUsed(instance_->NumberOfProcessors(), 0);
    std::vector<std::vector<vertex_idx>> inFastMem(instance->GetComputationalDag().NumVertices(),
                                                   std::vector<vertex_idx>(instance->NumberOfProcessors(), false));
    std::vector<vertex_idx> inSlowMem(instance_->GetComputationalDag().NumVertices(), false);

    if (needToLoadInputs_) {
        for (vertex_idx node = 0; node < instance_->NumberOfVertices(); ++node) {
            if (instance_->GetComputationalDag().in_degree(node) == 0) {
                inSlowMem[node] = true;
            }
        }
    }

    if (!has_red_in_beginning.empty()) {
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
            for (vertex_idx node : has_red_in_beginning[proc]) {
                mem_used[proc] += instance->GetComputationalDag().VertexMemWeight(node);
                in_fast_mem[node][proc] = true;
            }
        }
    }

    for (unsigned step = 0; step < numberOfSupersteps_; ++step) {
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
            // computation phase
            for (const auto &computeStep : computeStepsForProcSuperstep_[proc][step]) {
                if (!instance_->isCompatible(computeStep.node, proc)) {
                    return false;
                }

                for (vertex_idx pred : instance->GetComputationalDag().Parents(computeStep.node)) {
                    if (!in_fast_mem[pred][proc]) {
                        return false;
                    }
                }

                if (needToLoadInputs_ && instance_->GetComputationalDag().in_degree(computeStep.node) == 0) {
                    return false;
                }

                if (!in_fast_mem[computeStep.node][proc]) {
                    inFastMem[computeStep.node][proc] = true;
                    memUsed[proc] += instance_->GetComputationalDag().VertexMemWeight(computeStep.node);
                }

                if (memUsed[proc] > instance_->GetArchitecture().memoryBound(proc)) {
                    return false;
                }

                for (vertex_idx to_remove : computeStep.nodes_evicted_after) {
                    if (!in_fast_mem[to_remove][proc]) {
                        return false;
                    }

                    in_fast_mem[to_remove][proc] = false;
                    mem_used[proc] -= instance->GetComputationalDag().VertexMemWeight(to_remove);
                }
            }

            // communication phase - sendup and eviction
            for (vertex_idx node : nodes_sent_up[proc][step]) {
                if (!in_fast_mem[node][proc]) {
                    return false;
                }

                in_slow_mem[node] = true;
            }
            for (vertex_idx node : nodes_evicted_in_comm[proc][step]) {
                if (!in_fast_mem[node][proc]) {
                    return false;
                }

                in_fast_mem[node][proc] = false;
                mem_used[proc] -= instance->GetComputationalDag().VertexMemWeight(node);
            }
        }

        // communication phase - senddown
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
            for (vertex_idx node : nodes_sent_down[proc][step]) {
                if (!in_slow_mem[node]) {
                    return false;
                }

                if (!in_fast_mem[node][proc]) {
                    in_fast_mem[node][proc] = true;
                    mem_used[proc] += instance->GetComputationalDag().VertexMemWeight(node);
                }
            }
        }

        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
            if (memUsed[proc] > instance_->GetArchitecture().memoryBound(proc)) {
                return false;
            }
        }
    }

    if (needs_blue_at_end.empty()) {
        for (vertex_idx node = 0; node < instance_->NumberOfVertices(); ++node) {
            if (instance_->GetComputationalDag().OutDegree(node) == 0 && !in_slow_mem[node]) {
                return false;
            }
        }
    } else {
        for (vertex_idx node : needs_blue_at_end) {
            if (!in_slow_mem[node]) {
                return false;
            }
        }
    }

    return true;
}

template <typename GraphT>
std::vector<VMemwT<Graph_t>> PebblingSchedule<GraphT>::MinimumMemoryRequiredPerNodeType(
    const BspInstance<GraphT> &instance, const std::set<vertex_idx> &externalSources) {
    std::vector<VMemwT<Graph_t>> maxNeeded(instance.GetComputationalDag().NumVertexTypes(), 0);
    for (vertex_idx_t<Graph_t> node = 0; node < instance.GetComputationalDag().NumVertices(); ++node) {
        if (externalSources.find(node) != external_sources.end()) {
            continue;
        }

        VMemwT<Graph_t> needed = instance.GetComputationalDag().VertexMemWeight(node);
        const v_type_t<Graph_t> type = instance.GetComputationalDag().VertexType(node);
        for (vertex_idx_t<Graph_t> pred : instance.GetComputationalDag().Parents(node)) {
            needed += instance.GetComputationalDag().VertexMemWeight(pred);
        }

        if (needed > max_needed[type]) {
            maxNeeded[type] = needed;
        }
    }
    return max_needed;
}

template <typename GraphT>
std::vector<std::vector<std::vector<vertex_idx_t<Graph_t>>>> PebblingSchedule<GraphT>::ComputeTopOrdersDfs(
    const BspSchedule<GraphT> &schedule) const {
    size_t n = schedule.GetInstance().GetComputationalDag().NumVertices();
    unsigned numProcs = schedule.GetInstance().NumberOfProcessors();
    unsigned numSupsteps = schedule.NumberOfSupersteps();

    std::vector<std::vector<std::vector<vertex_idx>>> TopOrders(num_procs, std::vector<std::vector<vertex_idx>>(numSupsteps));

    std::vector<std::vector<std::deque<vertex_idx>>> Q(num_procs, std::vector<std::deque<vertex_idx>>(numSupsteps));
    std::vector<std::vector<std::vector<vertex_idx>>> NodesUpdated(num_procs, std::vector<std::vector<vertex_idx>>(numSupsteps));
    std::vector<unsigned> nrPred(n);
    std::vector<unsigned> predDone(n, 0);
    for (vertex_idx node = 0; node < n; ++node) {
        unsigned predecessors = 0;
        for (vertex_idx pred : schedule.GetInstance().GetComputationalDag().Parents(node)) {
            if (external_sources.find(pred) == external_sources.end()
                && schedule.assignedProcessor(node) == schedule.assignedProcessor(pred)
                && schedule.assignedSuperstep(node) == schedule.assignedSuperstep(pred)) {
                ++predecessors;
            }
        }
        nrPred[node] = predecessors;
        if (predecessors == 0 && external_sources.find(node) == external_sources.end()) {
            Q[schedule.assignedProcessor(node)][schedule.assignedSuperstep(node)].push_back(node);
        }
    }
    for (unsigned proc = 0; proc < numProcs; ++proc) {
        for (unsigned step = 0; step < numSupsteps; ++step) {
            while (!Q[proc][step].empty()) {
                vertex_idx node = Q[proc][step].front();
                Q[proc][step].pop_front();
                TopOrders[proc][step].push_back(node);
                for (vertex_idx succ : schedule.GetInstance().GetComputationalDag().Children(node)) {
                    if (schedule.assignedProcessor(node) == schedule.assignedProcessor(succ)
                        && schedule.assignedSuperstep(node) == schedule.assignedSuperstep(succ)) {
                        ++pred_done[succ];
                        if (pred_done[succ] == nr_pred[succ]) {
                            Q[proc][step].push_front(succ);
                        }
                    }
                }
            }
        }
    }

    return top_orders;
}

template <typename GraphT>
void PebblingSchedule<GraphT>::GetDataForMultiprocessorPebbling(
    std::vector<std::vector<std::vector<vertex_idx>>> &computeSteps,
    std::vector<std::vector<std::vector<vertex_idx>>> &sendUpSteps,
    std::vector<std::vector<std::vector<vertex_idx>>> &sendDownSteps,
    std::vector<std::vector<std::vector<vertex_idx>>> &nodesEvictedAfterStep) const {
    computeSteps.clear();
    computeSteps.resize(instance_->NumberOfProcessors());
    sendUpSteps.clear();
    sendUpSteps.resize(instance_->NumberOfProcessors());
    sendDownSteps.clear();
    sendDownSteps.resize(instance_->NumberOfProcessors());
    nodesEvictedAfterStep.clear();
    nodesEvictedAfterStep.resize(instance_->NumberOfProcessors());

    std::vector<memweight_type> memUsed(instance_->NumberOfProcessors(), 0);
    std::vector<std::set<vertex_idx>> inMem(instance_->NumberOfProcessors());
    if (!has_red_in_beginning.empty()) {
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
            for (vertex_idx node : has_red_in_beginning[proc]) {
                in_mem[proc].insert(node);
                mem_used[proc] += instance->GetComputationalDag().VertexMemWeight(node);
            }
        }
    }

    unsigned step = 0;

    for (unsigned superstep = 0; superstep < numberOfSupersteps_; ++superstep) {
        std::vector<unsigned> stepOnProc(instance_->NumberOfProcessors(), step);
        bool anyCompute = false;
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
            if (!computeStepsForProcSuperstep_[proc][superstep].empty()) {
                anyCompute = true;
            }
        }

        if (anyCompute) {
            for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
                computeSteps[proc].emplace_back();
                sendUpSteps[proc].emplace_back();
                sendDownSteps[proc].emplace_back();
                nodesEvictedAfterStep[proc].emplace_back();
            }
        }

        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
            std::vector<vertex_idx> evictList;
            for (unsigned stepIndex = 0; stepIndex < computeStepsForProcSuperstep_[proc][superstep].size(); ++stepIndex) {
                vertex_idx node = computeStepsForProcSuperstep_[proc][superstep][stepIndex].node;
                if (memUsed[proc] + instance_->GetComputationalDag().VertexMemWeight(node)
                    > instance_->GetArchitecture().memoryBound(proc)) {
                    // open new step
                    nodesEvictedAfterStep[proc][stepOnProc[proc]] = evict_list;
                    ++stepOnProc[proc];
                    for (vertex_idx to_evict : evict_list) {
                        mem_used[proc] -= instance->GetComputationalDag().VertexMemWeight(to_evict);
                    }

                    evictList.clear();
                    computeSteps[proc].emplace_back();
                    sendUpSteps[proc].emplace_back();
                    sendDownSteps[proc].emplace_back();
                    nodesEvictedAfterStep[proc].emplace_back();
                }

                computeSteps[proc][stepOnProc[proc]].emplace_back(node);
                memUsed[proc] += instance_->GetComputationalDag().VertexMemWeight(node);
                for (vertex_idx to_evict : compute_steps_for_proc_superstep[proc][superstep][stepIndex].nodes_evicted_after) {
                    evict_list.emplace_back(to_evict);
                }
            }

            if (!evict_list.empty()) {
                nodesEvictedAfterStep[proc][stepOnProc[proc]] = evict_list;
                for (vertex_idx to_evict : evict_list) {
                    mem_used[proc] -= instance->GetComputationalDag().VertexMemWeight(to_evict);
                }
            }
        }
        if (anyCompute) {
            for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
                ++stepOnProc[proc];
            }
        }

        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
            step = std::max(step, stepOnProc[proc]);
        }
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
            for (; stepOnProc[proc] < step; ++stepOnProc[proc]) {
                computeSteps[proc].emplace_back();
                sendUpSteps[proc].emplace_back();
                sendDownSteps[proc].emplace_back();
                nodesEvictedAfterStep[proc].emplace_back();
            }
        }

        bool anySendUp = false;
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
            if (!nodes_sent_up[proc][superstep].empty() || !nodes_evicted_in_comm[proc][superstep].empty()) {
                anySendUp = true;
            }
        }

        if (anySendUp) {
            for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
                computeSteps[proc].emplace_back();
                sendUpSteps[proc].emplace_back(nodes_sent_up[proc][superstep]);
                sendDownSteps[proc].emplace_back();
                nodesEvictedAfterStep[proc].emplace_back(nodes_evicted_in_comm[proc][superstep]);
                for (vertex_idx to_evict : nodes_evicted_in_comm[proc][superstep]) {
                    mem_used[proc] -= instance->GetComputationalDag().VertexMemWeight(to_evict);
                }
                ++stepOnProc[proc];
            }
            ++step;
        }

        bool anySendDown = false;
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
            if (!nodes_sent_down[proc][superstep].empty()) {
                anySendDown = true;
            }
        }

        if (anySendDown) {
            for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
                computeSteps[proc].emplace_back();
                sendUpSteps[proc].emplace_back();
                sendDownSteps[proc].emplace_back(nodes_sent_down[proc][superstep]);
                for (vertex_idx send_down : nodes_sent_down[proc][superstep]) {
                    mem_used[proc] += instance->GetComputationalDag().VertexMemWeight(send_down);
                }
                nodesEvictedAfterStep[proc].emplace_back();
                ++stepOnProc[proc];
            }
            ++step;
        }
    }
}

template <typename GraphT>
std::vector<std::set<vertex_idx_t<Graph_t>>> PebblingSchedule<GraphT>::GetMemContentAtEnd() const {
    std::vector<std::set<vertex_idx>> memContent(instance_->NumberOfProcessors());
    if (!has_red_in_beginning.empty()) {
        mem_content = has_red_in_beginning;
    }

    for (unsigned step = 0; step < numberOfSupersteps_; ++step) {
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
            // computation phase
            for (const auto &computeStep : computeStepsForProcSuperstep_[proc][step]) {
                memContent[proc].insert(computeStep.node);
                for (vertex_idx to_remove : computeStep.nodes_evicted_after) {
                    mem_content[proc].erase(to_remove);
                }
            }

            // communication phase - eviction
            for (vertex_idx node : nodes_evicted_in_comm[proc][step]) {
                mem_content[proc].erase(node);
            }

            // communication phase - senddown
            for (vertex_idx node : nodes_sent_down[proc][step]) {
                mem_content[proc].insert(node);
            }
        }
    }

    return mem_content;
}

template <typename GraphT>
void PebblingSchedule<GraphT>::RemoveEvictStepsFromEnd() {
    std::vector<memweight_type> memUsed(instance_->NumberOfProcessors(), 0);
    std::vector<memweight_type> bottleneck(instance_->NumberOfProcessors(), 0);
    std::vector<std::set<vertex_idx>> fastMemEnd = getMemContentAtEnd();
    for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
        for (vertex_idx node : fast_mem_end[proc]) {
            mem_used[proc] += instance->GetComputationalDag().VertexMemWeight(node);
        }

        bottleneck[proc] = instance_->GetArchitecture().memoryBound(proc) - mem_used[proc];
    }

    for (unsigned step = numberOfSupersteps_; step > 0;) {
        --step;

        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
            // communication phase - senddown
            for (vertex_idx node : nodes_sent_down[proc][step]) {
                mem_used[proc] -= instance->GetComputationalDag().VertexMemWeight(node);
            }

            // communication phase - eviction
            std::vector<vertex_idx> remaining;
            for (vertex_idx node : nodes_evicted_in_comm[proc][step]) {
                mem_used[proc] += instance->GetComputationalDag().VertexMemWeight(node);
                if (instance->GetComputationalDag().VertexMemWeight(node) <= bottleneck[proc]
                    && fast_mem_end[proc].find(node) == fast_mem_end[proc].end()) {
                    fast_mem_end[proc].insert(node);
                    bottleneck[proc] -= instance->GetComputationalDag().VertexMemWeight(node);
                } else {
                    remaining.push_back(node);
                }
            }
            nodes_evicted_in_comm[proc][step] = remaining;
            bottleneck[proc] = std::min(bottleneck[proc], instance_->GetArchitecture().memoryBound(proc) - mem_used[proc]);

            // computation phase
            for (unsigned stepIndex = static_cast<unsigned>(computeStepsForProcSuperstep_[proc][step].size()); stepIndex > 0;) {
                --stepIndex;
                auto &computeStep = computeStepsForProcSuperstep_[proc][step][stepIndex];

                std::vector<vertex_idx> remaining2;
                for (vertex_idx to_remove : computeStep.nodes_evicted_after) {
                    mem_used[proc] += instance->GetComputationalDag().VertexMemWeight(to_remove);
                    if (instance->GetComputationalDag().VertexMemWeight(to_remove) <= bottleneck[proc]
                        && fast_mem_end[proc].find(to_remove) == fast_mem_end[proc].end()) {
                        fast_mem_end[proc].insert(to_remove);
                        bottleneck[proc] -= instance->GetComputationalDag().VertexMemWeight(to_remove);
                    } else {
                        remaining_2.push_back(to_remove);
                    }
                }
                computeStep.nodes_evicted_after = remaining_2;
                bottleneck[proc] = std::min(bottleneck[proc], instance_->GetArchitecture().memoryBound(proc) - mem_used[proc]);

                memUsed[proc] -= instance_->GetComputationalDag().VertexMemWeight(computeStep.node);
            }
        }
    }

    if (!IsValid()) {
        std::cout << "ERROR: eviction removal process created an invalid schedule." << std::endl;
    }
}

template <typename GraphT>
void PebblingSchedule<GraphT>::CreateFromPartialPebblings(const BspInstance<GraphT> &bspInstance,
                                                          const std::vector<PebblingSchedule<GraphT>> &pebblings,
                                                          const std::vector<std::set<unsigned>> &processorsToParts,
                                                          const std::vector<std::map<vertex_idx, vertex_idx>> &originalNodeId,
                                                          const std::vector<std::map<unsigned, unsigned>> &originalProcId,
                                                          const std::vector<std::vector<std::set<vertex_idx>>> &hasRedsInBeginning) {
    instance_ = &bspInstance;

    unsigned nrParts = static_cast<unsigned>(processorsToParts.size());

    std::vector<std::set<vertex_idx>> inMem(instance_->NumberOfProcessors());
    std::vector<std::tuple<vertex_idx, unsigned, unsigned>> forceEvicts;

    computeStepsForProcSuperstep_.clear();
    nodes_sent_up.clear();
    nodes_sent_down.clear();
    nodes_evicted_in_comm.clear();
    computeStepsForProcSuperstep_.resize(instance_->NumberOfProcessors());
    nodes_sent_up.resize(instance->NumberOfProcessors());
    nodes_sent_down.resize(instance->NumberOfProcessors());
    nodes_evicted_in_comm.resize(instance->NumberOfProcessors());

    std::vector<unsigned> supstepIdx(instance_->NumberOfProcessors(), 0);

    std::vector<unsigned> getsBlueInSuperstep(instance_->NumberOfVertices(), UINT_MAX);
    for (vertex_idx node = 0; node < instance_->NumberOfVertices(); ++node) {
        if (instance_->GetComputationalDag().in_degree(node) == 0) {
            getsBlueInSuperstep[node] = 0;
        }
    }

    for (unsigned part = 0; part < nrParts; ++part) {
        unsigned startingStepIndex = 0;

        // find dependencies on previous subschedules
        for (vertex_idx node = 0; node < pebblings[part].instance->NumberOfVertices(); ++node) {
            if (pebblings[part].instance->GetComputationalDag().in_degree(node) == 0) {
                startingStepIndex = std::max(startingStepIndex, getsBlueInSuperstep[original_node_id[part].at(node)]);
            }
        }

        // sync starting points for the subset of processors
        for (unsigned proc : processorsToParts[part]) {
            startingStepIndex = std::max(startingStepIndex, supstepIdx[proc]);
        }
        for (unsigned proc : processorsToParts[part]) {
            while (supstepIdx[proc] < startingStepIndex) {
                computeStepsForProcSuperstep_[proc].emplace_back();
                nodes_sent_up[proc].emplace_back();
                nodes_sent_down[proc].emplace_back();
                nodes_evicted_in_comm[proc].emplace_back();
                ++supstepIdx[proc];
            }
        }

        // check and update according to initial states of red pebbles
        for (unsigned proc = 0; proc < processorsToParts[part].size(); ++proc) {
            unsigned procId = originalProcId[part].at(proc);
            std::set<vertex_idx> neededInRed, add_before, remove_before;
            for (vertex_idx node : has_reds_in_beginning[part][proc]) {
                vertex_idx node_id = original_node_id[part].at(node);
                needed_in_red.insert(node_id);
                if (in_mem[proc_id].find(node_id) == in_mem[proc_id].end()) {
                    add_before.insert(node_id);
                }
            }
            for (vertex_idx node : in_mem[proc_id]) {
                if (needed_in_red.find(node) == needed_in_red.end()) {
                    remove_before.insert(node);
                }
            }

            if ((!add_before.empty() || !remove_before.empty()) && supstep_idx[proc_id] == 0) {
                // this code is added just in case - this shouldn't happen in normal schedules
                computeStepsForProcSuperstep_[procId].emplace_back();
                nodes_sent_up[proc_id].emplace_back();
                nodes_sent_down[proc_id].emplace_back();
                nodes_evicted_in_comm[proc_id].emplace_back();
                ++supstepIdx[procId];
            }

            for (vertex_idx node : add_before) {
                in_mem[proc_id].insert(node);
                nodes_sent_down[proc_id].back().push_back(node);
            }
            for (vertex_idx node : remove_before) {
                in_mem[proc_id].erase(node);
                nodes_evicted_in_comm[proc_id].back().push_back(node);
                force_evicts.push_back(std::make_tuple(node, proc_id, nodes_evicted_in_comm[proc_id].size() - 1));
            }
        }

        for (unsigned supstep = 0; supstep < pebblings[part].NumberOfSupersteps(); ++supstep) {
            for (unsigned proc = 0; proc < processorsToParts[part].size(); ++proc) {
                unsigned procId = originalProcId[part].at(proc);
                computeStepsForProcSuperstep_[procId].emplace_back();
                nodes_sent_up[proc_id].emplace_back();
                nodes_sent_down[proc_id].emplace_back();
                nodes_evicted_in_comm[proc_id].emplace_back();

                // copy schedule with translated indeces
                for (const ComputeStep &computeStep : pebblings[part].GetComputeStepsForProcSuperstep(proc, supstep)) {
                    computeStepsForProcSuperstep_[procId].back().emplace_back();
                    computeStepsForProcSuperstep_[procId].back().back().node = original_node_id[part].at(computeStep.node_);
                    inMem[procId].insert(original_node_id[part].at(computeStep.node_));

                    for (vertex_idx local_id : computeStep.nodes_evicted_after) {
                        compute_steps_for_proc_superstep[proc_id].back().back().nodes_evicted_after.push_back(
                            original_node_id[part].at(local_id));
                        in_mem[proc_id].erase(original_node_id[part].at(local_id));
                    }
                }
                for (vertex_idx node : pebblings[part].GetNodesSentUp(proc, supstep)) {
                    vertex_idx node_id = original_node_id[part].at(node);
                    nodes_sent_up[proc_id].back().push_back(node_id);
                    gets_blue_in_superstep[node_id] = std::min(gets_blue_in_superstep[node_id], supstep_idx[proc_id]);
                }
                for (vertex_idx node : pebblings[part].GetNodesEvictedInComm(proc, supstep)) {
                    nodes_evicted_in_comm[proc_id].back().push_back(original_node_id[part].at(node));
                    in_mem[proc_id].erase(original_node_id[part].at(node));
                }
                for (vertex_idx node : pebblings[part].GetNodesSentDown(proc, supstep)) {
                    nodes_sent_down[proc_id].back().push_back(original_node_id[part].at(node));
                    in_mem[proc_id].insert(original_node_id[part].at(node));
                }

                ++supstepIdx[procId];
            }
        }
    }

    // padding supersteps in the end
    unsigned maxStepIndex = 0;
    for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
        maxStepIndex = std::max(maxStepIndex, supstepIdx[proc]);
    }
    for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
        while (supstepIdx[proc] < maxStepIndex) {
            computeStepsForProcSuperstep_[proc].emplace_back();
            nodes_sent_up[proc].emplace_back();
            nodes_sent_down[proc].emplace_back();
            nodes_evicted_in_comm[proc].emplace_back();
            ++supstepIdx[proc];
        }
    }
    numberOfSupersteps_ = maxStepIndex;
    needToLoadInputs_ = true;

    FixForceEvicts(force_evicts);
    TryToMergeSupersteps();
}

template <typename GraphT>
void PebblingSchedule<GraphT>::FixForceEvicts(const std::vector<std::tuple<vertex_idx, unsigned, unsigned>> forceEvictNodeProcStep) {
    // Some values were evicted only because they weren't present in the next part - see if we can undo those evictions
    for (auto force_evict : force_evict_node_proc_step) {
        vertex_idx node = std::get<0>(force_evict);
        unsigned proc = std::get<1>(force_evict);
        unsigned superstep = std::get<2>(force_evict);

        bool next_in_comp = false;
        bool next_in_comm = false;
        std::pair<unsigned, unsigned> where;

        for (unsigned find_supstep = superstep + 1; find_supstep < NumberOfSupersteps(); ++find_supstep) {
            for (unsigned stepIndex = 0; stepIndex < compute_steps_for_proc_superstep[proc][find_supstep].size(); ++stepIndex) {
                if (compute_steps_for_proc_superstep[proc][find_supstep][stepIndex].node == node) {
                    next_in_comp = true;
                    where = std::make_pair(find_supstep, stepIndex);
                    break;
                }
            }
            if (next_in_comp) {
                break;
            }
            for (vertex_idx send_down : nodes_sent_down[proc][find_supstep]) {
                if (send_down == node) {
                    next_in_comm = true;
                    where = std::make_pair(find_supstep, 0);
                    break;
                }
            }
            if (next_in_comm) {
                break;
            }
        }

        // check new schedule for validity
        if (!next_in_comp && !next_in_comm) {
            continue;
        }

        PebblingSchedule<Graph_t> test_schedule = *this;
        for (auto itr = test_schedule.nodes_evicted_in_comm[proc][superstep].begin();
             itr != test_schedule.nodes_evicted_in_comm[proc][superstep].end();
             ++itr) {
            if (*itr == node) {
                test_schedule.nodes_evicted_in_comm[proc][superstep].erase(itr);
                break;
            }
        }

        if (next_in_comp) {
            for (auto itr = test_schedule.compute_steps_for_proc_superstep[proc][where.first].begin();
                 itr != test_schedule.compute_steps_for_proc_superstep[proc][where.first].end();
                 ++itr) {
                if (itr->node == node) {
                    if (where.second > 0) {
                        auto previous_step = itr;
                        --previous_step;
                        for (vertex_idx to_evict : itr->nodes_evicted_after) {
                            previous_step->nodes_evicted_after.push_back(to_evict);
                        }
                    } else {
                        for (vertex_idx to_evict : itr->nodes_evicted_after) {
                            test_schedule.nodes_evicted_in_comm[proc][where.first - 1].push_back(to_evict);
                        }
                    }
                    test_schedule.compute_steps_for_proc_superstep[proc][where.first].erase(itr);
                    break;
                }
            }

            if (test_schedule.isValid()) {
                nodes_evicted_in_comm[proc][superstep] = test_schedule.nodes_evicted_in_comm[proc][superstep];
                compute_steps_for_proc_superstep[proc][where.first]
                    = test_schedule.compute_steps_for_proc_superstep[proc][where.first];
                nodes_evicted_in_comm[proc][where.first - 1] = test_schedule.nodes_evicted_in_comm[proc][where.first - 1];
            }
        } else if (next_in_comm) {
            for (auto itr = test_schedule.nodes_sent_down[proc][where.first].begin();
                 itr != test_schedule.nodes_sent_down[proc][where.first].end();
                 ++itr) {
                if (*itr == node) {
                    test_schedule.nodes_sent_down[proc][where.first].erase(itr);
                    break;
                }
            }

            if (test_schedule.isValid()) {
                nodes_evicted_in_comm[proc][superstep] = test_schedule.nodes_evicted_in_comm[proc][superstep];
                nodes_sent_down[proc][where.first] = test_schedule.nodes_sent_down[proc][where.first];
            }
        }
    }
}

template <typename GraphT>
void PebblingSchedule<GraphT>::TryToMergeSupersteps() {
    std::vector<bool> isRemoved(numberOfSupersteps_, false);

    for (unsigned step = 1; step < numberOfSupersteps_; ++step) {
        if (isRemoved[step]) {
            continue;
        }

        unsigned prevStep = step - 1;
        while (isRemoved[prevStep]) {
            --prevStep;
        }

        for (unsigned nextStep = step + 1; nextStep < numberOfSupersteps_; ++nextStep) {
            // Try to merge step and next_step
            PebblingSchedule testSchedule = *this;

            for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
                testSchedule.computeStepsForProcSuperstep_[proc][step].insert(
                    testSchedule.computeStepsForProcSuperstep_[proc][step].end(),
                    testSchedule.computeStepsForProcSuperstep_[proc][nextStep].begin(),
                    testSchedule.computeStepsForProcSuperstep_[proc][nextStep].end());
                testSchedule.computeStepsForProcSuperstep_[proc][nextStep].clear();

                testSchedule.nodes_sent_up[proc][step].insert(testSchedule.nodes_sent_up[proc][step].end(),
                                                              testSchedule.nodes_sent_up[proc][nextStep].begin(),
                                                              testSchedule.nodes_sent_up[proc][nextStep].end());
                testSchedule.nodes_sent_up[proc][nextStep].clear();

                testSchedule.nodes_sent_down[proc][prevStep].insert(testSchedule.nodes_sent_down[proc][prevStep].end(),
                                                                    testSchedule.nodes_sent_down[proc][step].begin(),
                                                                    testSchedule.nodes_sent_down[proc][step].end());
                testSchedule.nodes_sent_down[proc][step].clear();

                testSchedule.nodes_evicted_in_comm[proc][step].insert(testSchedule.nodes_evicted_in_comm[proc][step].end(),
                                                                      testSchedule.nodes_evicted_in_comm[proc][nextStep].begin(),
                                                                      testSchedule.nodes_evicted_in_comm[proc][nextStep].end());
                testSchedule.nodes_evicted_in_comm[proc][nextStep].clear();
            }

            if (testSchedule.IsValid()) {
                isRemoved[nextStep] = true;
                for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
                    computeStepsForProcSuperstep_[proc][step] = testSchedule.computeStepsForProcSuperstep_[proc][step];
                    computeStepsForProcSuperstep_[proc][nextStep].clear();

                    nodes_sent_up[proc][step] = test_schedule.nodes_sent_up[proc][step];
                    nodes_sent_up[proc][next_step].clear();

                    nodes_sent_down[proc][prev_step] = test_schedule.nodes_sent_down[proc][prev_step];
                    nodes_sent_down[proc][step] = nodes_sent_down[proc][next_step];
                    nodes_sent_down[proc][next_step].clear();

                    nodes_evicted_in_comm[proc][step] = test_schedule.nodes_evicted_in_comm[proc][step];
                    nodes_evicted_in_comm[proc][next_step].clear();
                }
            } else {
                break;
            }
        }
    }

    unsigned newNrSupersteps = 0;
    for (unsigned step = 0; step < numberOfSupersteps_; ++step) {
        if (!isRemoved[step]) {
            ++newNrSupersteps;
        }
    }

    if (newNrSupersteps == numberOfSupersteps_) {
        return;
    }

    PebblingSchedule<GraphT> shortenedSchedule = *this;
    shortenedSchedule.UpdateNumberOfSupersteps(newNrSupersteps);

    unsigned newIndex = 0;
    for (unsigned step = 0; step < numberOfSupersteps_; ++step) {
        if (isRemoved[step]) {
            continue;
        }

        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
            shortenedSchedule.computeStepsForProcSuperstep_[proc][newIndex] = computeStepsForProcSuperstep_[proc][step];
            shortened_schedule.nodes_sent_up[proc][new_index] = nodes_sent_up[proc][step];
            shortened_schedule.nodes_sent_down[proc][new_index] = nodes_sent_down[proc][step];
            shortened_schedule.nodes_evicted_in_comm[proc][new_index] = nodes_evicted_in_comm[proc][step];
        }

        ++newIndex;
    }

    *this = shortenedSchedule;

    if (!IsValid()) {
        std::cout << "ERROR: schedule is not valid after superstep merging." << std::endl;
    }
}

template <typename GraphT>
PebblingSchedule<GraphT> PebblingSchedule<GraphT>::ExpandMemSchedule(const BspInstance<GraphT> &originalInstance,
                                                                     const std::vector<vertex_idx> mappingToCoarse) const {
    std::map<vertex_idx, std::set<vertex_idx>> original_vertices_for_coarse_ID;
    for (vertex_idx node = 0; node < originalInstance.NumberOfVertices(); ++node) {
        original_vertices_for_coarse_ID[mapping_to_coarse[node]].insert(node);
    }

    PebblingSchedule<GraphT> fineSchedule;
    fineSchedule.instance_ = &originalInstance;
    fineSchedule.UpdateNumberOfSupersteps(numberOfSupersteps_);

    for (unsigned step = 0; step < numberOfSupersteps_; ++step) {
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
            // computation phase
            for (const auto &computeStep : computeStepsForProcSuperstep_[proc][step]) {
                vertex_idx node = computeStep.node;
                for (vertex_idx original_node : original_vertices_for_coarse_ID[node]) {
                    fine_schedule.compute_steps_for_proc_superstep[proc][step].emplace_back(original_node);
                }

                for (vertex_idx to_remove : computeStep.nodes_evicted_after) {
                    for (vertex_idx original_node : original_vertices_for_coarse_ID[to_remove]) {
                        fine_schedule.compute_steps_for_proc_superstep[proc][step].back().nodes_evicted_after.push_back(
                            original_node);
                    }
                }
            }

            // communication phase
            for (vertex_idx node : nodes_sent_up[proc][step]) {
                for (vertex_idx original_node : original_vertices_for_coarse_ID[node]) {
                    fine_schedule.nodes_sent_up[proc][step].push_back(original_node);
                }
            }

            for (vertex_idx node : nodes_evicted_in_comm[proc][step]) {
                for (vertex_idx original_node : original_vertices_for_coarse_ID[node]) {
                    fine_schedule.nodes_evicted_in_comm[proc][step].push_back(original_node);
                }
            }

            for (vertex_idx node : nodes_sent_down[proc][step]) {
                for (vertex_idx original_node : original_vertices_for_coarse_ID[node]) {
                    fine_schedule.nodes_sent_down[proc][step].push_back(original_node);
                }
            }
        }
    }

    fineSchedule.CleanSchedule();
    return fineSchedule;
}

template <typename GraphT>
BspSchedule<GraphT> PebblingSchedule<GraphT>::ConvertToBsp() const {
    std::vector<unsigned> nodeToProc(instance_->NumberOfVertices(), UINT_MAX),
        nodeToSupstep(instance_->NumberOfVertices(), UINT_MAX);

    for (unsigned step = 0; step < numberOfSupersteps_; ++step) {
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
            for (const auto &computeStep : computeStepsForProcSuperstep_[proc][step]) {
                const vertex_idx &node = computeStep.node;
                if (nodeToProc[node] == UINT_MAX) {
                    nodeToProc[node] = proc;
                    nodeToSupstep[node] = step;
                }
            }
        }
    }
    if (needToLoadInputs_) {
        for (vertex_idx node = 0; node < instance_->NumberOfVertices(); ++node) {
            if (instance_->GetComputationalDag().in_degree(node) == 0) {
                unsigned minSuperstep = UINT_MAX, procChosen = 0;
                for (vertex_idx succ : instance->GetComputationalDag().Children(node)) {
                    if (node_to_supstep[succ] < min_superstep) {
                        min_superstep = node_to_supstep[succ];
                        proc_chosen = node_to_proc[succ];
                    }
                }
                nodeToSupstep[node] = minSuperstep;
                nodeToProc[node] = procChosen;
            }
        }
    }

    BspSchedule<GraphT> schedule(*instance_, nodeToProc, nodeToSupstep);
    if (schedule.satisfiesPrecedenceConstraints() && schedule.satisfiesNodeTypeConstraints()) {
        schedule.setAutoCommunicationSchedule();
        return schedule;
    } else {
        std::cout << "ERROR: no direct conversion to Bsp schedule exists, using dummy schedule instead." << std::endl;
        return BspSchedule<GraphT>(*instance_);
    }
}

}    // namespace osp
