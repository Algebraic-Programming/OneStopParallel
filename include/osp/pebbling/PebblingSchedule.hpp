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
    static_assert(isComputationalDagV<GraphT>, "PebblingSchedule can only be used with computational DAGs.");

  private:
    using VertexIdx = VertexIdxT<GraphT>;
    using CostType = VWorkwT<GraphT>;
    using MemweightType = VMemwT<GraphT>;

    static_assert(std::is_same_v<VWorkwT<GraphT>, VCommwT<GraphT>>,
                  "PebblingSchedule requires work and comm. weights to have the same type.");

    const BspInstance<GraphT> *instance_;

    unsigned int numberOfSupersteps_;

    bool needToLoadInputs_ = true;

    struct ComputeStep {
        VertexIdx node;
        std::vector<VertexIdx> nodesEvictedAfter;

        ComputeStep() {}

        ComputeStep(VertexIdx node) : node(node) {}

        ComputeStep(VertexIdx node, const std::vector<VertexIdx> &evicted) : node(node), nodesEvictedAfter(evicted) {}
    };

    // executed nodes in order in a computation phase, for processor p and superstep s
    std::vector<std::vector<std::vector<ComputeStep>>> computeStepsForProcSuperstep_;

    // nodes evicted from cache in a given superstep's comm phase
    std::vector<std::vector<std::vector<VertexIdx>>> nodesEvictedInComm_;

    // nodes sent down to processor p in superstep s
    std::vector<std::vector<std::vector<VertexIdx>>> nodesSentDown_;

    // nodes sent up from processor p in superstep s
    std::vector<std::vector<std::vector<VertexIdx>>> nodesSentUp_;

    // set of nodes that need to have blue pebble at end, sinks by default, and
    // set of nodes on each processor that begin with red pebble, nothing by default
    // (TODO: maybe move to problem definition classes instead?)
    std::set<VertexIdx> needsBlueAtEnd_;
    std::vector<std::set<VertexIdx>> hasRedInBeginning_;

    // nodes that are from a previous part of a larger DAG, handled differently in conversion
    std::set<VertexIdx> externalSources_;

  public:
    enum CACHE_EVICTION_STRATEGY { FORESIGHT, LEAST_RECENTLY_USED, LARGEST_ID };

    /**
     * @brief Default constructor for the PebblingSchedule class.
     */
    PebblingSchedule() : instance_(nullptr), numberOfSupersteps_(0) {}

    PebblingSchedule(const BspInstance<GraphT> &inst) : instance_(&inst) {
        BspSchedule<GraphT> schedule(
            inst, std::vector<unsigned int>(inst.numberOfVertices(), 0), std::vector<unsigned int>(inst.numberOfVertices(), 0));
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
                     const std::vector<std::vector<std::vector<VertexIdx>>> &computeSteps,
                     const std::vector<std::vector<std::vector<std::vector<VertexIdx>>>> &nodesEvictedAfterCompute,
                     const std::vector<std::vector<std::vector<VertexIdx>>> &nodesSentUp,
                     const std::vector<std::vector<std::vector<VertexIdx>>> &nodesSentDown,
                     const std::vector<std::vector<std::vector<VertexIdx>>> &nodesEvictedInComm,
                     const std::set<VertexIdx> &needsBlueAtEnd = std::set<VertexIdx>(),
                     const std::vector<std::set<VertexIdx>> &hasRedInBeginning = std::vector<std::set<VertexIdx>>(),
                     const bool needToLoadInputs = false)
        : instance_(&inst),
          numberOfSupersteps_(0),
          needToLoadInputs_(needToLoadInputs),
          nodesEvictedInComm_(nodesEvictedInComm),
          nodesSentDown_(nodesSentDown),
          nodesSentUp_(nodesSentUp),
          needsBlueAtEnd_(needsBlueAtEnd),
          hasRedInBeginning_(hasRedInBeginning) {
        computeStepsForProcSuperstep_.resize(computeSteps.size(), std::vector<std::vector<ComputeStep>>(computeSteps[0].size()));
        for (unsigned proc = 0; proc < computeSteps.size(); ++proc) {
            numberOfSupersteps_ = std::max(numberOfSupersteps_, static_cast<unsigned>(computeSteps[proc].size()));
            for (unsigned supstep = 0; supstep < static_cast<unsigned>(computeSteps[proc].size()); ++supstep) {
                for (unsigned stepIndex = 0; stepIndex < static_cast<unsigned>(computeSteps[proc][supstep].size()); ++stepIndex) {
                    computeStepsForProcSuperstep_[proc][supstep].emplace_back(computeSteps[proc][supstep][stepIndex],
                                                                              nodesEvictedAfterCompute[proc][supstep][stepIndex]);
                }
            }
        }
    }

    PebblingSchedule(const BspSchedule<GraphT> &schedule, CACHE_EVICTION_STRATEGY evictRule = LARGEST_ID)
        : instance_(&schedule.getInstance()) {
        ConvertFromBsp(schedule, evictRule);
    }

    virtual ~PebblingSchedule() = default;

    // cost computation
    CostType ComputeCost() const;
    CostType ComputeAsynchronousCost() const;

    // remove unnecessary steps (e.g. from ILP solution)
    void CleanSchedule();

    // convert from unconstrained schedule
    void ConvertFromBsp(const BspSchedule<GraphT> &schedule, CACHE_EVICTION_STRATEGY evict_rule = LARGEST_ID);

    // auxiliary for conversion
    std::vector<std::vector<std::vector<VertexIdx>>> ComputeTopOrdersDfs(const BspSchedule<GraphT> &schedule) const;
    static bool HasValidSolution(const BspInstance<GraphT> &instance,
                                 const std::set<VertexIdx> &external_sources = std::set<VertexIdx>());
    void SplitSupersteps(const BspSchedule<GraphT> &schedule);
    void SetMemoryMovement(CACHE_EVICTION_STRATEGY evict_rule = LARGEST_ID);

    // delete current communication schedule, and switch to foresight policy instead
    void ResetToForesight();

    // other basic operations
    bool isValid() const;
    static std::vector<MemweightType> minimumMemoryRequiredPerNodeType(const BspInstance<GraphT> &instance,
                                                                       const std::set<VertexIdx> &external_sources
                                                                       = std::set<VertexIdx>());

    // expand a MemSchedule from a coarsened DAG to the original DAG
    PebblingSchedule<GraphT> ExpandMemSchedule(const BspInstance<GraphT> &originalInstance,
                                               const std::vector<VertexIdx> mappingToCoarse) const;

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

    void UpdateNumberOfSupersteps(unsigned new_number_of_supersteps);

    inline bool NeedsToLoadInputs() const { return needToLoadInputs_; }

    inline void SetNeedToLoadInputs(const bool loadInputs) { needToLoadInputs_ = loadInputs; }

    void GetDataForMultiprocessorPebbling(std::vector<std::vector<std::vector<VertexIdx>>> &computeSteps,
                                          std::vector<std::vector<std::vector<VertexIdx>>> &sendUpSteps,
                                          std::vector<std::vector<std::vector<VertexIdx>>> &sendDownSteps,
                                          std::vector<std::vector<std::vector<VertexIdx>>> &nodesEvictedAfterStep) const;

    // utility for partial ILPs
    std::vector<std::set<VertexIdx>> GetMemContentAtEnd() const;
    void RemoveEvictStepsFromEnd();

    void CreateFromPartialPebblings(const BspInstance<GraphT> &bspInstance,
                                    const std::vector<PebblingSchedule<GraphT>> &pebblings,
                                    const std::vector<std::set<unsigned>> &processorsToParts,
                                    const std::vector<std::map<VertexIdx, VertexIdx>> &originalNodeId,
                                    const std::vector<std::map<unsigned, unsigned>> &originalProcId,
                                    const std::vector<std::vector<std::set<VertexIdx>>> &hasRedsInBeginning);

    // auxiliary function to remove some unnecessary communications after assembling from partial pebblings
    void FixForceEvicts(const std::vector<std::tuple<VertexIdx, unsigned, unsigned>> forceEvictNodeProcStep);

    // auxiliary after partial pebblings: try to merge supersteps
    void TryToMergeSupersteps();

    const std::vector<ComputeStep> &GetComputeStepsForProcSuperstep(unsigned proc, unsigned supstep) const {
        return computeStepsForProcSuperstep_[proc][supstep];
    }

    const std::vector<VertexIdx> &GetNodesEvictedInComm(unsigned proc, unsigned supstep) const {
        return nodesEvictedInComm_[proc][supstep];
    }

    const std::vector<VertexIdx> &GetNodesSentDown(unsigned proc, unsigned supstep) const {
        return nodesSentDown_[proc][supstep];
    }

    const std::vector<VertexIdx> &GetNodesSentUp(unsigned proc, unsigned supstep) const { return nodesSentUp_[proc][supstep]; }

    void SetNeedsBlueAtEnd(const std::set<VertexIdx> &nodes) { needsBlueAtEnd_ = nodes; }

    void SetExternalSources(const std::set<VertexIdx> &nodes) { externalSources_ = nodes; }

    void SetHasRedInBeginning(const std::vector<std::set<VertexIdx>> &nodes) { hasRedInBeginning_ = nodes; }
};

template <typename GraphT>
void PebblingSchedule<GraphT>::UpdateNumberOfSupersteps(unsigned new_number_of_supersteps) {
    numberOfSupersteps_ = new_number_of_supersteps;

    computeStepsForProcSuperstep_.clear();
    computeStepsForProcSuperstep_.resize(instance_->NumberOfProcessors(),
                                         std::vector<std::vector<ComputeStep>>(numberOfSupersteps_));

    nodesEvictedInComm_.clear();
    nodesEvictedInComm_.resize(instance_->NumberOfProcessors(), std::vector<std::vector<VertexIdx>>(numberOfSupersteps_));

    nodesSentDown_.clear();
    nodesSentDown_.resize(instance_->NumberOfProcessors(), std::vector<std::vector<VertexIdx>>(numberOfSupersteps_));

    nodesSentUp_.clear();
    nodesSentUp_.resize(instance_->NumberOfProcessors(), std::vector<std::vector<VertexIdx>>(numberOfSupersteps_));
}

template <typename GraphT>
VWorkwT<GraphT> PebblingSchedule<GraphT>::ComputeCost() const {
    CostType totalCosts = 0;
    for (unsigned step = 0; step < numberOfSupersteps_; ++step) {
        // compute phase
        CostType maxWork = std::numeric_limits<CostType>::min();
        for (unsigned proc = 0; proc < instance_->getArchitecture().numberOfProcessors(); ++proc) {
            CostType work = 0;
            for (const auto &computeStep : computeStepsForProcSuperstep_[proc][step]) {
                work += instance_->getComputationalDag().VertexWorkWeight(computeStep.node);
            }

            if (work > maxWork) {
                maxWork = work;
            }
        }
        totalCosts += maxWork;

        // communication phase
        CostType maxSendUp = std::numeric_limits<CostType>::min();
        for (unsigned proc = 0; proc < instance_->getArchitecture().numberOfProcessors(); ++proc) {
            CostType sendUp = 0;
            for (VertexIdx node : nodesSentUp_[proc][step]) {
                sendUp += instance_->getComputationalDag().VertexCommWeight(node)
                          * instance_->getArchitecture().communicationCosts();
            }

            if (sendUp > maxSendUp) {
                maxSendUp = sendUp;
            }
        }
        totalCosts += maxSendUp;

        totalCosts += static_cast<CostType>(instance_->getArchitecture().synchronisationCosts());

        CostType maxSendDown = std::numeric_limits<CostType>::min();
        for (unsigned proc = 0; proc < instance_->getArchitecture().numberOfProcessors(); ++proc) {
            CostType sendDown = 0;
            for (VertexIdx node : nodesSentDown_[proc][step]) {
                sendDown += instance_->getComputationalDag().VertexCommWeight(node)
                            * instance_->getArchitecture().communicationCosts();
            }

            if (sendDown > maxSendDown) {
                maxSendDown = sendDown;
            }
        }
        totalCosts += maxSendDown;
    }

    return totalCosts;
}

template <typename GraphT>
VWorkwT<GraphT> PebblingSchedule<GraphT>::ComputeAsynchronousCost() const {
    std::vector<CostType> currentTimeAtProcessor(instance_->getArchitecture().numberOfProcessors(), 0);
    std::vector<CostType> timeWhenNodeGetsBlue(instance_->getComputationalDag().NumVertices(),
                                               std::numeric_limits<CostType>::max());
    if (needToLoadInputs_) {
        for (VertexIdx node = 0; node < instance_->numberOfVertices(); ++node) {
            if (instance_->getComputationalDag().InDegree(node) == 0) {
                timeWhenNodeGetsBlue[node] = 0;
            }
        }
    }

    for (unsigned step = 0; step < numberOfSupersteps_; ++step) {
        // compute phase
        for (unsigned proc = 0; proc < instance_->getArchitecture().numberOfProcessors(); ++proc) {
            for (const auto &computeStep : computeStepsForProcSuperstep_[proc][step]) {
                currentTimeAtProcessor[proc] += instance_->getComputationalDag().VertexWorkWeight(computeStep.node);
            }
        }

        // communication phase - send up
        for (unsigned proc = 0; proc < instance_->getArchitecture().numberOfProcessors(); ++proc) {
            for (VertexIdx node : nodesSentUp_[proc][step]) {
                currentTimeAtProcessor[proc] += instance_->getComputationalDag().VertexCommWeight(node)
                                                * instance_->getArchitecture().communicationCosts();
                if (timeWhenNodeGetsBlue[node] > currentTimeAtProcessor[proc]) {
                    timeWhenNodeGetsBlue[node] = currentTimeAtProcessor[proc];
                }
            }
        }

        // communication phase - send down
        for (unsigned proc = 0; proc < instance_->getArchitecture().numberOfProcessors(); ++proc) {
            for (VertexIdx node : nodesSentDown_[proc][step]) {
                if (currentTimeAtProcessor[proc] < timeWhenNodeGetsBlue[node]) {
                    currentTimeAtProcessor[proc] = timeWhenNodeGetsBlue[node];
                }
                currentTimeAtProcessor[proc] += instance_->getComputationalDag().VertexCommWeight(node)
                                                * instance_->getArchitecture().communicationCosts();
            }
        }
    }

    CostType makespan = 0;
    for (unsigned proc = 0; proc < instance_->getArchitecture().numberOfProcessors(); ++proc) {
        if (currentTimeAtProcessor[proc] > makespan) {
            makespan = currentTimeAtProcessor[proc];
        }
    }

    return makespan;
}

template <typename GraphT>
void PebblingSchedule<GraphT>::CleanSchedule() {
    if (!isValid()) {
        return;
    }

    // NOTE - this function removes unnecessary steps in most cases, but not all (some require e.g. multiple iterations)

    std::vector<std::vector<std::deque<bool>>> needed(instance_->numberOfVertices(),
                                                      std::vector<std::deque<bool>>(instance_->numberOfProcessors()));
    std::vector<std::vector<bool>> keepFalse(instance_->numberOfVertices(),
                                             std::vector<bool>(instance_->numberOfProcessors(), false));
    std::vector<std::vector<bool>> hasRedAfterCleaning(instance_->numberOfVertices(),
                                                       std::vector<bool>(instance_->numberOfProcessors(), false));

    std::vector<bool> everNeededAsBlue(instance_->numberOfVertices(), false);
    if (needsBlueAtEnd_.empty()) {
        for (VertexIdx node = 0; node < instance_->numberOfVertices(); ++node) {
            if (instance_->getComputationalDag().OutDegree(node) == 0) {
                everNeededAsBlue[node] = true;
            }
        }
    } else {
        for (VertexIdx node : needsBlueAtEnd_) {
            everNeededAsBlue[node] = true;
        }
    }

    for (unsigned step = 0; step < numberOfSupersteps_; ++step) {
        for (unsigned proc = 0; proc < instance_->numberOfProcessors(); ++proc) {
            for (VertexIdx node : nodesSentDown_[proc][step]) {
                everNeededAsBlue[node] = true;
            }
        }
    }

    if (!hasRedInBeginning_.empty()) {
        for (unsigned proc = 0; proc < instance_->numberOfProcessors(); ++proc) {
            for (VertexIdx node : hasRedInBeginning_[proc]) {
                hasRedAfterCleaning[node][proc] = true;
            }
        }
    }

    for (unsigned step = 0; step < numberOfSupersteps_; ++step) {
        // compute phase
        for (unsigned proc = 0; proc < instance_->getArchitecture().numberOfProcessors(); ++proc) {
            for (const auto &computeStep : computeStepsForProcSuperstep_[proc][step]) {
                VertexIdx node = computeStep.node;
                needed[node][proc].emplace_back(false);
                keepFalse[node][proc] = hasRedAfterCleaning[node][proc];
                for (VertexIdx pred : instance_->getComputationalDag().Parents(node)) {
                    hasRedAfterCleaning[pred][proc] = true;
                    if (!keepFalse[pred][proc]) {
                        needed[pred][proc].back() = true;
                    }
                }
                for (VertexIdx toEvict : computeStep.nodes_evicted_after) {
                    hasRedAfterCleaning[toEvict][proc] = false;
                }
            }
        }

        // send up phase
        for (unsigned proc = 0; proc < instance_->getArchitecture().numberOfProcessors(); ++proc) {
            for (VertexIdx node : nodesSentUp_[proc][step]) {
                if (everNeededAsBlue[node]) {
                    hasRedAfterCleaning[node][proc] = true;
                    if (!keepFalse[node][proc]) {
                        needed[node][proc].back() = true;
                    }
                }
            }
        }

        for (unsigned proc = 0; proc < instance_->getArchitecture().numberOfProcessors(); ++proc) {
            for (VertexIdx node : nodesEvictedInComm_[proc][step]) {
                hasRedAfterCleaning[node][proc] = false;
            }
        }

        // send down phase
        for (unsigned proc = 0; proc < instance_->getArchitecture().numberOfProcessors(); ++proc) {
            for (VertexIdx node : nodesSentDown_[proc][step]) {
                needed[node][proc].emplace_back(false);
                keepFalse[node][proc] = hasRedAfterCleaning[node][proc];
            }
        }
    }

    std::vector<std::vector<std::vector<ComputeStep>>> newComputeStepsForProcSuperstep(
        instance_->numberOfProcessors(), std::vector<std::vector<ComputeStep>>(numberOfSupersteps_));
    std::vector<std::vector<std::vector<VertexIdx>>> newNodesEvictedInComm(
        instance_->numberOfProcessors(), std::vector<std::vector<VertexIdx>>(numberOfSupersteps_));
    std::vector<std::vector<std::vector<VertexIdx>>> newNodesSentDown(instance_->numberOfProcessors(),
                                                                      std::vector<std::vector<VertexIdx>>(numberOfSupersteps_));
    std::vector<std::vector<std::vector<VertexIdx>>> newNodesSentUp(instance_->numberOfProcessors(),
                                                                    std::vector<std::vector<VertexIdx>>(numberOfSupersteps_));

    std::vector<std::vector<bool>> hasRed(instance_->numberOfVertices(), std::vector<bool>(instance_->numberOfProcessors(), false));
    if (!hasRedInBeginning_.empty()) {
        for (unsigned proc = 0; proc < instance_->numberOfProcessors(); ++proc) {
            for (VertexIdx node : hasRedInBeginning_[proc]) {
                hasRed[node][proc] = true;
            }
        }
    }

    std::vector<bool> hasBlue(instance_->numberOfVertices());
    std::vector<CostType> timeWhenNodeGetsBlue(instance_->getComputationalDag().NumVertices(),
                                               std::numeric_limits<CostType>::max());
    if (needToLoadInputs_) {
        for (VertexIdx node = 0; node < instance_->numberOfVertices(); ++node) {
            if (instance_->getComputationalDag().InDegree(node) == 0) {
                hasBlue[node] = true;
                timeWhenNodeGetsBlue[node] = 0;
            }
        }
    }

    std::vector<CostType> currentTimeAtProcessor(instance_->getArchitecture().numberOfProcessors(), 0);

    for (unsigned superstep = 0; superstep < numberOfSupersteps_; ++superstep) {
        // compute phase
        for (unsigned proc = 0; proc < instance_->getArchitecture().numberOfProcessors(); ++proc) {
            std::vector<bool> stepRemains(computeStepsForProcSuperstep_[proc][superstep].size(), false);
            std::vector<std::vector<VertexIdx>> newEvictAfter(computeStepsForProcSuperstep_[proc][superstep].size());

            unsigned newStepIndex = 0;
            for (unsigned stepIndex = 0; stepIndex < computeStepsForProcSuperstep_[proc][superstep].size(); ++stepIndex) {
                VertexIdx node = computeStepsForProcSuperstep_[proc][superstep][stepIndex].node;

                if (needed[node][proc].front()) {
                    newComputeStepsForProcSuperstep[proc][superstep].emplace_back(node, std::vector<VertexIdx>());
                    stepRemains[stepIndex] = true;
                    hasRed[node][proc] = true;
                    ++newStepIndex;
                    currentTimeAtProcessor[proc] += instance_->getComputationalDag().VertexWorkWeight(node);
                }

                needed[node][proc].pop_front();

                for (VertexIdx toEvict : computeStepsForProcSuperstep_[proc][superstep][stepIndex].nodes_evicted_after) {
                    if (hasRed[toEvict][proc]) {
                        newEvictAfter[stepIndex].push_back(toEvict);
                    }
                    hasRed[toEvict][proc] = false;
                }
            }

            // go backwards to fix cache eviction steps
            std::vector<VertexIdx> toEvict;
            for (size_t stepIndex = computeStepsForProcSuperstep_[proc][superstep].size() - 1;
                 stepIndex < computeStepsForProcSuperstep_[proc][superstep].size();
                 --stepIndex) {
                for (VertexIdx node : newEvictAfter[stepIndex]) {
                    toEvict.push_back(node);
                }

                if (stepRemains[stepIndex]) {
                    newComputeStepsForProcSuperstep[proc][superstep][newStepIndex - 1].nodes_evicted_after = toEvict;
                    toEvict.clear();
                    --newStepIndex;
                }
            }
            if (!toEvict.empty() && superstep >= 1) {
                for (VertexIdx node : toEvict) {
                    auto itr = std::find(
                        newNodesSentDown[proc][superstep - 1].begin(), newNodesSentDown[proc][superstep - 1].end(), node);
                    if (itr == newNodesSentDown[proc][superstep - 1].end()) {
                        newNodesEvictedInComm[proc][superstep - 1].push_back(node);
                    } else {
                        newNodesSentDown[proc][superstep - 1].erase(itr);
                    }
                }
            }
        }
        for (unsigned proc = 0; proc < instance_->getArchitecture().numberOfProcessors(); ++proc) {
            // send up phase
            for (VertexIdx node : nodesSentUp_[proc][superstep]) {
                if (!everNeededAsBlue[node]) {
                    continue;
                }

                CostType newTimeAtProcessor = currentTimeAtProcessor[proc]
                                              + instance_->getComputationalDag().VertexCommWeight(node)
                                                    * instance_->getArchitecture().communicationCosts();

                // only copy send up step if it is not obsolete in at least one of the two cases (sync or async schedule)
                if (!hasBlue[node] || newTimeAtProcessor < timeWhenNodeGetsBlue[node]) {
                    newNodesSentUp[proc][superstep].push_back(node);
                    hasBlue[node] = true;
                    currentTimeAtProcessor[proc] = newTimeAtProcessor;
                    if (timeWhenNodeGetsBlue[node] > newTimeAtProcessor) {
                        timeWhenNodeGetsBlue[node] = newTimeAtProcessor;
                    }
                }
            }
        }

        // comm phase evict
        for (unsigned proc = 0; proc < instance_->getArchitecture().numberOfProcessors(); ++proc) {
            for (VertexIdx node : nodesEvictedInComm_[proc][superstep]) {
                if (hasRed[node][proc]) {
                    newNodesEvictedInComm[proc][superstep].push_back(node);
                    hasRed[node][proc] = false;
                }
            }
        }

        for (unsigned proc = 0; proc < instance_->getArchitecture().numberOfProcessors(); ++proc) {
            // send down phase
            for (VertexIdx node : nodesSentDown_[proc][superstep]) {
                if (needed[node][proc].front()) {
                    newNodesSentDown[proc][superstep].push_back(node);
                    hasRed[node][proc] = true;
                    if (currentTimeAtProcessor[proc] < timeWhenNodeGetsBlue[node]) {
                        currentTimeAtProcessor[proc] = timeWhenNodeGetsBlue[node];
                    }
                    currentTimeAtProcessor[proc] += instance_->getComputationalDag().VertexCommWeight(node)
                                                    * instance_->getArchitecture().communicationCosts();
                }
                needed[node][proc].pop_front();
            }
        }
    }

    computeStepsForProcSuperstep_ = newComputeStepsForProcSuperstep;
    nodesEvictedInComm_ = newNodesEvictedInComm;
    nodesSentDown_ = newNodesSentDown;
    nodesSentUp_ = newNodesSentUp;
}

template <typename GraphT>
void PebblingSchedule<GraphT>::ConvertFromBsp(const BspSchedule<GraphT> &schedule, CACHE_EVICTION_STRATEGY evict_rule) {
    instance_ = &schedule.getInstance();

    // check if conversion possible at all
    if (!HasValidSolution(schedule.getInstance(), externalSources_)) {
        std::cout << "Conversion failed." << std::endl;
        return;
    }

    // split supersteps
    SplitSupersteps(schedule);

    // track memory
    SetMemoryMovement(evict_rule);
}

template <typename GraphT>
bool PebblingSchedule<GraphT>::HasValidSolution(const BspInstance<GraphT> &instance, const std::set<VertexIdx> &external_sources) {
    std::vector<MemweightType> memoryRequired = MinimumMemoryRequiredPerNodeType(instance);
    std::vector<bool> hasEnoughMemory(instance.GetComputationalDag().NumVertexTypes(), true);
    for (VertexIdx node = 0; node < instance.NumberOfVertices(); ++node) {
        if (external_sources.find(node) == external_sources.end()) {
            hasEnoughMemory[instance.GetComputationalDag().VertexType(node)] = false;
        }
    }

    for (VTypeT<GraphT> nodeType = 0; nodeType < instance.GetComputationalDag().NumVertexTypes(); ++nodeType) {
        for (unsigned proc = 0; proc < instance.NumberOfProcessors(); ++proc) {
            if (instance.IsCompatibleType(nodeType, instance.GetArchitecture().ProcessorType(proc))
                && instance.GetArchitecture().MemoryBound(proc) >= memoryRequired[nodeType]) {
                hasEnoughMemory[nodeType] = true;
                break;
            }
        }
    }

    for (VTypeT<GraphT> nodeType = 0; nodeType < instance.GetComputationalDag().NumVertexTypes(); ++nodeType) {
        if (!hasEnoughMemory[nodeType]) {
            std::cout << "No valid solution exists. Minimum memory required for node type " << nodeType << " is "
                      << memoryRequired[nodeType] << std::endl;
            return false;
        }
    }
    return true;
}

template <typename GraphT>
void PebblingSchedule<GraphT>::SplitSupersteps(const BspSchedule<GraphT> &schedule) {
    // get DFS topological order in each superstep
    std::vector<std::vector<std::vector<VertexIdx>>> topOrders = ComputeTopOrdersDfs(schedule);

    std::vector<unsigned> topOrderIdx(instance_->GetComputationalDag().NumVertices(), 0);
    for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
        for (unsigned step = 0; step < schedule.numberOfSupersteps(); ++step) {
            for (unsigned idx = 0; idx < topOrders[proc][step].size(); ++idx) {
                topOrderIdx[topOrders[proc][step][idx]] = idx;
            }
        }
    }

    // split supersteps as needed
    std::vector<unsigned> newSuperstepId(instance_->GetComputationalDag().NumVertices());
    unsigned superstepIndex = 0;
    for (unsigned step = 0; step < schedule.numberOfSupersteps(); ++step) {
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
                unsigned endLowerBound = startIdx, endUpperBound = static_cast<unsigned>(topOrders[proc][step].size() - 1);
                while (endLowerBound < endUpperBound) {
                    unsigned endCurrent;

                    if (doublingPhase) {
                        if (endLowerBound == startIdx) {
                            endCurrent = startIdx + 1;
                        } else {
                            endCurrent = std::min(startIdx + 2 * (endLowerBound - startIdx),
                                                  static_cast<unsigned>(topOrders[proc][step].size()) - 1);
                        }
                    } else {
                        endCurrent = endLowerBound + (endUpperBound - endLowerBound + 1) / 2;
                    }

                    // check if this segment is valid
                    bool valid = true;

                    std::map<VertexIdx, bool> neededAfter;
                    for (unsigned idx = startIdx; idx <= endCurrent; ++idx) {
                        VertexIdx node = topOrders[proc][step][idx];
                        neededAfter[node] = false;
                        if (needsBlueAtEnd_.empty()) {
                            neededAfter[node] = (instance_->GetComputationalDag().OutDegree(node) == 0);
                        } else {
                            neededAfter[node] = (needsBlueAtEnd_.find(node) != needsBlueAtEnd_.end());
                        }
                        for (VertexIdx succ : instance_->GetComputationalDag().Children(node)) {
                            if (schedule.AssignedSuperstep(succ) > step) {
                                neededAfter[node] = true;
                            }
                            if (schedule.AssignedSuperstep(succ) == step && topOrderIdx[succ] <= endCurrent) {
                                neededAfter[node] = true;
                            }
                        }
                    }

                    std::map<VertexIdx, VertexIdx> lastUsedBy;
                    std::set<VertexIdx> valuesNeeded;
                    for (unsigned idx = startIdx; idx <= endCurrent; ++idx) {
                        VertexIdx node = topOrders[proc][step][idx];
                        for (VertexIdx pred : instance_->GetComputationalDag().Parents(node)) {
                            if (schedule.AssignedSuperstep(pred) < step
                                || (schedule.AssignedSuperstep(pred) == step && !neededAfter[pred])) {
                                lastUsedBy[pred] = node;
                            }
                            if (schedule.AssignedSuperstep(pred) < step
                                || (schedule.AssignedSuperstep(pred) == step && topOrderIdx[pred] < startIdx)
                                || (needToLoadInputs_ && instance_->GetComputationalDag().InDegree(pred) == 0)
                                || externalSources_.find(pred) != externalSources_.end()) {
                                valuesNeeded.insert(pred);
                            }
                        }
                    }

                    MemweightType memNeeded = 0;
                    for (VertexIdx node : valuesNeeded) {
                        memNeeded += instance_->GetComputationalDag().VertexMemWeight(node);
                    }

                    for (unsigned idx = startIdx; idx <= endCurrent; ++idx) {
                        VertexIdx node = topOrders[proc][step][idx];

                        if (needToLoadInputs_ && instance_->GetComputationalDag().InDegree(node) == 0) {
                            continue;
                        }

                        memNeeded += instance_->GetComputationalDag().VertexMemWeight(node);
                        if (memNeeded > instance_->GetArchitecture().MemoryBound(proc)) {
                            valid = false;
                            break;
                        }

                        for (VertexIdx pred : instance_->GetComputationalDag().Parents(node)) {
                            if (lastUsedBy[pred] == node) {
                                memNeeded -= instance_->GetComputationalDag().VertexMemWeight(pred);
                            }
                        }
                    }

                    if (valid) {
                        endLowerBound = endCurrent;
                        if (endCurrent == topOrders[proc][step].size() - 1) {
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
                    newSuperstepId[topOrders[proc][step][idx]] = superstepIndex + stepIdx;
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
    for (VertexIdx node = 0; node < instance_->NumberOfVertices(); ++node) {
        if (!needToLoadInputs_ || instance_->GetComputationalDag().InDegree(node) > 0) {
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
    std::cout << schedule.numberOfSupersteps() << " -> " << numberOfSupersteps_ << std::endl;

    // TODO: might not need offset for first step when beginning with red pebbles

    for (unsigned step = 0; step < schedule.numberOfSupersteps(); ++step) {
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
            for (VertexIdx node : topOrders[proc][step]) {
                if (!needToLoadInputs_ || instance_->GetComputationalDag().InDegree(node) > 0) {
                    computeStepsForProcSuperstep_[proc][reindexToShrink[newSuperstepId[node]] + offset].emplace_back(node);
                }
            }
        }
    }
}

template <typename GraphT>
void PebblingSchedule<GraphT>::SetMemoryMovement(CACHE_EVICTION_STRATEGY evict_rule) {
    const size_t n = instance_->GetComputationalDag().NumVertices();

    std::vector<MemweightType> memUsed(instance_->NumberOfProcessors(), 0);
    std::vector<std::set<VertexIdx>> inMem(instance_->NumberOfProcessors());

    std::vector<bool> inSlowMem(n, false);
    if (needToLoadInputs_) {
        for (VertexIdx node = 0; node < n; ++node) {
            if (instance_->GetComputationalDag().InDegree(node) == 0) {
                inSlowMem[node] = true;
            }
        }
    }

    std::vector<std::set<std::pair<std::pair<unsigned, unsigned>, VertexIdx>>> evictable(instance_->NumberOfProcessors());
    std::vector<std::set<VertexIdx>> nonEvictable(instance_->NumberOfProcessors());

    // iterator to its position in "evictable" - for efficient delete
    std::vector<std::vector<decltype(evictable[0].begin())>> placeInEvictable(
        n, std::vector<decltype(evictable[0].begin())>(instance_->NumberOfProcessors()));
    for (VertexIdx node = 0; node < n; ++node) {
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
            placeInEvictable[node][proc] = evictable[proc].end();
        }
    }

    // utility for LRU eviction strategy
    std::vector<std::vector<unsigned>> nodeLastUsedOnProc;
    if (evict_rule == CACHE_EVICTION_STRATEGY::LEAST_RECENTLY_USED) {
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
                VertexIdx node = computeStepsForProcSuperstep_[proc][superstep][stepIndex].node;
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
                VertexIdx node = computeStepsForProcSuperstep_[proc][superstep][stepIndex].node;
                computedInCurrentSuperstep[node] = true;
                for (VertexIdx pred : instance_->GetComputationalDag().Parents(node)) {
                    if (!computedInCurrentSuperstep[pred]) {
                        mustBePreserved[pred] = true;
                    }
                }
            }
            for (unsigned stepIndex = 0; stepIndex < computeStepsForProcSuperstep_[proc][superstep].size(); ++stepIndex) {
                computedInCurrentSuperstep[computeStepsForProcSuperstep_[proc][superstep][stepIndex].node] = false;
            }
        }
    }
    if (needsBlueAtEnd_.empty()) {
        for (VertexIdx node = 0; node < n; ++node) {
            if (instance_->GetComputationalDag().OutDegree(node) == 0) {
                mustBePreserved[node] = true;
            }
        }
    } else {
        for (VertexIdx node : needsBlueAtEnd_) {
            mustBePreserved[node] = true;
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
                VertexIdx node = computeStepsForProcSuperstep_[proc][superstep][stepIndex].node;
                for (VertexIdx pred : instance_->GetComputationalDag().Parents(node)) {
                    nodeUsedAtProcLists[pred][proc].back().emplace_back(superstep, stepIndex);
                }

                nodeUsedAtProcLists[node][proc].emplace_back();
            }
        }
    }

    // set up initial content of fast memories
    if (!hasRedInBeginning_.empty()) {
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
            inMem = hasRedInBeginning_;
            for (VertexIdx node : inMem[proc]) {
                memUsed[proc] += instance_->GetComputationalDag().VertexMemWeight(node);

                std::pair<unsigned, unsigned> prio;
                if (evict_rule == CACHE_EVICTION_STRATEGY::FORESIGHT) {
                    prio = nodeUsedAtProcLists[node][proc].front().front();
                } else if (evict_rule == CACHE_EVICTION_STRATEGY::LEAST_RECENTLY_USED) {
                    prio = std::make_pair(UINT_MAX - nodeLastUsedOnProc[node][proc], static_cast<unsigned>(node));
                } else if (evict_rule == CACHE_EVICTION_STRATEGY::LARGEST_ID) {
                    prio = std::make_pair(static_cast<unsigned>(node), 0);
                }

                placeInEvictable[node][proc] = evictable[proc].emplace(prio, node).first;
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
            std::set<VertexIdx> newValuesNeeded;
            for (unsigned stepIndex = 0; stepIndex < computeStepsForProcSuperstep_[proc][superstep].size(); ++stepIndex) {
                VertexIdx node = computeStepsForProcSuperstep_[proc][superstep][stepIndex].node;
                computedInCurrentSuperstep[node] = true;
                for (VertexIdx pred : instance_->GetComputationalDag().Parents(node)) {
                    if (!computedInCurrentSuperstep[pred]) {
                        nonEvictable[proc].insert(pred);

                        if (placeInEvictable[pred][proc] != evictable[proc].end()) {
                            evictable[proc].erase(placeInEvictable[pred][proc]);
                            placeInEvictable[pred][proc] = evictable[proc].end();
                        }

                        if (inMem[proc].find(pred) == inMem[proc].end()) {
                            newValuesNeeded.insert(pred);
                        }
                    }
                }
            }
            for (unsigned stepIndex = 0; stepIndex < computeStepsForProcSuperstep_[proc][superstep].size(); ++stepIndex) {
                computedInCurrentSuperstep[computeStepsForProcSuperstep_[proc][superstep][stepIndex].node] = false;
            }

            for (VertexIdx node : newValuesNeeded) {
                inMem[proc].insert(node);
                memUsed[proc] += instance_->GetComputationalDag().VertexMemWeight(node);
                nodesSentDown_[proc][superstep - 1].push_back(node);
                if (!inSlowMem[node]) {
                    inSlowMem[node] = true;
                    nodesSentUp_[selectedProcessor[node]][selectedStep[node].first].push_back(node);
                }
            }

            MemweightType firstNodeWeight
                = instance_->GetComputationalDag().VertexMemWeight(computeStepsForProcSuperstep_[proc][superstep][0].node);

            while (memUsed[proc] + firstNodeWeight
                   > instance_->GetArchitecture().MemoryBound(proc))    // no sliding pebbles for now
            {
                if (evictable[proc].empty()) {
                    std::cout << "ERROR: Cannot create valid memory movement for these superstep lists." << std::endl;
                    return;
                }
                VertexIdx evicted = (--evictable[proc].end())->second;
                evictable[proc].erase(--evictable[proc].end());
                placeInEvictable[evicted][proc] = evictable[proc].end();

                memUsed[proc] -= instance_->GetComputationalDag().VertexMemWeight(evicted);
                inMem[proc].erase(evicted);

                nodesEvictedInComm_[proc][superstep - 1].push_back(evicted);
            }

            // indicates if the node will be needed after (and thus cannot be deleted during) this compute phase
            std::map<VertexIdx, bool> neededAfter;

            // during compute phase
            for (unsigned stepIndex = 0; stepIndex < computeStepsForProcSuperstep_[proc][superstep].size(); ++stepIndex) {
                VertexIdx node = computeStepsForProcSuperstep_[proc][superstep][stepIndex].node;
                MemweightType nodeWeight = instance_->GetComputationalDag().VertexMemWeight(node);

                if (stepIndex > 0) {
                    // evict nodes to make space
                    while (memUsed[proc] + nodeWeight > instance_->GetArchitecture().MemoryBound(proc)) {
                        if (evictable[proc].empty()) {
                            std::cout << "ERROR: Cannot create valid memory movement for these superstep lists." << std::endl;
                            return;
                        }
                        VertexIdx evicted = (--evictable[proc].end())->second;
                        evictable[proc].erase(--evictable[proc].end());
                        placeInEvictable[evicted][proc] = evictable[proc].end();

                        memUsed[proc] -= instance_->GetComputationalDag().VertexMemWeight(evicted);
                        inMem[proc].erase(evicted);

                        computeStepsForProcSuperstep_[proc][superstep][stepIndex - 1].nodesEvictedAfter.push_back(evicted);
                    }
                }

                inMem[proc].insert(node);
                memUsed[proc] += nodeWeight;

                nonEvictable[proc].insert(node);

                if (evict_rule == CACHE_EVICTION_STRATEGY::LEAST_RECENTLY_USED)    // update usage times for LRU strategy
                {
                    ++totalStepCountOnProc[proc];
                    nodeLastUsedOnProc[node][proc] = totalStepCountOnProc[proc];
                    for (VertexIdx pred : instance_->GetComputationalDag().Parents(node)) {
                        nodeLastUsedOnProc[pred][proc] = totalStepCountOnProc[proc];
                    }
                }

                if (selectedProcessor[node] == proc && selectedStep[node] == std::make_pair(superstep, stepIndex)
                    && mustBePreserved[node]) {
                    neededAfter[node] = true;
                } else {
                    neededAfter[node] = false;
                }

                nodeUsedAtProcLists[node][proc].pop_front();

                for (VertexIdx pred : instance_->GetComputationalDag().Parents(node)) {
                    nodeUsedAtProcLists[pred][proc].front().pop_front();

                    if (neededAfter[pred]) {
                        continue;
                    }

                    // autoevict
                    if (nodeUsedAtProcLists[pred][proc].front().empty()) {
                        inMem[proc].erase(pred);
                        nonEvictable[proc].erase(pred);
                        memUsed[proc] -= instance_->GetComputationalDag().VertexMemWeight(pred);
                        computeStepsForProcSuperstep_[proc][superstep][stepIndex].nodesEvictedAfter.push_back(pred);
                    } else if (nodeUsedAtProcLists[pred][proc].front().front().first > superstep) {
                        nonEvictable[proc].erase(pred);

                        std::pair<unsigned, unsigned> prio;
                        if (evict_rule == CACHE_EVICTION_STRATEGY::FORESIGHT) {
                            prio = nodeUsedAtProcLists[pred][proc].front().front();
                        } else if (evict_rule == CACHE_EVICTION_STRATEGY::LEAST_RECENTLY_USED) {
                            prio = std::make_pair(UINT_MAX - nodeLastUsedOnProc[pred][proc], static_cast<unsigned>(pred));
                        } else if (evict_rule == CACHE_EVICTION_STRATEGY::LARGEST_ID) {
                            prio = std::make_pair(static_cast<unsigned>(pred), 0);
                        }

                        placeInEvictable[pred][proc] = evictable[proc].emplace(prio, pred).first;
                    }
                }
            }

            // after compute phase
            for (VertexIdx node : nonEvictable[proc]) {
                if (nodeUsedAtProcLists[node][proc].front().empty()) {
                    memUsed[proc] -= instance_->GetComputationalDag().VertexMemWeight(node);
                    inMem[proc].erase(node);
                    nodesEvictedInComm_[proc][superstep].push_back(node);
                    if ((instance_->GetComputationalDag().OutDegree(node) == 0
                         || needsBlueAtEnd_.find(node) != needsBlueAtEnd_.end())
                        && !inSlowMem[node]) {
                        inSlowMem[node] = true;
                        nodesSentUp_[proc][superstep].push_back(node);
                    }
                } else {
                    std::pair<unsigned, unsigned> prio;
                    if (evict_rule == CACHE_EVICTION_STRATEGY::FORESIGHT) {
                        prio = nodeUsedAtProcLists[node][proc].front().front();
                    } else if (evict_rule == CACHE_EVICTION_STRATEGY::LEAST_RECENTLY_USED) {
                        prio = std::make_pair(UINT_MAX - nodeLastUsedOnProc[node][proc], static_cast<unsigned>(node));
                    } else if (evict_rule == CACHE_EVICTION_STRATEGY::LARGEST_ID) {
                        prio = std::make_pair(static_cast<unsigned>(node), 0);
                    }

                    placeInEvictable[node][proc] = evictable[proc].emplace(prio, node).first;

                    if (needsBlueAtEnd_.find(node) != needsBlueAtEnd_.end() && !inSlowMem[node]) {
                        inSlowMem[node] = true;
                        nodesSentUp_[proc][superstep].push_back(node);
                    }
                }
            }
            nonEvictable[proc].clear();
        }
    }
}

template <typename GraphT>
void PebblingSchedule<GraphT>::ResetToForesight() {
    nodesEvictedInComm_.clear();
    nodesEvictedInComm_.resize(instance_->numberOfProcessors(), std::vector<std::vector<VertexIdx>>(numberOfSupersteps_));

    nodesSentDown_.clear();
    nodesSentDown_.resize(instance_->numberOfProcessors(), std::vector<std::vector<VertexIdx>>(numberOfSupersteps_));

    nodesSentUp_.clear();
    nodesSentUp_.resize(instance_->numberOfProcessors(), std::vector<std::vector<VertexIdx>>(numberOfSupersteps_));

    SetMemoryMovement(CACHE_EVICTION_STRATEGY::FORESIGHT);
}

template <typename GraphT>
bool PebblingSchedule<GraphT>::isValid() const {
    std::vector<MemweightType> memUsed(instance_->NumberOfProcessors(), 0);
    std::vector<std::vector<VertexIdx>> inFastMem(instance_->GetComputationalDag().NumVertices(),
                                                  std::vector<VertexIdx>(instance_->NumberOfProcessors(), false));
    std::vector<VertexIdx> inSlowMem(instance_->GetComputationalDag().NumVertices(), false);

    if (needToLoadInputs_) {
        for (VertexIdx node = 0; node < instance_->NumberOfVertices(); ++node) {
            if (instance_->GetComputationalDag().InDegree(node) == 0) {
                inSlowMem[node] = true;
            }
        }
    }

    if (!hasRedInBeginning_.empty()) {
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
            for (VertexIdx node : hasRedInBeginning_[proc]) {
                memUsed[proc] += instance_->GetComputationalDag().VertexMemWeight(node);
                inFastMem[node][proc] = true;
            }
        }
    }

    for (unsigned step = 0; step < numberOfSupersteps_; ++step) {
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
            // computation phase
            for (const auto &computeStep : computeStepsForProcSuperstep_[proc][step]) {
                if (!instance_->IsCompatible(computeStep.node, proc)) {
                    return false;
                }

                for (VertexIdx pred : instance_->GetComputationalDag().Parents(computeStep.node)) {
                    if (!inFastMem[pred][proc]) {
                        return false;
                    }
                }

                if (needToLoadInputs_ && instance_->GetComputationalDag().InDegree(computeStep.node) == 0) {
                    return false;
                }

                if (!inFastMem[computeStep.node][proc]) {
                    inFastMem[computeStep.node][proc] = true;
                    memUsed[proc] += instance_->GetComputationalDag().VertexMemWeight(computeStep.node);
                }

                if (memUsed[proc] > instance_->GetArchitecture().MemoryBound(proc)) {
                    return false;
                }

                for (VertexIdx toRemove : computeStep.nodesEvictedAfter) {
                    if (!inFastMem[toRemove][proc]) {
                        return false;
                    }

                    inFastMem[toRemove][proc] = false;
                    memUsed[proc] -= instance_->GetComputationalDag().VertexMemWeight(toRemove);
                }
            }

            // communication phase - sendup and eviction
            for (VertexIdx node : nodesSentUp_[proc][step]) {
                if (!inFastMem[node][proc]) {
                    return false;
                }

                inSlowMem[node] = true;
            }
            for (VertexIdx node : nodesEvictedInComm_[proc][step]) {
                if (!inFastMem[node][proc]) {
                    return false;
                }

                inFastMem[node][proc] = false;
                memUsed[proc] -= instance_->GetComputationalDag().VertexMemWeight(node);
            }
        }

        // communication phase - senddown
        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
            for (VertexIdx node : nodesSentDown_[proc][step]) {
                if (!inSlowMem[node]) {
                    return false;
                }

                if (!inFastMem[node][proc]) {
                    inFastMem[node][proc] = true;
                    memUsed[proc] += instance_->GetComputationalDag().VertexMemWeight(node);
                }
            }
        }

        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
            if (memUsed[proc] > instance_->GetArchitecture().MemoryBound(proc)) {
                return false;
            }
        }
    }

    if (needsBlueAtEnd_.empty()) {
        for (VertexIdx node = 0; node < instance_->NumberOfVertices(); ++node) {
            if (instance_->GetComputationalDag().OutDegree(node) == 0 && !inSlowMem[node]) {
                return false;
            }
        }
    } else {
        for (VertexIdx node : needsBlueAtEnd_) {
            if (!inSlowMem[node]) {
                return false;
            }
        }
    }

    return true;
}

template <typename GraphT>
std::vector<VMemwT<GraphT>> PebblingSchedule<GraphT>::MinimumMemoryRequiredPerNodeType(const BspInstance<GraphT> &instance,
                                                                                       const std::set<VertexIdx> &external_sources) {
    std::vector<VMemwT<GraphT>> maxNeeded(instance.GetComputationalDag().NumVertexTypes(), 0);
    for (VertexIdxT<GraphT> node = 0; node < instance.GetComputationalDag().NumVertices(); ++node) {
        if (external_sources.find(node) != external_sources.end()) {
            continue;
        }

        VMemwT<GraphT> needed = instance.GetComputationalDag().VertexMemWeight(node);
        const VTypeT<GraphT> type = instance.GetComputationalDag().VertexType(node);
        for (VertexIdxT<GraphT> pred : instance.GetComputationalDag().Parents(node)) {
            needed += instance.GetComputationalDag().VertexMemWeight(pred);
        }

        if (needed > maxNeeded[type]) {
            maxNeeded[type] = needed;
        }
    }
    return maxNeeded;
}

template <typename GraphT>
std::vector<std::vector<std::vector<VertexIdxT<GraphT>>>> PebblingSchedule<GraphT>::ComputeTopOrdersDfs(
    const BspSchedule<GraphT> &schedule) const {
    size_t n = schedule.getInstance().GetComputationalDag().NumVertices();
    unsigned numProcs = schedule.getInstance().NumberOfProcessors();
    unsigned numSupsteps = schedule.numberOfSupersteps();

    std::vector<std::vector<std::vector<VertexIdx>>> topOrders(numProcs, std::vector<std::vector<VertexIdx>>(numSupsteps));

    std::vector<std::vector<std::deque<VertexIdx>>> q(numProcs, std::vector<std::deque<VertexIdx>>(numSupsteps));
    std::vector<std::vector<std::vector<VertexIdx>>> nodesUpdated(numProcs, std::vector<std::vector<VertexIdx>>(numSupsteps));
    std::vector<unsigned> nrPred(n);
    std::vector<unsigned> predDone(n, 0);
    for (VertexIdx node = 0; node < n; ++node) {
        unsigned predecessors = 0;
        for (VertexIdx pred : schedule.getInstance().GetComputationalDag().Parents(node)) {
            if (externalSources_.find(pred) == externalSources_.end()
                && schedule.AssignedProcessor(node) == schedule.AssignedProcessor(pred)
                && schedule.AssignedSuperstep(node) == schedule.AssignedSuperstep(pred)) {
                ++predecessors;
            }
        }
        nrPred[node] = predecessors;
        if (predecessors == 0 && externalSources_.find(node) == externalSources_.end()) {
            q[schedule.AssignedProcessor(node)][schedule.AssignedSuperstep(node)].push_back(node);
        }
    }
    for (unsigned proc = 0; proc < numProcs; ++proc) {
        for (unsigned step = 0; step < numSupsteps; ++step) {
            while (!q[proc][step].empty()) {
                VertexIdx node = q[proc][step].front();
                q[proc][step].pop_front();
                topOrders[proc][step].push_back(node);
                for (VertexIdx succ : schedule.getInstance().GetComputationalDag().Children(node)) {
                    if (schedule.AssignedProcessor(node) == schedule.AssignedProcessor(succ)
                        && schedule.AssignedSuperstep(node) == schedule.AssignedSuperstep(succ)) {
                        ++predDone[succ];
                        if (predDone[succ] == nrPred[succ]) {
                            q[proc][step].push_front(succ);
                        }
                    }
                }
            }
        }
    }

    return topOrders;
}

template <typename GraphT>
void PebblingSchedule<GraphT>::GetDataForMultiprocessorPebbling(
    std::vector<std::vector<std::vector<VertexIdx>>> &computeSteps,
    std::vector<std::vector<std::vector<VertexIdx>>> &sendUpSteps,
    std::vector<std::vector<std::vector<VertexIdx>>> &sendDownSteps,
    std::vector<std::vector<std::vector<VertexIdx>>> &nodesEvictedAfterStep) const {
    computeSteps.clear();
    computeSteps.resize(instance_->numberOfProcessors());
    sendUpSteps.clear();
    sendUpSteps.resize(instance_->numberOfProcessors());
    sendDownSteps.clear();
    sendDownSteps.resize(instance_->numberOfProcessors());
    nodesEvictedAfterStep.clear();
    nodesEvictedAfterStep.resize(instance_->numberOfProcessors());

    std::vector<MemweightType> memUsed(instance_->numberOfProcessors(), 0);
    std::vector<std::set<VertexIdx>> inMem(instance_->numberOfProcessors());
    if (!hasRedInBeginning_.empty()) {
        for (unsigned proc = 0; proc < instance_->numberOfProcessors(); ++proc) {
            for (VertexIdx node : hasRedInBeginning_[proc]) {
                inMem[proc].insert(node);
                memUsed[proc] += instance_->getComputationalDag().VertexMemWeight(node);
            }
        }
    }

    unsigned step = 0;

    for (unsigned superstep = 0; superstep < numberOfSupersteps_; ++superstep) {
        std::vector<unsigned> stepOnProc(instance_->numberOfProcessors(), step);
        bool anyCompute = false;
        for (unsigned proc = 0; proc < instance_->numberOfProcessors(); ++proc) {
            if (!computeStepsForProcSuperstep_[proc][superstep].empty()) {
                anyCompute = true;
            }
        }

        if (anyCompute) {
            for (unsigned proc = 0; proc < instance_->numberOfProcessors(); ++proc) {
                computeSteps[proc].emplace_back();
                sendUpSteps[proc].emplace_back();
                sendDownSteps[proc].emplace_back();
                nodesEvictedAfterStep[proc].emplace_back();
            }
        }

        for (unsigned proc = 0; proc < instance_->numberOfProcessors(); ++proc) {
            std::vector<VertexIdx> evictList;
            for (unsigned stepIndex = 0; stepIndex < computeStepsForProcSuperstep_[proc][superstep].size(); ++stepIndex) {
                VertexIdx node = computeStepsForProcSuperstep_[proc][superstep][stepIndex].node;
                if (memUsed[proc] + instance_->getComputationalDag().VertexMemWeight(node)
                    > instance_->getArchitecture().memoryBound(proc)) {
                    // open new step
                    nodesEvictedAfterStep[proc][stepOnProc[proc]] = evictList;
                    ++stepOnProc[proc];
                    for (VertexIdx toEvict : evictList) {
                        memUsed[proc] -= instance_->getComputationalDag().VertexMemWeight(toEvict);
                    }

                    evictList.clear();
                    computeSteps[proc].emplace_back();
                    sendUpSteps[proc].emplace_back();
                    sendDownSteps[proc].emplace_back();
                    nodesEvictedAfterStep[proc].emplace_back();
                }

                computeSteps[proc][stepOnProc[proc]].emplace_back(node);
                memUsed[proc] += instance_->getComputationalDag().VertexMemWeight(node);
                for (VertexIdx toEvict : computeStepsForProcSuperstep_[proc][superstep][stepIndex].nodes_evicted_after) {
                    evictList.emplace_back(toEvict);
                }
            }

            if (!evictList.empty()) {
                nodesEvictedAfterStep[proc][stepOnProc[proc]] = evictList;
                for (VertexIdx toEvict : evictList) {
                    memUsed[proc] -= instance_->getComputationalDag().VertexMemWeight(toEvict);
                }
            }
        }
        if (anyCompute) {
            for (unsigned proc = 0; proc < instance_->numberOfProcessors(); ++proc) {
                ++stepOnProc[proc];
            }
        }

        for (unsigned proc = 0; proc < instance_->numberOfProcessors(); ++proc) {
            step = std::max(step, stepOnProc[proc]);
        }
        for (unsigned proc = 0; proc < instance_->numberOfProcessors(); ++proc) {
            for (; stepOnProc[proc] < step; ++stepOnProc[proc]) {
                computeSteps[proc].emplace_back();
                sendUpSteps[proc].emplace_back();
                sendDownSteps[proc].emplace_back();
                nodesEvictedAfterStep[proc].emplace_back();
            }
        }

        bool anySendUp = false;
        for (unsigned proc = 0; proc < instance_->numberOfProcessors(); ++proc) {
            if (!nodesSentUp_[proc][superstep].empty() || !nodesEvictedInComm_[proc][superstep].empty()) {
                anySendUp = true;
            }
        }

        if (anySendUp) {
            for (unsigned proc = 0; proc < instance_->numberOfProcessors(); ++proc) {
                computeSteps[proc].emplace_back();
                sendUpSteps[proc].emplace_back(nodesSentUp_[proc][superstep]);
                sendDownSteps[proc].emplace_back();
                nodesEvictedAfterStep[proc].emplace_back(nodesEvictedInComm_[proc][superstep]);
                for (VertexIdx toEvict : nodesEvictedInComm_[proc][superstep]) {
                    memUsed[proc] -= instance_->getComputationalDag().VertexMemWeight(toEvict);
                }
                ++stepOnProc[proc];
            }
            ++step;
        }

        bool anySendDown = false;
        for (unsigned proc = 0; proc < instance_->numberOfProcessors(); ++proc) {
            if (!nodesSentDown_[proc][superstep].empty()) {
                anySendDown = true;
            }
        }

        if (anySendDown) {
            for (unsigned proc = 0; proc < instance_->numberOfProcessors(); ++proc) {
                computeSteps[proc].emplace_back();
                sendUpSteps[proc].emplace_back();
                sendDownSteps[proc].emplace_back(nodesSentDown_[proc][superstep]);
                for (VertexIdx sendDown : nodesSentDown_[proc][superstep]) {
                    memUsed[proc] += instance_->getComputationalDag().VertexMemWeight(sendDown);
                }
                nodesEvictedAfterStep[proc].emplace_back();
                ++stepOnProc[proc];
            }
            ++step;
        }
    }
}

template <typename GraphT>
std::vector<std::set<VertexIdxT<GraphT>>> PebblingSchedule<GraphT>::GetMemContentAtEnd() const {
    std::vector<std::set<VertexIdx>> memContent(instance_->numberOfProcessors());
    if (!hasRedInBeginning_.empty()) {
        memContent = hasRedInBeginning_;
    }

    for (unsigned step = 0; step < numberOfSupersteps_; ++step) {
        for (unsigned proc = 0; proc < instance_->numberOfProcessors(); ++proc) {
            // computation phase
            for (const auto &computeStep : computeStepsForProcSuperstep_[proc][step]) {
                memContent[proc].insert(computeStep.node);
                for (VertexIdx toRemove : computeStep.nodes_evicted_after) {
                    memContent[proc].erase(toRemove);
                }
            }

            // communication phase - eviction
            for (VertexIdx node : nodesEvictedInComm_[proc][step]) {
                memContent[proc].erase(node);
            }

            // communication phase - senddown
            for (VertexIdx node : nodesSentDown_[proc][step]) {
                memContent[proc].insert(node);
            }
        }
    }

    return memContent;
}

template <typename GraphT>
void PebblingSchedule<GraphT>::RemoveEvictStepsFromEnd() {
    std::vector<MemweightType> memUsed(instance_->numberOfProcessors(), 0);
    std::vector<MemweightType> bottleneck(instance_->numberOfProcessors(), 0);
    std::vector<std::set<VertexIdx>> fastMemEnd = GetMemContentAtEnd();
    for (unsigned proc = 0; proc < instance_->numberOfProcessors(); ++proc) {
        for (VertexIdx node : fastMemEnd[proc]) {
            memUsed[proc] += instance_->getComputationalDag().VertexMemWeight(node);
        }

        bottleneck[proc] = instance_->getArchitecture().memoryBound(proc) - memUsed[proc];
    }

    for (unsigned step = numberOfSupersteps_; step > 0;) {
        --step;

        for (unsigned proc = 0; proc < instance_->numberOfProcessors(); ++proc) {
            // communication phase - senddown
            for (VertexIdx node : nodesSentDown_[proc][step]) {
                memUsed[proc] -= instance_->getComputationalDag().VertexMemWeight(node);
            }

            // communication phase - eviction
            std::vector<VertexIdx> remaining;
            for (VertexIdx node : nodesEvictedInComm_[proc][step]) {
                memUsed[proc] += instance_->getComputationalDag().VertexMemWeight(node);
                if (instance_->getComputationalDag().VertexMemWeight(node) <= bottleneck[proc]
                    && fastMemEnd[proc].find(node) == fastMemEnd[proc].end()) {
                    fastMemEnd[proc].insert(node);
                    bottleneck[proc] -= instance_->getComputationalDag().VertexMemWeight(node);
                } else {
                    remaining.push_back(node);
                }
            }
            nodesEvictedInComm_[proc][step] = remaining;
            bottleneck[proc] = std::min(bottleneck[proc], instance_->getArchitecture().memoryBound(proc) - memUsed[proc]);

            // computation phase
            for (unsigned stepIndex = static_cast<unsigned>(computeStepsForProcSuperstep_[proc][step].size()); stepIndex > 0;) {
                --stepIndex;
                auto &computeStep = computeStepsForProcSuperstep_[proc][step][stepIndex];

                std::vector<VertexIdx> remaining2;
                for (VertexIdx toRemove : computeStep.nodes_evicted_after) {
                    memUsed[proc] += instance_->getComputationalDag().VertexMemWeight(toRemove);
                    if (instance_->getComputationalDag().VertexMemWeight(toRemove) <= bottleneck[proc]
                        && fastMemEnd[proc].find(toRemove) == fastMemEnd[proc].end()) {
                        fastMemEnd[proc].insert(toRemove);
                        bottleneck[proc] -= instance_->getComputationalDag().VertexMemWeight(toRemove);
                    } else {
                        remaining2.push_back(toRemove);
                    }
                }
                computeStep.nodes_evicted_after = remaining2;
                bottleneck[proc] = std::min(bottleneck[proc], instance_->getArchitecture().memoryBound(proc) - memUsed[proc]);

                memUsed[proc] -= instance_->getComputationalDag().VertexMemWeight(computeStep.node);
            }
        }
    }

    if (!isValid()) {
        std::cout << "ERROR: eviction removal process created an invalid schedule." << std::endl;
    }
}

template <typename GraphT>
void PebblingSchedule<GraphT>::CreateFromPartialPebblings(const BspInstance<GraphT> &bspInstance,
                                                          const std::vector<PebblingSchedule<GraphT>> &pebblings,
                                                          const std::vector<std::set<unsigned>> &processorsToParts,
                                                          const std::vector<std::map<VertexIdx, VertexIdx>> &originalNodeId,
                                                          const std::vector<std::map<unsigned, unsigned>> &originalProcId,
                                                          const std::vector<std::vector<std::set<VertexIdx>>> &hasRedsInBeginning) {
    instance_ = &bspInstance;

    unsigned nrParts = static_cast<unsigned>(processorsToParts.size());

    std::vector<std::set<VertexIdx>> inMem(instance_->numberOfProcessors());
    std::vector<std::tuple<VertexIdx, unsigned, unsigned>> forceEvicts;

    computeStepsForProcSuperstep_.clear();
    nodesSentUp_.clear();
    nodesSentDown_.clear();
    nodesEvictedInComm_.clear();
    computeStepsForProcSuperstep_.resize(instance_->numberOfProcessors());
    nodesSentUp_.resize(instance_->numberOfProcessors());
    nodesSentDown_.resize(instance_->numberOfProcessors());
    nodesEvictedInComm_.resize(instance_->numberOfProcessors());

    std::vector<unsigned> supstepIdx(instance_->numberOfProcessors(), 0);

    std::vector<unsigned> getsBlueInSuperstep(instance_->numberOfVertices(), UINT_MAX);
    for (VertexIdx node = 0; node < instance_->numberOfVertices(); ++node) {
        if (instance_->getComputationalDag().InDegree(node) == 0) {
            getsBlueInSuperstep[node] = 0;
        }
    }

    for (unsigned part = 0; part < nrParts; ++part) {
        unsigned startingStepIndex = 0;

        // find dependencies on previous subschedules
        for (VertexIdx node = 0; node < pebblings[part].instance->numberOfVertices(); ++node) {
            if (pebblings[part].instance->getComputationalDag().InDegree(node) == 0) {
                startingStepIndex = std::max(startingStepIndex, getsBlueInSuperstep[originalNodeId[part].at(node)]);
            }
        }

        // sync starting points for the subset of processors
        for (unsigned proc : processorsToParts[part]) {
            startingStepIndex = std::max(startingStepIndex, supstepIdx[proc]);
        }
        for (unsigned proc : processorsToParts[part]) {
            while (supstepIdx[proc] < startingStepIndex) {
                computeStepsForProcSuperstep_[proc].emplace_back();
                nodesSentUp_[proc].emplace_back();
                nodesSentDown_[proc].emplace_back();
                nodesEvictedInComm_[proc].emplace_back();
                ++supstepIdx[proc];
            }
        }

        // check and update according to initial states of red pebbles
        for (unsigned proc = 0; proc < processorsToParts[part].size(); ++proc) {
            unsigned procId = originalProcId[part].at(proc);
            std::set<VertexIdx> neededInRed, addBefore, removeBefore;
            for (VertexIdx node : hasRedsInBeginning[part][proc]) {
                VertexIdx nodeId = originalNodeId[part].at(node);
                neededInRed.insert(nodeId);
                if (inMem[procId].find(nodeId) == inMem[procId].end()) {
                    addBefore.insert(nodeId);
                }
            }
            for (VertexIdx node : inMem[procId]) {
                if (neededInRed.find(node) == neededInRed.end()) {
                    removeBefore.insert(node);
                }
            }

            if ((!addBefore.empty() || !removeBefore.empty()) && supstepIdx[procId] == 0) {
                // this code is added just in case - this shouldn't happen in normal schedules
                computeStepsForProcSuperstep_[procId].emplace_back();
                nodesSentUp_[procId].emplace_back();
                nodesSentDown_[procId].emplace_back();
                nodesEvictedInComm_[procId].emplace_back();
                ++supstepIdx[procId];
            }

            for (VertexIdx node : addBefore) {
                inMem[procId].insert(node);
                nodesSentDown_[procId].back().push_back(node);
            }
            for (VertexIdx node : removeBefore) {
                inMem[procId].erase(node);
                nodesEvictedInComm_[procId].back().push_back(node);
                forceEvicts.push_back(std::make_tuple(node, procId, nodesEvictedInComm_[procId].size() - 1));
            }
        }

        for (unsigned supstep = 0; supstep < pebblings[part].numberOfSupersteps(); ++supstep) {
            for (unsigned proc = 0; proc < processorsToParts[part].size(); ++proc) {
                unsigned procId = originalProcId[part].at(proc);
                computeStepsForProcSuperstep_[procId].emplace_back();
                nodesSentUp_[procId].emplace_back();
                nodesSentDown_[procId].emplace_back();
                nodesEvictedInComm_[procId].emplace_back();

                // copy schedule with translated indeces
                for (const ComputeStep &computeStep : pebblings[part].GetComputeStepsForProcSuperstep(proc, supstep)) {
                    computeStepsForProcSuperstep_[procId].back().emplace_back();
                    computeStepsForProcSuperstep_[procId].back().back().node = originalNodeId[part].at(computeStep.node);
                    inMem[procId].insert(originalNodeId[part].at(computeStep.node));

                    for (VertexIdx localId : computeStep.nodesEvictedAfter) {
                        computeStepsForProcSuperstep_[procId].back().back().nodes_evicted_after.push_back(
                            originalNodeId[part].at(localId));
                        inMem[procId].erase(originalNodeId[part].at(localId));
                    }
                }
                for (VertexIdx node : pebblings[part].GetNodesSentUp(proc, supstep)) {
                    VertexIdx nodeId = originalNodeId[part].at(node);
                    nodesSentUp_[procId].back().push_back(nodeId);
                    getsBlueInSuperstep[nodeId] = std::min(getsBlueInSuperstep[nodeId], supstepIdx[procId]);
                }
                for (VertexIdx node : pebblings[part].GetNodesEvictedInComm(proc, supstep)) {
                    nodesEvictedInComm_[procId].back().push_back(originalNodeId[part].at(node));
                    inMem[procId].erase(originalNodeId[part].at(node));
                }
                for (VertexIdx node : pebblings[part].GetNodesSentDown(proc, supstep)) {
                    nodesSentDown_[procId].back().push_back(originalNodeId[part].at(node));
                    inMem[procId].insert(originalNodeId[part].at(node));
                }

                ++supstepIdx[procId];
            }
        }
    }

    // padding supersteps in the end
    unsigned maxStepIndex = 0;
    for (unsigned proc = 0; proc < instance_->numberOfProcessors(); ++proc) {
        maxStepIndex = std::max(maxStepIndex, supstepIdx[proc]);
    }
    for (unsigned proc = 0; proc < instance_->numberOfProcessors(); ++proc) {
        while (supstepIdx[proc] < maxStepIndex) {
            computeStepsForProcSuperstep_[proc].emplace_back();
            nodesSentUp_[proc].emplace_back();
            nodesSentDown_[proc].emplace_back();
            nodesEvictedInComm_[proc].emplace_back();
            ++supstepIdx[proc];
        }
    }
    numberOfSupersteps_ = maxStepIndex;
    needToLoadInputs_ = true;

    FixForceEvicts(forceEvicts);
    TryToMergeSupersteps();
}

template <typename GraphT>
void PebblingSchedule<GraphT>::FixForceEvicts(const std::vector<std::tuple<VertexIdx, unsigned, unsigned>> forceEvictNodeProcStep) {
    // Some values were evicted only because they weren't present in the next part - see if we can undo those evictions
    for (auto forceEvict : forceEvictNodeProcStep) {
        VertexIdx node = std::get<0>(forceEvict);
        unsigned proc = std::get<1>(forceEvict);
        unsigned superstep = std::get<2>(forceEvict);

        bool nextInComp = false;
        bool nextInComm = false;
        std::pair<unsigned, unsigned> where;

        for (unsigned findSupstep = superstep + 1; findSupstep < NumberOfSupersteps(); ++findSupstep) {
            for (unsigned stepIndex = 0; stepIndex < computeStepsForProcSuperstep_[proc][findSupstep].size(); ++stepIndex) {
                if (computeStepsForProcSuperstep_[proc][findSupstep][stepIndex].node == node) {
                    nextInComp = true;
                    where = std::make_pair(findSupstep, stepIndex);
                    break;
                }
            }
            if (nextInComp) {
                break;
            }
            for (VertexIdx sendDown : nodesSentDown_[proc][findSupstep]) {
                if (sendDown == node) {
                    nextInComm = true;
                    where = std::make_pair(findSupstep, 0);
                    break;
                }
            }
            if (nextInComm) {
                break;
            }
        }

        // check new schedule for validity
        if (!nextInComp && !nextInComm) {
            continue;
        }

        PebblingSchedule<GraphT> testSchedule = *this;
        for (auto itr = testSchedule.nodesEvictedInComm_[proc][superstep].begin();
             itr != testSchedule.nodesEvictedInComm_[proc][superstep].end();
             ++itr) {
            if (*itr == node) {
                testSchedule.nodesEvictedInComm_[proc][superstep].erase(itr);
                break;
            }
        }

        if (nextInComp) {
            for (auto itr = testSchedule.computeStepsForProcSuperstep_[proc][where.first].begin();
                 itr != testSchedule.computeStepsForProcSuperstep_[proc][where.first].end();
                 ++itr) {
                if (itr->node == node) {
                    if (where.second > 0) {
                        auto previousStep = itr;
                        --previousStep;
                        for (VertexIdx toEvict : itr->nodes_evicted_after) {
                            previousStep->nodes_evicted_after.push_back(toEvict);
                        }
                    } else {
                        for (VertexIdx toEvict : itr->nodes_evicted_after) {
                            testSchedule.nodesEvictedInComm_[proc][where.first - 1].push_back(toEvict);
                        }
                    }
                    testSchedule.computeStepsForProcSuperstep_[proc][where.first].erase(itr);
                    break;
                }
            }

            if (testSchedule.isValid()) {
                nodesEvictedInComm_[proc][superstep] = testSchedule.nodesEvictedInComm_[proc][superstep];
                computeStepsForProcSuperstep_[proc][where.first] = testSchedule.computeStepsForProcSuperstep_[proc][where.first];
                nodesEvictedInComm_[proc][where.first - 1] = testSchedule.nodesEvictedInComm_[proc][where.first - 1];
            }
        } else if (nextInComm) {
            for (auto itr = testSchedule.nodesSentDown_[proc][where.first].begin();
                 itr != testSchedule.nodesSentDown_[proc][where.first].end();
                 ++itr) {
                if (*itr == node) {
                    testSchedule.nodesSentDown_[proc][where.first].erase(itr);
                    break;
                }
            }

            if (testSchedule.isValid()) {
                nodesEvictedInComm_[proc][superstep] = testSchedule.nodesEvictedInComm_[proc][superstep];
                nodesSentDown_[proc][where.first] = testSchedule.nodesSentDown_[proc][where.first];
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

            for (unsigned proc = 0; proc < instance_->numberOfProcessors(); ++proc) {
                testSchedule.computeStepsForProcSuperstep_[proc][step].insert(
                    testSchedule.computeStepsForProcSuperstep_[proc][step].end(),
                    testSchedule.computeStepsForProcSuperstep_[proc][nextStep].begin(),
                    testSchedule.computeStepsForProcSuperstep_[proc][nextStep].end());
                testSchedule.computeStepsForProcSuperstep_[proc][nextStep].clear();

                testSchedule.nodesSentUp_[proc][step].insert(testSchedule.nodesSentUp_[proc][step].end(),
                                                             testSchedule.nodesSentUp_[proc][nextStep].begin(),
                                                             testSchedule.nodesSentUp_[proc][nextStep].end());
                testSchedule.nodesSentUp_[proc][nextStep].clear();

                testSchedule.nodesSentDown_[proc][prevStep].insert(testSchedule.nodesSentDown_[proc][prevStep].end(),
                                                                   testSchedule.nodesSentDown_[proc][step].begin(),
                                                                   testSchedule.nodesSentDown_[proc][step].end());
                testSchedule.nodesSentDown_[proc][step].clear();

                testSchedule.nodesEvictedInComm_[proc][step].insert(testSchedule.nodesEvictedInComm_[proc][step].end(),
                                                                    testSchedule.nodesEvictedInComm_[proc][nextStep].begin(),
                                                                    testSchedule.nodesEvictedInComm_[proc][nextStep].end());
                testSchedule.nodesEvictedInComm_[proc][nextStep].clear();
            }

            if (testSchedule.isValid()) {
                isRemoved[nextStep] = true;
                for (unsigned proc = 0; proc < instance_->numberOfProcessors(); ++proc) {
                    computeStepsForProcSuperstep_[proc][step] = testSchedule.computeStepsForProcSuperstep_[proc][step];
                    computeStepsForProcSuperstep_[proc][nextStep].clear();

                    nodesSentUp_[proc][step] = testSchedule.nodesSentUp_[proc][step];
                    nodesSentUp_[proc][nextStep].clear();

                    nodesSentDown_[proc][prevStep] = testSchedule.nodesSentDown_[proc][prevStep];
                    nodesSentDown_[proc][step] = nodesSentDown_[proc][nextStep];
                    nodesSentDown_[proc][nextStep].clear();

                    nodesEvictedInComm_[proc][step] = testSchedule.nodesEvictedInComm_[proc][step];
                    nodesEvictedInComm_[proc][nextStep].clear();
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

        for (unsigned proc = 0; proc < instance_->numberOfProcessors(); ++proc) {
            shortenedSchedule.computeStepsForProcSuperstep_[proc][newIndex] = computeStepsForProcSuperstep_[proc][step];
            shortenedSchedule.nodesSentUp_[proc][newIndex] = nodesSentUp_[proc][step];
            shortenedSchedule.nodesSentDown_[proc][newIndex] = nodesSentDown_[proc][step];
            shortenedSchedule.nodesEvictedInComm_[proc][newIndex] = nodesEvictedInComm_[proc][step];
        }

        ++newIndex;
    }

    *this = shortenedSchedule;

    if (!isValid()) {
        std::cout << "ERROR: schedule is not valid after superstep merging." << std::endl;
    }
}

template <typename GraphT>
PebblingSchedule<GraphT> PebblingSchedule<GraphT>::ExpandMemSchedule(const BspInstance<GraphT> &originalInstance,
                                                                     const std::vector<VertexIdx> mappingToCoarse) const {
    std::map<VertexIdx, std::set<VertexIdx>> originalVerticesForCoarseId;
    for (VertexIdx node = 0; node < originalInstance.numberOfVertices(); ++node) {
        originalVerticesForCoarseId[mappingToCoarse[node]].insert(node);
    }

    PebblingSchedule<GraphT> fineSchedule;
    fineSchedule.instance_ = &originalInstance;
    fineSchedule.UpdateNumberOfSupersteps(numberOfSupersteps_);

    for (unsigned step = 0; step < numberOfSupersteps_; ++step) {
        for (unsigned proc = 0; proc < instance_->numberOfProcessors(); ++proc) {
            // computation phase
            for (const auto &computeStep : computeStepsForProcSuperstep_[proc][step]) {
                VertexIdx node = computeStep.node;
                for (VertexIdx originalNode : originalVerticesForCoarseId[node]) {
                    fineSchedule.computeStepsForProcSuperstep_[proc][step].emplace_back(originalNode);
                }

                for (VertexIdx toRemove : computeStep.nodes_evicted_after) {
                    for (VertexIdx originalNode : originalVerticesForCoarseId[toRemove]) {
                        fineSchedule.computeStepsForProcSuperstep_[proc][step].back().nodes_evicted_after.push_back(originalNode);
                    }
                }
            }

            // communication phase
            for (VertexIdx node : nodesSentUp_[proc][step]) {
                for (VertexIdx originalNode : originalVerticesForCoarseId[node]) {
                    fineSchedule.nodesSentUp_[proc][step].push_back(originalNode);
                }
            }

            for (VertexIdx node : nodesEvictedInComm_[proc][step]) {
                for (VertexIdx originalNode : originalVerticesForCoarseId[node]) {
                    fineSchedule.nodesEvictedInComm_[proc][step].push_back(originalNode);
                }
            }

            for (VertexIdx node : nodesSentDown_[proc][step]) {
                for (VertexIdx originalNode : originalVerticesForCoarseId[node]) {
                    fineSchedule.nodesSentDown_[proc][step].push_back(originalNode);
                }
            }
        }
    }

    fineSchedule.CleanSchedule();
    return fineSchedule;
}

template <typename GraphT>
BspSchedule<GraphT> PebblingSchedule<GraphT>::ConvertToBsp() const {
    std::vector<unsigned> nodeToProc(instance_->numberOfVertices(), UINT_MAX),
        nodeToSupstep(instance_->numberOfVertices(), UINT_MAX);

    for (unsigned step = 0; step < numberOfSupersteps_; ++step) {
        for (unsigned proc = 0; proc < instance_->numberOfProcessors(); ++proc) {
            for (const auto &computeStep : computeStepsForProcSuperstep_[proc][step]) {
                const VertexIdx &node = computeStep.node;
                if (nodeToProc[node] == UINT_MAX) {
                    nodeToProc[node] = proc;
                    nodeToSupstep[node] = step;
                }
            }
        }
    }
    if (needToLoadInputs_) {
        for (VertexIdx node = 0; node < instance_->numberOfVertices(); ++node) {
            if (instance_->getComputationalDag().InDegree(node) == 0) {
                unsigned minSuperstep = UINT_MAX, procChosen = 0;
                for (VertexIdx succ : instance_->getComputationalDag().Children(node)) {
                    if (nodeToSupstep[succ] < minSuperstep) {
                        minSuperstep = nodeToSupstep[succ];
                        procChosen = nodeToProc[succ];
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
