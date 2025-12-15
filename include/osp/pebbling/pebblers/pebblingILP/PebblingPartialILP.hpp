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

#include "osp/bsp/model/BspInstance.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/graph_algorithms/subgraph_algorithms.hpp"
#include "osp/pebbling/PebblingSchedule.hpp"
#include "osp/pebbling/pebblers/pebblingILP/MultiProcessorPebbling.hpp"
#include "osp/pebbling/pebblers/pebblingILP/partialILP/AcyclicDagDivider.hpp"
#include "osp/pebbling/pebblers/pebblingILP/partialILP/SubproblemMultiScheduling.hpp"

namespace osp {

template <typename GraphT>
class PebblingPartialILP : public Scheduler<GraphT> {
    static_assert(IsComputationalDagV<GraphT>, "PebblingSchedule can only be used with computational DAGs.");
    static_assert(std::is_same_v<VWorkwT<GraphT>, VCommwT<GraphT>>,
                  "PebblingSchedule requires work and comm. weights to have the same type.");

    using VertexIdx = VertexIdxT<GraphT>;
    using cost_type = VWorkwT<GraphT>;

    unsigned minPartitionSize_ = 50, maxPartitionSize_ = 100;
    unsigned timeSecondsForSubIlPs_ = 600;

    bool asynchronous_ = false;
    bool verbose_ = false;

    std::map<std::pair<unsigned, unsigned>, unsigned> partAndNodetypeToNewIndex_;

  public:
    PebblingPartialILP() {}

    virtual ~PebblingPartialILP() = default;

    ReturnStatus ComputePebbling(PebblingSchedule<GraphT> &schedule);

    // not used, only here for using scheduler class base functionality (status enums, timelimits, etc)
    virtual ReturnStatus ComputeSchedule(BspSchedule<GraphT> &schedule) override;

    GraphT ContractByPartition(const BspInstance<GraphT> &instance, const std::vector<unsigned> &nodeToPartAssignment);

    /**
     * @brief Get the name of the schedule.
     *
     * @return The name of the schedule.
     */
    virtual std::string GetScheduleName() const override { return "PebblingPartialILP"; }

    // getters and setters for problem parameters
    inline std::pair<unsigned, unsigned> GetMinAndMaxSize() const { return std::make_pair(minPartitionSize_, maxPartitionSize_); }

    inline void SetMinSize(const unsigned minSize) {
        minPartitionSize_ = minSize;
        maxPartitionSize_ = 2 * minSize;
    }

    inline void SetMinAndMaxSize(const std::pair<unsigned, unsigned> minAndMax) {
        minPartitionSize_ = min_and_max.first;
        maxPartitionSize_ = min_and_max.second;
    }

    inline void SetAsync(const bool async) { asynchronous_ = async; }

    inline void SetSecondsForSubIlp(const unsigned seconds) { timeSecondsForSubIlPs_ = seconds; }

    inline void SetVerbose(const bool verbose) { verbose_ = verbose; }
};

template <typename GraphT>
ReturnStatus PebblingPartialILP<GraphT>::ComputePebbling(PebblingSchedule<GraphT> &schedule) {
    const BspInstance<GraphT> &instance = schedule.GetInstance();

    if (!PebblingSchedule<GraphT>::hasValidSolution(instance)) {
        return ReturnStatus::ERROR;
    }

    // STEP 1: divide DAG acyclicly with partitioning ILP

    AcyclicDagDivider<GraphT> dagDivider;
    dagDivider.SetMinAndMaxSize({minPartitionSize_, maxPartitionSize_});
    std::vector<unsigned> assignmentToParts = dagDivider.ComputePartitioning(instance);
    unsigned nrParts = *std::max_element(assignmentToParts.begin(), assignmentToParts.end()) + 1;

    // TODO remove source nodes before this?
    GraphT contractedDag = contractByPartition(instance, assignmentToParts);

    // STEP 2: develop high-level multischedule on parts

    BspInstance<GraphT> contractedInstance(
        contractedDag, instance.GetArchitecture(), instance.getNodeProcessorCompatibilityMatrix());

    SubproblemMultiScheduling<GraphT> multiScheduler;
    std::vector<std::set<unsigned>> processorsToPartsAndTypes;
    multiScheduler.ComputeMultiSchedule(contractedInstance, processorsToPartsAndTypes);

    std::vector<std::set<unsigned>> processorsToParts(nrParts);
    for (unsigned part = 0; part < nrParts; ++part) {
        for (unsigned type = 0; type < instance.GetComputationalDag().NumVertexTypes(); ++type) {
            if (partAndNodetypeToNewIndex.find({part, type}) != partAndNodetypeToNewIndex.end()) {
                unsigned newIndex = partAndNodetypeToNewIndex[{part, type}];
                for (unsigned proc : processorsToPartsAndTypes[newIndex]) {
                    processorsToParts[part].insert(proc);
                }
            }
        }
    }

    // AUX: check for isomorphism

    // create set of nodes & external sources for all parts, and the nodes that need to have blue pebble at the end
    std::vector<std::set<VertexIdx>> nodesInPart(nrParts), extraSources(nrParts);
    std::vector<std::map<VertexIdx, VertexIdx>> originalNodeId(nrParts);
    std::vector<std::map<unsigned, unsigned>> originalProcId(nrParts);
    for (VertexIdx node = 0; node < instance.NumberOfVertices(); ++node) {
        if (instance.GetComputationalDag().InDegree(node) > 0) {
            nodesInPart[assignmentToParts[node]].insert(node);
        } else {
            extraSources[assignmentToParts[node]].insert(node);
        }
        for (const VertexIdx &pred : instance.GetComputationalDag().Parents(node)) {
            if (assignmentToParts[node] != assignmentToParts[pred]) {
                extraSources[assignmentToParts[node]].insert(pred);
            }
        }
    }

    std::vector<GraphT> subDags;
    for (unsigned part = 0; part < nrParts; ++part) {
        GraphT dag;
        create_induced_subgraph(instance.GetComputationalDag(), dag, nodesInPart[part], extraSources[part]);
        subDags.push_back(dag);

        // set source nodes to a new type, so that they are compatible with any processor
        unsigned artificialTypeForSources = subDags.back().NumVertexTypes();
        for (VertexIdx nodeIdx = 0; nodeIdx < extraSources[part].size(); ++nodeIdx) {
            subDags.back().SetVertexType(nodeIdx, artificialTypeForSources);
        }
    }

    std::vector<unsigned> isomorphicTo(nrParts, UINT_MAX);

    std::cout << "Number of parts: " << nrParts << std::endl;

    for (unsigned part = 0; part < nrParts; ++part) {
        for (unsigned otherPart = part + 1; otherPart < nrParts; ++otherPart) {
            if (isomorphicTo[otherPart] < UINT_MAX) {
                continue;
            }

            bool isomorphic = true;
            if (!checkOrderedIsomorphism(subDags[part], subDags[otherPart])) {
                continue;
            }

            std::vector<unsigned> procAssignedPerType(instance.GetArchitecture().GetNumberOfProcessorTypes(), 0);
            std::vector<unsigned> otherProcAssignedPerType(instance.GetArchitecture().GetNumberOfProcessorTypes(), 0);
            for (unsigned proc : processorsToParts[part]) {
                ++procAssignedPerType[instance.GetArchitecture().ProcessorType(proc)];
            }
            for (unsigned proc : processorsToParts[otherPart]) {
                ++otherProcAssignedPerType[instance.GetArchitecture().ProcessorType(proc)];
            }

            for (unsigned procType = 0; procType < instance.GetArchitecture().GetNumberOfProcessorTypes(); ++procType) {
                if (procAssignedPerType[procType] != otherProcAssignedPerType[procType]) {
                    isomorphic = false;
                }
            }

            if (!isomorphic) {
                continue;
            }

            isomorphicTo[otherPart] = part;
            std::cout << "Part " << otherPart << " is isomorphic to " << part << std::endl;
        }
    }

    // PART 3: solve a small ILP for each part
    std::vector<std::set<VertexIdx>> inFastMem(instance.NumberOfProcessors());
    std::vector<PebblingSchedule<GraphT>> pebbling(nrParts);
    std::vector<BspArchitecture<GraphT>> subArch(nrParts);
    std::vector<BspInstance<GraphT>> subInstance(nrParts);

    // to handle the initial memory content for isomorphic parts
    std::vector<std::vector<std::set<VertexIdx>>> hasRedsInBeginning(
        nrParts, std::vector<std::set<VertexIdx>>(instance.NumberOfProcessors()));

    for (unsigned part = 0; part < nrParts; ++part) {
        std::cout << "part " << part << std::endl;

        // set up sub-DAG
        GraphT &subDag = subDags[part];
        std::map<VertexIdx, VertexIdx> localId;
        VertexIdx nodeIdx = 0;
        for (VertexIdx node : extraSources[part]) {
            localId[node] = nodeIdx;
            originalNodeId[part][nodeIdx] = node;
            ++nodeIdx;
        }
        for (VertexIdx node : nodesInPart[part]) {
            localId[node] = nodeIdx;
            originalNodeId[part][nodeIdx] = node;
            ++nodeIdx;
        }

        std::set<VertexIdx> needsBlueAtEnd;
        for (VertexIdx node : nodesInPart[part]) {
            for (const VertexIdx &succ : instance.GetComputationalDag().Children(node)) {
                if (assignmentToParts[node] != assignmentToParts[succ]) {
                    needsBlueAtEnd.insert(localId[node]);
                }
            }

            if (instance.GetComputationalDag().OutDegree(node) == 0) {
                needsBlueAtEnd.insert(localId[node]);
            }
        }

        // set up sub-architecture
        subArch[part].setNumberOfProcessors(static_cast<unsigned>(processorsToParts[part].size()));
        unsigned procIndex = 0;
        for (unsigned proc : processorsToParts[part]) {
            subArch[part].setProcessorType(procIndex, instance.GetArchitecture().ProcessorType(proc));
            subArch[part].setMemoryBound(instance.GetArchitecture().memoryBound(proc), procIndex);
            originalProcId[part][procIndex] = proc;
            ++procIndex;
        }
        subArch[part].setCommunicationCosts(instance.GetArchitecture().CommunicationCosts());
        subArch[part].setSynchronisationCosts(instance.GetArchitecture().SynchronisationCosts());
        // no NUMA parameters for now

        // skip if isomorphic to previous part
        if (isomorphicTo[part] < UINT_MAX) {
            pebbling[part] = pebbling[isomorphicTo[part]];
            hasRedsInBeginning[part] = has_reds_in_beginning[isomorphicTo[part]];
            continue;
        }

        // set node-processor compatibility matrix
        std::vector<std::vector<bool>> compMatrix = instance.getNodeProcessorCompatibilityMatrix();
        compMatrix.emplace_back(instance.GetArchitecture().GetNumberOfProcessorTypes(), true);
        subInstance[part] = BspInstance(subDag, subArch[part], comp_matrix);

        // currently we only allow the input laoding scenario - the case where this is false is unmaintained/untested
        bool needToLoadInputs = true;

        // keep in fast memory what's relevant, remove the rest
        for (unsigned proc = 0; proc < processorsToParts[part].size(); ++proc) {
            hasRedsInBeginning[part][proc].clear();
            std::set<VertexIdx> newContentFastMem;
            for (VertexIdx node : inFastMem[originalProcId[part][proc]]) {
                if (localId.find(node) != localId.end()) {
                    hasRedsInBeginning[part][proc].insert(localId[node]);
                    newContentFastMem.insert(node);
                }
            }

            inFastMem[originalProcId[part][proc]] = newContentFastMem;
        }

        // heuristic solution for baseline
        PebblingSchedule<GraphT> heuristicPebbling;
        GreedyBspScheduler<GraphT> greedyScheduler;
        BspSchedule<GraphT> bspHeuristic(subInstance[part]);
        greedyScheduler.ComputeSchedule(bspHeuristic);

        std::set<VertexIdx> extraSourceIds;
        for (VertexIdx idx = 0; idx < extra_sources[part].size(); ++idx) {
            extraSourceIds.insert(idx);
        }

        heuristicPebbling.SetNeedToLoadInputs(true);
        heuristicPebbling.SetExternalSources(extraSourceIds);
        heuristicPebbling.SetNeedsBlueAtEnd(needsBlueAtEnd);
        heuristicPebbling.SetHasRedInBeginning(hasRedsInBeginning[part]);
        heuristicPebbling.ConvertFromBsp(bspHeuristic, PebblingSchedule<GraphT>::CACHE_EVICTION_STRATEGY::FORESIGHT);

        heuristicPebbling.RemoveEvictStepsFromEnd();
        pebbling[part] = heuristicPebbling;
        cost_type heuristicCost = asynchronous_ ? heuristicPebbling.ComputeAsynchronousCost() : heuristicPebbling.ComputeCost();

        if (!heuristicPebbling.IsValid()) {
            std::cout << "ERROR: Pebbling heuristic INVALID!" << std::endl;
        }

        // solution with subILP
        MultiProcessorPebbling<GraphT> mpp;
        mpp.SetVerbose(verbose_);
        mpp.SetTimeLimitSeconds(timeSecondsForSubIlPs_);
        mpp.SetMaxTime(2 * maxPartitionSize_);    // just a heuristic choice, does not guarantee feasibility!
        mpp.SetNeedsBlueAtEnd(needsBlueAtEnd);
        mpp.SetNeedToLoadInputs(needToLoadInputs);
        mpp.SetHasRedInBeginning(hasRedsInBeginning[part]);

        PebblingSchedule<GraphT> pebblingILP(subInstance[part]);
        ReturnStatus status = mpp.ComputePebblingWithInitialSolution(heuristicPebbling, pebblingILP, asynchronous_);
        if (status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::BEST_FOUND) {
            if (!pebblingILP.isValid()) {
                std::cout << "ERROR: Pebbling ILP INVALID!" << std::endl;
            }

            pebblingILP.removeEvictStepsFromEnd();
            cost_type ilpCost = asynchronous_ ? pebblingILP.computeAsynchronousCost() : pebblingILP.computeCost();
            if (ILP_cost < heuristicCost) {
                pebbling[part] = pebblingILP;
                std::cout << "ILP chosen instead of greedy. (" << ILP_cost << " < " << heuristicCost << ")" << std::endl;
            } else {
                std::cout << "Greedy chosen instead of ILP. (" << heuristicCost << " < " << ILP_cost << ")" << std::endl;
            }

            // save fast memory content for next subproblem
            std::vector<std::set<VertexIdx>> fastMemContentAtEnd = pebbling[part].getMemContentAtEnd();
            for (unsigned proc = 0; proc < processorsToParts[part].size(); ++proc) {
                inFastMem[originalProcId[part][proc]].clear();
                for (VertexIdx node : fastMemContentAtEnd[proc]) {
                    inFastMem[originalProcId[part][proc]].insert(originalNodeId[part][node]);
                }
            }
        } else {
            std::cout << "ILP found no solution; using greedy instead (cost = " << heuristicCost << ")." << std::endl;
        }
    }

    // AUX: assemble final schedule from subschedules
    schedule.CreateFromPartialPebblings(instance, pebbling, processorsToParts, originalNodeId, originalProcId, hasRedsInBeginning);
    schedule.cleanSchedule();
    return schedule.isValid() ? ReturnStatus::OSP_SUCCESS : ReturnStatus::ERROR;
}

template <typename GraphT>
GraphT PebblingPartialILP<GraphT>::ContractByPartition(const BspInstance<GraphT> &instance,
                                                       const std::vector<unsigned> &nodeToPartAssignment) {
    const auto &g = instance.GetComputationalDag();

    partAndNodeTypeToNewIndex.clear();

    unsigned nrNewNodes = 0;
    for (VertexIdx node = 0; node < instance.NumberOfVertices(); ++node) {
        if (partAndNodeTypeToNewIndex.find({nodeToPartAssignment[node], g.VertexType(node)}) == partAndNodeTypeToNewIndex.end()) {
            partAndNodeTypeToNewIndex[{nodeToPartAssignment[node], g.VertexType(node)}] = nrNewNodes;
            ++nrNewNodes;
        }
    }

    GraphT contracted;
    for (VertexIdx node = 0; node < nrNewNodes; ++node) {
        contracted.add_vertex(0, 0, 0);
    }

    std::set<std::pair<VertexIdx, VertexIdx>> edges;

    for (VertexIdx node = 0; node < instance.NumberOfVertices(); ++node) {
        VertexIdx nodeNewIndex = partAndNodeTypeToNewIndex[{nodeToPartAssignment[node], g.VertexType(node)}];
        for (const VertexIdx &succ : instance.GetComputationalDag().Children(node)) {
            if (nodeToPartAssignment[node] != nodeToPartAssignment[succ]) {
                edges.emplace(nodeNewIndex, partAndNodeTypeToNewIndex[{nodeToPartAssignment[succ], g.VertexType(succ)}]);
            }
        }

        contracted.SetVertexWorkWeight(nodeNewIndex, contracted.VertexWorkWeight(nodeNewIndex) + g.VertexWorkWeight(node));
        contracted.SetVertexCommWeight(nodeNewIndex, contracted.VertexCommWeight(nodeNewIndex) + g.VertexCommWeight(node));
        contracted.SetVertexMemWeight(nodeNewIndex, contracted.VertexMemWeight(nodeNewIndex) + g.VertexMemWeight(node));
        contracted.SetVertexType(nodeNewIndex, g.VertexType(node));
    }

    for (auto edge : edges) {
        if constexpr (HasEdgeWeightsV<GraphT>) {
            contracted.add_edge(edge.first, edge.second, 1);
        } else {
            contracted.add_edge(edge.first, edge.second);
        }
    }

    return contracted;
}

template <typename GraphT>
ReturnStatus PebblingPartialILP<GraphT>::ComputeSchedule(BspSchedule<GraphT> &) {
    return ReturnStatus::ERROR;
}

}    // namespace osp
