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
#include <cmath>

#include "osp/partitioning/model/partitioning.hpp"

namespace osp {

template <typename HypergraphT>
class GenericFM {
    using IndexType = typename HypergraphT::VertexIdx;
    using WorkwType = typename HypergraphT::VertexWorkWeightType;
    using MemwType = typename HypergraphT::VertexMemWeightType;
    using CommwType = typename HypergraphT::VertexCommWeightType;

  protected:
    unsigned maxNumberOfPasses_ = 10;
    IndexType maxNodesInPart_ = 0;

    // auxiliary for RecursiveFM
    std::vector<IndexType> GetMaxNodesOnLevel(IndexType nrNodes, unsigned nrParts) const;

  public:
    void ImprovePartitioning(Partitioning<HypergraphT> &partition);

    void RecursiveFM(Partitioning<HypergraphT> &partition);

    inline unsigned GetMaxNumberOfPasses() const { return maxNumberOfPasses_; }

    inline void SetMaxNumberOfPasses(unsigned passes) { maxNumberOfPasses_ = passes; }

    inline IndexType GetMaxNodesInPart() const { return maxNodesInPart_; }

    inline void SetMaxNodesInPart(IndexType maxNodes) { maxNodesInPart_ = maxNodes; }
};

template <typename HypergraphT>
void GenericFM<HypergraphT>::ImprovePartitioning(Partitioning<HypergraphT> &partition) {
    // Note: this algorithm disregards hyperedge weights, in order to keep the size of the gain bucket array bounded!

    if (partition.GetInstance().GetNumberOfPartitions() != 2) {
        std::cout << "Error: FM can only be used for 2 partitions." << std::endl;
        return;
    }

    if (!partition.SatisfiesBalanceConstraint()) {
        std::cout << "Error: initial partition to FM does not satisfy balance constraint." << std::endl;
        return;
    }

    const Hypergraph<IndexType, WorkwType, MemwType, CommwType> &hgraph = partition.GetInstance().GetHypergraph();

    IndexType maxDegree = 0;
    for (IndexType node = 0; node < hgraph.NumVertices(); ++node) {
        maxDegree = std::max(maxDegree, static_cast<IndexType>(hgraph.GetIncidentHyperedges(node).size()));
    }

    if (maxNodesInPart_ == 0) {    // if not initialized
        maxNodesInPart_ = static_cast<IndexType>(ceil(static_cast<double>(hgraph.NumVertices())
                                                      * static_cast<double>(partition.GetInstance().GetMaxWorkWeightPerPartition())
                                                      / static_cast<double>(ComputeTotalVertexWorkWeight(hgraph))));
    }

    for (unsigned passIdx = 0; passIdx < maxNumberOfPasses_; ++passIdx) {
        std::vector<unsigned> nodeToNewPart = partition.AssignedPartitions();
        std::vector<bool> locked(hgraph.NumVertices(), false);
        std::vector<int> gain(hgraph.NumVertices(), 0);
        std::vector<std::vector<IndexType> > nrNodesInHyperedgeOnSide(hgraph.NumHyperedges(), std::vector<IndexType>(2, 0));
        int cost = 0;

        IndexType leftSide = 0;
        for (IndexType node = 0; node < hgraph.NumVertices(); ++node) {
            if (partition.AssignedPartition(node) == 0) {
                ++leftSide;
            }
        }

        if (leftSide > maxNodesInPart_ || hgraph.NumVertices() - leftSide > maxNodesInPart_) {
            if (passIdx == 0) {
                std::cout << "Error: initial partitioning of FM is not balanced." << std::endl;
                return;
            } else {
                std::cout << "Error during FM: partitionming somehow became imbalanced." << std::endl;
                return;
            }
        }

        // Initialize gain values
        for (IndexType hyperedge = 0; hyperedge < hgraph.NumHyperedges(); ++hyperedge) {
            for (IndexType node : hgraph.GetVerticesInHyperedge(hyperedge)) {
                ++nrNodesInHyperedgeOnSide[hyperedge][partition.AssignedPartition(node)];
            }

            if (hgraph.GetVerticesInHyperedge(hyperedge).size() < 2) {
                continue;
            }

            for (unsigned part = 0; part < 2; ++part) {
                if (nrNodesInHyperedgeOnSide[hyperedge][part] == 1) {
                    for (IndexType node : hgraph.GetVerticesInHyperedge(hyperedge)) {
                        if (partition.AssignedPartition(node) == part) {
                            ++gain[node];
                        }
                    }
                }

                if (nrNodesInHyperedgeOnSide[hyperedge][part] == 0) {
                    for (IndexType node : hgraph.GetVerticesInHyperedge(hyperedge)) {
                        if (partition.AssignedPartition(node) != part) {
                            --gain[node];
                        }
                    }
                }
            }
        }

        // build gain bucket array
        std::vector<int> maxGain(2, -static_cast<int>(maxDegree) - 1);
        std::vector<std::vector<std::vector<IndexType> > > gainBucketArray(
            2, std::vector<std::vector<IndexType> >(2 * maxDegree + 1));
        for (IndexType node = 0; node < hgraph.NumVertices(); ++node) {
            const unsigned &part = partition.AssignedPartition(node);
            gainBucketArray[part][static_cast<unsigned>(gain[node] + static_cast<int>(maxDegree))].push_back(node);
            maxGain[part] = std::max(maxGain[part], gain[node]);
        }

        IndexType bestIndex = 0;
        int bestCost = 0;
        std::vector<IndexType> movedNodes;

        // the pass itself: make moves
        while (movedNodes.size() < hgraph.NumVertices()) {
            // select move
            IndexType toMove = std::numeric_limits<IndexType>::max();
            unsigned chosenPart = std::numeric_limits<unsigned>::max();

            unsigned gainIndex = static_cast<unsigned>(std::max(maxGain[0], maxGain[1]) + static_cast<int>(maxDegree));
            while (gainIndex < std::numeric_limits<unsigned>::max()) {
                bool canChooseLeft = (hgraph.NumVertices() - leftSide < maxNodesInPart_) && !gainBucketArray[0][gainIndex].empty();
                bool canChooseRight = (leftSide < maxNodesInPart_) && !gainBucketArray[1][gainIndex].empty();

                if (canChooseLeft && canChooseRight) {
                    chosenPart = (leftSide >= hgraph.NumVertices() / 2) ? 1 : 0;
                } else if (canChooseLeft) {
                    chosenPart = 0;
                } else if (canChooseRight) {
                    chosenPart = 1;
                }

                if (chosenPart < 2) {
                    toMove = gainBucketArray[chosenPart][gainIndex].back();
                    gainBucketArray[chosenPart][gainIndex].pop_back();
                    break;
                }
                --gainIndex;
            }

            if (toMove == std::numeric_limits<IndexType>::max()) {
                break;
            }

            // make move

            movedNodes.push_back(toMove);
            cost -= gain[toMove];
            if (cost < bestCost) {
                bestCost = cost;
                bestIndex = static_cast<IndexType>(movedNodes.size()) + 1;
            }
            locked[toMove] = true;
            nodeToNewPart[toMove] = 1 - nodeToNewPart[toMove];

            if (chosenPart == 0) {
                --leftSide;
            } else {
                ++leftSide;
            }

            unsigned otherPart = 1 - chosenPart;

            // update gain values
            for (IndexType hyperedge : hgraph.GetIncidentHyperedges(toMove)) {
                if (nrNodesInHyperedgeOnSide[hyperedge][chosenPart] == 1) {
                    for (IndexType node : hgraph.GetVerticesInHyperedge(hyperedge)) {
                        if (locked[node]) {
                            continue;
                        }

                        std::vector<IndexType> &vec
                            = gainBucketArray[otherPart][static_cast<unsigned>(gain[node] + static_cast<int>(maxDegree))];
                        vec.erase(std::remove(vec.begin(), vec.end(), node), vec.end());
                        --gain[node];
                        gainBucketArray[otherPart][static_cast<unsigned>(gain[node] + static_cast<int>(maxDegree))].push_back(node);
                    }
                } else if (nrNodesInHyperedgeOnSide[hyperedge][chosenPart] == 2) {
                    for (IndexType node : hgraph.GetVerticesInHyperedge(hyperedge)) {
                        if (nodeToNewPart[node] == chosenPart && !locked[node]) {
                            std::vector<IndexType> &vec
                                = gainBucketArray[chosenPart][static_cast<unsigned>(gain[node] + static_cast<int>(maxDegree))];
                            vec.erase(std::remove(vec.begin(), vec.end(), node), vec.end());
                            ++gain[node];
                            gainBucketArray[chosenPart][static_cast<unsigned>(gain[node] + static_cast<int>(maxDegree))].push_back(
                                node);
                            maxGain[chosenPart] = std::max(maxGain[chosenPart], gain[node]);
                            break;
                        }
                    }
                }
                if (nrNodesInHyperedgeOnSide[hyperedge][otherPart] == 1) {
                    for (IndexType node : hgraph.GetVerticesInHyperedge(hyperedge)) {
                        if (nodeToNewPart[node] == otherPart && !locked[node]) {
                            std::vector<IndexType> &vec
                                = gainBucketArray[otherPart][static_cast<unsigned>(gain[node] + static_cast<int>(maxDegree))];
                            vec.erase(std::remove(vec.begin(), vec.end(), node), vec.end());
                            --gain[node];
                            gainBucketArray[otherPart][static_cast<unsigned>(gain[node] + static_cast<int>(maxDegree))].push_back(
                                node);
                            break;
                        }
                    }
                } else if (nrNodesInHyperedgeOnSide[hyperedge][otherPart] == 0) {
                    for (IndexType node : hgraph.GetVerticesInHyperedge(hyperedge)) {
                        if (locked[node]) {
                            continue;
                        }

                        std::vector<IndexType> &vec
                            = gainBucketArray[chosenPart][static_cast<unsigned>(gain[node] + static_cast<int>(maxDegree))];
                        vec.erase(std::remove(vec.begin(), vec.end(), node), vec.end());
                        ++gain[node];
                        gainBucketArray[chosenPart][static_cast<unsigned>(gain[node] + static_cast<int>(maxDegree))].push_back(
                            node);
                        maxGain[chosenPart] = std::max(maxGain[chosenPart], gain[node]);
                    }
                }
                --nrNodesInHyperedgeOnSide[hyperedge][chosenPart];
                ++nrNodesInHyperedgeOnSide[hyperedge][otherPart];
            }
        }

        // apply best configuration seen
        if (bestIndex == 0) {
            break;
        }

        for (IndexType nodeIdx = 0; nodeIdx < bestIndex && nodeIdx < static_cast<IndexType>(movedNodes.size()); ++nodeIdx) {
            partition.SetAssignedPartition(movedNodes[nodeIdx], 1U - partition.AssignedPartition(movedNodes[nodeIdx]));
        }
    }
}

template <typename HypergraphT>
void GenericFM<HypergraphT>::RecursiveFM(Partitioning<HypergraphT> &partition) {
    const unsigned &nrParts = partition.GetInstance().GetNumberOfPartitions();
    const IndexType &nrNodes = partition.GetInstance().GetHypergraph().NumVertices();

    using Hgraph = Hypergraph<IndexType, WorkwType, MemwType, CommwType>;

    // Note: this is just a simple recursive heuristic for the case when the partitions are a small power of 2
    if (nrParts != 4 && nrParts != 8 && nrParts != 16 && nrParts != 32) {
        std::cout << "Error: Recursive FM can only be used for 4, 8, 16 or 32 partitions currently." << std::endl;
        return;
    }

    for (IndexType node = 0; node < nrNodes; ++node) {
        partition.SetAssignedPartition(node, static_cast<unsigned>(node % 2));
    }

    if (maxNodesInPart_ == 0) {    // if not initialized
        maxNodesInPart_ = static_cast<IndexType>(
            ceil(static_cast<double>(nrNodes) * static_cast<double>(partition.GetInstance().GetMaxWorkWeightPerPartition())
                 / static_cast<double>(ComputeTotalVertexWorkWeight(partition.GetInstance().GetHypergraph()))));
    }

    const std::vector<IndexType> maxNodesOnLevel = GetMaxNodesOnLevel(nrNodes, nrParts);

    unsigned parts = 1;
    unsigned level = 0;
    std::vector<Hgraph> subHgraphs({partition.GetInstance().GetHypergraph()});
    unsigned startIndex = 0;

    std::map<IndexType, std::pair<unsigned, IndexType> > nodeToNewHgraphAndId;
    std::map<std::pair<unsigned, IndexType>, IndexType> hgraphAndIdToOldIdx;
    for (IndexType node = 0; node < nrNodes; ++node) {
        nodeToNewHgraphAndId[node] = std::make_pair(0, node);
        hgraphAndIdToOldIdx[std::make_pair(0, node)] = node;
    }

    while (parts < nrParts) {
        unsigned endIdx = static_cast<unsigned>(subHgraphs.size());
        for (unsigned subHgraphIndex = startIndex; subHgraphIndex < endIdx; ++subHgraphIndex) {
            const Hgraph &hgraph = subHgraphs[subHgraphIndex];
            PartitioningProblem instance(hgraph, 2);
            Partitioning subPartition(instance);
            for (IndexType node = 0; node < hgraph.NumVertices(); ++node) {
                subPartition.SetAssignedPartition(node, node % 2);
            }

            GenericFM subFm;
            subFm.SetMaxNodesInPart(maxNodesOnLevel[level]);
            // std::cout<<"Hgraph of size "<<hgraph.NumVertices()<<" split into two parts of at most "<<max_nodes_on_level[level]<<std::endl;
            subFm.ImprovePartitioning(subPartition);

            std::vector<unsigned> currentIdx(2, 0);
            std::vector<std::vector<bool> > partIndicator(2, std::vector<bool>(hgraph.NumVertices(), false));
            for (IndexType node = 0; node < hgraph.NumVertices(); ++node) {
                const unsigned partId = subPartition.AssignedPartition(node);
                const IndexType originalId = hgraphAndIdToOldIdx[std::make_pair(subHgraphIndex, node)];
                nodeToNewHgraphAndId[originalId] = std::make_pair(subHgraphs.size() + partId, currentIdx[partId]);
                hgraphAndIdToOldIdx[std::make_pair(subHgraphs.size() + partId, currentIdx[partId])] = originalId;
                ++currentIdx[partId];
                partIndicator[partId][node] = true;
            }

            for (unsigned part = 0; part < 2; ++part) {
                subHgraphs.push_back(CreateInducedHypergraph(subHgraphs[subHgraphIndex], partIndicator[part]));
            }

            ++startIndex;
        }

        parts *= 2;
        ++level;
    }

    for (IndexType node = 0; node < nrNodes; ++node) {
        partition.SetAssignedPartition(node,
                                       nodeToNewHgraphAndId[node].first - (static_cast<unsigned>(subHgraphs.size()) - nrParts));
    }
}

template <typename HypergraphT>
std::vector<typename HypergraphT::vertex_idx> GenericFM<HypergraphT>::GetMaxNodesOnLevel(typename HypergraphT::vertex_idx nrNodes,
                                                                                         unsigned nrParts) const {
    std::vector<IndexType> maxNodesOnLevel;
    std::vector<IndexType> limitPerLevel({static_cast<IndexType>(ceil(static_cast<double>(nrNodes) / 2.0))});
    for (unsigned parts = nrParts / 4; parts > 0; parts /= 2) {
        limitPerLevel.push_back(static_cast<IndexType>(ceil(static_cast<double>(limitPerLevel.back()) / 2.0)));
    }

    maxNodesOnLevel.push_back(maxNodesInPart_);
    for (unsigned parts = 2; parts < nrParts; parts *= 2) {
        IndexType nextLimit = maxNodesOnLevel.back() * 2;
        if (nextLimit > limitPerLevel.back()) {
            --nextLimit;
        }

        limitPerLevel.pop_back();
        maxNodesOnLevel.push_back(nextLimit);
    }

    std::reverse(maxNodesOnLevel.begin(), maxNodesOnLevel.end());
    return maxNodesOnLevel;
}

}    // namespace osp
