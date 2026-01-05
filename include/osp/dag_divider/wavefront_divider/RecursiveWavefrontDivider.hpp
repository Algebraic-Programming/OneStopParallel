/*
Copyright 2024 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,.
See the License for the specific language governing permissions and
limitations under the License.

@author Toni Boehnlein, Benjamin Lozes, Pal Andras Papp, Raphael S. Steiner
*/
#pragma once

#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>
#include <vector>

#include "AbstractWavefrontDivider.hpp"
#include "SequenceGenerator.hpp"
#include "SequenceSplitter.hpp"

namespace osp {

/**
 * @class RecursiveWavefrontDivider
 * @brief Recursively divides a DAG by applying a splitting algorithm to subgraphs.
 *
 * This divider first computes the wavefronts for the entire DAG. It then uses a
 * configured splitting algorithm to find all cut points. For each resulting
 * section, it recursively repeats the process, allowing for a hierarchical
 * division of the DAG.
 */
template <typename GraphT>
class RecursiveWavefrontDivider : public AbstractWavefrontDivider<GraphT> {
  public:
    constexpr static bool enableDebugPrint_ = true;

    RecursiveWavefrontDivider() {
        // Set a sensible default splitter on construction.
        UseLargestStepSplitter(3.0, 4);
    }

    std::vector<std::vector<std::vector<VertexIdxT<GraphT>>>> Divide(const GraphT &dag) override {
        this->dagPtr_ = &dag;
        if constexpr (enableDebugPrint_) {
            std::cout << "[DEBUG] Starting recursive-scan division." << std::endl;
        }

        auto globalLevelSets = this->ComputeWavefronts();
        if (globalLevelSets.empty()) {
            return {};
        }

        std::vector<std::vector<std::vector<VertexIdxT<GraphT>>>> allSections;
        DivideRecursive(globalLevelSets.cbegin(), globalLevelSets.cend(), globalLevelSets, allSections, 0);
        return allSections;
    }

    RecursiveWavefrontDivider &SetMetric(SequenceMetric metric) {
        sequenceMetric_ = metric;
        return *this;
    }

    RecursiveWavefrontDivider &UseVarianceSplitter(double mult, double threshold, size_t minLen = 1) {
        splitter_ = std::make_unique<VarianceSplitter>(mult, threshold, minLen);
        minSubseqLen_ = minLen;
        return *this;
    }

    RecursiveWavefrontDivider &UseLargestStepSplitter(double threshold, size_t minLen) {
        splitter_ = std::make_unique<LargestStepSplitter>(threshold, minLen);
        minSubseqLen_ = minLen;
        return *this;
    }

    RecursiveWavefrontDivider &UseThresholdScanSplitter(double diffThreshold, double absThreshold, size_t minLen = 1) {
        splitter_ = std::make_unique<ThresholdScanSplitter>(diffThreshold, absThreshold, minLen);
        minSubseqLen_ = minLen;
        return *this;
    }

    RecursiveWavefrontDivider &SetMaxDepth(size_t maxDepth) {
        maxDepth_ = maxDepth;
        return *this;
    }

  private:
    using VertexType = VertexIdxT<GraphT>;
    using LevelSetConstIterator = typename std::vector<std::vector<VertexType>>::const_iterator;
    using DifferenceType = typename std::iterator_traits<LevelSetConstIterator>::difference_type;

    SequenceMetric sequenceMetric_ = SequenceMetric::COMPONENT_COUNT;
    std::unique_ptr<SequenceSplitter> splitter_;
    size_t minSubseqLen_ = 4;
    size_t maxDepth_ = std::numeric_limits<size_t>::max();

    void DivideRecursive(LevelSetConstIterator levelBegin,
                         LevelSetConstIterator levelEnd,
                         const std::vector<std::vector<VertexType>> &globalLevelSets,
                         std::vector<std::vector<std::vector<VertexType>>> &allSections,
                         size_t currentDepth) const {
        const auto currentRangeSize = static_cast<size_t>(std::distance(levelBegin, levelEnd));
        size_t startLevelIdx = static_cast<size_t>(std::distance(globalLevelSets.cbegin(), levelBegin));
        size_t endLevelIdx = static_cast<size_t>(std::distance(globalLevelSets.cbegin(), levelEnd));

        // --- Base Cases for Recursion ---
        if (currentDepth >= maxDepth_ || currentRangeSize < minSubseqLen_) {
            if constexpr (enableDebugPrint_) {
                std::cout << "[DEBUG depth " << currentDepth << "] Base case reached. Creating section from levels "
                          << startLevelIdx << " to " << endLevelIdx << "." << std::endl;
            }
            // Ensure the section is not empty before adding
            if (startLevelIdx < endLevelIdx) {
                allSections.push_back(this->GetComponentsForRange(startLevelIdx, endLevelIdx, globalLevelSets));
            }
            return;
        }

        // --- Create a view of the levels for the current sub-problem ---
        std::vector<std::vector<VertexType>> subLevelSets(levelBegin, levelEnd);

        SequenceGenerator<GraphT> generator(*(this->dagPtr_), subLevelSets);
        std::vector<double> sequence = generator.Generate(sequenceMetric_);

        if constexpr (enableDebugPrint_) {
            std::cout << "[DEBUG depth " << currentDepth << "] Analyzing sequence: ";
            for (const auto &val : sequence) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }

        std::vector<size_t> localCuts = splitter_->Split(sequence);

        // --- Base Case: No further cuts found ---
        if (localCuts.empty()) {
            if constexpr (enableDebugPrint_) {
                std::cout << "[DEBUG depth " << currentDepth << "] No cuts found. Creating section from levels " << startLevelIdx
                          << " to " << endLevelIdx << "." << std::endl;
            }
            allSections.push_back(this->GetComponentsForRange(startLevelIdx, endLevelIdx, globalLevelSets));
            return;
        }

        if constexpr (enableDebugPrint_) {
            std::cout << "[DEBUG depth " << currentDepth << "] Found " << localCuts.size() << " cuts: ";
            for (const auto c : localCuts) {
                std::cout << c << ", ";
            }
            std::cout << "in level range [" << startLevelIdx << ", " << endLevelIdx << "). Recursing." << std::endl;
        }

        // --- Recurse on the new, smaller sub-problems ---
        std::sort(localCuts.begin(), localCuts.end());
        localCuts.erase(std::unique(localCuts.begin(), localCuts.end()), localCuts.end());

        auto currentSubBegin = levelBegin;
        for (const auto &localCutIdx : localCuts) {
            auto cutIterator = levelBegin + static_cast<DifferenceType>(localCutIdx);
            if (cutIterator > currentSubBegin) {
                DivideRecursive(currentSubBegin, cutIterator, globalLevelSets, allSections, currentDepth + 1);
            }
            currentSubBegin = cutIterator;
        }
        // Recurse on the final segment from the last cut to the end.
        if (currentSubBegin < levelEnd) {
            DivideRecursive(currentSubBegin, levelEnd, globalLevelSets, allSections, currentDepth + 1);
        }
    }
};

}    // namespace osp
