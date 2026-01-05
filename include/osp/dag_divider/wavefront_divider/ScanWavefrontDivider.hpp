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
#include <memory>
#include <vector>

#include "AbstractWavefrontDivider.hpp"
#include "SequenceGenerator.hpp"
#include "SequenceSplitter.hpp"

namespace osp {

/**
 * @class ScanWavefrontDivider
 * @brief Divides a DAG by scanning all wavefronts and applying a splitting algorithm.
 * This revised version uses a fluent API for safer and clearer algorithm configuration.
 */
template <typename GraphT>
class ScanWavefrontDivider : public AbstractWavefrontDivider<GraphT> {
  public:
    constexpr static bool enableDebugPrint_ = true;

    ScanWavefrontDivider() { UseLargestStepSplitter(3.0, 4); }

    std::vector<std::vector<std::vector<VertexIdxT<GraphT>>>> Divide(const GraphT &dag) override {
        this->dagPtr_ = &dag;
        if constexpr (enableDebugPrint_) {
            std::cout << "[DEBUG] Starting scan-all division." << std::endl;
        }

        std::vector<std::vector<VertexIdxT<GraphT>>> levelSets = this->ComputeWavefronts();
        if (levelSets.empty()) {
            return {};
        }

        SequenceGenerator<GraphT> generator(dag, levelSets);
        std::vector<double> sequence = generator.Generate(sequenceMetric_);

        if constexpr (enableDebugPrint_) {
            std::cout << "[DEBUG]   Metric: " << static_cast<int>(sequenceMetric_) << std::endl;
            std::cout << "[DEBUG]   Generated sequence: ";
            for (const auto &val : sequence) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }

        std::vector<size_t> cutLevels = splitter_->Split(sequence);
        std::sort(cutLevels.begin(), cutLevels.end());
        cutLevels.erase(std::unique(cutLevels.begin(), cutLevels.end()), cutLevels.end());

        if constexpr (enableDebugPrint_) {
            std::cout << "[DEBUG]   Final cut levels: ";
            for (const auto &level : cutLevels) {
                std::cout << level << " ";
            }
            std::cout << std::endl;
        }

        return CreateVertexMapsFromCuts(cutLevels, levelSets);
    }

    ScanWavefrontDivider &SetMetric(SequenceMetric metric) {
        sequenceMetric_ = metric;
        return *this;
    }

    ScanWavefrontDivider &UseVarianceSplitter(double mult, double threshold, size_t minLen = 1) {
        splitter_ = std::make_unique<VarianceSplitter>(mult, threshold, minLen);
        return *this;
    }

    ScanWavefrontDivider &UseLargestStepSplitter(double threshold, size_t minLen) {
        splitter_ = std::make_unique<LargestStepSplitter>(threshold, minLen);
        return *this;
    }

    ScanWavefrontDivider &UseThresholdScanSplitter(double diffThreshold, double absThreshold, size_t minLen = 1) {
        splitter_ = std::make_unique<ThresholdScanSplitter>(diffThreshold, absThreshold, minLen);
        return *this;
    }

  private:
    using VertexType = VertexIdxT<GraphT>;

    SequenceMetric sequenceMetric_ = SequenceMetric::COMPONENT_COUNT;
    std::unique_ptr<SequenceSplitter> splitter_;

    std::vector<std::vector<std::vector<VertexType>>> CreateVertexMapsFromCuts(
        const std::vector<size_t> &cutLevels, const std::vector<std::vector<VertexType>> &levelSets) const {
        if (cutLevels.empty()) {
            // If there are no cuts, return a single section with all components.
            return {this->GetComponentsForRange(0, levelSets.size(), levelSets)};
        }

        std::vector<std::vector<std::vector<VertexType>>> vertexMaps;
        size_t startLevel = 0;

        for (const auto &cutLevel : cutLevels) {
            if (startLevel < cutLevel) {    // Avoid creating empty sections
                vertexMaps.push_back(this->GetComponentsForRange(startLevel, cutLevel, levelSets));
            }
            startLevel = cutLevel;
        }
        // Add the final section from the last cut to the end of the levels
        if (startLevel < levelSets.size()) {
            vertexMaps.push_back(this->GetComponentsForRange(startLevel, levelSets.size(), levelSets));
        }

        return vertexMaps;
    }
};

}    // namespace osp
