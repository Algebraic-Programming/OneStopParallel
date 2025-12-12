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

#include <numeric>
#include <vector>

#include "WavefrontStatisticsCollector.hpp"

namespace osp {

enum class SequenceMetric { COMPONENT_COUNT, AVAILABLE_PARALLELISM };

/**
 * @class SequenceGenerator
 * @brief Helper to generate a numerical sequence based on a chosen metric.
 */
template <typename GraphT>
class SequenceGenerator {
    using VertexType = VertexIdxT<GraphT>;

  public:
    SequenceGenerator(const GraphT &dag, const std::vector<std::vector<VertexType>> &levelSets)
        : dag_(dag), level_sets_(level_sets) {}

    std::vector<double> Generate(SequenceMetric metric) const {
        switch (metric) {
            case SequenceMetric::AVAILABLE_PARALLELISM:
                return GenerateAvailableParallelism();
            case SequenceMetric::COMPONENT_COUNT:
            default:
                return GenerateComponentCount();
        }
    }

  private:
    std::vector<double> GenerateComponentCount() const {
        WavefrontStatisticsCollector<GraphT> collector(dag_, level_sets_);
        auto fwdStats = collector.compute_forward();
        std::vector<double> seq;
        seq.reserve(fwdStats.size());
        for (const auto &stat : fwdStats) {
            seq.push_back(static_cast<double>(stat.connected_components_vertices.size()));
        }
        return seq;
    }

    std::vector<double> GenerateAvailableParallelism() const {
        std::vector<double> seq;
        seq.reserve(level_sets_.size());
        double cumulativeWork = 0.0;
        for (size_t i = 0; i < level_sets_.size(); ++i) {
            double levelWork = 0.0;
            for (const auto &vertex : level_sets_[i]) {
                level_work += dag_.VertexWorkWeight(vertex);
            }
            cumulativeWork += levelWork;
            seq.push_back(cumulativeWork / (static_cast<double>(i) + 1.0));
        }
        return seq;
    }

    const GraphT &dag_;
    const std::vector<std::vector<VertexType>> &levelSets_;
};

}    // end namespace osp
