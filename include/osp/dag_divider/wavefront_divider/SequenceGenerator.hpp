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

#include <vector>
#include <algorithm>
#include "WavefrontStatisticsCollector.hpp"

namespace osp {

enum class SequenceMetric { COMPONENT_COUNT, AVAILABLE_PARALLELISM };

/**
 * @class SequenceGenerator
 * @brief Helper to generate a numerical sequence based on a chosen metric.
 */
template<typename Graph_t>
class SequenceGenerator {
    using VertexType = vertex_idx_t<Graph_t>;

public:
    SequenceGenerator(const Graph_t& dag, const std::vector<std::vector<VertexType>>& level_sets)
        : dag_(dag), level_sets_(level_sets) {}

    std::vector<double> generate(SequenceMetric metric) {
        switch (metric) {
            case SequenceMetric::AVAILABLE_PARALLELISM:
                return generate_available_parallelism();
            case SequenceMetric::COMPONENT_COUNT:
            default:
                return generate_component_count();
        }
    }

private:
    std::vector<double> generate_component_count() {
        WavefrontStatisticsCollector<Graph_t> collector(dag_, level_sets_);
        auto fwd_stats = collector.compute_forward();
        std::vector<double> seq;
        seq.reserve(fwd_stats.size());
        for (const auto& stat : fwd_stats) {
            seq.push_back(static_cast<double>(stat.connected_components_vertices.size()));
        }
        return seq;
    }

    std::vector<double> generate_available_parallelism() {
        std::vector<double> seq;
        seq.reserve(level_sets_.size());
        double cumulative_work = 0.0;
        for (size_t i = 0; i < level_sets_.size(); ++i) {
            for (const auto& vertex : level_sets_[i]) {
                cumulative_work += dag_.vertex_work_weight(vertex);
            }
            seq.push_back(cumulative_work / (i + 1.0));
        }
        return seq;
    }

    const Graph_t& dag_;
    const std::vector<std::vector<VertexType>>& level_sets_;
};

};