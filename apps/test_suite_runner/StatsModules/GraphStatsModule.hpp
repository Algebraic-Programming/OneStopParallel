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

#include <map>
#include <string>
#include <vector>

#include "IStatsModule.hpp"
#include "osp/bsp/model/IBspScheduleEval.hpp"
#include "osp/graph_implementations/boost_graphs/boost_graph.hpp"    // For graph_t

namespace osp {

template <typename TargetObjectType>
class GraphStatsModule : public IStatisticModule<TargetObjectType> {
  public:
  private:
    const std::vector<std::string> metricHeaders_ = {"num_vertices", "num_edges", "avg_wavefront_size"};

  public:
    std::vector<std::string> GetMetricHeaders() const override { return metricHeaders_; }

    std::map<std::string, std::string> RecordStatistics(const TargetObjectType &schedule,
                                                        std::ofstream & /*log_stream*/) const override {
        std::map<std::string, std::string> stats;

        const auto &graph = schedule.GetInstance().GetComputationalDag();

        stats["num_vertices"] = std::to_string(graph.NumVertices());
        stats["num_edges"] = std::to_string(graph.NumEdges());
        stats["avg_wavefront_size"]
            = std::to_string(static_cast<double>(graph.NumVertices()) / static_cast<double>(longestPath(graph)));
        return stats;
    }
};

}    // namespace osp
