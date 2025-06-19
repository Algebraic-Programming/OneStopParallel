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

#include "IStatisticModule.hpp"
#include "bsp/model/BspSchedule.hpp" // Still needed
#include "graph_implementations/boost_graphs/boost_graph.hpp" // For graph_t
#include <string>
#include <vector>
#include <map>

namespace osp {

// Assuming graph_t is osp::boost_graph_int_t for this context
using graph_t_for_stats = osp::boost_graph_int_t; 

class CommStatsModule : public IStatisticModule<osp::BspSchedule<graph_t_for_stats>> { 
public:

private:
    const std::vector<std::string> metric_headers = {
        "TotalCommCost", "BufferedSendingCosts"
    };

public:

    std::vector<std::string> get_metric_headers() const override {
        return metric_headers;
    }

    std::map<std::string, std::string> record_statistics(
                            const osp::BspSchedule<graph_t_for_stats>& schedule, 
                            std::ofstream& /*log_stream*/) const override {
        std::map<std::string, std::string> stats;
        stats["TotalCommCost"] = std::to_string(schedule.computeCostsTotalCommunication());
        stats["BufferedSendingCosts"] = std::to_string(schedule.computeCostsBufferedSending());
        return stats;
    }
};

} // namespace osp
