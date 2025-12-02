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

#include "IStatsModule.hpp"
#include "osp/bsp/model/BspSchedule.hpp" // Still needed
#include "osp/bsp/model/cost/BufferedSendingCost.hpp"
#include "osp/bsp/model/cost/TotalCommunicationCost.hpp"
#include "osp/bsp/model/cost/TotalLambdaCommunicationCost.hpp"
#include <map>
#include <string>
#include <vector>

namespace osp {

template<typename Graph_t>
class BspCommStatsModule : public IStatisticModule<BspSchedule<Graph_t>> {
  public:
  private:
    const std::vector<std::string> metric_headers = {
        "TotalCommCost", "TotalLambdaCommCost", "BufferedSendingCosts"};

  public:
    std::vector<std::string> get_metric_headers() const override {
        return metric_headers;
    }

    std::map<std::string, std::string> record_statistics(
        const BspSchedule<Graph_t> &schedule,
        std::ofstream & /*log_stream*/) const override {
        std::map<std::string, std::string> stats;
        stats["TotalCommCost"] = std::to_string(TotalCommunicationCost<Graph_t>()(schedule));
        stats["TotalLambdaCommCost"] = std::to_string(TotalLambdaCommunicationCost<Graph_t>()(schedule));
        stats["BufferedSendingCosts"] = std::to_string(BufferedSendingCost<Graph_t>()(schedule));
        return stats;
    }
};

} // namespace osp
