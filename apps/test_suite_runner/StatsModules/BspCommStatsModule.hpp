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
#include "osp/bsp/model/BspSchedule.hpp"    // Still needed
#include "osp/bsp/model/cost/BufferedSendingCost.hpp"
#include "osp/bsp/model/cost/TotalCommunicationCost.hpp"
#include "osp/bsp/model/cost/TotalLambdaCommunicationCost.hpp"

namespace osp {

template <typename GraphT>
class BspCommStatsModule : public IStatisticModule<BspSchedule<GraphT>> {
  public:
  private:
    const std::vector<std::string> metricHeaders_ = {"TotalCommCost", "TotalLambdaCommCost", "BufferedSendingCosts"};

  public:
    std::vector<std::string> GetMetricHeaders() const override { return metricHeaders_; }

    std::map<std::string, std::string> RecordStatistics(const BspSchedule<GraphT> &schedule,
                                                        std::ofstream & /*log_stream*/) const override {
        std::map<std::string, std::string> stats;
        stats["TotalCommCost"] = std::to_string(TotalCommunicationCost<GraphT>()(schedule));
        stats["TotalLambdaCommCost"] = std::to_string(TotalLambdaCommunicationCost<GraphT>()(schedule));
        stats["BufferedSendingCosts"] = std::to_string(BufferedSendingCost<GraphT>()(schedule));
        return stats;
    }
};

}    // namespace osp
