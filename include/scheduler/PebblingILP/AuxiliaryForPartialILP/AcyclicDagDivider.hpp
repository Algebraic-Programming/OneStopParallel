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

#include "scheduler/Scheduler.hpp"
#include "model/BspInstance.hpp"
#include "scheduler/PebblingILP/AuxiliaryForPartialILP/AcyclicPartitioningILP.hpp"

class AcyclicDagDivider {

  protected:

    std::vector<unsigned> node_to_part;

    unsigned minPartitionSize = 40, maxPartitionSize = 80;
    bool ignore_sources_in_size = true;

    std::vector<unsigned> getTopologicalSplit(const ComputationalDag &G, std::pair<unsigned, unsigned> min_and_max, const std::vector<bool>& is_original_source) const;

    unsigned static getSplitCost(const ComputationalDag &G, const std::vector<unsigned>& node_to_part);

  public:
    AcyclicDagDivider() {}

    virtual ~AcyclicDagDivider() = default;

    std::vector<unsigned> computePartitioning(const BspInstance &instance);

    // getters and setters for problem parameters
    inline std::pair<unsigned, unsigned> getMinAndMaxSize() const { return std::make_pair(minPartitionSize, maxPartitionSize); }
    inline void setMinAndMaxSize(const std::pair<unsigned, unsigned> min_and_max) {minPartitionSize = min_and_max.first; maxPartitionSize = min_and_max.second; }
    inline void setIgnoreSources(const bool ignore_) {ignore_sources_in_size = ignore_; }
};