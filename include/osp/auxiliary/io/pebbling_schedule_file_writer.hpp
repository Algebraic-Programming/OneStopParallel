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

#include <fstream>
#include <iostream>

#include "osp/pebbling/PebblingSchedule.hpp"

namespace osp {
namespace file_writer {

template <typename GraphT>
void WriteTxt(std::ostream &os, const PebblingSchedule<GraphT> &schedule) {
    using vertex_idx = vertex_idx_t<Graph_t>;

    os << "%% PebblingSchedule for " << schedule.getInstance().numberOfProcessors() << " processors and "
       << schedule.numberOfSupersteps() << " supersteps." << std::endl;

    for (unsigned step = 0; step < schedule.numberOfSupersteps(); ++step) {
        for (unsigned proc = 0; proc < schedule.getInstance().numberOfProcessors(); ++proc) {
            const auto &computeSteps = schedule.GetComputeStepsForProcSuperstep(proc, step);
            for (const auto &computeStep : computeSteps) {
                os << "Compute " << computeStep.node << " on proc " << proc << " in superstep " << step << std::endl;
                for (vertex_idx to_evict : computeStep.nodes_evicted_after) {
                    os << "Evict " << to_evict << " from proc " << proc << " in superstep " << step << std::endl;
                }
            }
        }
        for (unsigned proc = 0; proc < schedule.getInstance().numberOfProcessors(); ++proc) {
            const std::vector<vertex_idx> &nodesSentUp = schedule.GetNodesSentUp(proc, step);
            for (vertex_idx node : nodesSentUp) {
                os << "Send up " << node << " from proc " << proc << " in superstep " << step << std::endl;
            }
        }
        for (unsigned proc = 0; proc < schedule.getInstance().numberOfProcessors(); ++proc) {
            const std::vector<vertex_idx> &nodesEvictedInComm = schedule.GetNodesEvictedInComm(proc, step);
            for (vertex_idx node : nodesEvictedInComm) {
                os << "Evict " << node << " from proc " << proc << " in superstep " << step << std::endl;
            }
        }
        for (unsigned proc = 0; proc < schedule.getInstance().numberOfProcessors(); ++proc) {
            const std::vector<vertex_idx> &nodesSentDown = schedule.GetNodesSentDown(proc, step);
            for (vertex_idx node : nodesSentDown) {
                os << "Send down " << node << " to proc " << proc << " in superstep " << step << std::endl;
            }
        }
    }
}

template <typename GraphT>
void WriteTxt(const std::string &filename, const PebblingSchedule<GraphT> &schedule) {
    std::ofstream os(filename);
    write_txt(os, schedule);
}

}    // namespace file_writer
}    // namespace osp
