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

#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/model/BspScheduleCS.hpp"

namespace osp {
namespace file_writer {

template <typename GraphT>
void WriteTxt(std::ostream &os, const BspSchedule<GraphT> &schedule) {
    os << "%% BspSchedule for " << schedule.GetInstance().NumberOfProcessors() << " processors and "
       << schedule.NumberOfSupersteps() << " supersteps." << std::endl;
    os << schedule.GetInstance().NumberOfVertices() << " " << schedule.GetInstance().NumberOfProcessors() << " "
       << schedule.NumberOfSupersteps() << std::endl;

    for (const auto &vertex : schedule.GetInstance().GetComputationalDag().vertices()) {
        os << vertex << " " << schedule.AssignedProcessor(vertex) << " " << schedule.AssignedSuperstep(vertex) << std::endl;
    }
}

template <typename GraphT>
void WriteTxt(const std::string &filename, const BspSchedule<GraphT> &schedule) {
    std::ofstream os(filename);
    WriteTxt(os, schedule);
}

template <typename GraphT>
void WriteTxt(std::ostream &os, const BspScheduleCS<GraphT> &schedule) {
    os << "%% BspSchedule for " << schedule.GetInstance().NumberOfProcessors() << " processors and "
       << schedule.NumberOfSupersteps() << " supersteps." << std::endl;
    os << schedule.GetInstance().NumberOfVertices() << " " << schedule.GetInstance().NumberOfProcessors() << " "
       << schedule.NumberOfSupersteps() << " ";
    if (schedule.GetCommunicationSchedule().empty()) {
        os << 0 << " ";
    } else {
        os << 1 << " ";
    }

    os << std::endl;

    for (const auto &vertex : schedule.GetInstance().GetComputationalDag().Vertices()) {
        os << vertex << " " << schedule.AssignedProcessor(vertex) << " " << schedule.AssignedSuperstep(vertex) << std::endl;
    }

    if (schedule.GetCommunicationSchedule().empty()) {
        os << "%% No communication schedule available." << std::endl;
    } else {
        os << "%% Communication schedule available." << std::endl;

        for (const auto &[key, val] : schedule.GetCommunicationSchedule()) {
            os << std::get<0>(key) << " " << std::get<1>(key) << " " << std::get<2>(key) << " " << val << std::endl;
        }
    }
}

template <typename GraphT>
void WriteTxt(const std::string &filename, const BspScheduleCS<GraphT> &schedule) {
    std::ofstream os(filename);
    WriteTxt(os, schedule);
}

template <typename GraphT>
void WriteSankey(std::ostream &os, const BspScheduleCS<GraphT> &schedule) {
    // Computing workloads
    std::vector<std::vector<VWorkwT<GraphT>>> procWorkloads(
        schedule.NumberOfSupersteps(), std::vector<VWorkwT<GraphT>>(schedule.GetInstance().NumberOfProcessors(), 0));

    for (size_t node = 0; node < schedule.GetInstance().NumberOfVertices(); node++) {
        procWorkloads[schedule.AssignedSuperstep(node)][schedule.AssignedProcessor(node)]
            += schedule.GetInstance().GetComputationalDag().VertexWorkWeight(node);
    }

    // Computing communicationloads
    std::vector<std::vector<std::vector<VCommwT<GraphT>>>> commloads(
        schedule.NumberOfSupersteps() - 1,
        std::vector<std::vector<VCommwT<GraphT>>>(schedule.GetInstance().NumberOfProcessors(),
                                                  std::vector<VCommwT<GraphT>>(schedule.GetInstance().NumberOfProcessors(), 0)));

    for (const auto &[commTriple, sstep] : schedule.GetCommunicationSchedule()) {
        commloads[sstep][std::get<1>(commTriple)][std::get<2>(commTriple)]
            += schedule.GetInstance().GetComputationalDag().VertexCommWeight(std::get<0>(commTriple));
    }

    os << "BspSchedule: Number of Processors, Number of Supersteps" << std::endl;
    os << schedule.GetInstance().NumberOfProcessors() << "," << schedule.NumberOfSupersteps() << std::endl;

    os << "Processor workloads in Superstep" << std::endl;
    for (const auto &sstep : procWorkloads) {
        for (size_t procInd = 0; procInd < sstep.size(); procInd++) {
            if (procInd != 0) {
                os << ",";
            }
            os << sstep[procInd];
        }
        os << std::endl;
    }

    os << "Communication between Processors in Supersteps" << std::endl;
    for (size_t sstep = 0; sstep < commloads.size(); sstep++) {
        for (size_t sendProc = 0; sendProc < schedule.GetInstance().NumberOfProcessors(); sendProc++) {
            for (size_t receiveProc = 0; receiveProc < schedule.GetInstance().NumberOfProcessors(); receiveProc++) {
                // if (commloads[ sstep ][ send_proc ][ receive_proc ] == 0) continue;
                os << sstep + 1 << "," << sendProc + 1 << "," << receiveProc + 1 << "," << commloads[sstep][sendProc][receiveProc]
                   << std::endl;
            }
        }
    }
}

template <typename GraphT>
void WriteSankey(const std::string &filename, const BspScheduleCS<GraphT> &schedule) {
    std::ofstream os(filename);
    WriteSankey(os, schedule);
}

}    // namespace file_writer
}    // namespace osp
