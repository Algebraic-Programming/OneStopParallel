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

#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/model/BspScheduleCS.hpp"
#include <fstream>
#include <iostream>

namespace osp { namespace file_writer {

template<typename Graph_t>
void write_txt(std::ostream &os, const BspSchedule<Graph_t> &schedule) {

    os << "\%\% BspSchedule for " << schedule.getInstance().numberOfProcessors() << " processors and "
       << schedule.numberOfSupersteps() << " supersteps." << std::endl;
    os << schedule.getInstance().numberOfVertices() << " " << schedule.getInstance().numberOfProcessors() << " "
       << schedule.numberOfSupersteps() << std::endl;

    for (const auto &vertex : schedule.getInstance().getComputationalDag().vertices()) {
        os << vertex << " " << schedule.assignedProcessor(vertex) << " " << schedule.assignedSuperstep(vertex)
           << std::endl;
    }
}

template<typename Graph_t>
void write_txt(const std::string &filename, const BspSchedule<Graph_t> &schedule) {
    std::ofstream os(filename);
    write_txt(os, schedule);
}

template<typename Graph_t>
void write_txt(std::ostream &os, const BspScheduleCS<Graph_t> &schedule) {

    os << "\%\% BspSchedule for " << schedule.getInstance().numberOfProcessors() << " processors and "
       << schedule.numberOfSupersteps() << " supersteps." << std::endl;
    os << schedule.getInstance().numberOfVertices() << " " << schedule.getInstance().numberOfProcessors() << " "
       << schedule.numberOfSupersteps() << " ";
    if (schedule.getCommunicationSchedule().empty()) {
        os << 0 << " ";
    } else {
        os << 1 << " ";
    }

    os << std::endl;

    for (const auto &vertex : schedule.getInstance().getComputationalDag().vertices()) {
        os << vertex << " " << schedule.assignedProcessor(vertex) << " " << schedule.assignedSuperstep(vertex)
           << std::endl;
    }

    if (schedule.getCommunicationSchedule().empty()) {
        os << "\%\% No communication schedule available." << std::endl;
    } else {

        os << "\%\% Communication schedule available." << std::endl;

        for (const auto &[key, val] : schedule.getCommunicationSchedule()) {
            os << std::get<0>(key) << " " << std::get<1>(key) << " " << std::get<2>(key) << " " << val << std::endl;
        }
    }
}

template<typename Graph_t>
void write_txt(const std::string &filename, const BspScheduleCS<Graph_t> &schedule) {
    std::ofstream os(filename);
    write_txt(os, schedule);
}

template<typename Graph_t>
void write_sankey(std::ostream &os, const BspScheduleCS<Graph_t> &schedule) {
    // Computing workloads
    std::vector<std::vector<v_workw_t<Graph_t>>> proc_workloads(
        schedule.numberOfSupersteps(), std::vector<v_workw_t<Graph_t>>(schedule.getInstance().numberOfProcessors(), 0));

    for (size_t node = 0; node < schedule.getInstance().numberOfVertices(); node++) {
        proc_workloads[schedule.assignedSuperstep(node)][schedule.assignedProcessor(node)] +=
            schedule.getInstance().getComputationalDag().vertex_work_weight(node);
    }

    // Computing communicationloads
    std::vector<std::vector<std::vector<v_commw_t<Graph_t>>>> commloads(
                      schedule.numberOfSupersteps() - 1,
                      std::vector<std::vector<v_commw_t<Graph_t>>>(
                          schedule.getInstance().numberOfProcessors(),
                          std::vector<v_commw_t<Graph_t>>(schedule.getInstance().numberOfProcessors(), 0)));

    for (const auto &[comm_triple, sstep] : schedule.getCommunicationSchedule()) {
        commloads[sstep][std::get<1>(comm_triple)][std::get<2>(comm_triple)] +=
            schedule.getInstance().getComputationalDag().vertex_comm_weight(std::get<0>(comm_triple));
    }

    os << "BspSchedule: Number of Processors, Number of Supersteps" << std::endl;
    os << schedule.getInstance().numberOfProcessors() << "," << schedule.numberOfSupersteps() << std::endl;

    os << "Processor workloads in Superstep" << std::endl;
    for (const auto &sstep : proc_workloads) {
        for (size_t proc_ind = 0; proc_ind < sstep.size(); proc_ind++) {
            if (proc_ind != 0) {
                os << ",";
            }
            os << sstep[proc_ind];
        }
        os << std::endl;
    }

    os << "Communication between Processors in Supersteps" << std::endl;
    for (size_t sstep = 0; sstep < commloads.size(); sstep++) {
        for (size_t send_proc = 0; send_proc < schedule.getInstance().numberOfProcessors(); send_proc++) {
            for (size_t receive_proc = 0; receive_proc < schedule.getInstance().numberOfProcessors(); receive_proc++) {
                // if (commloads[ sstep ][ send_proc ][ receive_proc ] == 0) continue;
                os << sstep + 1 << "," << send_proc + 1 << "," << receive_proc + 1 << ","
                   << commloads[sstep][send_proc][receive_proc] << std::endl;
            }
        }
    }
}

template<typename Graph_t>
void write_sankey(const std::string &filename, const BspScheduleCS<Graph_t> &schedule) {
    std::ofstream os(filename);
    write_sankey(os, schedule);
}

}} // namespace osp::file_writer