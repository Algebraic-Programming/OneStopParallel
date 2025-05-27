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

#include "bsp/model/BspSchedule.hpp"
#include "bsp/model/BspScheduleCS.hpp"
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
            os << get<0>(key) << " " << get<1>(key) << " " << get<2>(key) << " " << val << std::endl;
        }
    }
}

template<typename Graph_t>
void write_txt(const std::string &filename, const BspScheduleCS<Graph_t> &schedule) {
    std::ofstream os(filename);
    write_txt(os, schedule);
}

}} // namespace osp::file_writer