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

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <filesystem>
#include <iostream>
#include <string>
#include <tuple>

#include "../ConfigParser.hpp"
#include "osp/bsp/model/BspScheduleRecomp.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyBspScheduler.hpp"
#include "osp/bsp/scheduler/GreedySchedulers/GreedyRecomputer.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"
#include "run_bsp_scheduler.hpp"

namespace osp {

const std::set<std::string> GetAvailableBspRecompSchedulerNames() { return {"GreedyRecomputer"}; }

template <typename GraphT>
ReturnStatus RunBspRecompScheduler(const ConfigParser &parser,
                                   const boost::property_tree::ptree &algorithm,
                                   BspScheduleRecomp<GraphT> &schedule) {
    // const unsigned timeLimit = parser.global_params.get_child("timeLimit").get_value<unsigned>();
    //  const bool use_memory_constraint = parser.global_params.get_child("use_memory_constraints").get_value<bool>();

    std::cout << "Running algorithm: " << algorithm.get_child("name").get_value<std::string>() << std::endl;

    if (algorithm.get_child("name").get_value<std::string>() == "GreedyRecomputer") {
        BspSchedule<GraphT> bspSchedule(schedule.GetInstance());

        ReturnStatus status = RunBspScheduler(parser, algorithm.get_child("parameters").get_child("scheduler"), bspSchedule);

        BspScheduleCS<GraphT> initialSchedule(std::move(bspSchedule));

        if (status == ReturnStatus::ERROR) {
            return ReturnStatus::ERROR;
        }

        GreedyRecomputer<GraphT> scheduler;

        return scheduler.ComputeRecompScheduleBasic(initialSchedule, schedule);

    } else {
        throw std::invalid_argument("Parameter error: Unknown algorithm.\n");
    }
}

}    // namespace osp
