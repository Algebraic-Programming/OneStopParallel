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

const std::set<std::string> get_available_bsp_recomp_scheduler_names() { return {"GreedyRecomputer"}; }

template <typename Graph_t>
RETURN_STATUS run_bsp_recomp_scheduler(const ConfigParser &parser,
                                       const boost::property_tree::ptree &algorithm,
                                       BspScheduleRecomp<Graph_t> &schedule) {
    // const unsigned timeLimit = parser.global_params.get_child("timeLimit").get_value<unsigned>();
    //  const bool use_memory_constraint = parser.global_params.get_child("use_memory_constraints").get_value<bool>();

    std::cout << "Running algorithm: " << algorithm.get_child("name").get_value<std::string>() << std::endl;

    if (algorithm.get_child("name").get_value<std::string>() == "GreedyRecomputer") {
        BspSchedule<Graph_t> bsp_schedule(schedule.getInstance());

        RETURN_STATUS status = run_bsp_scheduler(parser, algorithm.get_child("parameters").get_child("scheduler"), bsp_schedule);

        BspScheduleCS<Graph_t> initial_schedule(std::move(bsp_schedule));

        if (status == RETURN_STATUS::ERROR) {
            return RETURN_STATUS::ERROR;
        }

        GreedyRecomputer<Graph_t> scheduler;

        return scheduler.computeRecompScheduleBasic(initial_schedule, schedule);

    } else {
        throw std::invalid_argument("Parameter error: Unknown algorithm.\n");
    }
}

}    // namespace osp
