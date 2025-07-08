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

#include <set>

#include "AbstractTestSuiteRunner.hpp"
#include "StatsModules/BasicBspStatsModule.hpp"
#include "StatsModules/GraphStatsModule.hpp"
#include "StringToScheduler/run_bsp_recomp_scheduler.hpp"
#include "StringToScheduler/run_bsp_scheduler.hpp"
#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/model/BspScheduleRecomp.hpp"
#include "osp/bsp/model/IBspScheduleEval.hpp"
#include "osp/auxiliary/io/bsp_schedule_file_writer.hpp"

namespace osp {

template<typename concrete_graph_t>
class BspScheduleRecompTestSuiteRunner
    : public AbstractTestSuiteRunner<IBspScheduleEval<concrete_graph_t>, concrete_graph_t> {
  private:
    bool use_memory_constraint_for_bsp;

  protected:
    RETURN_STATUS compute_target_object_impl(const BspInstance<concrete_graph_t> &instance, std::unique_ptr<IBspScheduleEval<concrete_graph_t>>& schedule, const pt::ptree &algo_config,
                                             long long &computation_time_ms) override {

        std::string algo_name = algo_config.get_child("name").get_value<std::string>();
        const std::set<std::string> scheduler_names = get_available_bsp_scheduler_names();
        const std::set<std::string> scheduler_recomp_names = get_available_bsp_recomp_scheduler_names();

        if (scheduler_names.find(algo_name) != scheduler_names.end()) {

            auto bsp_schedule = std::make_unique<BspSchedule<concrete_graph_t>>(instance);

            const auto start_time = std::chrono::high_resolution_clock::now();

            RETURN_STATUS status = run_bsp_scheduler(this->parser, algo_config, *bsp_schedule);

            const auto finish_time = std::chrono::high_resolution_clock::now();
            computation_time_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(finish_time - start_time).count();

            schedule = std::move(bsp_schedule);

            return status;

        } else if (scheduler_recomp_names.find(algo_name) != scheduler_recomp_names.end()) {

            auto bsp_recomp_schedule = std::make_unique<BspScheduleRecomp<concrete_graph_t>>(instance);

            const auto start_time = std::chrono::high_resolution_clock::now();

            RETURN_STATUS status = run_bsp_recomp_scheduler(this->parser, algo_config, *bsp_recomp_schedule);

            const auto finish_time = std::chrono::high_resolution_clock::now();
            computation_time_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(finish_time - start_time).count();

            schedule = std::move(bsp_recomp_schedule);

            return status;
        } else {

            std::cerr << "No matching category found for algorithm" << std::endl;
            return RETURN_STATUS::ERROR;
        }
    }

    void create_and_register_statistic_modules(const std::string &module_name) override {
        if (module_name == "BasicBspStats") {
            this->active_stats_modules.push_back(
                std::make_unique<BasicBspStatsModule<IBspScheduleEval<concrete_graph_t>>>());
        } else if (module_name == "GraphStats") {
            this->active_stats_modules.push_back(
                std::make_unique<GraphStatsModule<IBspScheduleEval<concrete_graph_t>>>());
        }
    }

    // TODO
    // void write_target_object_hook(const BspSchedule<concrete_graph_t> &schedule, const std::string &graph_name,
    //                               const std::string &machine_name, const std::string &algo_name) const override {
    //     std::string file_path =
    //         this->output_target_object_dir_path + graph_name + "_" + machine_name + "_" + algo_name +
    //         "_schedule.txt";
    //     file_writer::write_txt(file_path, schedule);
    // }

  public:
    BspScheduleRecompTestSuiteRunner() : AbstractTestSuiteRunner<IBspScheduleEval<concrete_graph_t>, concrete_graph_t>() {}
};

} // namespace osp
