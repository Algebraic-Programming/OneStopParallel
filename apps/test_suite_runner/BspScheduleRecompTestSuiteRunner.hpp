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
#include "osp/auxiliary/io/bsp_schedule_file_writer.hpp"
#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/model/BspScheduleRecomp.hpp"
#include "osp/bsp/model/IBspScheduleEval.hpp"

namespace osp {

template <typename ConcreteGraphT>
class BspScheduleRecompTestSuiteRunner : public AbstractTestSuiteRunner<IBspScheduleEval<ConcreteGraphT>, ConcreteGraphT> {
  private:
    bool useMemoryConstraintForBsp_;

  protected:
    ReturnStatus ComputeTargetObjectImpl(const BspInstance<ConcreteGraphT> &instance,
                                         std::unique_ptr<IBspScheduleEval<ConcreteGraphT>> &schedule,
                                         const pt::ptree &algoConfig,
                                         long long &computationTimeMs) override {
        std::string algoName = algoConfig.get_child("id").get_value<std::string>();
        const std::set<std::string> schedulerNames = GetAvailableBspSchedulerNames();
        const std::set<std::string> schedulerRecompNames = GetAvailableBspRecompSchedulerNames();

        if (schedulerNames.find(algoName) != schedulerNames.end()) {
            auto bspSchedule = std::make_unique<BspSchedule<ConcreteGraphT>>(instance);

            const auto startTime = std::chrono::high_resolution_clock::now();

            ReturnStatus status = RunBspScheduler(this->parser_, algoConfig, *bspSchedule);

            const auto finishTime = std::chrono::high_resolution_clock::now();
            computationTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(finishTime - startTime).count();

            schedule = std::move(bspSchedule);

            return status;

        } else if (schedulerRecompNames.find(algoName) != schedulerRecompNames.end()) {
            auto bspRecompSchedule = std::make_unique<BspScheduleRecomp<ConcreteGraphT>>(instance);

            const auto startTime = std::chrono::high_resolution_clock::now();

            ReturnStatus status = RunBspRecompScheduler(this->parser_, algoConfig, *bspRecompSchedule);

            const auto finishTime = std::chrono::high_resolution_clock::now();
            computationTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(finishTime - startTime).count();

            schedule = std::move(bspRecompSchedule);

            return status;
        } else {
            std::cerr << "No matching category found for algorithm" << std::endl;
            return ReturnStatus::ERROR;
        }
    }

    void CreateAndRegisterStatisticModules(const std::string &moduleName) override {
        if (moduleName == "BasicBspStats") {
            this->activeStatsModules_.push_back(std::make_unique<BasicBspStatsModule<IBspScheduleEval<ConcreteGraphT>>>());
        } else if (moduleName == "GraphStats") {
            this->activeStatsModules_.push_back(std::make_unique<GraphStatsModule<IBspScheduleEval<ConcreteGraphT>>>());
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
    BspScheduleRecompTestSuiteRunner() : AbstractTestSuiteRunner<IBspScheduleEval<ConcreteGraphT>, ConcreteGraphT>() {}
};

}    // namespace osp
