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

#include "AbstractTestSuiteRunner.hpp"
#include "StatsModules/BasicBspStatsModule.hpp"
#include "StatsModules/BspCommStatsModule.hpp"
#include "StatsModules/BspSptrsvStatsModule.hpp"
#include "StatsModules/GraphStatsModule.hpp"
#include "StringToScheduler/run_bsp_scheduler.hpp"
#include "osp/auxiliary/io/bsp_schedule_file_writer.hpp"
#include "osp/bsp/model/BspSchedule.hpp"

namespace osp {

template <typename ConcreteGraphT>
class BspScheduleTestSuiteRunner : public AbstractTestSuiteRunner<BspSchedule<ConcreteGraphT>, ConcreteGraphT> {
  private:
  protected:
    ReturnStatus ComputeTargetObjectImpl(const BspInstance<ConcreteGraphT> &instance,
                                         std::unique_ptr<BspSchedule<ConcreteGraphT>> &schedule,
                                         const pt::ptree &algoConfig,
                                         long long &computationTimeMs) override {
        schedule = std::make_unique<BspSchedule<ConcreteGraphT>>(instance);

        const auto startTime = std::chrono::high_resolution_clock::now();

        ReturnStatus status = run_bsp_scheduler(this->parser_, algoConfig, *schedule);

        const auto finishTime = std::chrono::high_resolution_clock::now();
        computationTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(finishTime - startTime).count();

        return status;
    }

    void CreateAndRegisterStatisticModules(const std::string &moduleName) override {
        if (moduleName == "BasicBspStats") {
            this->active_stats_modules.push_back(std::make_unique<BasicBspStatsModule<BspSchedule<ConcreteGraphT>>>());
        } else if (moduleName == "BspCommStats") {
            this->active_stats_modules.push_back(std::make_unique<BspCommStatsModule<ConcreteGraphT>>());
#ifdef EIGEN_FOUND
        } else if (moduleName == "BspSptrsvStats") {
            this->active_stats_modules.push_back(std::make_unique<BspSptrsvStatsModule<BspSchedule<ConcreteGraphT>>>(NO_PERMUTE));
        } else if (moduleName == "BspSptrsvPermLoopProcessorsStats") {
            this->active_stats_modules.push_back(
                std::make_unique<BspSptrsvStatsModule<BspSchedule<ConcreteGraphT>>>(LOOP_PROCESSORS));
        } else if (moduleName == "BspSptrsvPermSnakeProcessorsStats") {
            this->active_stats_modules.push_back(
                std::make_unique<BspSptrsvStatsModule<BspSchedule<ConcreteGraphT>>>(SNAKE_PROCESSORS));
#endif
        } else if (moduleName == "GraphStats") {
            this->active_stats_modules.push_back(std::make_unique<GraphStatsModule<BspSchedule<ConcreteGraphT>>>());
        }
    }

    // TODO
    // void write_target_object_hook(const BspSchedule<concrete_graph_t> &schedule, const std::string &graph_name,
    //                               const std::string &machine_name, const std::string &algo_name) const override {
    //     std::string file_path =
    //         this->output_target_object_dir_path + graph_name + "_" + machine_name + "_" + algo_name + "_schedule.txt";
    //     file_writer::write_txt(file_path, schedule);
    // }

  public:
    BspScheduleTestSuiteRunner() : AbstractTestSuiteRunner<BspSchedule<ConcreteGraphT>, ConcreteGraphT>() {}
};

}    // namespace osp
