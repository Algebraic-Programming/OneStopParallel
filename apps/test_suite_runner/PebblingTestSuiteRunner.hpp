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
#include "StatsModules/IStatsModule.hpp"
#include "StringToScheduler/run_pebbler.hpp"
#include "osp/pebbling/PebblingSchedule.hpp"

namespace osp {

template <typename GraphT>
class BasicPebblingStatsModule : public IStatisticModule<PebblingSchedule<GraphT>> {
  public:
  private:
    const std::vector<std::string> metricHeaders_ = {"PebblingCost", "AsynchronousPebblingCost", "Supersteps"};

  public:
    std::vector<std::string> get_metric_headers() const override { return metric_headers; }

    std::map<std::string, std::string> record_statistics(const PebblingSchedule<GraphT> &schedule,
                                                         std::ofstream & /*log_stream*/) const override {
        std::map<std::string, std::string> stats;
        stats["PebblingCost"] = std::to_string(schedule.computeCosts());
        stats["AsynchronousPebblingCost"] = std::to_string(computeAsynchronousCost());
        stats["Supersteps"] = std::to_string(schedule.NumberOfSupersteps());
        return stats;
    }
};

template <typename ConcreteGraphT>
class PebblingTestSuiteRunner : public AbstractTestSuiteRunner<PebblingSchedule<ConcreteGraphT>, ConcreteGraphT> {
  private:
    bool useMemoryConstraint_;

  protected:
    ReturnStatus compute_target_object_impl(const BspInstance<ConcreteGraphT> &instance,
                                            std::unique_ptr<PebblingSchedule<concrete_graph_t>> &schedule,
                                            const pt::ptree &algoConfig,
                                            long long &computationTimeMs) override {
        schedule = std::make_unique<PebblingSchedule<ConcreteGraphT>>(instance);

        const auto startTime = std::chrono::high_resolution_clock::now();

        ReturnStatus status = run_pebbler(this->parser, algoConfig, *schedule);

        const auto finishTime = std::chrono::high_resolution_clock::now();
        computationTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(finishTime - startTime).count();

        return status;
    }

    void create_and_register_statistic_modules(const std::string &moduleName) override {
        if (moduleName == "BasicPebblingStats") {
            this->active_stats_modules.push_back(std::make_unique<BasicPebblingStatsModule<ConcreteGraphT>>());
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
    PebblingTestSuiteRunner() : AbstractTestSuiteRunner<PebblingSchedule<ConcreteGraphT>, ConcreteGraphT>() {}
};

}    // namespace osp
