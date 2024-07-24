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

#include <memory>
#include <numeric>
#include <set>
#include <vector>

#include "algorithms/ImprovementScheduler.hpp"
#include "algorithms/Scheduler.hpp"
#include "model/BspInstance.hpp"
#include "model/BspSchedule.hpp"
#include "model/ComputationalDag.hpp"

class InstanceContractor : public Scheduler {
  private:
  protected:
    const BspInstance *original_inst;

    Scheduler *sched;
    ImprovementScheduler *improver;

    std::vector<std::unique_ptr<BspInstance>> dag_history;
    std::vector<std::unique_ptr<std::unordered_map<VertexType, VertexType>>> contraction_maps;
    std::vector<std::unique_ptr<std::unordered_map<VertexType, std::set<VertexType>>>> expansion_maps;
    long int active_graph;
    BspSchedule active_schedule;

    RETURN_STATUS add_contraction(const std::vector<std::unordered_set<VertexType>> &partition);
    virtual RETURN_STATUS run_contractions() = 0;
    void compactify_dag_history();
    RETURN_STATUS compute_initial_schedule();
    RETURN_STATUS expand_active_schedule();
    RETURN_STATUS improve_active_schedule();
    RETURN_STATUS run_expansions();

    void clear_computation_data();

  public:
    virtual void setTimeLimitSeconds(unsigned int limit) override;
    virtual void setTimeLimitHours(unsigned int limit) override;

    InstanceContractor() : Scheduler(), original_inst(nullptr), sched(nullptr), improver(nullptr){};
    InstanceContractor(Scheduler *sched_, ImprovementScheduler *improver_ = nullptr)
        : Scheduler(), original_inst(nullptr), sched(sched_), improver(improver_), active_graph(-1){};
    InstanceContractor(unsigned timelimit, Scheduler *sched_, ImprovementScheduler *improver_)
        : Scheduler(timelimit), original_inst(nullptr), sched(sched_), improver(improver_), active_graph(-1){};
    virtual ~InstanceContractor() = default;

    inline void setInitialScheduler(Scheduler *const sched_) { sched = sched_; };
    inline void setImprovementScheduler(ImprovementScheduler *const improver_) { improver = improver_; };

    std::pair<ComputationalDag, std::unordered_map<VertexType, VertexType>>
    get_contracted_graph_and_mapping(const ComputationalDag &graph);
    std::pair<RETURN_STATUS, BspSchedule> computeSchedule(const BspInstance &instance) override;

    virtual std::string getCoarserName() const = 0;
    std::string getScheduleName() const override {
        if (improver == nullptr) {
            return "C" + getCoarserName() + "-S" + sched->getScheduleName();
        } else {
            return "C" + getCoarserName() + "-S" + sched->getScheduleName() + "-I" + improver->getScheduleName();
        }
    };

    static BspSchedule expand_schedule(const BspSchedule &schedule,
                                       std::pair<ComputationalDag, std::unordered_map<VertexType, VertexType>> pair,
                                       const BspInstance &instance);
};