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

#include "scheduler/SSPImprovementScheduler.hpp"
#include "scheduler/SSPScheduler.hpp"
#include "model/BspInstance.hpp"
#include "model/BspSchedule.hpp"
#include "model/ComputationalDag.hpp"

class SSPInstanceContractor : public SSPScheduler {
  private:
    const BspInstance *original_inst;
  protected:
    const BspInstance * const getOriginalInstance() const { return original_inst; };

    SSPScheduler *sched;
    SSPImprovementScheduler *improver;

    std::vector<std::unique_ptr<BspInstance>> dag_history;
    std::vector<std::unique_ptr<std::unordered_map<VertexType, VertexType>>> contraction_maps;
    std::vector<std::unique_ptr<std::unordered_map<VertexType, std::set<VertexType>>>> expansion_maps;
    long int active_graph;
    SspSchedule active_schedule;

    RETURN_STATUS add_contraction(const std::vector<std::unordered_set<VertexType>> &partition);
    virtual RETURN_STATUS run_contractions() = 0;
    void compactify_dag_history();
    RETURN_STATUS compute_initial_schedule(unsigned stale);
    RETURN_STATUS expand_active_schedule();
    RETURN_STATUS improve_active_schedule();
    RETURN_STATUS run_expansions();

    void clear_computation_data();

  public:
    virtual void setTimeLimitSeconds(unsigned int limit) override;
    virtual void setTimeLimitHours(unsigned int limit) override;
    virtual void setUseMemoryConstraint(bool use_memory_constraint_) override;

    SSPInstanceContractor() : SSPScheduler(), original_inst(nullptr), sched(nullptr), improver(nullptr){};
    SSPInstanceContractor(SSPScheduler *sched_, SSPImprovementScheduler *improver_ = nullptr)
        : SSPScheduler(), original_inst(nullptr), sched(sched_), improver(improver_), active_graph(-1){};
    SSPInstanceContractor(unsigned timelimit, SSPScheduler *sched_, SSPImprovementScheduler *improver_)
        : SSPScheduler(timelimit), original_inst(nullptr), sched(sched_), improver(improver_), active_graph(-1){};
    virtual ~SSPInstanceContractor() = default;

    inline void setInitialScheduler(SSPScheduler *const sched_) { sched = sched_; };
    inline void setImprovementScheduler(SSPImprovementScheduler *const improver_) { improver = improver_; };

    std::pair<ComputationalDag, std::unordered_map<VertexType, VertexType>>
    get_contracted_graph_and_mapping(const ComputationalDag &graph);
    std::pair<RETURN_STATUS, SspSchedule> computeSspSchedule(const BspInstance &instance, unsigned stale) override;

    virtual std::string getCoarserName() const = 0;
    std::string getScheduleName() const override {
        if (improver == nullptr) {
            return "C" + getCoarserName() + "-S" + sched->getScheduleName();
        } else {
            return "C" + getCoarserName() + "-S" + sched->getScheduleName() + "-I" + improver->getScheduleName();
        }
    };

    static SspSchedule expand_schedule(const SspSchedule &schedule,
                                       std::pair<ComputationalDag, std::unordered_map<VertexType, VertexType>> pair,
                                       const BspInstance &instance);
};