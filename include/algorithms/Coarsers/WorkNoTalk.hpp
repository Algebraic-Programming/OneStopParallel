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

#include <cmath>

#include "algorithms/InstanceContractor.hpp"
#include "coarsen/coarsen_history.hpp"

/**
 * @brief Acyclic graph contractor
 * 
 * WARNING: DO NOT USE! IT SUCKS!!!
 */
class WorkNoTalk : public InstanceContractor {
    private:
        const CoarsenParams coarsen_par;
        const float comm_to_work_ratio;
        const unsigned min_nodes;
        Thue_Morse_Sequence thue_coin;
        Biased_Random balanced_random;

    protected:
        int run_and_add_contraction(const contract_edge_sort edge_sort_type);
        RETURN_STATUS run_contractions() override;

        unsigned workload;
        float comm_cost_multiplier;

        unsigned compute_communication();

    public:
        WorkNoTalk(CoarsenParams coarsen_par_ = CoarsenParams(), float comm_to_work_ratio_ = 2, unsigned min_nodes_ = 0) : InstanceContractor(), coarsen_par(coarsen_par_), comm_to_work_ratio(comm_to_work_ratio_), min_nodes(min_nodes_) { }
        WorkNoTalk(Scheduler* sched_, CoarsenParams coarsen_par_ = CoarsenParams(), float comm_to_work_ratio_ = 2, unsigned min_nodes_ = 0) : WorkNoTalk(sched_, nullptr, coarsen_par_, comm_to_work_ratio_, min_nodes_) { }
        WorkNoTalk(Scheduler* sched_, ImprovementScheduler* improver_, CoarsenParams coarsen_par_ = CoarsenParams(), float comm_to_work_ratio_ = 2, unsigned min_nodes_ = 0) : InstanceContractor(sched_, improver_), coarsen_par(coarsen_par_), comm_to_work_ratio(comm_to_work_ratio_), min_nodes(min_nodes_) { }
        WorkNoTalk(unsigned timelimit, Scheduler* sched_, CoarsenParams coarsen_par_ = CoarsenParams(), float comm_to_work_ratio_ = 2, unsigned min_nodes_ = 0) : WorkNoTalk(timelimit, sched_, nullptr, coarsen_par_, comm_to_work_ratio_, min_nodes_) { }
        WorkNoTalk(unsigned timelimit, Scheduler* sched_, ImprovementScheduler* improver_, CoarsenParams coarsen_par_ = CoarsenParams(), float comm_to_work_ratio_ = 2, unsigned min_nodes_ = 0) : InstanceContractor(timelimit, sched_, improver_), coarsen_par(coarsen_par_), comm_to_work_ratio(comm_to_work_ratio_), min_nodes(min_nodes_) { }
        virtual ~WorkNoTalk() = default;

        std::string getCoarserName() const override { return "WorkNoTalk"; }
};