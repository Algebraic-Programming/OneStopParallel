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

#include "scheduler/InstanceContractor.hpp"
#include "scheduler/ContractRefineScheduler/coarsen/coarsen_history.hpp"

/**
 * @brief Acyclic graph contractor based on (Herrmann, Julien, et al. "Acyclic partitioning of large directed acyclic graphs." 2017 17th IEEE/ACM international symposium on cluster, cloud and grid computing (CCGRID). IEEE, 2017.))
 * @brief with additional improvements such as slowing down the coarsen, decisions when not to glue, and the introduction of randomness to the gluing criteria
 * 
 */
class SquashA : public InstanceContractor {
    private:
        const CoarsenParams coarsen_par;
        unsigned min_nodes;
        Thue_Morse_Sequence thue_coin;
        Biased_Random balanced_random;

    protected:
        int run_and_add_contraction(const contract_edge_sort edge_sort_type);
        RETURN_STATUS run_contractions() override;

    public:
        SquashA(CoarsenParams coarsen_par_ = CoarsenParams(), unsigned min_nodes_ = 0) : InstanceContractor(), coarsen_par(coarsen_par_), min_nodes(min_nodes_) { }
        SquashA(Scheduler* sched_, CoarsenParams coarsen_par_ = CoarsenParams(), unsigned min_nodes_ = 0) : SquashA(sched_, nullptr, coarsen_par_, min_nodes_) { }
        SquashA(Scheduler* sched_, ImprovementScheduler* improver_, CoarsenParams coarsen_par_ = CoarsenParams(), unsigned min_nodes_ = 0) : InstanceContractor(sched_, improver_), coarsen_par(coarsen_par_), min_nodes(min_nodes_) { }
        SquashA(unsigned timelimit, Scheduler* sched_, CoarsenParams coarsen_par_ = CoarsenParams(), unsigned min_nodes_ = 0) : SquashA(timelimit, sched_, nullptr, coarsen_par_, min_nodes_) { }
        SquashA(unsigned timelimit, Scheduler* sched_, ImprovementScheduler* improver_, CoarsenParams coarsen_par_ = CoarsenParams(), unsigned min_nodes_ = 0) : InstanceContractor(timelimit, sched_, improver_), coarsen_par(coarsen_par_), min_nodes(min_nodes_) { }
        virtual ~SquashA() = default;

        std::string getCoarserName() const override { return "SquashA"; }
};