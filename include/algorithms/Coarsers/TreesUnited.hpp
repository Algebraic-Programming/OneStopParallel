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

#include "algorithms/InstanceContractor.hpp"

/**
 * @brief Parameters for TreesUnited coarsener
 * 
 */
struct TreesUnited_parameters {
    bool save_in_trees;
    bool save_out_trees;

    bool use_approx_transitive_reduction;

    bool first_save_in_tree = true;

    TreesUnited_parameters(bool save_in_trees_ = true, bool save_out_trees_ = true, bool use_approx_transitive_reduction_ = false, bool first_save_in_tree_ = true) : save_in_trees(save_in_trees_), save_out_trees(save_out_trees_), use_approx_transitive_reduction(use_approx_transitive_reduction_), first_save_in_tree(first_save_in_tree_) {};
    ~TreesUnited_parameters() = default;
};


/**
 * @brief Acyclic graph contractor that contracts trees from/insprired by (Zarebavani, Behrooz, et al. "HDagg: hybrid aggregation of loop-carried dependence iterations in sparse matrix computations." 2022 IEEE International Parallel and Distributed Processing Symposium (IPDPS). IEEE, 2022.)
 * 
 */
class TreesUnited : public InstanceContractor {
    private:
        TreesUnited_parameters parameters;

        void run_in_tree_contraction();
        void run_out_tree_contraction();

    protected:
        RETURN_STATUS run_contractions() override;

    public:
        TreesUnited(TreesUnited_parameters parameters_ = TreesUnited_parameters()) : InstanceContractor(), parameters(parameters_) { }
        TreesUnited(Scheduler* sched_, TreesUnited_parameters parameters_ = TreesUnited_parameters()) : TreesUnited(sched_, nullptr, parameters_) { }
        TreesUnited(Scheduler* sched_, ImprovementScheduler* improver_, TreesUnited_parameters parameters_ = TreesUnited_parameters()) : InstanceContractor(sched_, improver_), parameters(parameters_) { }
        TreesUnited(unsigned timelimit, Scheduler* sched_, TreesUnited_parameters parameters_ = TreesUnited_parameters()) : TreesUnited(timelimit, sched_, nullptr, parameters_) { }
        TreesUnited(unsigned timelimit, Scheduler* sched_, ImprovementScheduler* improver_, TreesUnited_parameters parameters_ = TreesUnited_parameters()) : InstanceContractor(timelimit, sched_, improver_), parameters(parameters_) { }
        virtual ~TreesUnited() = default;

        std::string getCoarserName() const override { return "TreesUnited"; }
};