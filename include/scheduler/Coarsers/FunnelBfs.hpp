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

#include <limits.h>

#include "scheduler/InstanceContractor.hpp"
#include "auxiliary/auxiliary.hpp"

/**
 * @brief Parameters for Funnel coarsener
 * 
 */
struct FunnelBfs_parameters {
    bool funnel_incoming;
    bool funnel_outgoing;
    bool first_funnel_incoming;

    bool use_approx_transitive_reduction;

    long unsigned max_work_weight;
    long unsigned max_memory_weight;
    unsigned max_depth;

    FunnelBfs_parameters(   long unsigned max_work_weight_ = ULONG_MAX,
                            long unsigned max_memory_weight_ = ULONG_MAX,
                            unsigned max_depth_ = UINT_MAX,
                            bool funnel_incoming_ = true,
                            bool funnel_outgoing_ = false,
                            bool first_funnel_incoming_ = true,
                            bool use_approx_transitive_reduction_ = true) :
                                funnel_incoming(funnel_incoming_),
                                funnel_outgoing(funnel_outgoing_),
                                first_funnel_incoming(first_funnel_incoming_),
                                use_approx_transitive_reduction(use_approx_transitive_reduction_),
                                max_work_weight(max_work_weight_),
                                max_memory_weight(max_memory_weight_),
                                max_depth(max_depth_) {};
    ~FunnelBfs_parameters() = default;
};


/**
 * @brief Acyclic graph contractor that contracts groups of nodes with only one vertex with incoming/outgoing edges (from outside the group)
 * 
 */
class FunnelBfs : public InstanceContractor {
    private:
        FunnelBfs_parameters parameters;

        bool use_architecture_memory_contraints;

        void run_in_contraction();

        void run_out_contraction();

        bool isCompatibleNodeType(const VertexType& new_node, const VertexType& old_node, const BspInstance& instance);

    protected:
        RETURN_STATUS run_contractions() override;

    public:
        FunnelBfs(FunnelBfs_parameters parameters_ = FunnelBfs_parameters()) : InstanceContractor(), parameters(parameters_), use_architecture_memory_contraints(false) { }
        FunnelBfs(Scheduler* sched_, FunnelBfs_parameters parameters_ = FunnelBfs_parameters()) : FunnelBfs(sched_, nullptr, parameters_) { }
        FunnelBfs(Scheduler* sched_, ImprovementScheduler* improver_, FunnelBfs_parameters parameters_ = FunnelBfs_parameters()) : InstanceContractor(sched_, improver_), parameters(parameters_), use_architecture_memory_contraints(false) { }
        FunnelBfs(unsigned timelimit, Scheduler* sched_, FunnelBfs_parameters parameters_ = FunnelBfs_parameters()) : FunnelBfs(timelimit, sched_, nullptr, parameters_) { }
        FunnelBfs(unsigned timelimit, Scheduler* sched_, ImprovementScheduler* improver_, FunnelBfs_parameters parameters_ = FunnelBfs_parameters()) : InstanceContractor(timelimit, sched_, improver_), parameters(parameters_), use_architecture_memory_contraints(false) { }
        virtual ~FunnelBfs() = default;

        std::string getCoarserName() const override { return "FunnelBfs"; }

        virtual void setUseMemoryConstraint(bool use_architecture_memory_contraints_) override {
            use_architecture_memory_contraints = use_architecture_memory_contraints_;
            if (sched) {
                sched->setUseMemoryConstraint(use_architecture_memory_contraints_);
            }
            if (improver) {
                improver->setUseMemoryConstraint(use_architecture_memory_contraints_);
            }
        }
};