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

#include "scheduler/Coarsers/FunnelBfs.hpp"
#include "scheduler/SSPInstanceContractor.hpp"
#include "auxiliary/auxiliary.hpp"


/**
 * @brief Acyclic graph contractor that contracts groups of nodes with only one vertex with incoming/outgoing edges (from outside the group)
 * 
 */
class FunnelBfsSSP : public SSPInstanceContractor {
    private:
        FunnelBfs_parameters parameters;

        bool use_architecture_memory_contraints;

        void run_in_contraction();

        void run_out_contraction();

        bool isCompatibleNodeType(const VertexType& new_node, const VertexType& old_node, const BspInstance& instance);

    protected:
        RETURN_STATUS run_contractions() override;

    public:
        FunnelBfsSSP(FunnelBfs_parameters parameters_ = FunnelBfs_parameters()) : SSPInstanceContractor(), parameters(parameters_), use_architecture_memory_contraints(false) { }
        FunnelBfsSSP(SSPScheduler* sched_, FunnelBfs_parameters parameters_ = FunnelBfs_parameters()) : FunnelBfsSSP(sched_, nullptr, parameters_) { }
        FunnelBfsSSP(SSPScheduler* sched_, SSPImprovementScheduler* improver_, FunnelBfs_parameters parameters_ = FunnelBfs_parameters()) : SSPInstanceContractor(sched_, improver_), parameters(parameters_), use_architecture_memory_contraints(false) { }
        FunnelBfsSSP(unsigned timelimit, SSPScheduler* sched_, FunnelBfs_parameters parameters_ = FunnelBfs_parameters()) : FunnelBfsSSP(timelimit, sched_, nullptr, parameters_) { }
        FunnelBfsSSP(unsigned timelimit, SSPScheduler* sched_, SSPImprovementScheduler* improver_, FunnelBfs_parameters parameters_ = FunnelBfs_parameters()) : SSPInstanceContractor(timelimit, sched_, improver_), parameters(parameters_), use_architecture_memory_contraints(false) { }
        virtual ~FunnelBfsSSP() = default;

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