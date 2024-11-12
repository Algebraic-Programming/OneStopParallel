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

#include "scheduler/Coarsers/Funnel.hpp"
#include "scheduler/SSPInstanceContractor.hpp"
#include "auxiliary/auxiliary.hpp"


/**
 * @brief Acyclic graph contractor that contracts groups of nodes with only one vertex with incoming/outgoing edges (from outside the group)
 * 
 */
class FunnelSSP : public SSPInstanceContractor {
    private:
        Funnel_parameters parameters;

        void expand_in_group_dfs(const std::unordered_set<EdgeType, EdgeType_hash>& edge_mask, std::unordered_set<VertexType>& group, std::unordered_map<VertexType, unsigned>& children_not_in_group, long unsigned& group_weight, const double& max_weight, const VertexType active_node, const VertexType sink_node, bool& failed_to_add);
        void run_in_contraction();

        void expand_out_group_dfs(const std::unordered_set<EdgeType, EdgeType_hash>& edge_mask, std::unordered_set<VertexType>& group, std::unordered_map<VertexType, unsigned>& parents_not_in_group, long unsigned& group_weight, const double& max_weight, const VertexType active_node, const VertexType source_node, bool& failed_to_add);
        void run_out_contraction();

        bool isCompatibleNodeType(const VertexType& new_node, const VertexType& old_node, const BspInstance& instance);

    protected:
        RETURN_STATUS run_contractions() override;

    public:
        FunnelSSP(Funnel_parameters parameters_ = Funnel_parameters()) : SSPInstanceContractor(), parameters(parameters_) { }
        FunnelSSP(SSPScheduler* sched_, Funnel_parameters parameters_ = Funnel_parameters()) : FunnelSSP(sched_, nullptr, parameters_) { }
        FunnelSSP(SSPScheduler* sched_, SSPImprovementScheduler* improver_, Funnel_parameters parameters_ = Funnel_parameters()) : SSPInstanceContractor(sched_, improver_), parameters(parameters_) { }
        FunnelSSP(unsigned timelimit, SSPScheduler* sched_, Funnel_parameters parameters_ = Funnel_parameters()) : FunnelSSP(timelimit, sched_, nullptr, parameters_) { }
        FunnelSSP(unsigned timelimit, SSPScheduler* sched_, SSPImprovementScheduler* improver_, Funnel_parameters parameters_ = Funnel_parameters()) : SSPInstanceContractor(timelimit, sched_, improver_), parameters(parameters_) { }
        virtual ~FunnelSSP() = default;

        std::string getCoarserName() const override { return "FunnelSSP"; }
};