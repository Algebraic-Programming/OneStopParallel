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

#include "scheduler/InstanceContractor.hpp"
#include "auxiliary/auxiliary.hpp"

/**
 * @brief Parameters for Funnel coarsener
 * 
 */
struct Funnel_parameters {
    bool funnel_incoming;
    bool funnel_outgoing;
    bool first_funnel_incoming;

    bool use_approx_transitive_reduction;

    float max_relative_weight;

    Funnel_parameters(  float max_relative_weight_ = 20.0,
                        bool funnel_incoming_ = true,
                        bool funnel_outgoing_ = false,
                        bool first_funnel_incoming_ = true,
                        bool use_approx_transitive_reduction_ = true) :
                            funnel_incoming(funnel_incoming_),
                            funnel_outgoing(funnel_outgoing_),
                            first_funnel_incoming(first_funnel_incoming_),
                            use_approx_transitive_reduction(use_approx_transitive_reduction_),
                            max_relative_weight(max_relative_weight_) {};
    ~Funnel_parameters() = default;
};


/**
 * @brief Acyclic graph contractor that contracts groups of nodes with only one vertex with incoming/outgoing edges (from outside the group)
 * 
 */
class Funnel : public InstanceContractor {
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
        Funnel(Funnel_parameters parameters_ = Funnel_parameters()) : InstanceContractor(), parameters(parameters_) { }
        Funnel(Scheduler* sched_, Funnel_parameters parameters_ = Funnel_parameters()) : Funnel(sched_, nullptr, parameters_) { }
        Funnel(Scheduler* sched_, ImprovementScheduler* improver_, Funnel_parameters parameters_ = Funnel_parameters()) : InstanceContractor(sched_, improver_), parameters(parameters_) { }
        Funnel(unsigned timelimit, Scheduler* sched_, Funnel_parameters parameters_ = Funnel_parameters()) : Funnel(timelimit, sched_, nullptr, parameters_) { }
        Funnel(unsigned timelimit, Scheduler* sched_, ImprovementScheduler* improver_, Funnel_parameters parameters_ = Funnel_parameters()) : InstanceContractor(timelimit, sched_, improver_), parameters(parameters_) { }
        virtual ~Funnel() = default;

        std::string getCoarserName() const override { return "Funnel"; }
};