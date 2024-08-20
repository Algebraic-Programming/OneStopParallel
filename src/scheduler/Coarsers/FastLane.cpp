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

#include "scheduler/Coarsers/FastLane.hpp"

RETURN_STATUS FastLane::run_contractions() {
    Union_Find_Universe<VertexType> universe;

    const ComputationalDag& graph = dag_history.back()->getComputationalDag();

    for (const auto& edge : graph.edges()) {
        if (graph.children(edge.m_source).size() == 1 && graph.parents(edge.m_target).size() == 1) {
            if (!universe.is_in_universe(edge.m_source)) universe.add_object(edge.m_source,graph.nodeWorkWeight(edge.m_source));
            if (!universe.is_in_universe(edge.m_target)) universe.add_object(edge.m_target,graph.nodeWorkWeight(edge.m_target));
            universe.join_by_name(edge.m_source,edge.m_target);
        }
    }

    std::vector<std::vector<VertexType>> partition = universe.get_connected_components();
    
    std::vector<std::unordered_set<VertexType>> partition_other_format(partition.size());
    for (size_t i = 0; i < partition.size(); i++) {
        for (auto vert : partition[i]) {
            partition_other_format[i].emplace(vert);
        }
    }

    add_contraction(partition_other_format);
    return SUCCESS;
}
