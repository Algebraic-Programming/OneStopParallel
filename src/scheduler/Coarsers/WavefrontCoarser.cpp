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

#include "scheduler/Coarsers/WavefrontCoarser.hpp"

RETURN_STATUS WavefrontCoarser::run_contractions() {
    if (min_wavefront_size == 0) auto_wavefront_size();

    std::cout   << "Coarsen Step: " << dag_history.size()
                << ", Number of nodes: " << dag_history.back()->numberOfVertices()
                << ", Number of edges: " << dag_history.back()->getComputationalDag().numberOfEdges()
                << ", Log ratio: " << std::log(dag_history.back()->getComputationalDag().numberOfEdges()) / std::log(dag_history.back()->numberOfVertices()) << std::endl;

    unsigned no_decrease_counter = 0;
    bool enough_parallelisation = check_wavefront_size();
    while (enough_parallelisation && no_decrease_counter < 5) {
        const ComputationalDag& graph = dag_history.back()->getComputationalDag();
        std::vector<std::unordered_set<VertexType>> partition;

        std::vector<int> poset = graph.get_strict_poset_integer_map(5, 1.5);
        std::map<int, std::vector<VertexType>> level_sets;
        for (VertexType node = 0; node < graph.numberOfVertices(); node++) {
            if (level_sets.find(poset[node]) == level_sets.cend() ) {
                level_sets.emplace(poset[node], std::vector<VertexType>({node}));
            } else {
                level_sets[poset[node]].push_back(node);
            }
        }

        int parity = randInt(2);
        for (const auto& map_element : level_sets) {
            if ((map_element.first - parity) % 2 != 0) continue;
            
            const int level_1 = map_element.first;
            const int level_2 = level_1 + 1;
            if (level_sets.find(level_2) == level_sets.cend()) continue;

            Union_Find_Universe<VertexType> uf(level_sets[level_1]);
            uf.add_object(level_sets[level_2]);

            std::unordered_set<VertexType> second_level_set(level_sets[level_2].begin(), level_sets[level_2].end());
            for (const auto& node : level_sets[level_1]) {
                for (const auto& chld : graph.children(node)) {
                    if (second_level_set.find(chld) != second_level_set.cend()) {
                        uf.join_by_name(node, chld);
                    }
                }
            }

            std::vector<std::vector<VertexType>> components = uf.get_connected_components();
            for (const auto& comp : components) {
                partition.emplace_back(comp.begin(), comp.end());
            }
            
        }
                
        size_t number_vert_prior = graph.numberOfVertices();
        add_contraction(partition);
        size_t number_vert_posterior = dag_history.back()->numberOfVertices();
        if ( (double) number_vert_prior / (double) number_vert_posterior < std::pow(17.0 / 16.0, 0.15) ) {
            no_decrease_counter++;
        } else {
            no_decrease_counter = 0;
        }

        enough_parallelisation = check_wavefront_size();

        std::cout   << "Coarsen Step: " << dag_history.size()
                << ", Number of nodes: " << dag_history.back()->numberOfVertices()
                << ", Number of edges: " << dag_history.back()->getComputationalDag().numberOfEdges()
                << ", Log ratio: " << std::log(dag_history.back()->getComputationalDag().numberOfEdges()) / std::log(dag_history.back()->numberOfVertices()) << std::endl;
    }
    
    
    return SUCCESS;
}

bool WavefrontCoarser::check_wavefront_size() const {
    const ComputationalDag& graph = dag_history.back()->getComputationalDag();
    size_t length = graph.longestPath()+1;

    return ((double) graph.numberOfVertices() / (double) length) > min_wavefront_size;
}

void WavefrontCoarser::auto_wavefront_size() {
    min_wavefront_size = (double) getOriginalInstance()->numberOfProcessors() * 5;
}