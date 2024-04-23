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
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>

#include "model/ComputationalDag.hpp"
#include "structures/dag.hpp"
#include "structures/union_find.hpp"
#include "auxiliary/Balanced_Coin_Flips.hpp"

class CoarsenHistory {
    private:
        friend class CoarseRefineScheduler;

        const CoarsenParams coarsen_par;
        std::vector<std::unique_ptr<DAG>> dag_evolution;

        Thue_Morse_Sequence thue_coin;
        Biased_Random balanced_random;


        /// @brief Surjective maps contraction_maps[i]: Vertices(dag_evolution[i]) -> Vertices(dag_evolution[i+1])
        std::vector<std::unique_ptr<std::unordered_map<int, int>>> contraction_maps;

        /// @brief expansion_maps[i]: Vertices(dag_evolution[i+1]) -> Powerset(Vertices(dag_evolution[i]))
        /// Such that the image consists a partition of Vertices(dag_evolution[i])
        std::vector<std::unique_ptr<std::unordered_map<int, std::unordered_set<int>>>> expansion_maps;

        /**
         * @brief Adds a dag to the history
         * 
         * @param graph new dag
         * @param contraction_map verticies map from last dag in history to new one 
         */
        void add_dag( std::unique_ptr<DAG> graph, std::unique_ptr<std::unordered_map<int, int>> contraction_map);

        void add_graph_contraction( const std::vector<std::unordered_set<int>>& sets_to_contract );
        void add_graph_contraction( const std::vector<std::vector<int>>& sets_to_contract );


    public:
        CoarsenHistory(const DAG& graph, const CoarsenParams coarsen_par_ = CoarsenParams() );
        CoarsenHistory(const ComputationalDag& graph, const CoarsenParams coarsen_par_ = CoarsenParams() );

        /**
         * @brief Adds a contracted graph to the Dag history if contraction attemps did indeed decrease the number of nodes
         * 
         * @param edge_sort_type either by edge collapse or by weight diminishing
         * @param override_coarsen_param can add noise to the partially ordered set to integer mapping
         * @return int decrease in number of nodes
         */
        int run_and_add_contraction(const contract_edge_sort edge_sort_type , const CoarsenParams& override_coarsen_param);

        int run_and_add_contraction(const contract_edge_sort edge_sort_type) {
            return run_and_add_contraction(edge_sort_type, coarsen_par);
        }

        void run_dag_evolution(const int min_number_of_nodes = 0);

        // only for testing
        const std::vector<DAG> retrieve_dag_evolution() const {
            std::vector<DAG> dag_hist_references;
            dag_hist_references.reserve( dag_evolution.size() );
            for (long unsigned i = 0; i < dag_evolution.size(); i++) {
                dag_hist_references.push_back( *(dag_evolution[i]) );
            }
            return dag_hist_references;
        }

        // only for testing
        const std::vector<std::unordered_map<int, int>> retrieve_contr_maps() const {
            std::vector<std::unordered_map<int, int>> contr_maps;
            contr_maps.reserve( contraction_maps.size() );
            for (long unsigned i = 0; i < contraction_maps.size(); i++) {
                contr_maps.push_back( *(contraction_maps[i]) );
            }
            return contr_maps;
        }

        //only for testing
        const std::vector<std::unordered_map<int, std::unordered_set<int>>> retrieve_expansion_maps() const {
            std::vector<std::unordered_map<int, std::unordered_set<int>>> exp_maps;
            exp_maps.reserve( expansion_maps.size() );
            for (long unsigned i = 0; i < expansion_maps.size(); i++) {
                exp_maps.push_back( *(expansion_maps[i]) );
            }
            return exp_maps;
        }

};