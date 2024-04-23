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

#include "auxiliary/Erdos_Renyi_graph.hpp"

// //old slow implementation
// DAG erdos_renyi_graph_gen( unsigned num_vertices, unsigned chance ) {
//     std::vector<std::vector<int>> in_(num_vertices);
//     std::vector<std::vector<int>> out_(num_vertices);
//     for (unsigned i = 0; i< num_vertices; i++) {
//         for (unsigned j = i+1; j< num_vertices; j++) {
//             if (randInt(num_vertices) < chance ) {
//                 in_[j].emplace_back(i);
//                 out_[i].emplace_back(j);
//             }
//         }
//     }

//     return DAG(in_,out_,std::vector<int>(num_vertices,1),std::vector<int>(num_vertices,1));
// }


DAG erdos_renyi_graph_gen( unsigned num_vertices, unsigned chance ) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<std::vector<int>> in_(num_vertices);
    std::vector<std::vector<int>> out_(num_vertices);

    for (unsigned i = 0; i< num_vertices; i++) {
        std::binomial_distribution<> bino_dist( num_vertices-1-i , double(chance)/double(num_vertices)  );
        unsigned out_edges_num = bino_dist(gen);

        std::unordered_set<unsigned> out_edges;
        while ( out_edges.size() < out_edges_num ) {
            unsigned edge = i+1 + randInt( num_vertices-i-1 );

            if ( out_edges.find( edge ) != out_edges.cend() ) continue;

            out_edges.emplace(edge);
        }


        for (auto& j : out_edges) {
            in_[j].emplace_back(i);
            out_[i].emplace_back(j);
        }
    }

    return DAG(in_,out_,std::vector<int>(num_vertices,1),std::vector<int>(num_vertices,1));
}