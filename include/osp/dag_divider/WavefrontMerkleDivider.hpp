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

#include <vector>
#include <cmath>
#include <unordered_map>
#include "osp/bsp/scheduler/Scheduler.hpp"
#include "MerkleHashComputer.hpp"
#include "osp/auxiliary/datastructures/union_find.hpp"
#include "DagDivider.hpp"
#include "osp/graph_algorithms/subgraph_algorithms.hpp"
#include "osp/graph_algorithms/directed_graph_path_util.hpp"

namespace osp {


template<typename Graph_t>
struct subgraph {

    std::vector<vertex_idx_t<Graph_t>> vertices;
    std::size_t hash;

    v_workw_t<Graph_t> work_weight;
    v_memw_t<Graph_t> memory_weight;

    unsigned start_wavefront;
    unsigned end_wavefront;

    // std::vector<VertexType> sinks;

    subgraph() = default;
    subgraph(const std::vector<vertex_idx_t<Graph_t>> &vertices_arg, std::size_t hash_arg, v_workw_t<Graph_t> work_weight_arg, v_memw_t<Graph_t> memory_weight_arg, unsigned start_wavefront_arg, unsigned end_wavefront_arg)
        : vertices(vertices_arg), hash(hash_arg), work_weight(work_weight_arg), memory_weight(memory_weight_arg), start_wavefront(start_wavefront_arg), end_wavefront(end_wavefront_arg) {}

    subgraph(vertex_idx_t<Graph_t> vertex, std::size_t hash_arg, v_workw_t<Graph_t> work_weight_arg, v_memw_t<Graph_t> memory_weight_arg, unsigned wavefront_arg)
        : vertices({vertex}), hash(hash_arg), work_weight(work_weight_arg), memory_weight(memory_weight_arg), start_wavefront(wavefront_arg), end_wavefront(wavefront_arg) {}


    void extend_by_vertex(vertex_idx_t<Graph_t> vertex, v_workw_t<Graph_t> work_weight_arg, v_memw_t<Graph_t> memory_weight_arg, unsigned wavefront_arg, std::size_t hash_arg) {
        vertices.push_back(vertex);
        work_weight += work_weight_arg;
        memory_weight += memory_weight_arg;
        end_wavefront = std::max(end_wavefront, wavefront_arg);
    
        hash = hash_arg; // not sure yet what this will be useful
    }
    

};


/**
 * @class WavefrontComponentDivider
 * @brief Divides the wavefronts of a computational DAG into consecutive groups or sections.
 * The sections are created with the aim of containing a high number of connected components.
 * The class also provides functionality to detect groups of isomorphic components within the sections.
 *
 *
 */
template<typename Graph_t, typename node_hash_func_t = default_node_hash_func<vertex_idx_t<Graph_t>>>
class WavefrontMerkleDivider : public IDagDivider<Graph_t> {

    static_assert(is_computational_dag_v<Graph_t>,
                  "WavefrontComponentDivider can only be used with computational DAGs.");

  private:

    using MerkleHashComputer_fw_t = MerkleHashComputer<Graph_t, node_hash_func_t, true>;
    using MerkleHashComputer_bw_t = MerkleHashComputer<Graph_t, node_hash_func_t, false>;
    using subgraph_t = subgraph<Graph_t>;
    using VertexType = vertex_idx_t<Graph_t>;
    using work_weight_t = v_workw_t<Graph_t>;
    using mem_weight_t = v_memw_t<Graph_t>;

    std::vector<unsigned> sub_graph_id;
    std::vector<subgraph_t> sub_graphs;

    std::unordered_map<unsigned, subgraph_t> active_subgraphs;

    unsigned next_id;

    work_weight_t threshold = 1000;
    
    
    inline subgraph_t create_subgraph(const Graph_t &dag, std::size_t hash_arg, const VertexType &v, unsigned wavefront) {

        return subgraph_t(v, hash_arg, dag.vertex_work_weight(v), dag.vertex_mem_weight(v), wavefront);

    }
  
    template<typename RandomAccessIterator>
    void merge_active_subgraphs(const RandomAccessIterator& start, const RandomAccessIterator& end, VertexType v, unsigned end_wavefront, std::size_t hash_arg, const Graph_t &dag) {

        subgraph_t subg(v, hash_arg, dag.vertex_work_weight(v), dag.vertex_mem_weight(v), end_wavefront);

        for (auto it = start; it != end; ++it) {
        
            const subgraph_t & g = active_subgraphs[*it];

            for (const auto & v : g.vertices) {
                subg.vertices.push_back(v);
                sub_graph_id[v] = next_id;
            }           

            subg.work_weight += g.work_weight;
            subg.memory_weight += g.memory_weight;
            subg.start_wavefront = std::min(subg.start_wavefront, g.start_wavefront); // max or min?
            
            active_subgraphs.erase(*it);
        
        }

        add_active_subgraph(std::move(subg));
    }
    

    std::pair<bool,bool> check_extension(const std::vector<VertexType> & orbit, std::unordered_map<VertexType, std::set<unsigned>> & extends_sugraphs) {

        bool no_merge_extension = true;
        std::set<unsigned>() all_extensions;
        for (const auto & v : orbit) {
            extends_sugraphs[v] = std::set<unsigned>();
            for (const auto & parent : dag.parents(v)) {   
                
                const unsigned parent_id = sub_graph_id[parent];

                if (active_subgraphs.find(parent_id) != active_subgraphs.end()) {
                    extends_sugraphs[v].insert(parent_id);    
                    all_extensions.insert(parent_id);
                }
            }

            if (no_merge_extension && extends_sugraphs[v].size() > 1) {
                no_merge_extension = false;
            }            
        }
        
        return {no_merge_extension, all_extensions.size() == orbit.size()};
    }

    inline void add_active_subgraph(subgraph_t && subg) {
        active_subgraphs[next_id] = subg; 
        next_id++;  
    }

  public:
    WavefrontMerkleDivider() = default;

    std::vector<std::vector<std::vector<vertex_idx_t<Graph_t>>>> divide(const Graph_t &dag) override {  

        next_id = 0;
        sub_graph_id.resize(dag.num_vertices(), std::numeric_limits<unsigned>::max());
        std::vector<std::vector<VertexType>> level_sets = compute_wavefronts(dag);
        MerkleHashComputer_fw_t m_fw_hash(dag);
        MerkleHashComputer_bw_t m_bw_hash(dag);

        unsigned wf_index = 0;
        for (const auto& level : level_sets) {

            std::unordered_map<std::size_t, std::vector<VertexType>> fw_orbits;
            std::unordered_map<std::size_t, std::vector<VertexType>> bw_orbits;

            std::cout << "Wavefront " << wf_index << std::endl;
            for (const auto v : level) {
                if(fw_orbits.find(m_fw_hash.get_vertex_hash(v)) == fw_orbits.end()) {
                    fw_orbits[m_fw_hash.get_vertex_hash(v)] = {v};
                } else {
                    fw_orbits[m_fw_hash.get_vertex_hash(v)].push_back(v);
                }

                if(bw_orbits.find(m_bw_hash.get_vertex_hash(v)) == bw_orbits.end()) {
                    bw_orbits[m_bw_hash.get_vertex_hash(v)] = {v};
                } else {
                    bw_orbits[m_bw_hash.get_vertex_hash(v)].push_back(v);
                }


            }

            std::cout << "fw: ";
            for (const auto& [key, value] : fw_orbits) { 
                                
                std::cout << "{";
                for (const auto & v: value) {
                    std::cout << v << ", ";
                } 
                std::cout << "}, ";

            }
            std::cout << std::endl;

            std::cout << "bw: ";
            for (const auto& [key, value] : bw_orbits) { 
                                
                std::cout << "{";
                for (const auto & v: value) {
                    std::cout << v << ", ";
                } 
                std::cout << "}, ";

            }
            std::cout << std::endl;
            

 

            for (const auto v : level) {
                add_active_subgraph(create_subgraph(dag, m_fw_hash.get_vertex_hash(v), v, wf_index));
            }
                


            for (const auto& [hash, orbit] : fw_orbits) { 
                            
                if (orbit.size() == 1) {

                    const VertexType v = orbit[0];

                    work_weight_t acc_weight = 0;
                    std::set<unsigned> visited_parents;
                    for (const auto & parent : dag.parents(v)) {
                        const unsigned parent_id = sub_graph_id[parent];
                        if (active_subgraphs.find(parent_id) != active_subgraphs.end()) {
                        
                            auto pair = visited_parents.insert(parent_id);
                            
                            if (pair.second)
                                acc_weight += active_subgraphs[parent_id].work_weight;                            
                        } 
                    }

                    if (acc_weight > threshold) {

                        // for (const auto & id : visited_parents) {
                        //     sub_graphs.push_back(std::move(active_subgraphs[id]));
                        //     active_subgraphs.erase(id);
                        // }

                        add_active_subgraph(create_subgraph(dag, m_fw_hash.get_vertex_hash(v), v, wf_index));


                    } else {
                        merge_active_subgraphs(visited_parents.begin(), visited_parents.end(), v, wf_index, m_fw_hash, dag);
                    }
                    

                } else {

                    std::unordered_map<VertexType, std::set<unsigned>> & extends_sugraphs;
                    auto bool_pair = check_extension(orbit, extends_sugraphs )
                    if(bool_pair.first && bool_pair.second) { 

                        for (const auto & [v : ext_subg ] : extends_sugraphs) {
                            active_subgraphs[*ext_subg.begin()].extend_by_vertex(v, dag.vertex_work_weight(v), dag.vertex_mem_weight(v), wf_index, m_fw_hash.get_vertex_hash(v));
                        }

                    } else if (bool_pair.first) {
                    
                        // possible all
                    
                    } else {

                        for (const auto & [v : ext_subg ] : extends_sugraphs) {

                            work_weight_t acc_weight = 0;
                            for (const unsigned & subgraph_id : ext_subg) {
                                acc_weight += active_subgraphs[subgraph_id].work_weight;
                            }

                            if (acc_weight > threshold) {
                                add_active_subgraph(create_subgraph(dag, m_fw_hash.get_vertex_hash(v), v, wf_index));
                            } else {
                                merge_active_subgraphs(ext_subg.begin(), ext_subg.end(), v, wf_index, m_fw_hash, dag);
                            }
                        }
                    }
                }
            }  
                                
        
            wf_index++;
        
        }
     


        std::vector<std::vector<std::vector<vertex_idx_t<Graph_t>>>> vertex_maps;
    
        return vertex_maps;
        //
    }

};

} // namespace osp