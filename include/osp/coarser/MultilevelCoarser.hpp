/*
Copyright 2025 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author Toni Boehnlein, Pal Andras Papp, Raphael S. Steiner
*/

#pragma once

#include <memory>
#include <numeric>
#include <set>
#include <vector>

#include "osp/coarser/Coarser.hpp"
#include "osp/bsp/model/BspInstance.hpp"
#include "osp/coarser/coarser_util.hpp"


namespace osp {

template<typename Graph_t, typename Graph_t_coarse>
class MultilevelCoarseAndSchedule;

template<typename Graph_t, typename Graph_t_coarse>
class MultilevelCoarser : public Coarser<Graph_t, Graph_t_coarse> {
    friend class MultilevelCoarseAndSchedule<Graph_t, Graph_t_coarse>;
  private:
    const Graph_t *original_graph;
  protected:
    inline const Graph_t * getOriginalGraph() const { return original_graph; };

    std::vector<std::unique_ptr<Graph_t_coarse>> dag_history;
    std::vector<std::unique_ptr<std::vector<vertex_idx_t<Graph_t_coarse>>>> contraction_maps;

    RETURN_STATUS add_contraction(const std::vector<vertex_idx_t<Graph_t_coarse>> &contraction_map);
    RETURN_STATUS add_contraction(std::vector<vertex_idx_t<Graph_t_coarse>> &&contraction_map);
    RETURN_STATUS add_contraction(const std::vector<vertex_idx_t<Graph_t_coarse>> &contraction_map, const Graph_t_coarse &contracted_graph);
    RETURN_STATUS add_contraction(std::vector<vertex_idx_t<Graph_t_coarse>> &&contraction_map, Graph_t_coarse &&contracted_graph);
    void add_identity_contraction();
    
    std::vector<vertex_idx_t<Graph_t_coarse>> getCombinedContractionMap() const;

    virtual RETURN_STATUS run_contractions() = 0;
    void compactify_dag_history();

    void clear_computation_data();

  public:
    MultilevelCoarser() : original_graph(nullptr) {};
    MultilevelCoarser(const Graph_t &graph) : original_graph(&graph) {};
    virtual ~MultilevelCoarser() = default;


    bool coarsenDag(const Graph_t &dag_in, Graph_t_coarse &coarsened_dag,
                            std::vector<vertex_idx_t<Graph_t_coarse>> &vertex_contraction_map) override;

    
    RETURN_STATUS run(const Graph_t &graph);
    RETURN_STATUS run(const BspInstance<Graph_t> &inst);

    virtual std::string getCoarserName() const override = 0;
};



template<typename Graph_t, typename Graph_t_coarse>
RETURN_STATUS MultilevelCoarser<Graph_t, Graph_t_coarse>::run(const Graph_t &graph) {
    clear_computation_data();
    original_graph = &graph;

    RETURN_STATUS status = RETURN_STATUS::OSP_SUCCESS;
    status = std::max(status, run_contractions());

    if (dag_history.size() == 0) {
        add_identity_contraction();
    }

    return status;
}

template<typename Graph_t, typename Graph_t_coarse>
RETURN_STATUS MultilevelCoarser<Graph_t, Graph_t_coarse>::run(const BspInstance< Graph_t > &inst) {
    return run(inst.getComputationalDag());
}

template<typename Graph_t, typename Graph_t_coarse>
void MultilevelCoarser<Graph_t, Graph_t_coarse>::clear_computation_data() {
    dag_history.clear();
    dag_history.shrink_to_fit();
    
    contraction_maps.clear();
    contraction_maps.shrink_to_fit();
}


template<typename Graph_t, typename Graph_t_coarse>
void MultilevelCoarser<Graph_t, Graph_t_coarse>::compactify_dag_history() {
    if (dag_history.size() < 3) return;

    size_t dag_indx_first = dag_history.size() - 2;
    size_t map_indx_first = contraction_maps.size() - 2;

    size_t dag_indx_second = dag_history.size() - 1;
    size_t map_indx_second = contraction_maps.size() - 1;

    if ( (static_cast<double>( dag_history[dag_indx_first-1]->num_vertices() ) / static_cast<double>( dag_history[dag_indx_second-1]->num_vertices() )) > 1.25 ) return;
    

    // Compute combined contraction_map
    std::unique_ptr<std::vector<vertex_idx_t<Graph_t_coarse>>> combi_contraction_map = std::make_unique<std::vector<vertex_idx_t<Graph_t_coarse>>>( contraction_maps[map_indx_first]->size() );
    for (std::size_t vert = 0; vert < contraction_maps[map_indx_first]->size(); ++vert) {
        combi_contraction_map->at(vert) = contraction_maps[map_indx_second]->at( contraction_maps[map_indx_first]->at( vert ) );
    }

    // Delete ComputationalDag
    auto dag_it = dag_history.begin();
    std::advance(dag_it, dag_indx_first);
    dag_history.erase(dag_it);

    // Delete contraction map
    auto contr_map_it = contraction_maps.begin();
    std::advance(contr_map_it, map_indx_second);
    contraction_maps.erase(contr_map_it);

    // Replace contraction map
    contraction_maps[map_indx_first] = std::move(combi_contraction_map);
}


template<typename Graph_t, typename Graph_t_coarse>
RETURN_STATUS MultilevelCoarser<Graph_t, Graph_t_coarse>::add_contraction(const std::vector<vertex_idx_t<Graph_t_coarse>> &contraction_map) {
    std::unique_ptr<Graph_t_coarse> new_graph = std::make_unique<Graph_t_coarse>();

    contraction_maps.emplace_back(contraction_map);

    bool success = false;

    if (dag_history.size() == 0) {
        success = coarser_util::construct_coarse_dag<Graph_t, Graph_t_coarse>(*(getOriginalGraph()), *new_graph, *(contraction_maps.back()) );
    } else {
        success = coarser_util::construct_coarse_dag<Graph_t_coarse, Graph_t_coarse>(*(dag_history.back()), *new_graph, *(contraction_maps.back()) );
    }

    dag_history.emplace_back( std::move(new_graph) );

    if (success) {
        compactify_dag_history();
        return RETURN_STATUS::OSP_SUCCESS;
    } else {
        return RETURN_STATUS::ERROR;
    }
}

template<typename Graph_t, typename Graph_t_coarse>
RETURN_STATUS MultilevelCoarser<Graph_t, Graph_t_coarse>::add_contraction(std::vector<vertex_idx_t<Graph_t_coarse>> &&contraction_map) {
    std::unique_ptr<Graph_t_coarse> new_graph = std::make_unique<Graph_t_coarse>();
    
    std::unique_ptr<std::vector<vertex_idx_t<Graph_t_coarse>>> contr_map_ptr(new std::vector<vertex_idx_t<Graph_t_coarse>>(std::move(contraction_map)));
    contraction_maps.emplace_back(std::move(contr_map_ptr));

    bool success = false;

    if (dag_history.size() == 0) {
        success = coarser_util::construct_coarse_dag<Graph_t, Graph_t_coarse>(*(getOriginalGraph()), *new_graph, *(contraction_maps.back()) );
    } else {
        success = coarser_util::construct_coarse_dag<Graph_t_coarse, Graph_t_coarse>(*(dag_history.back()), *new_graph, *(contraction_maps.back()) );
    }

    dag_history.emplace_back( std::move(new_graph) );

    if (success) {
        compactify_dag_history();
        return RETURN_STATUS::OSP_SUCCESS;
    } else {
        return RETURN_STATUS::ERROR;
    }
}


template<typename Graph_t, typename Graph_t_coarse>
RETURN_STATUS MultilevelCoarser<Graph_t, Graph_t_coarse>::add_contraction(const std::vector<vertex_idx_t<Graph_t_coarse>> &contraction_map, const Graph_t_coarse &contracted_graph) {
    std::unique_ptr<Graph_t_coarse> graph_ptr(new Graph_t_coarse(contracted_graph));
    dag_history.emplace_back(std::move(graph_ptr));
    
    std::unique_ptr<std::vector<vertex_idx_t<Graph_t_coarse>>> contr_map_ptr(new std::vector<vertex_idx_t<Graph_t_coarse>>(contraction_map));
    contraction_maps.emplace_back(std::move(contr_map_ptr));

    compactify_dag_history();
    return RETURN_STATUS::OSP_SUCCESS;
}

template<typename Graph_t, typename Graph_t_coarse>
RETURN_STATUS MultilevelCoarser<Graph_t, Graph_t_coarse>::add_contraction(std::vector<vertex_idx_t<Graph_t_coarse>> &&contraction_map, Graph_t_coarse &&contracted_graph) {
    std::unique_ptr<Graph_t_coarse> graph_ptr(new Graph_t_coarse(std::move(contracted_graph)));
    dag_history.emplace_back(std::move(graph_ptr));

    std::unique_ptr<std::vector<vertex_idx_t<Graph_t_coarse>>> contr_map_ptr(new std::vector<vertex_idx_t<Graph_t_coarse>>(std::move(contraction_map)));
    contraction_maps.emplace_back(std::move(contr_map_ptr));

    compactify_dag_history();
    return RETURN_STATUS::OSP_SUCCESS;
}


template<typename Graph_t, typename Graph_t_coarse>
std::vector<vertex_idx_t<Graph_t_coarse>> MultilevelCoarser<Graph_t, Graph_t_coarse>::getCombinedContractionMap() const {
    std::vector<vertex_idx_t<Graph_t_coarse>> combinedContractionMap(original_graph->num_vertices());
    std::iota(combinedContractionMap.begin(), combinedContractionMap.end(), 0);

    for (std::size_t j = 0; j < contraction_maps.size(); ++j) {
        for (std::size_t i = 0; i < combinedContractionMap.size(); ++i) {
            combinedContractionMap[i] = contraction_maps[j]->at( combinedContractionMap[i] );
        }
    }

    return combinedContractionMap;
}



template<typename Graph_t, typename Graph_t_coarse>
bool MultilevelCoarser<Graph_t, Graph_t_coarse>::coarsenDag(const Graph_t &dag_in, Graph_t_coarse &coarsened_dag,
                                                                    std::vector<vertex_idx_t<Graph_t_coarse>> &vertex_contraction_map) {
    clear_computation_data();

    RETURN_STATUS status = run(dag_in);

    if (status != RETURN_STATUS::OSP_SUCCESS && status != RETURN_STATUS::BEST_FOUND) return false;

    assert(dag_history.size() != 0);
    coarsened_dag = *(dag_history.back());

    vertex_contraction_map = getCombinedContractionMap();

    return true;
}

template<typename Graph_t, typename Graph_t_coarse>
void MultilevelCoarser<Graph_t, Graph_t_coarse>::add_identity_contraction() {
    std::size_t n_vert;
    if (dag_history.size() == 0) {
        n_vert = static_cast<std::size_t>( original_graph->num_vertices() );
    } else {
        n_vert = static_cast<std::size_t>( dag_history.back()->num_vertices() );
    }
    
    std::vector<vertex_idx_t<Graph_t_coarse>> contraction_map( n_vert );
    std::iota(contraction_map.begin(), contraction_map.end(), 0);

    add_contraction(std::move(contraction_map));
    compactify_dag_history();
}





} // end namespace osp