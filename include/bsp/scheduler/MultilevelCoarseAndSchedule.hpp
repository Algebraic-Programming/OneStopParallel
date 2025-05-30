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

#include "coarser/coarser_util.hpp"
#include "model/BspInstance.hpp"
#include "model/BspSchedule.hpp"
#include "scheduler/ImprovementScheduler.hpp"
#include "scheduler/Scheduler.hpp"


namespace osp {

template<typename Graph_t, typename Graph_t_coarse>
class MultilevelCoarseAndSchedule : public Scheduler<Graph_t> {
  private:
    const BspInstance<Graph_t> *original_inst;
  protected:
    inline const BspInstance<Graph_t> * const getOriginalInstance() const { return original_inst; };

    Scheduler<Graph_t_coarse> *sched;
    ImprovementScheduler<Graph_t_coarse> *improver;

    std::vector<std::unique_ptr<BspInstance<Graph_t_coarse>>> dag_history;
    std::vector<std::unique_ptr<std::vector<vertex_idx_t<Graph_t_coarse>>>> contraction_maps;
    long int active_graph;
    std::unique_ptr< BspSchedule<Graph_t_coarse> > active_schedule;


    RETURN_STATUS add_contraction(const std::vector<vertex_idx_t<Graph_t_coarse>> &contraction_map);
    RETURN_STATUS add_contraction(std::vector<vertex_idx_t<Graph_t_coarse>> &&contraction_map);
    RETURN_STATUS add_contraction(const std::vector<vertex_idx_t<Graph_t_coarse>> &contraction_map, const Graph_t_coarse &contracted_graph);
    RETURN_STATUS add_contraction(std::vector<vertex_idx_t<Graph_t_coarse>> &&contraction_map, Graph_t_coarse &&contracted_graph);
    RETURN_STATUS add_contraction(const std::vector<vertex_idx_t<Graph_t_coarse>> &contraction_map, const BspInstance<Graph_t_coarse> &contracted_instance);
    RETURN_STATUS add_contraction(std::vector<vertex_idx_t<Graph_t_coarse>> &&contraction_map, BspInstance<Graph_t_coarse> &&contracted_instance);
    
    virtual RETURN_STATUS run_contractions() = 0;
    void compactify_dag_history();
    RETURN_STATUS compute_initial_schedule();
    RETURN_STATUS expand_active_schedule();
    RETURN_STATUS expand_active_schedule_to_original_schedule(BspSchedule<Graph_t>& schedule);
    RETURN_STATUS improve_active_schedule();
    RETURN_STATUS run_expansions(BspSchedule<Graph_t>& schedule);

    void clear_computation_data();

  public:
    MultilevelCoarseAndSchedule() : Scheduler<Graph_t>(), original_inst(nullptr), sched(nullptr), improver(nullptr){};
    MultilevelCoarseAndSchedule(Scheduler<Graph_t_coarse> *sched_, ImprovementScheduler<Graph_t_coarse> *improver_ = nullptr)
        : Scheduler<Graph_t>(), original_inst(nullptr), sched(sched_), improver(improver_), active_graph(-1L){};
    MultilevelCoarseAndSchedule(unsigned timelimit, Scheduler<Graph_t_coarse> *sched_, ImprovementScheduler<Graph_t_coarse> *improver_)
        : Scheduler<Graph_t>(timelimit), original_inst(nullptr), sched(sched_), improver(improver_), active_graph(-1L){};
    virtual ~MultilevelCoarseAndSchedule() = default;

    inline void setInitialScheduler(Scheduler<Graph_t_coarse> *const sched_) { sched = sched_; };
    inline void setImprovementScheduler(ImprovementScheduler<Graph_t_coarse> *const improver_) { improver = improver_; };

    RETURN_STATUS computeSchedule(BspSchedule<Graph_t> &schedule) override;

    virtual std::string getCoarserName() const = 0;
    std::string getScheduleName() const override {
        if (improver == nullptr) {
            return "C:" + getCoarserName() + "-S:" + sched->getScheduleName();
        } else {
            return "C:" + getCoarserName() + "-S:" + sched->getScheduleName() + "-I:" + improver->getScheduleName();
        }
    };

};





template<typename Graph_t, typename Graph_t_coarse>
RETURN_STATUS MultilevelCoarseAndSchedule<Graph_t, Graph_t_coarse>::compute_initial_schedule() {
    active_graph = static_cast<long int>(dag_history.size());
    active_graph--;

    assert((active_graph >= 0) && "Must have done at least one coarsening!");
    
    RETURN_STATUS status;
    
    active_schedule = std::make_unique< BspSchedule<Graph_t_coarse> >( *(dag_history[active_graph]) );
    status = sched->computeSchedule( active_schedule );

    RETURN_STATUS ret = improve_active_schedule();

    status = std::max(ret,status);
    return status;
}

template<typename Graph_t, typename Graph_t_coarse>
RETURN_STATUS MultilevelCoarseAndSchedule<Graph_t, Graph_t_coarse>::improve_active_schedule() {
    if (improver) {
        return improver->improveSchedule( active_schedule );
    }
    return SUCCESS;
}




template<typename Graph_t, typename Graph_t_coarse>
RETURN_STATUS MultilevelCoarseAndSchedule<Graph_t, Graph_t_coarse>::expand_active_schedule() {
    assert((active_graph > 0) && ( (long unsigned) active_graph < dag_history.size()));

    BspSchedule<Graph_t_coarse> expanded_schedule( *(dag_history[ active_graph - 1 ]) );

    for (const auto &node : dag_history[active_graph - 1]->getComputationalDag().vertices() ) {
        expanded_schedule.setAssignedProcessor(node, active_schedule.assignedProcessor(contraction_maps[active_graph - 1]->at(node)) );
        expanded_schedule.setAssignedSuperstep(node, active_schedule.assignedSuperstep(contraction_maps[active_graph - 1]->at(node)) );
    }

    active_graph--;
    active_schedule = std::make_unique<BspSchedule<Graph_t_coarse>>( move(expanded_schedule) );
    
    return SUCCESS;
}


template<typename Graph_t, typename Graph_t_coarse>
RETURN_STATUS MultilevelCoarseAndSchedule<Graph_t, Graph_t_coarse>::expand_active_schedule_to_original_schedule(BspSchedule<Graph_t>& schedule) {
    assert(active_graph == 0);

    for (const auto &node : dag_history[active_graph - 1]->getComputationalDag().vertices() ) {
        schedule.setAssignedProcessor(node, active_schedule.assignedProcessor(contraction_maps[active_graph - 1]->at(node)) );
        schedule.setAssignedSuperstep(node, active_schedule.assignedSuperstep(contraction_maps[active_graph - 1]->at(node)) );
    }

    active_graph--;
    active_schedule = std::unique_ptr<BspSchedule<Graph_t_coarse>>();
    
    return SUCCESS;
}






template<typename Graph_t, typename Graph_t_coarse>
RETURN_STATUS MultilevelCoarseAndSchedule<Graph_t, Graph_t_coarse>::run_expansions(BspSchedule<Graph_t>& schedule) {
    assert(active_graph >= 0 && (long unsigned) active_graph == dag_history.size()-1);

    RETURN_STATUS status = SUCCESS;
    
    while(active_graph > 0) {
        status = std::max(status, expand_active_schedule());
        status = std::max(status, improve_active_schedule());
    }

    status = std::max(status, expand_active_schedule_to_original_schedule(schedule));
    
    return status;
}


template<typename Graph_t, typename Graph_t_coarse>
void MultilevelCoarseAndSchedule<Graph_t, Graph_t_coarse>::clear_computation_data() {
    dag_history.clear();
    dag_history.shrink_to_fit();
    
    contraction_maps.clear();
    contraction_maps.shrink_to_fit();

    active_graph = -1;
    active_schedule = std::unique_ptr<BspSchedule<Graph_t_coarse>>();
}



template<typename Graph_t, typename Graph_t_coarse>
RETURN_STATUS MultilevelCoarseAndSchedule<Graph_t, Graph_t_coarse>::computeSchedule(BspSchedule<Graph_t>& schedule) {
    clear_computation_data();

    RETURN_STATUS status = SUCCESS;

    status = std::max(status, run_contractions());
    assert( (dag_history.size() == contraction_maps.size()) );

    if ( dag_history.size() == 0 ) {
        std::vector<vertex_idx_t<Graph_t_coarse>> contraction_map( schedule.getInstance().numberOfVertices() );
        std::iota(contraction_map.begin(), contraction_map.end(), 0);
        
        add_contraction(move(contraction_map));
    }

    assert(dag_history.size() > 0);

    status = std::max(status, compute_initial_schedule());
    status = std::max(status, run_expansions());

    assert(active_graph == -1);

    clear_computation_data();

    return status;
}



template<typename Graph_t, typename Graph_t_coarse>
void MultilevelCoarseAndSchedule<Graph_t, Graph_t_coarse>::compactify_dag_history() {
    if (dag_history.size() < 3) return;

    size_t dag_indx_first = dag_history.size() - 2;
    size_t map_indx_first = contraction_maps.size() - 2;

    size_t dag_indx_second = dag_history.size() - 1;
    size_t map_indx_second = contraction_maps.size() - 1;

    if ( ((double) dag_history[dag_indx_first-1]->numberOfVertices() / (double) dag_history[dag_indx_second-1]->numberOfVertices()) > 1.25 ) return;
    

    // Compute combined contraction_map
    std::unique_ptr<std::vector<vertex_idx_t<Graph_t_coarse>>> combi_contraction_map = std::make_unique<std::vector<vertex_idx_t<Graph_t_coarse>>>( contraction_maps[map_indx_first]->size() );
    for (std::size_t vert = 0; vert < contraction_maps[map_indx_first]->size(); ++vert) {
        combi_contraction_map[vert] = contraction_maps[map_indx_second]->at( contraction_maps[map_indx_first]->at( vert ) );
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
    contraction_maps[map_indx_first] = move(combi_contraction_map);
}


template<typename Graph_t, typename Graph_t_coarse>
RETURN_STATUS MultilevelCoarseAndSchedule<Graph_t, Graph_t_coarse>::add_contraction(const std::vector<vertex_idx_t<Graph_t_coarse>> &contraction_map) {
    dag_history.emplace_back({});
    contraction_maps.emplace_back(contraction_map);

    dag_history.back()->getArchitecture() = getOriginalInstance()->getArchitecture();

    bool success = false;

    if (dag_history.size() == 1) {
        success = coarser_util::construct_coarse_dag<Graph_t, Graph_t_coarse>(getOriginalInstance()->getComputationalDag(), dag_history.back()->getComputationalDag(), *(contraction_maps.back()) );
    } else {
        success = coarser_util::construct_coarse_dag<Graph_t, Graph_t_coarse>(dag_history.at( dag_history.size() - 2 )->getComputationalDag(), dag_history.back()->getComputationalDag(), *(contraction_maps.back()) );
    }

    if (success) {
        return SUCCESS;
    } else {
        return ERROR;
    }
}

template<typename Graph_t, typename Graph_t_coarse>
RETURN_STATUS MultilevelCoarseAndSchedule<Graph_t, Graph_t_coarse>::add_contraction(std::vector<vertex_idx_t<Graph_t_coarse>> &&contraction_map) {
    dag_history.emplace_back({});
    contraction_maps.emplace_back(move(contraction_map));

    dag_history.back()->getArchitecture() = getOriginalInstance()->getArchitecture();

    bool success = false;

    if (dag_history.size() == 1) {
        success = coarser_util::construct_coarse_dag<Graph_t, Graph_t_coarse>(getOriginalInstance()->getComputationalDag(), dag_history.back()->getComputationalDag(), *(contraction_maps.back()) );
    } else {
        success = coarser_util::construct_coarse_dag<Graph_t, Graph_t_coarse>(dag_history.at( dag_history.size() - 2 )->getComputationalDag(), dag_history.back()->getComputationalDag(), *(contraction_maps.back()) );
    }

    if (success) {
        return SUCCESS;
    } else {
        return ERROR;
    }
}


template<typename Graph_t, typename Graph_t_coarse>
RETURN_STATUS MultilevelCoarseAndSchedule<Graph_t, Graph_t_coarse>::add_contraction(const std::vector<vertex_idx_t<Graph_t_coarse>> &contraction_map, const Graph_t_coarse &contracted_graph) {
    dag_history.emplace_back({contracted_graph, getOriginalInstance()->getArchitecture()});
    contraction_maps.emplace_back(contraction_map);

    return SUCCESS;
}

template<typename Graph_t, typename Graph_t_coarse>
RETURN_STATUS MultilevelCoarseAndSchedule<Graph_t, Graph_t_coarse>::add_contraction(std::vector<vertex_idx_t<Graph_t_coarse>> &&contraction_map, Graph_t_coarse &&contracted_graph) {
    BspArchitecture<Graph_t> architecture = getOriginalInstance()->getArchitecture();
    dag_history.emplace_back({move(contracted_graph), move(architecture)});
    contraction_maps.emplace_back(move(contraction_map));

    return SUCCESS;
}



template<typename Graph_t, typename Graph_t_coarse>
RETURN_STATUS MultilevelCoarseAndSchedule<Graph_t, Graph_t_coarse>::add_contraction(const std::vector<vertex_idx_t<Graph_t_coarse>> &contraction_map, const BspInstance<Graph_t_coarse> &contracted_instance) {
    dag_history.emplace_back(contracted_instance);
    contraction_maps.emplace_back(contraction_map);

    return SUCCESS;
}

template<typename Graph_t, typename Graph_t_coarse>
RETURN_STATUS MultilevelCoarseAndSchedule<Graph_t, Graph_t_coarse>::add_contraction(std::vector<vertex_idx_t<Graph_t_coarse>> &&contraction_map, BspInstance<Graph_t_coarse> &&contracted_instance) {
    dag_history.emplace_back(move(contracted_instance));
    contraction_maps.emplace_back(move(contraction_map));

    return SUCCESS;
}








} // end namespace osp