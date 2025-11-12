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

#include "osp/bsp/model/BspInstance.hpp"
#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/scheduler/ImprovementScheduler.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/coarser/coarser_util.hpp"
#include "osp/coarser/MultilevelCoarser.hpp"


namespace osp {

template<typename Graph_t, typename Graph_t_coarse>
class MultilevelCoarseAndSchedule : public Scheduler<Graph_t> {
  private:
    const BspInstance<Graph_t> *original_inst;
  protected:
    inline const BspInstance<Graph_t> * getOriginalInstance() const { return original_inst; };

    Scheduler<Graph_t_coarse> *sched;
    ImprovementScheduler<Graph_t_coarse> *improver;

    MultilevelCoarser<Graph_t, Graph_t_coarse> *ml_coarser;
    long int active_graph;
    std::unique_ptr< BspInstance<Graph_t_coarse> > active_instance;
    std::unique_ptr< BspSchedule<Graph_t_coarse> > active_schedule;

    RETURN_STATUS compute_initial_schedule();
    RETURN_STATUS expand_active_schedule();
    RETURN_STATUS expand_active_schedule_to_original_schedule(BspSchedule<Graph_t>& schedule);
    RETURN_STATUS improve_active_schedule();
    RETURN_STATUS run_expansions(BspSchedule<Graph_t>& schedule);

    void clear_computation_data();

  public:
    MultilevelCoarseAndSchedule() : Scheduler<Graph_t>(), original_inst(nullptr), sched(nullptr), improver(nullptr), ml_coarser(nullptr), active_graph(-1L) {};
    MultilevelCoarseAndSchedule(Scheduler<Graph_t_coarse> &sched_, MultilevelCoarser<Graph_t, Graph_t_coarse> &ml_coarser_)
        : Scheduler<Graph_t>(), original_inst(nullptr), sched(&sched_), improver(nullptr), ml_coarser(&ml_coarser_), active_graph(-1L) {};
    MultilevelCoarseAndSchedule(Scheduler<Graph_t_coarse> &sched_, ImprovementScheduler<Graph_t_coarse> &improver_, MultilevelCoarser<Graph_t, Graph_t_coarse> &ml_coarser_)
        : Scheduler<Graph_t>(), original_inst(nullptr), sched(&sched_), improver(&improver_), ml_coarser(&ml_coarser_), active_graph(-1L) {};
    MultilevelCoarseAndSchedule(unsigned timelimit, Scheduler<Graph_t_coarse> &sched_, MultilevelCoarser<Graph_t, Graph_t_coarse> &ml_coarser_)
        : Scheduler<Graph_t>(timelimit), original_inst(nullptr), sched(&sched_), improver(nullptr), ml_coarser(&ml_coarser_), active_graph(-1L) {};
    MultilevelCoarseAndSchedule(unsigned timelimit, Scheduler<Graph_t_coarse> &sched_, ImprovementScheduler<Graph_t_coarse> &improver_, MultilevelCoarser<Graph_t, Graph_t_coarse> &ml_coarser_)
        : Scheduler<Graph_t>(timelimit), original_inst(nullptr), sched(&sched_), improver(&improver_), ml_coarser(&ml_coarser_), active_graph(-1L) {};
    virtual ~MultilevelCoarseAndSchedule() = default;

    inline void setInitialScheduler(Scheduler<Graph_t_coarse> &sched_) { sched = &sched_; };
    inline void setImprovementScheduler(ImprovementScheduler<Graph_t_coarse> &improver_) { improver = &improver_; };
    inline void setMultilevelCoarser(MultilevelCoarser<Graph_t, Graph_t_coarse> &ml_coarser_) { ml_coarser = &ml_coarser_; };

    RETURN_STATUS computeSchedule(BspSchedule<Graph_t> &schedule) override;

    std::string getScheduleName() const override {
        if (improver == nullptr) {
            return "C:" + ml_coarser->getCoarserName() + "-S:" + sched->getScheduleName();
        } else {
            return "C:" + ml_coarser->getCoarserName() + "-S:" + sched->getScheduleName() + "-I:" + improver->getScheduleName();
        }
    };

};





template<typename Graph_t, typename Graph_t_coarse>
RETURN_STATUS MultilevelCoarseAndSchedule<Graph_t, Graph_t_coarse>::compute_initial_schedule() {
    active_graph = static_cast<long int>(ml_coarser->dag_history.size());
    active_graph--;

    assert((active_graph >= 0L) && "Must have done at least one coarsening!");
    
    RETURN_STATUS status;

    active_instance = std::make_unique< BspInstance<Graph_t_coarse> >( *(ml_coarser->dag_history.at( static_cast<std::size_t>(active_graph) )), original_inst->getArchitecture());
    active_schedule = std::make_unique< BspSchedule<Graph_t_coarse> >( *active_instance );
    status = sched->computeSchedule( *active_schedule );
    assert(active_schedule->satisfiesPrecedenceConstraints());

    RETURN_STATUS ret = improve_active_schedule();
    status = std::max(ret,status);

    return status;
}

template<typename Graph_t, typename Graph_t_coarse>
RETURN_STATUS MultilevelCoarseAndSchedule<Graph_t, Graph_t_coarse>::improve_active_schedule() {
    if (improver) {
        if (active_instance->getComputationalDag().num_vertices() == 0) return RETURN_STATUS::OSP_SUCCESS;
        return improver->improveSchedule( *active_schedule );
    }
    return RETURN_STATUS::OSP_SUCCESS;
}




template<typename Graph_t, typename Graph_t_coarse>
RETURN_STATUS MultilevelCoarseAndSchedule<Graph_t, Graph_t_coarse>::expand_active_schedule() {
    assert((active_graph > 0L) && ( static_cast<long unsigned>(active_graph) < ml_coarser->dag_history.size()));

    std::unique_ptr< BspInstance<Graph_t_coarse> > expanded_instance = std::make_unique< BspInstance<Graph_t_coarse> >( *(ml_coarser->dag_history.at( static_cast<std::size_t>(active_graph) - 1 )), original_inst->getArchitecture());
    std::unique_ptr< BspSchedule<Graph_t_coarse> > expanded_schedule = std::make_unique< BspSchedule<Graph_t_coarse> >( *expanded_instance );

    for (const auto &node : expanded_instance->getComputationalDag().vertices() ) {
        expanded_schedule->setAssignedProcessor(node, active_schedule->assignedProcessor( ml_coarser->contraction_maps.at( static_cast<std::size_t>(active_graph) )->at(node) ) );
        expanded_schedule->setAssignedSuperstep(node, active_schedule->assignedSuperstep( ml_coarser->contraction_maps.at( static_cast<std::size_t>(active_graph) )->at(node) ) );
    }

    assert(expanded_schedule->satisfiesPrecedenceConstraints());

    // std::cout << "exp_inst:  " << expanded_instance.get() << " n: " << expanded_instance->numberOfVertices() << " m: " << expanded_instance->getComputationalDag().num_edges() << std::endl;
    // std::cout << "exp_sched: " << &expanded_schedule->getInstance() << " n: " << expanded_schedule->getInstance().numberOfVertices() << " m: " << expanded_schedule->getInstance().getComputationalDag().num_edges() << std::endl;


    active_graph--;
    std::swap(expanded_instance, active_instance);
    std::swap(expanded_schedule, active_schedule);

    // std::cout << "act_inst:  " << active_instance.get() << " n: " << active_instance->numberOfVertices() << " m: " << active_instance->getComputationalDag().num_edges() << std::endl;
    // std::cout << "act_sched: " << &active_schedule->getInstance() << " n: " << active_schedule->getInstance().numberOfVertices() << " m: " << active_schedule->getInstance().getComputationalDag().num_edges() << std::endl;
    
    assert(active_schedule->satisfiesPrecedenceConstraints());    
    return RETURN_STATUS::OSP_SUCCESS;
}


template<typename Graph_t, typename Graph_t_coarse>
RETURN_STATUS MultilevelCoarseAndSchedule<Graph_t, Graph_t_coarse>::expand_active_schedule_to_original_schedule(BspSchedule<Graph_t>& schedule) {
    assert(active_graph == 0L);

    for (const auto &node : getOriginalInstance()->getComputationalDag().vertices() ) {
        schedule.setAssignedProcessor(node, active_schedule->assignedProcessor( ml_coarser->contraction_maps.at( static_cast<std::size_t>(active_graph) )->at(node)) );
        schedule.setAssignedSuperstep(node, active_schedule->assignedSuperstep( ml_coarser->contraction_maps.at( static_cast<std::size_t>(active_graph) )->at(node)) );
    }

    active_graph--;
    active_instance = std::unique_ptr< BspInstance<Graph_t_coarse> >();
    active_schedule = std::unique_ptr< BspSchedule<Graph_t_coarse> >();

    assert(schedule.satisfiesPrecedenceConstraints());
    
    return RETURN_STATUS::OSP_SUCCESS;
}






template<typename Graph_t, typename Graph_t_coarse>
RETURN_STATUS MultilevelCoarseAndSchedule<Graph_t, Graph_t_coarse>::run_expansions(BspSchedule<Graph_t>& schedule) {
    assert(active_graph >= 0L && static_cast<long unsigned>(active_graph) == ml_coarser->dag_history.size() - 1);

    RETURN_STATUS status = RETURN_STATUS::OSP_SUCCESS;
    
    while(active_graph > 0L) {
        status = std::max(status, expand_active_schedule());
        status = std::max(status, improve_active_schedule());
    }

    status = std::max(status, expand_active_schedule_to_original_schedule(schedule));
    
    return status;
}


template<typename Graph_t, typename Graph_t_coarse>
void MultilevelCoarseAndSchedule<Graph_t, Graph_t_coarse>::clear_computation_data() {
    active_graph = -1L;
    active_instance = std::unique_ptr< BspInstance<Graph_t_coarse> >();
    active_schedule = std::unique_ptr< BspSchedule<Graph_t_coarse> >();
}



template<typename Graph_t, typename Graph_t_coarse>
RETURN_STATUS MultilevelCoarseAndSchedule<Graph_t, Graph_t_coarse>::computeSchedule(BspSchedule<Graph_t>& schedule) {
    clear_computation_data();

    original_inst = &schedule.getInstance();

    RETURN_STATUS status = RETURN_STATUS::OSP_SUCCESS;

    status = std::max(status, ml_coarser->run(*original_inst));

    if constexpr (std::is_same_v<Graph_t, Graph_t_coarse>) {
        if ( ml_coarser->dag_history.size() == 0 ) {
            status = std::max(status, sched->computeSchedule(schedule));
        } else {
            status = std::max(status, compute_initial_schedule());
            status = std::max(status, run_expansions(schedule));
        }
    } else {    
        assert(ml_coarser->dag_history.size() > 0);
    
        status = std::max(status, compute_initial_schedule());
        status = std::max(status, run_expansions(schedule));
    }

    assert(active_graph == -1L);

    clear_computation_data();

    return status;
}







} // end namespace osp