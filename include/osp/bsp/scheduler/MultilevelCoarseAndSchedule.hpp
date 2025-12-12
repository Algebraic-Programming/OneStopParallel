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
#include "osp/coarser/MultilevelCoarser.hpp"
#include "osp/coarser/coarser_util.hpp"

namespace osp {

template <typename GraphT, typename GraphTCoarse>
class MultilevelCoarseAndSchedule : public Scheduler<GraphT> {
  private:
    const BspInstance<GraphT> *originalInst_;

  protected:
    inline const BspInstance<GraphT> *GetOriginalInstance() const { return originalInst_; };

    Scheduler<GraphTCoarse> *sched_;
    ImprovementScheduler<GraphTCoarse> *improver_;

    MultilevelCoarser<GraphT, GraphTCoarse> *mlCoarser_;
    long int activeGraph_;
    std::unique_ptr<BspInstance<GraphTCoarse>> activeInstance_;
    std::unique_ptr<BspSchedule<GraphTCoarse>> activeSchedule_;

    ReturnStatus ComputeInitialSchedule();
    ReturnStatus ExpandActiveSchedule();
    ReturnStatus ExpandActiveScheduleToOriginalSchedule(BspSchedule<GraphT> &schedule);
    ReturnStatus ImproveActiveSchedule();
    ReturnStatus RunExpansions(BspSchedule<GraphT> &schedule);

    void ClearComputationData();

  public:
    MultilevelCoarseAndSchedule()
        : Scheduler<GraphT>(), originalInst_(nullptr), sched_(nullptr), improver_(nullptr), mlCoarser_(nullptr), activeGraph_(-1L) {
          };
    MultilevelCoarseAndSchedule(Scheduler<GraphTCoarse> &sched, MultilevelCoarser<GraphT, GraphTCoarse> &mlCoarser)
        : Scheduler<GraphT>(),
          originalInst_(nullptr),
          sched_(&sched),
          improver_(nullptr),
          mlCoarser_(&mlCoarser),
          activeGraph_(-1L) {};
    MultilevelCoarseAndSchedule(Scheduler<GraphTCoarse> &sched,
                                ImprovementScheduler<GraphTCoarse> &improver,
                                MultilevelCoarser<GraphT, GraphTCoarse> &mlCoarser)
        : Scheduler<GraphT>(),
          originalInst_(nullptr),
          sched_(&sched),
          improver_(&improver),
          mlCoarser_(&mlCoarser),
          activeGraph_(-1L) {};
    virtual ~MultilevelCoarseAndSchedule() = default;

    inline void SetInitialScheduler(Scheduler<GraphTCoarse> &sched) { sched_ = &sched; };

    inline void SetImprovementScheduler(ImprovementScheduler<GraphTCoarse> &improver) { improver_ = &improver; };

    inline void SetMultilevelCoarser(MultilevelCoarser<GraphT, GraphTCoarse> &mlCoarser) { mlCoarser_ = &mlCoarser; };

    ReturnStatus computeSchedule(BspSchedule<GraphT> &schedule) override;

    std::string getScheduleName() const override {
        if (improver_ == nullptr) {
            return "C:" + mlCoarser_->getCoarserName() + "-S:" + sched_->getScheduleName();
        } else {
            return "C:" + mlCoarser_->getCoarserName() + "-S:" + sched_->getScheduleName() + "-I:" + improver_->getScheduleName();
        }
    };
};

template <typename GraphT, typename GraphTCoarse>
ReturnStatus MultilevelCoarseAndSchedule<GraphT, GraphTCoarse>::ComputeInitialSchedule() {
    activeGraph_ = static_cast<long int>(mlCoarser_->dag_history.size());
    activeGraph_--;

    assert((activeGraph_ >= 0L) && "Must have done at least one coarsening!");

    ReturnStatus status;

    activeInstance_ = std::make_unique<BspInstance<GraphTCoarse>>(
        *(mlCoarser_->dag_history.at(static_cast<std::size_t>(activeGraph_))), originalInst_->GetArchitecture());
    activeSchedule_ = std::make_unique<BspSchedule<GraphTCoarse>>(*activeInstance_);
    status = sched_->computeSchedule(*activeSchedule_);
    assert(activeSchedule_->satisfiesPrecedenceConstraints());

    ReturnStatus ret = improve_active_schedule();
    status = std::max(ret, status);

    return status;
}

template <typename GraphT, typename GraphTCoarse>
ReturnStatus MultilevelCoarseAndSchedule<GraphT, GraphTCoarse>::ImproveActiveSchedule() {
    if (improver_) {
        if (activeInstance_->GetComputationalDag().NumVertices() == 0) {
            return ReturnStatus::OSP_SUCCESS;
        }
        return improver_->improveSchedule(*activeSchedule_);
    }
    return ReturnStatus::OSP_SUCCESS;
}

template <typename GraphT, typename GraphTCoarse>
ReturnStatus MultilevelCoarseAndSchedule<GraphT, GraphTCoarse>::ExpandActiveSchedule() {
    assert((activeGraph_ > 0L) && (static_cast<long unsigned>(activeGraph_) < mlCoarser_->dag_history.size()));

    std::unique_ptr<BspInstance<GraphTCoarse>> expandedInstance = std::make_unique<BspInstance<GraphTCoarse>>(
        *(mlCoarser_->dag_history.at(static_cast<std::size_t>(activeGraph_) - 1)), originalInst_->GetArchitecture());
    std::unique_ptr<BspSchedule<GraphTCoarse>> expandedSchedule = std::make_unique<BspSchedule<GraphTCoarse>>(*expandedInstance);

    for (const auto &node : expandedInstance->GetComputationalDag().vertices()) {
        expandedSchedule->SetAssignedProcessor(
            node,
            activeSchedule_->AssignedProcessor(mlCoarser_->contraction_maps.at(static_cast<std::size_t>(activeGraph_))->at(node)));
        expandedSchedule->setAssignedSuperstep(
            node,
            activeSchedule_->AssignedSuperstep(mlCoarser_->contraction_maps.at(static_cast<std::size_t>(activeGraph_))->at(node)));
    }

    assert(expandedSchedule->satisfiesPrecedenceConstraints());

    // std::cout << "exp_inst:  " << expanded_instance.get() << " n: " << expanded_instance->NumberOfVertices() << " m:
    // " << expanded_instance->GetComputationalDag().NumEdges() << std::endl; std::cout << "exp_sched: " <<
    // &expanded_schedule->GetInstance() << " n: " << expanded_schedule->GetInstance().NumberOfVertices() << " m: " <<
    // expanded_schedule->GetInstance().GetComputationalDag().NumEdges() << std::endl;

    activeGraph_--;
    std::swap(expandedInstance, activeInstance_);
    std::swap(expandedSchedule, activeSchedule_);

    // std::cout << "act_inst:  " << active_instance.get() << " n: " << active_instance->NumberOfVertices() << " m: " <<
    // active_instance->GetComputationalDag().NumEdges() << std::endl; std::cout << "act_sched: " <<
    // &active_schedule->GetInstance() << " n: " << active_schedule->GetInstance().NumberOfVertices() << " m: " <<
    // active_schedule->GetInstance().GetComputationalDag().NumEdges() << std::endl;

    assert(activeSchedule_->satisfiesPrecedenceConstraints());
    return ReturnStatus::OSP_SUCCESS;
}

template <typename GraphT, typename GraphTCoarse>
ReturnStatus MultilevelCoarseAndSchedule<GraphT, GraphTCoarse>::ExpandActiveScheduleToOriginalSchedule(BspSchedule<GraphT> &schedule) {
    assert(activeGraph_ == 0L);

    for (const auto &node : GetOriginalInstance()->GetComputationalDag().vertices()) {
        schedule.SetAssignedProcessor(
            node,
            activeSchedule_->AssignedProcessor(mlCoarser_->contraction_maps.at(static_cast<std::size_t>(activeGraph_))->at(node)));
        schedule.setAssignedSuperstep(
            node,
            activeSchedule_->AssignedSuperstep(mlCoarser_->contraction_maps.at(static_cast<std::size_t>(activeGraph_))->at(node)));
    }

    activeGraph_--;
    activeInstance_ = std::unique_ptr<BspInstance<GraphTCoarse>>();
    activeSchedule_ = std::unique_ptr<BspSchedule<GraphTCoarse>>();

    assert(schedule.satisfiesPrecedenceConstraints());

    return ReturnStatus::OSP_SUCCESS;
}

template <typename GraphT, typename GraphTCoarse>
ReturnStatus MultilevelCoarseAndSchedule<GraphT, GraphTCoarse>::RunExpansions(BspSchedule<GraphT> &schedule) {
    assert(activeGraph_ >= 0L && static_cast<long unsigned>(activeGraph_) == mlCoarser_->dag_history.size() - 1);

    ReturnStatus status = ReturnStatus::OSP_SUCCESS;

    while (activeGraph_ > 0L) {
        status = std::max(status, expand_active_schedule());
        status = std::max(status, improve_active_schedule());
    }

    status = std::max(status, expand_active_schedule_to_original_schedule(schedule));

    return status;
}

template <typename GraphT, typename GraphTCoarse>
void MultilevelCoarseAndSchedule<GraphT, GraphTCoarse>::ClearComputationData() {
    activeGraph_ = -1L;
    activeInstance_ = std::unique_ptr<BspInstance<GraphTCoarse>>();
    activeSchedule_ = std::unique_ptr<BspSchedule<GraphTCoarse>>();
}

template <typename GraphT, typename GraphTCoarse>
ReturnStatus MultilevelCoarseAndSchedule<GraphT, GraphTCoarse>::ComputeSchedule(BspSchedule<GraphT> &schedule) {
    ClearComputationData();

    originalInst_ = &schedule.GetInstance();

    ReturnStatus status = ReturnStatus::OSP_SUCCESS;

    status = std::max(status, mlCoarser_->run(*originalInst_));

    if constexpr (std::is_same_v<GraphT, GraphTCoarse>) {
        if (mlCoarser_->dag_history.size() == 0) {
            status = std::max(status, sched_->computeSchedule(schedule));
        } else {
            status = std::max(status, compute_initial_schedule());
            status = std::max(status, run_expansions(schedule));
        }
    } else {
        assert(mlCoarser_->dag_history.size() > 0);

        status = std::max(status, compute_initial_schedule());
        status = std::max(status, run_expansions(schedule));
    }

    assert(activeGraph_ == -1L);

    ClearComputationData();

    return status;
}

}    // end namespace osp
