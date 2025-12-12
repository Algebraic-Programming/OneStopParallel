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

// #define KL_DEBUG

#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/model/IBspSchedule.hpp"
#include "osp/bsp/model/util/SetSchedule.hpp"
#include "osp/bsp/model/util/VectorSchedule.hpp"
#include "osp/bsp/scheduler/ImprovementScheduler.hpp"
#include "osp/bsp/scheduler/LocalSearch/LocalSearchMemoryConstraintModules.hpp"
#include "osp/graph_algorithms/directed_graph_util.hpp"

namespace osp {

template <typename GraphT>
struct KlMove {
    vertex_idx_t<Graph_t> node_;

    double gain_;
    double changeInCost_;

    unsigned fromProc_;
    unsigned fromStep_;

    unsigned toProc_;
    unsigned toStep_;

    KlMove() : node(0), gain_(0), changeInCost_(0), fromProc_(0), fromStep_(0), toProc_(0), toStep_(0) {}

    KlMove(vertexIdxT_<Graph_t> _node,
           double _gain,
           double _change_cost,
           unsigned _from_proc,
           unsigned _from_step,
           unsigned _to_proc,
           unsigned _to_step)
        : node(_node),
          Gain(_gain),
          ChangeInCost(_change_cost),
          FromProc(_from_proc),
          FromStep(_from_step),
          ToProc(_to_proc),
          ToStep(_to_step) {}

    bool operator<(kl_move const &rhs) const {
        return (gain < rhs.gain) or (gain <= rhs.gain and change_in_cost < rhs.change_in_cost)
               or (gain <= rhs.gain and change_in_cost <= rhs.change_in_cost and node > rhs.node);
    }

    kl_move reverse_move() const { return kl_move(node, -gain, -change_in_cost, to_proc, to_step, from_proc, from_step); }
};

class IklCostFunction {
  public:
    virtual double ComputeCurrentCosts() = 0;

    virtual ~IklCostFunction() = default;
};

template <typename GraphT, typename MemoryConstraintT>
class KlCurrentSchedule {
  private:
    using VertexType = vertex_idx_t<Graph_t>;
    using EdgeType = edge_desc_t<Graph_t>;

  public:
    KlCurrentSchedule(IklCostFunction *costF) : costF_(costF) {
#ifdef KL_DEBUG
        if constexpr (use_memory_constraint) {
            std::cout << "KLCurrentSchedule constructor with memory constraint" << std::endl;
        } else {
            std::cout << "KLCurrentSchedule constructor without memory constraint" << std::endl;
        }
#endif
    }

    virtual ~KlCurrentSchedule() = default;

    IklCostFunction *costF_;

    const BspInstance<GraphT> *instance_;

    VectorSchedule<GraphT> vectorSchedule_;
    SetSchedule<GraphT> setSchedule_;

    constexpr static bool useMemoryConstraint_ = is_local_search_memory_constraint_v<MemoryConstraint_t>;

    MemoryConstraintT memoryConstraint_;

    std::vector<std::vector<VWorkwT<Graph_t>>> stepProcessorWork_;

    std::vector<VWorkwT<Graph_t>> stepMaxWork_;
    std::vector<VWorkwT<Graph_t>> stepSecondMaxWork_;

    double currentCost_ = 0;

    bool currentFeasible_ = true;
    std::unordered_set<EdgeType> currentViolations_;    // edges

    std::unordered_map<VertexType, EdgeType> newViolations_;
    std::unordered_set<EdgeType> resolvedViolations_;

    void RemoveSuperstep(unsigned step) {
        if (step > 0) {
            vectorSchedule_.mergeSupersteps(step - 1, step);
            setSchedule_.mergeSupersteps(step - 1, step);

            ComputeWorkMemoryDatastructures(step - 1, step);

        } else {
            vectorSchedule_.mergeSupersteps(0, 1);
            setSchedule_.mergeSupersteps(0, 1);

            ComputeWorkMemoryDatastructures(0, 0);
        }

        for (unsigned i = step + 1; i < NumSteps(); i++) {
            step_max_work[i] = step_max_work[i + 1];
            step_second_max_work[i] = step_second_max_work[i + 1];

            for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
                step_processor_work[i][proc] = step_processor_work[i + 1][proc];

                if constexpr (useMemoryConstraint_) {
                    memoryConstraint_.override_superstep(i, proc, i + 1, proc);
                }
            }
        }

        step_second_max_work[num_steps()] = 0;
        step_max_work[num_steps()] = 0;

        if constexpr (useMemoryConstraint_) {
            memoryConstraint_.reset_superstep(NumSteps());
        }

        RecomputeCurrentViolations();
        costF_->ComputeCurrentCosts();
    }

    void ResetSuperstep(unsigned step) {
        if (step > 0) {
            ComputeWorkMemoryDatastructures(step - 1, step - 1);
            if (step < NumSteps() - 1) {
                ComputeWorkMemoryDatastructures(step + 1, step + 1);
            }
        } else {
            ComputeWorkMemoryDatastructures(1, 1);
        }

        step_second_max_work[step] = 0;
        step_max_work[step] = 0;

        if constexpr (useMemoryConstraint_) {
            memoryConstraint_.reset_superstep(step);
        }

        RecomputeCurrentViolations();
        costF_->ComputeCurrentCosts();
    }

    void RecomputeNeighboringSupersteps(unsigned step) {
        if (step > 0) {
            ComputeWorkMemoryDatastructures(step - 1, step);
            if (step < NumSteps() - 1) {
                ComputeWorkMemoryDatastructures(step + 1, step + 1);
            }
        } else {
            ComputeWorkMemoryDatastructures(0, 0);
            if (NumSteps() > 1) {
                ComputeWorkMemoryDatastructures(1, 1);
            }
        }
    }

    inline unsigned NumSteps() const { return vectorSchedule_.NumberOfSupersteps(); }

    virtual void SetCurrentSchedule(const IBspSchedule<GraphT> &schedule) {
        if (NumSteps() == schedule.NumberOfSupersteps()) {
#ifdef KL_DEBUG
            std::cout << "KLCurrentSchedule set current schedule, same nr supersteps" << std::endl;
#endif

            for (unsigned step = 0; step < NumSteps(); step++) {
                for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
                    setSchedule_.step_processor_vertices[step][proc].clear();
                }
            }

            for (const auto &node : instance_->GetComputationalDag().vertices()) {
                vectorSchedule_.setAssignedProcessor(node, schedule.assignedProcessor(node));
                vectorSchedule_.setAssignedSuperstep(node, schedule.assignedSuperstep(node));

                setSchedule_.step_processor_vertices[schedule.assignedSuperstep(node)][schedule.assignedProcessor(node)].insert(
                    node);
            }

        } else {
#ifdef KL_DEBUG
            std::cout << "KLCurrentSchedule set current schedule, different nr supersteps" << std::endl;
#endif

            vectorSchedule_ = VectorSchedule(schedule);
            setSchedule_ = SetSchedule(schedule);

            InitializeSuperstepDatastructures();
        }

        ComputeWorkMemoryDatastructures(0, NumSteps() - 1);
        RecomputeCurrentViolations();

        costF_->ComputeCurrentCosts();

#ifdef KL_DEBUG
        std::cout << "KLCurrentSchedule set current schedule done, costs: " << current_cost
                  << " number of supersteps: " << num_steps() << std::endl;
#endif
    }

    virtual void InitializeSuperstepDatastructures() {
#ifdef KL_DEBUG
        std::cout << "KLCurrentSchedule initialize datastructures" << std::endl;
#endif

        const unsigned numProcs = instance_->NumberOfProcessors();

        if constexpr (useMemoryConstraint_) {
            memoryConstraint_.initialize(setSchedule_, vectorSchedule_);
        }

        step_processor_work = std::vector<std::vector<VWorkwT<Graph_t>>>(num_steps(), std::vector<VWorkwT<Graph_t>>(num_procs, 0));
        step_max_work = std::vector<VWorkwT<Graph_t>>(num_steps(), 0);
        step_second_max_work = std::vector<VWorkwT<Graph_t>>(num_steps(), 0);
    }

    virtual void CleanupSuperstepDatastructures() {
        step_processor_work.clear();
        step_max_work.clear();
        step_second_max_work.clear();

        if constexpr (useMemoryConstraint_) {
            memoryConstraint_.clear();
        }
    }

    virtual void ComputeWorkMemoryDatastructures(unsigned startStep, unsigned endStep) {
        if constexpr (useMemoryConstraint_) {
            memoryConstraint_.compute_memory_datastructure(startStep, endStep);
        }

        for (unsigned step = startStep; step <= endStep; step++) {
            step_max_work[step] = 0;
            step_second_max_work[step] = 0;

            for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
                step_processor_work[step][proc] = 0;

                for (const auto &node : setSchedule_.step_processor_vertices[step][proc]) {
                    step_processor_work[step][proc] += instance->GetComputationalDag().VertexWorkWeight(node);
                }

                if (step_processor_work[step][proc] > step_max_work[step]) {
                    step_second_max_work[step] = step_max_work[step];
                    step_max_work[step] = step_processor_work[step][proc];

                } else if (step_processor_work[step][proc] > step_second_max_work[step]) {
                    step_second_max_work[step] = step_processor_work[step][proc];
                }
            }
        }
    }

    virtual void RecomputeCurrentViolations() {
        current_violations.clear();

#ifdef KL_DEBUG
        std::cout << "Recompute current violations:" << std::endl;
#endif

        for (const auto &edge : Edges(instance_->GetComputationalDag())) {
            const auto &sourceV = Source(edge, instance_->GetComputationalDag());
            const auto &targetV = Traget(edge, instance_->GetComputationalDag());

            if (vectorSchedule_.assignedSuperstep(sourceV) >= vectorSchedule_.assignedSuperstep(targetV)) {
                if (vectorSchedule_.assignedProcessor(sourceV) != vectorSchedule_.assignedProcessor(targetV)
                    || vectorSchedule_.assignedSuperstep(sourceV) > vectorSchedule_.assignedSuperstep(targetV)) {
                    current_violations.insert(edge);

#ifdef KL_DEBUG
                    std::cout << "Edge: " << source_v << " -> " << target_v << std::endl;
#endif
                }
            }
        }

        if (current_violations.size() > 0) {
            currentFeasible_ = false;
        } else {
#ifdef KL_DEBUG
            std::cout << "Current schedule is feasible" << std::endl;
#endif

            currentFeasible_ = true;
        }
    };

    virtual void ApplyMove(KlMove<GraphT> move) {
        vectorSchedule_.setAssignedProcessor(move.node, move.to_proc);
        vectorSchedule_.setAssignedSuperstep(move.node, move.to_step);

        setSchedule_.step_processor_vertices[move.from_step][move.from_proc].erase(move.node);
        setSchedule_.step_processor_vertices[move.to_step][move.to_proc].insert(move.node);

        currentCost_ += move.change_in_cost;

        step_processor_work[move.to_step][move.to_proc] += instance->GetComputationalDag().VertexWorkWeight(move.node);
        step_processor_work[move.from_step][move.from_proc] -= instance->GetComputationalDag().VertexWorkWeight(move.node);

        UpdateMaxWorkDatastructures(move);
        update_violations(move.node);

        if constexpr (useMemoryConstraint_) {
            memoryConstraint_.apply_move(move.node, move.from_proc, move.from_step, move.to_proc, move.to_step);
        }
    }

    virtual void InitializeCurrentSchedule(const IBspSchedule<GraphT> &schedule) {
#ifdef KL_DEBUG
        std::cout << "KLCurrentSchedule initialize current schedule" << std::endl;
#endif

        vectorSchedule_ = VectorSchedule<GraphT>(schedule);
        setSchedule_ = SetSchedule<GraphT>(schedule);

        InitializeSuperstepDatastructures();

        ComputeWorkMemoryDatastructures(0, NumSteps() - 1);
        RecomputeCurrentViolations();

        costF_->ComputeCurrentCosts();
    }

  private:
    void UpdateViolations(VertexType node) {
        new_violations.clear();
        resolved_violations.clear();

        for (const auto &edge : OutEdges(node, instance->GetComputationalDag())) {
            const auto &child = Traget(edge, instance->GetComputationalDag());

            if (current_violations.find(edge) == current_violations.end()) {
                if (vector_schedule.assignedSuperstep(node) >= vector_schedule.assignedSuperstep(child)) {
                    if (vector_schedule.assignedProcessor(node) != vector_schedule.assignedProcessor(child)
                        || vector_schedule.assignedSuperstep(node) > vector_schedule.assignedSuperstep(child)) {
                        current_violations.insert(edge);
                        new_violations[child] = edge;
                    }
                }
            } else {
                if (vector_schedule.assignedSuperstep(node) <= vector_schedule.assignedSuperstep(child)) {
                    if (vector_schedule.assignedProcessor(node) == vector_schedule.assignedProcessor(child)
                        || vector_schedule.assignedSuperstep(node) < vector_schedule.assignedSuperstep(child)) {
                        current_violations.erase(edge);
                        resolved_violations.insert(edge);
                    }
                }
            }
        }

        for (const auto &edge : InEdges(node, instance->GetComputationalDag())) {
            const auto &parent = Source(edge, instance->GetComputationalDag());

            if (current_violations.find(edge) == current_violations.end()) {
                if (vector_schedule.assignedSuperstep(node) <= vector_schedule.assignedSuperstep(parent)) {
                    if (vector_schedule.assignedProcessor(node) != vector_schedule.assignedProcessor(parent)
                        || vector_schedule.assignedSuperstep(node) < vector_schedule.assignedSuperstep(parent)) {
                        current_violations.insert(edge);
                        new_violations[parent] = edge;
                    }
                }
            } else {
                if (vector_schedule.assignedSuperstep(node) >= vector_schedule.assignedSuperstep(parent)) {
                    if (vector_schedule.assignedProcessor(node) == vector_schedule.assignedProcessor(parent)
                        || vector_schedule.assignedSuperstep(node) > vector_schedule.assignedSuperstep(parent)) {
                        current_violations.erase(edge);
                        resolved_violations.insert(edge);
                    }
                }
            }
        }

#ifdef KL_DEBUG

        if (new_violations.size() > 0) {
            std::cout << "New violations: " << std::endl;
            for (const auto &edge : new_violations) {
                std::cout << "Edge: " << Source(edge.second, instance->GetComputationalDag()) << " -> "
                          << Traget(edge.second, instance->GetComputationalDag()) << std::endl;
            }
        }

        if (resolved_violations.size() > 0) {
            std::cout << "Resolved violations: " << std::endl;
            for (const auto &edge : resolved_violations) {
                std::cout << "Edge: " << Source(edge, instance->GetComputationalDag()) << " -> "
                          << Traget(edge, instance->GetComputationalDag()) << std::endl;
            }
        }

#endif

        if (current_violations.size() > 0) {
            currentFeasible_ = false;
        } else {
            currentFeasible_ = true;
        }
    }

    void UpdateMaxWorkDatastructures(KlMove<GraphT> move) {
        if (move.from_step == move.to_step) {
            RecomputeSuperstepMaxWork(move.from_step);

        } else {
            RecomputeSuperstepMaxWork(move.from_step);
            RecomputeSuperstepMaxWork(move.to_step);
        }
    }

    void RecomputeSuperstepMaxWork(unsigned step) {
        step_max_work[step] = 0;
        step_second_max_work[step] = 0;

        for (unsigned proc = 0; proc < instance_->NumberOfProcessors(); proc++) {
            if (step_processor_work[step][proc] > step_max_work[step]) {
                step_second_max_work[step] = step_max_work[step];
                step_max_work[step] = step_processor_work[step][proc];

            } else if (step_processor_work[step][proc] > step_second_max_work[step]) {
                step_second_max_work[step] = step_processor_work[step][proc];
            }
        }
    }
};

template <typename GraphT, typename MemoryConstraintT>
class KlCurrentScheduleMaxComm : public KlCurrentSchedule<GraphT, MemoryConstraintT> {
  public:
    std::vector<std::vector<VCommwT<Graph_t>>> stepProcessorSend_;
    std::vector<VCommwT<Graph_t>> stepMaxSend_;
    std::vector<VCommwT<Graph_t>> stepMaxReceive_;

    std::vector<std::vector<VCommwT<Graph_t>>> stepProcessorReceive_;
    std::vector<VCommwT<Graph_t>> stepSecondMaxSend_;
    std::vector<VCommwT<Graph_t>> stepSecondMaxReceive_;
};

}    // namespace osp
