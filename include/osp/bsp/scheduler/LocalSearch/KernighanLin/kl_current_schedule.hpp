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
    VertexIdxT<GraphT> node;

    double gain;
    double changeInCost;

    unsigned fromProc;
    unsigned fromStep;

    unsigned toProc;
    unsigned toStep;

    KlMove() : node(0), gain(0), changeInCost(0), fromProc(0), fromStep(0), toProc(0), toStep(0) {}

    KlMove(VertexIdxT<GraphT> node,
           double gain,
           double changeCost,
           unsigned fromProc,
           unsigned fromStep,
           unsigned toProc,
           unsigned toStep)
        : node(node), gain(gain), changeInCost(changeCost), fromProc(fromProc), fromStep(fromStep), toProc(toProc), toStep(toStep) {}

    bool operator<(KlMove const &rhs) const {
        return (gain < rhs.gain) or (gain <= rhs.gain and changeInCost < rhs.changeInCost)
               or (gain <= rhs.gain and changeInCost <= rhs.changeInCost and node > rhs.node);
    }

    KlMove ReverseMove() const { return KlMove(node, -gain, -changeInCost, toProc, toStep, fromProc, fromStep); }
};

class IklCostFunction {
  public:
    virtual double ComputeCurrentCosts() = 0;

    virtual ~IklCostFunction() = default;
};

template <typename GraphT, typename MemoryConstraintT>
class KlCurrentSchedule {
  private:
    using VertexType = VertexIdxT<GraphT>;
    using EdgeType = EdgeDescT<GraphT>;

  public:
    KlCurrentSchedule(IklCostFunction *costF) : costF(costF) {
#ifdef KL_DEBUG
        if constexpr (use_memory_constraint) {
            std::cout << "KLCurrentSchedule constructor with memory constraint" << std::endl;
        } else {
            std::cout << "KLCurrentSchedule constructor without memory constraint" << std::endl;
        }
#endif
    }

    virtual ~KlCurrentSchedule() = default;

    IklCostFunction *costF;

    const BspInstance<GraphT> *instance;

    VectorSchedule<GraphT> vectorSchedule;
    SetSchedule<GraphT> setSchedule;

    constexpr static bool useMemoryConstraint = is_local_search_memory_constraint_v<MemoryConstraintT>;

    MemoryConstraintT memoryConstraint;

    std::vector<std::vector<VWorkwT<GraphT>>> stepProcessorWork;

    std::vector<VWorkwT<GraphT>> stepMaxWork;
    std::vector<VWorkwT<GraphT>> stepSecondMaxWork;

    double currentCost = 0;

    bool currentFeasible = true;
    std::unordered_set<EdgeType> currentViolations;    // edges

    std::unordered_map<VertexType, EdgeType> newViolations;
    std::unordered_set<EdgeType> resolvedViolations;

    void RemoveSuperstep(unsigned step) {
        if (step > 0) {
            vectorSchedule.mergeSupersteps(step - 1, step);
            setSchedule.mergeSupersteps(step - 1, step);

            ComputeWorkMemoryDatastructures(step - 1, step);

        } else {
            vectorSchedule.mergeSupersteps(0, 1);
            setSchedule.mergeSupersteps(0, 1);

            ComputeWorkMemoryDatastructures(0, 0);
        }

        for (unsigned i = step + 1; i < NumSteps(); i++) {
            stepMaxWork[i] = stepMaxWork[i + 1];
            stepSecondMaxWork[i] = stepSecondMaxWork[i + 1];

            for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {
                stepProcessorWork[i][proc] = stepProcessorWork[i + 1][proc];

                if constexpr (useMemoryConstraint) {
                    memoryConstraint.override_superstep(i, proc, i + 1, proc);
                }
            }
        }

        stepSecondMaxWork[NumSteps()] = 0;
        stepMaxWork[NumSteps()] = 0;

        if constexpr (useMemoryConstraint) {
            memoryConstraint.reset_superstep(NumSteps());
        }

        RecomputeCurrentViolations();
        costF->ComputeCurrentCosts();
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

        stepSecondMaxWork[step] = 0;
        stepMaxWork[step] = 0;

        if constexpr (useMemoryConstraint) {
            memoryConstraint.reset_superstep(step);
        }

        RecomputeCurrentViolations();
        costF->ComputeCurrentCosts();
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

    inline unsigned NumSteps() const { return vectorSchedule.numberOfSupersteps(); }

    virtual void SetCurrentSchedule(const IBspSchedule<GraphT> &schedule) {
        if (NumSteps() == schedule.numberOfSupersteps()) {
#ifdef KL_DEBUG
            std::cout << "KLCurrentSchedule set current schedule, same nr supersteps" << std::endl;
#endif

            for (unsigned step = 0; step < NumSteps(); step++) {
                for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {
                    setSchedule.step_processor_vertices[step][proc].clear();
                }
            }

            for (const auto &node : instance->getComputationalDag().vertices()) {
                vectorSchedule.setAssignedProcessor(node, schedule.assignedProcessor(node));
                vectorSchedule.setAssignedSuperstep(node, schedule.assignedSuperstep(node));

                setSchedule.step_processor_vertices[schedule.assignedSuperstep(node)][schedule.assignedProcessor(node)].insert(
                    node);
            }

        } else {
#ifdef KL_DEBUG
            std::cout << "KLCurrentSchedule set current schedule, different nr supersteps" << std::endl;
#endif

            vectorSchedule = VectorSchedule(schedule);
            setSchedule = SetSchedule(schedule);

            InitializeSuperstepDatastructures();
        }

        ComputeWorkMemoryDatastructures(0, NumSteps() - 1);
        RecomputeCurrentViolations();

        costF->ComputeCurrentCosts();

#ifdef KL_DEBUG
        std::cout << "KLCurrentSchedule set current schedule done, costs: " << current_cost
                  << " number of supersteps: " << num_steps() << std::endl;
#endif
    }

    virtual void InitializeSuperstepDatastructures() {
#ifdef KL_DEBUG
        std::cout << "KLCurrentSchedule initialize datastructures" << std::endl;
#endif

        const unsigned numProcs = instance->numberOfProcessors();

        if constexpr (useMemoryConstraint) {
            memoryConstraint.initialize(setSchedule, vectorSchedule);
        }

        stepProcessorWork = std::vector<std::vector<VWorkwT<GraphT>>>(NumSteps(), std::vector<VWorkwT<GraphT>>(numProcs, 0));
        stepMaxWork = std::vector<VWorkwT<GraphT>>(NumSteps(), 0);
        stepSecondMaxWork = std::vector<VWorkwT<GraphT>>(NumSteps(), 0);
    }

    virtual void CleanupSuperstepDatastructures() {
        stepProcessorWork.clear();
        stepMaxWork.clear();
        stepSecondMaxWork.clear();

        if constexpr (useMemoryConstraint) {
            memoryConstraint.clear();
        }
    }

    virtual void ComputeWorkMemoryDatastructures(unsigned startStep, unsigned endStep) {
        if constexpr (useMemoryConstraint) {
            memoryConstraint.compute_memory_datastructure(startStep, endStep);
        }

        for (unsigned step = startStep; step <= endStep; step++) {
            stepMaxWork[step] = 0;
            stepSecondMaxWork[step] = 0;

            for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {
                stepProcessorWork[step][proc] = 0;

                for (const auto &node : setSchedule.step_processor_vertices[step][proc]) {
                    stepProcessorWork[step][proc] += instance->getComputationalDag().vertex_work_weight(node);
                }

                if (stepProcessorWork[step][proc] > stepMaxWork[step]) {
                    stepSecondMaxWork[step] = stepMaxWork[step];
                    stepMaxWork[step] = stepProcessorWork[step][proc];

                } else if (stepProcessorWork[step][proc] > stepSecondMaxWork[step]) {
                    stepSecondMaxWork[step] = stepProcessorWork[step][proc];
                }
            }
        }
    }

    virtual void RecomputeCurrentViolations() {
        currentViolations.clear();

#ifdef KL_DEBUG
        std::cout << "Recompute current violations:" << std::endl;
#endif

        for (const auto &edge : edges(instance->getComputationalDag())) {
            const auto &sourceV = source(edge, instance->getComputationalDag());
            const auto &targetV = target(edge, instance->getComputationalDag());

            if (vectorSchedule.assignedSuperstep(sourceV) >= vectorSchedule.assignedSuperstep(targetV)) {
                if (vectorSchedule.assignedProcessor(sourceV) != vectorSchedule.assignedProcessor(targetV)
                    || vectorSchedule.assignedSuperstep(sourceV) > vectorSchedule.assignedSuperstep(targetV)) {
                    currentViolations.insert(edge);

#ifdef KL_DEBUG
                    std::cout << "Edge: " << source_v << " -> " << target_v << std::endl;
#endif
                }
            }
        }

        if (currentViolations.size() > 0) {
            currentFeasible = false;
        } else {
#ifdef KL_DEBUG
            std::cout << "Current schedule is feasible" << std::endl;
#endif

            currentFeasible = true;
        }
    };

    virtual void ApplyMove(KlMove<GraphT> move) {
        vectorSchedule.setAssignedProcessor(move.node, move.to_proc);
        vectorSchedule.setAssignedSuperstep(move.node, move.to_step);

        setSchedule.step_processor_vertices[move.from_step][move.from_proc].erase(move.node);
        setSchedule.step_processor_vertices[move.to_step][move.to_proc].insert(move.node);

        currentCost += move.change_in_cost;

        stepProcessorWork[move.to_step][move.to_proc] += instance->getComputationalDag().vertex_work_weight(move.node);
        stepProcessorWork[move.from_step][move.from_proc] -= instance->getComputationalDag().vertex_work_weight(move.node);

        UpdateMaxWorkDatastructures(move);
        UpdateViolations(move.node);

        if constexpr (useMemoryConstraint) {
            memoryConstraint.apply_move(move.node, move.from_proc, move.from_step, move.to_proc, move.to_step);
        }
    }

    virtual void InitializeCurrentSchedule(const IBspSchedule<GraphT> &schedule) {
#ifdef KL_DEBUG
        std::cout << "KLCurrentSchedule initialize current schedule" << std::endl;
#endif

        vectorSchedule = VectorSchedule<GraphT>(schedule);
        setSchedule = SetSchedule<GraphT>(schedule);

        InitializeSuperstepDatastructures();

        ComputeWorkMemoryDatastructures(0, NumSteps() - 1);
        RecomputeCurrentViolations();

        costF->ComputeCurrentCosts();
    }

  private:
    void UpdateViolations(VertexType node) {
        newViolations.clear();
        resolvedViolations.clear();

        for (const auto &edge : out_edges(node, instance->getComputationalDag())) {
            const auto &child = target(edge, instance->getComputationalDag());

            if (currentViolations.find(edge) == currentViolations.end()) {
                if (vectorSchedule.assignedSuperstep(node) >= vectorSchedule.assignedSuperstep(child)) {
                    if (vectorSchedule.assignedProcessor(node) != vectorSchedule.assignedProcessor(child)
                        || vectorSchedule.assignedSuperstep(node) > vectorSchedule.assignedSuperstep(child)) {
                        currentViolations.insert(edge);
                        newViolations[child] = edge;
                    }
                }
            } else {
                if (vectorSchedule.assignedSuperstep(node) <= vectorSchedule.assignedSuperstep(child)) {
                    if (vectorSchedule.assignedProcessor(node) == vectorSchedule.assignedProcessor(child)
                        || vectorSchedule.assignedSuperstep(node) < vectorSchedule.assignedSuperstep(child)) {
                        currentViolations.erase(edge);
                        resolvedViolations.insert(edge);
                    }
                }
            }
        }

        for (const auto &edge : in_edges(node, instance->getComputationalDag())) {
            const auto &parent = source(edge, instance->getComputationalDag());

            if (currentViolations.find(edge) == currentViolations.end()) {
                if (vectorSchedule.assignedSuperstep(node) <= vectorSchedule.assignedSuperstep(parent)) {
                    if (vectorSchedule.assignedProcessor(node) != vectorSchedule.assignedProcessor(parent)
                        || vectorSchedule.assignedSuperstep(node) < vectorSchedule.assignedSuperstep(parent)) {
                        currentViolations.insert(edge);
                        newViolations[parent] = edge;
                    }
                }
            } else {
                if (vectorSchedule.assignedSuperstep(node) >= vectorSchedule.assignedSuperstep(parent)) {
                    if (vectorSchedule.assignedProcessor(node) == vectorSchedule.assignedProcessor(parent)
                        || vectorSchedule.assignedSuperstep(node) > vectorSchedule.assignedSuperstep(parent)) {
                        currentViolations.erase(edge);
                        resolvedViolations.insert(edge);
                    }
                }
            }
        }

#ifdef KL_DEBUG

        if (new_violations.size() > 0) {
            std::cout << "New violations: " << std::endl;
            for (const auto &edge : new_violations) {
                std::cout << "Edge: " << source(edge.second, instance->getComputationalDag()) << " -> "
                          << target(edge.second, instance->getComputationalDag()) << std::endl;
            }
        }

        if (resolved_violations.size() > 0) {
            std::cout << "Resolved violations: " << std::endl;
            for (const auto &edge : resolved_violations) {
                std::cout << "Edge: " << source(edge, instance->getComputationalDag()) << " -> "
                          << target(edge, instance->getComputationalDag()) << std::endl;
            }
        }

#endif

        if (currentViolations.size() > 0) {
            currentFeasible = false;
        } else {
            currentFeasible = true;
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
        stepMaxWork[step] = 0;
        stepSecondMaxWork[step] = 0;

        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {
            if (stepProcessorWork[step][proc] > stepMaxWork[step]) {
                stepSecondMaxWork[step] = stepMaxWork[step];
                stepMaxWork[step] = stepProcessorWork[step][proc];

            } else if (stepProcessorWork[step][proc] > stepSecondMaxWork[step]) {
                stepSecondMaxWork[step] = stepProcessorWork[step][proc];
            }
        }
    }
};

template <typename GraphT, typename MemoryConstraintT>
class KlCurrentScheduleMaxComm : public KlCurrentSchedule<GraphT, MemoryConstraintT> {
  public:
    std::vector<std::vector<VCommwT<GraphT>>> stepProcessorSend;
    std::vector<VCommwT<GraphT>> stepMaxSend;
    std::vector<VCommwT<GraphT>> stepMaxReceive;

    std::vector<std::vector<VCommwT<GraphT>>> stepProcessorReceive;
    std::vector<VCommwT<GraphT>> stepSecondMaxSend;
    std::vector<VCommwT<GraphT>> stepSecondMaxReceive;
};

}    // namespace osp
