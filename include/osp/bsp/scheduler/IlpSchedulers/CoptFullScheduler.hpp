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

#include <callbackbase.h>
#include <coptcpp_pch.h>

#include "osp/auxiliary/io/DotFileWriter.hpp"
#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/model/BspScheduleCS.hpp"
#include "osp/bsp/model/BspScheduleRecomp.hpp"
#include "osp/bsp/model/MaxBspSchedule.hpp"
#include "osp/bsp/model/MaxBspScheduleCS.hpp"
#include "osp/bsp/model/util/VectorSchedule.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"

namespace osp {

/**
 * @class CoptFullScheduler
 * @brief A class that represents a scheduler using the COPT solver for computing the schedule of a BSP instance.
 *
 * The `CoptFullScheduler` class is a subclass of the `Scheduler` class and provides an implementation of the
 * `computeSchedule` method using the COPT solver. It uses an ILP (Integer Linear Programming) formulation to find an
 * optimal schedule for a given BSP instance. The scheduler supports various options such as setting an initial
 * solution, setting the maximum number of supersteps, enabling/disabling writing intermediate solutions, and setting
 * communication constraints.
 *
 * The COPT solver is used to solve the ILP formulation and find the optimal schedule. It provides methods to set up the
 * ILP model, define variables and constraints, and solve the ILP problem. The scheduler constructs a `Model` object
 * from the COPT library to represent the ILP model and uses various callbacks to define the objective function,
 * constraints, and solution handling.
 *
 * To compute the schedule, the `computeSchedule` method is called with a `BspInstance` object representing the BSP
 * instance for which the schedule needs to be computed. The method returns a pair containing the return status and the
 * computed `BspSchedule`.
 *
 * The `CoptFullScheduler` class also provides methods to set the initial solution, set the maximum number of
 * supersteps, enable/disable writing intermediate solutions, and get information about the best gap, objective value,
 * and bound found by the solver.
 */
template <typename GraphT>
class CoptFullScheduler : public Scheduler<GraphT> {
    static_assert(IsComputationalDagV<Graph_t>, "CoptFullScheduler can only be used with computational DAGs.");

  private:
    bool allowRecomputation_;
    bool useMemoryConstraint_;
    bool useInitialScheduleRecomp_ = false;
    bool useInitialSchedule_ = false;
    bool writeSolutionsFound_;
    bool isMaxBsp_ = false;

    unsigned timeLimitSeconds_ = 0;

    const BspScheduleCS<GraphT> *initialSchedule_;
    const BspScheduleRecomp<GraphT> *initialScheduleRecomp_;

    std::string writeSolutionsPath_;
    std::string solutionFilePrefix_;

    class WriteSolutionCallback : public CallbackBase {
      private:
        unsigned counter_;
        unsigned maxNumberSolution_;

        double bestObj_;

      public:
        WriteSolutionCallback()
            : counter_(0),
              maxNumberSolution_(500),
              bestObj_(COPT_INFINITY),
              allowRecomputationCb_(false),
              writeSolutionsPathCb_(""),
              solutionFilePrefixCb_(""),
              instancePtr_(),
              nodeToProcessorSuperstepVarPtr_(),
              commProcessorToProcessorSuperstepNodeVarPtr_() {}

        bool allowRecomputationCb_;
        std::string writeSolutionsPathCb_;
        std::string solutionFilePrefixCb_;
        const BspInstance<GraphT> *instancePtr_;

        std::vector<std::vector<VarArray>> *nodeToProcessorSuperstepVarPtr_;
        std::vector<std::vector<std::vector<VarArray>>> *commProcessorToProcessorSuperstepNodeVarPtr_;

        void Callback() override {
            if (Where() == COPT_CBCONTEXT_MIPSOL && counter_ < maxNumberSolution_ && GetIntInfo(COPT_CBINFO_HASINCUMBENT)) {
                try {
                    if (GetDblInfo(COPT_CBINFO_BESTOBJ) < bestObj_ && 0.0 < GetDblInfo(COPT_CBINFO_BESTBND)) {
                        bestObj_ = GetDblInfo(COPT_CBINFO_BESTOBJ);

                        if (allowRecomputationCb_) {
                            auto sched = ConstructBspScheduleRecompFromCallback();
                            DotFileWriter schedWriter;
                            schedWriter.write_schedule_recomp(writeSolutionsPathCb_ + "intmed_sol_" + solutionFilePrefixCb_ + "_"
                                                                  + std::to_string(counter_) + "_schedule.dot",
                                                              sched);

                        } else {
                            BspSchedule<GraphT> sched = ConstructBspScheduleFromCallback();
                            DotFileWriter schedWriter;
                            schedWriter.write_schedule(writeSolutionsPathCb_ + "intmed_sol_" + solutionFilePrefixCb_ + "_"
                                                           + std::to_string(counter_) + "_schedule.dot",
                                                       sched);
                        }
                        counter_++;
                    }

                } catch (const std::exception &e) {}
            }
        }

        BspScheduleCS<GraphT> ConstructBspScheduleFromCallback() {
            BspScheduleCS<GraphT> schedule(*instancePtr_);

            for (const auto &node : instancePtr_->vertices()) {
                for (unsigned int processor = 0; processor < instancePtr_->NumberOfProcessors(); processor++) {
                    for (unsigned step = 0; step < static_cast<unsigned>((*nodeToProcessorSuperstepVarPtr_)[0][0].Size()); step++) {
                        if (GetSolution((*nodeToProcessorSuperstepVarPtr_)[node][processor][static_cast<int>(step)]) >= .99) {
                            schedule.setAssignedProcessor(node, processor);
                            schedule.setAssignedSuperstep(node, step);
                        }
                    }
                }
            }

            for (const auto &node : instancePtr_->vertices()) {
                for (unsigned int pFrom = 0; pFrom < instancePtr_->NumberOfProcessors(); pFrom++) {
                    for (unsigned int pTo = 0; pTo < instancePtr_->NumberOfProcessors(); pTo++) {
                        if (pFrom != pTo) {
                            for (int step = 0; step < (*nodeToProcessorSuperstepVarPtr_)[0][0].Size(); step++) {
                                if (GetSolution((*commProcessorToProcessorSuperstepNodeVarPtr_)[pFrom][pTo][static_cast<unsigned>(
                                        step)][static_cast<int>(node)])
                                    >= .99) {
                                    schedule.addCommunicationScheduleEntry(node, pFrom, pTo, static_cast<unsigned>(step));
                                }
                            }
                        }
                    }
                }
            }

            return schedule;
        }

        BspScheduleRecomp<GraphT> ConstructBspScheduleRecompFromCallback() {
            unsigned numberOfSupersteps = 0;
            BspScheduleRecomp<GraphT> schedule(*instancePtr_);

            for (unsigned int node = 0; node < instancePtr_->NumberOfVertices(); node++) {
                for (unsigned int processor = 0; processor < instancePtr_->NumberOfProcessors(); processor++) {
                    for (unsigned step = 0; step < static_cast<unsigned>((*nodeToProcessorSuperstepVarPtr_)[0][0].Size()); step++) {
                        if (GetSolution((*nodeToProcessorSuperstepVarPtr_)[node][processor][static_cast<int>(step)]) >= .99) {
                            schedule.assignments(node).emplace_back(processor, step);

                            if (step >= numberOfSupersteps) {
                                numberOfSupersteps = step + 1;
                            }
                        }
                    }
                }
            }

            schedule.setNumberOfSupersteps(numberOfSupersteps);

            for (unsigned int node = 0; node < instancePtr_->NumberOfVertices(); node++) {
                for (unsigned int pFrom = 0; pFrom < instancePtr_->NumberOfProcessors(); pFrom++) {
                    for (unsigned int pTo = 0; pTo < instancePtr_->NumberOfProcessors(); pTo++) {
                        if (pFrom != pTo) {
                            for (unsigned step = 0; step < static_cast<unsigned>((*nodeToProcessorSuperstepVarPtr_)[0][0].Size());
                                 step++) {
                                if (GetSolution(
                                        (*commProcessorToProcessorSuperstepNodeVarPtr_)[pFrom][pTo][step][static_cast<int>(node)])
                                    >= .99) {
                                    schedule.addCommunicationScheduleEntry(node, pFrom, pTo, step);
                                }
                            }
                        }
                    }
                }
            }

            return schedule;
        }
    };

    // WriteSolutionCallback solution_callback;

  protected:
    unsigned int maxNumberSupersteps_;

    VarArray superstepUsedVar_;
    std::vector<std::vector<VarArray>> nodeToProcessorSuperstepVar_;
    std::vector<std::vector<std::vector<VarArray>>> commProcessorToProcessorSuperstepNodeVar_;

    VarArray maxCommSuperstepVar_;
    VarArray maxWorkSuperstepVar_;

    void ConstructBspScheduleFromSolution(BspScheduleCS<GraphT> &schedule, bool cleanup = false) {
        const auto &instance = schedule.GetInstance();

        unsigned numberOfSupersteps = 0;

        for (unsigned step = 0; step < maxNumberSupersteps_; step++) {
            if (superstepUsedVar_[static_cast<int>(step)].Get(COPT_DBLINFO_VALUE) >= .99) {
                numberOfSupersteps++;
            }
        }

        for (const auto &node : instance.vertices()) {
            for (unsigned processor = 0; processor < instance.NumberOfProcessors(); processor++) {
                for (unsigned step = 0; step < maxNumberSupersteps_; step++) {
                    if (nodeToProcessorSuperstepVar_[node][processor][static_cast<int>(step)].Get(COPT_DBLINFO_VALUE) >= .99) {
                        schedule.setAssignedProcessor(node, processor);
                        schedule.setAssignedSuperstep(node, step);
                    }
                }
            }
        }

        if (isMaxBsp_ && numberOfSupersteps > 0) {    // can ignore last 2 comm phases in this case
            --numberOfSupersteps;
        }

        schedule.getCommunicationSchedule().clear();
        for (const auto &node : instance.vertices()) {
            for (unsigned int pFrom = 0; pFrom < instance.NumberOfProcessors(); pFrom++) {
                for (unsigned int pTo = 0; pTo < instance.NumberOfProcessors(); pTo++) {
                    if (pFrom != pTo) {
                        for (unsigned int step = 0; step < numberOfSupersteps - 1; step++) {
                            if (commProcessorToProcessorSuperstepNodeVar_[pFrom][pTo][step][static_cast<int>(node)].Get(
                                    COPT_DBLINFO_VALUE)
                                >= .99) {
                                schedule.addCommunicationScheduleEntry(node, pFrom, pTo, step);
                            }
                        }
                    }
                }
            }
        }

        if (cleanup) {
            nodeToProcessorSuperstepVar_.clear();
            commProcessorToProcessorSuperstepNodeVar_.clear();
        }
    }

    void ConstructBspScheduleRecompFromSolution(BspScheduleRecomp<GraphT> &schedule, bool cleanup) {
        unsigned numberOfSupersteps = 0;

        for (unsigned step = 0; step < maxNumberSupersteps_; step++) {
            if (superstepUsedVar_[static_cast<int>(step)].Get(COPT_DBLINFO_VALUE) >= .99) {
                numberOfSupersteps++;
            }
        }

        schedule.setNumberOfSupersteps(numberOfSupersteps);

        for (unsigned node = 0; node < schedule.GetInstance().NumberOfVertices(); node++) {
            for (unsigned processor = 0; processor < schedule.GetInstance().NumberOfProcessors(); processor++) {
                for (unsigned step = 0; step < numberOfSupersteps - 1; step++) {
                    if (nodeToProcessorSuperstepVar_[node][processor][static_cast<int>(step)].Get(COPT_DBLINFO_VALUE) >= .99) {
                        schedule.assignments(node).emplace_back(processor, step);
                    }
                }
            }
        }

        schedule.getCommunicationSchedule().clear();
        for (unsigned int node = 0; node < schedule.GetInstance().NumberOfVertices(); node++) {
            for (unsigned int pFrom = 0; pFrom < schedule.GetInstance().NumberOfProcessors(); pFrom++) {
                for (unsigned int pTo = 0; pTo < schedule.GetInstance().NumberOfProcessors(); pTo++) {
                    if (pFrom != pTo) {
                        for (unsigned int step = 0; step < maxNumberSupersteps_; step++) {
                            if (commProcessorToProcessorSuperstepNodeVar_[pFrom][pTo][step][static_cast<int>(node)].Get(
                                    COPT_DBLINFO_VALUE)
                                >= .99) {
                                schedule.addCommunicationScheduleEntry(node, pFrom, pTo, step);
                            }
                        }
                    }
                }
            }
        }

        if (cleanup) {
            nodeToProcessorSuperstepVar_.clear();
            commProcessorToProcessorSuperstepNodeVar_.clear();
        }
    }

    void LoadInitialSchedule(Model &model, const BspInstance<GraphT> &instance) {
        if (useInitialScheduleRecomp_
            && (maxNumberSupersteps_ < initialScheduleRecomp_->NumberOfSupersteps()
                || instance.NumberOfProcessors() != initialScheduleRecomp_->GetInstance().NumberOfProcessors()
                || instance.NumberOfVertices() != initialScheduleRecomp_->GetInstance().NumberOfVertices())) {
            throw std::invalid_argument("Invalid Argument while computeScheduleRecomp[Recomp]: instance parameters do not "
                                        "agree with those of the initial schedule's instance!");
        }

        if (!useInitialScheduleRecomp_ & useInitialSchedule_
            && (maxNumberSupersteps_ < initialSchedule_->NumberOfSupersteps()
                || instance.NumberOfProcessors() != initialSchedule_->GetInstance().NumberOfProcessors()
                || instance.NumberOfVertices() != initialSchedule_->GetInstance().NumberOfVertices())) {
            throw std::invalid_argument("Invalid Argument while computeScheduleRecomp[Recomp]: instance parameters do not "
                                        "agree with those of the initial schedule's instance!");
        }

        const auto &dag = useInitialScheduleRecomp_ ? initialScheduleRecomp_->GetInstance().GetComputationalDag()
                                                    : initialSchedule_->GetInstance().GetComputationalDag();

        const auto &arch = useInitialScheduleRecomp_ ? initialScheduleRecomp_->GetInstance().GetArchitecture()
                                                     : initialSchedule_->GetInstance().GetArchitecture();

        const unsigned &numProcessors = useInitialScheduleRecomp_ ? initialScheduleRecomp_->GetInstance().NumberOfProcessors()
                                                                  : initialSchedule_->GetInstance().NumberOfProcessors();

        const unsigned &numSupersteps = useInitialScheduleRecomp_ ? initialScheduleRecomp_->NumberOfSupersteps()
                                                                  : initialSchedule_->NumberOfSupersteps();

        const auto &cs = useInitialScheduleRecomp_ ? initialScheduleRecomp_->getCommunicationSchedule()
                                                   : initialSchedule_->getCommunicationSchedule();

        assert(maxNumberSupersteps_ <= static_cast<unsigned>(std::numeric_limits<int>::max()));
        for (unsigned step = 0; step < maxNumberSupersteps_; step++) {
            if (step < numSupersteps) {
                model.SetMipStart(superstepUsedVar_[static_cast<int>(step)], 1);

            } else {
                model.SetMipStart(superstepUsedVar_[static_cast<int>(step)], 0);
            }

            // model.SetMipStart(max_work_superstep_var[step], COPT_INFINITY);
            // model.SetMipStart(max_comm_superstep_var[step], COPT_INFINITY);
        }

        std::vector<std::set<std::pair<unsigned, unsigned>>> computed(dag.NumVertices());
        for (const auto &node : dag.vertices()) {
            if (useInitialScheduleRecomp_) {
                for (const std::pair<unsigned, unsigned> &assignment : initialScheduleRecomp_->assignments(node)) {
                    computed[node].emplace(assignment);
                }
            } else {
                computed[node].emplace(initialSchedule_->assignedProcessor(node), initialSchedule_->assignedSuperstep(node));
            }
        }

        std::vector<std::vector<unsigned>> firstAt(dag.NumVertices(),
                                                   std::vector<unsigned>(numProcessors, std::numeric_limits<unsigned>::max()));
        for (const auto &node : dag.vertices()) {
            if (useInitialScheduleRecomp_) {
                for (const std::pair<unsigned, unsigned> &assignment : initialScheduleRecomp_->assignments(node)) {
                    firstAt[node][assignment.first] = std::min(firstAt[node][assignment.first], assignment.second);
                }
            } else {
                firstAt[node][initialSchedule_->assignedProcessor(node)] = std::min(
                    firstAt[node][initialSchedule_->assignedProcessor(node)], initialSchedule_->assignedSuperstep(node));
            }
        }

        unsigned staleness = isMaxBsp_ ? 2 : 1;
        for (const auto &node : dag.vertices()) {
            for (unsigned p1 = 0; p1 < numProcessors; p1++) {
                for (unsigned step = 0; step < maxNumberSupersteps_; step++) {
                    for (unsigned p2 = 0; p2 < numProcessors; p2++) {
                        if (p1 != p2) {
                            const auto &key = std::make_tuple(node, p1, p2);
                            if (cs.find(key) != cs.end()) {
                                if (cs.at(key) == step) {
                                    model.SetMipStart(
                                        commProcessorToProcessorSuperstepNodeVar_[p1][p2][step][static_cast<int>(node)], 1);
                                    firstAt[node][p2] = std::min(firstAt[node][p2], step + staleness);
                                } else {
                                    model.SetMipStart(
                                        commProcessorToProcessorSuperstepNodeVar_[p1][p2][step][static_cast<int>(node)], 0);
                                }
                            }
                        }
                    }
                }
            }
        }

        for (const auto &node : dag.vertices()) {
            for (unsigned proc = 0; proc < numProcessors; proc++) {
                for (unsigned step = 0; step < maxNumberSupersteps_; step++) {
                    if (step >= firstAt[node][proc]) {
                        model.SetMipStart(commProcessorToProcessorSuperstepNodeVar_[proc][proc][step][static_cast<int>(node)], 1);
                    } else {
                        model.SetMipStart(commProcessorToProcessorSuperstepNodeVar_[proc][proc][step][static_cast<int>(node)], 0);
                    }
                }
            }
        }

        for (const auto &node : dag.vertices()) {
            for (unsigned proc = 0; proc < numProcessors; proc++) {
                for (unsigned step = 0; step < maxNumberSupersteps_; step++) {
                    if (computed[node].find(std::make_pair(proc, step)) != computed[node].end()) {
                        model.SetMipStart(nodeToProcessorSuperstepVar_[node][proc][static_cast<int>(step)], 1);

                    } else {
                        model.SetMipStart(nodeToProcessorSuperstepVar_[node][proc][static_cast<int>(step)], 0);
                    }
                }
            }
        }

        std::vector<std::vector<v_workw_t<Graph_t>>> work(max_number_supersteps,
                                                          std::vector<v_workw_t<Graph_t>>(num_processors, 0));

        if (useInitialScheduleRecomp_) {
            for (const auto &node : initialScheduleRecomp_->GetInstance().vertices()) {
                for (const std::pair<unsigned, unsigned> &assignment : initialScheduleRecomp_->assignments(node)) {
                    work[assignment.second][assignment.first] += dag.VertexWorkWeight(node);
                }
            }
        } else {
            for (const auto &node : initialSchedule_->GetInstance().vertices()) {
                work[initialSchedule_->assignedSuperstep(node)][initialSchedule_->assignedProcessor(node)]
                    += dag.VertexWorkWeight(node);
            }
        }

        std::vector<std::vector<v_commw_t<Graph_t>>> send(max_number_supersteps,
                                                          std::vector<v_commw_t<Graph_t>>(num_processors, 0));

        std::vector<std::vector<v_commw_t<Graph_t>>> rec(max_number_supersteps, std::vector<v_commw_t<Graph_t>>(num_processors, 0));

        for (const auto &[key, val] : cs) {
            send[val][std::get<1>(key)]
                += dag.VertexCommWeight(std::get<0>(key)) * arch.sendCosts(std::get<1>(key), std::get<2>(key));

            rec[val][std::get<2>(key)]
                += dag.VertexCommWeight(std::get<0>(key)) * arch.sendCosts(std::get<1>(key), std::get<2>(key));
        }

        for (unsigned step = 0; step < maxNumberSupersteps_; step++) {
            v_workw_t<Graph_t> maxWork = 0;
            for (unsigned i = 0; i < numProcessors; i++) {
                if (max_work < work[step][i]) {
                    maxWork = work[step][i];
                }
            }

            v_commw_t<Graph_t> maxComm = 0;
            for (unsigned i = 0; i < numProcessors; i++) {
                if (max_comm < send[step][i]) {
                    maxComm = send[step][i];
                }
                if (max_comm < rec[step][i]) {
                    maxComm = rec[step][i];
                }
            }

            model.SetMipStart(maxWorkSuperstepVar_[static_cast<int>(step)], max_work);
            model.SetMipStart(maxCommSuperstepVar_[static_cast<int>(step)], max_comm);
        }

        model.LoadMipStart();
        model.SetIntParam(COPT_INTPARAM_MIPSTARTMODE, 2);
    }

    void SetupVariablesConstraintsObjective(const BspInstance<GraphT> &instance, Model &model) {
        /*
       Variables
       */

        assert(maxNumberSupersteps_ <= static_cast<unsigned>(std::numeric_limits<int>::max()));
        assert(instance.NumberOfProcessors() <= static_cast<unsigned>(std::numeric_limits<int>::max()));

        // variables indicating if superstep is used at all
        superstepUsedVar_ = model.AddVars(static_cast<int>(maxNumberSupersteps_), COPT_BINARY, "superstep_used");

        VarArray superstepHasComm, mergeableSuperstepPenalty;
        if (isMaxBsp_) {
            // variables indicating if there is any communication in superstep
            superstepHasComm = model.AddVars(static_cast<int>(maxNumberSupersteps_), COPT_BINARY, "superstep_has_comm");
            // variables that incentivize the schedule to be continuous - needs to be done differently for maxBsp
            mergeableSuperstepPenalty
                = model.AddVars(static_cast<int>(maxNumberSupersteps_), COPT_BINARY, "mergeable_superstep_penalty");
        }

        // variables for assigments of nodes to processor and superstep
        nodeToProcessorSuperstepVar_ = std::vector<std::vector<VarArray>>(instance.NumberOfVertices(),
                                                                          std::vector<VarArray>(instance.NumberOfProcessors()));

        for (const auto &node : instance.vertices()) {
            for (unsigned int processor = 0; processor < instance.NumberOfProcessors(); processor++) {
                nodeToProcessorSuperstepVar_[node][processor]
                    = model.AddVars(static_cast<int>(maxNumberSupersteps_), COPT_BINARY, "node_to_processor_superstep");
            }
        }

        /*
        Constraints
          */
        if (useMemoryConstraint_) {
            for (unsigned int processor = 0; processor < instance.NumberOfProcessors(); processor++) {
                for (unsigned step = 0; step < maxNumberSupersteps_; step++) {
                    Expr expr;
                    for (const auto &node : instance.vertices()) {
                        expr += nodeToProcessorSuperstepVar_[node][processor][static_cast<int>(step)]
                                * instance.GetComputationalDag().VertexMemWeight(node);
                    }

                    model.AddConstr(expr <= instance.GetArchitecture().memoryBound(processor));
                }
            }
        }

        //  use consecutive supersteps starting from 0
        model.AddConstr(superstepUsedVar_[0] == 1);

        for (unsigned int step = 0; step < maxNumberSupersteps_ - 1; step++) {
            model.AddConstr(superstepUsedVar_[static_cast<int>(step)] >= superstepUsedVar_[static_cast<int>(step + 1)]);
        }

        // superstep is used at all
        for (unsigned int step = 0; step < maxNumberSupersteps_; step++) {
            Expr expr;
            for (const auto &node : instance.vertices()) {
                for (unsigned int processor = 0; processor < instance.NumberOfProcessors(); processor++) {
                    expr += nodeToProcessorSuperstepVar_[node][processor][static_cast<int>(step)];
                }
            }
            model.AddConstr(expr <= static_cast<double>(instance.NumberOfVertices() * instance.NumberOfProcessors())
                                        * superstepUsedVar_[static_cast<int>(step)]);
        }

        // nodes are assigend depending on whether recomputation is allowed or not
        for (const auto &node : instance.vertices()) {
            Expr expr;
            for (unsigned int processor = 0; processor < instance.NumberOfProcessors(); processor++) {
                for (unsigned int step = 0; step < maxNumberSupersteps_; step++) {
                    expr += nodeToProcessorSuperstepVar_[node][processor].GetVar(static_cast<int>(step));
                }
            }

            model.AddConstr(allowRecomputation_ ? expr >= .99 : expr == 1);
        }
        if (allowRecomputation_) {
            std::cout << "setting up constraints with recomputation: " << allowRecomputation_ << std::endl;
        }

        commProcessorToProcessorSuperstepNodeVar_ = std::vector<std::vector<std::vector<VarArray>>>(
            instance.NumberOfProcessors(),
            std::vector<std::vector<VarArray>>(instance.NumberOfProcessors(), std::vector<VarArray>(maxNumberSupersteps_)));

        for (unsigned int p1 = 0; p1 < instance.NumberOfProcessors(); p1++) {
            for (unsigned int p2 = 0; p2 < instance.NumberOfProcessors(); p2++) {
                for (unsigned int step = 0; step < maxNumberSupersteps_; step++) {
                    commProcessorToProcessorSuperstepNodeVar_[p1][p2][step] = model.AddVars(
                        static_cast<int>(instance.NumberOfVertices()), COPT_BINARY, "comm_processor_to_processor_superstep_node");
                }
            }
        }

        // precedence constraint: if task is computed then all of its predecessors must have been present
        for (const auto &node : instance.vertices()) {
            if (instance.GetComputationalDag().in_degree(node) > 0) {
                for (unsigned int step = 0; step < maxNumberSupersteps_; step++) {
                    for (unsigned int processor = 0; processor < instance.NumberOfProcessors(); processor++) {
                        Expr expr;
                        for (const auto &parent : instance.GetComputationalDag().parents(node)) {
                            expr += commProcessorToProcessorSuperstepNodeVar_[processor][processor][step][static_cast<int>(parent)];
                        }

                        model.AddConstr(expr >= static_cast<double>(instance.GetComputationalDag().in_degree(node))
                                                    * nodeToProcessorSuperstepVar_[node][processor][static_cast<int>(step)]);
                    }
                }
            }
        }

        // combines two constraints: node can only be communicated if it is present; and node is present if it was
        // computed or communicated
        for (unsigned int step = 0; step < maxNumberSupersteps_; step++) {
            for (unsigned int processor = 0; processor < instance.NumberOfProcessors(); processor++) {
                for (const auto &node : instance.vertices()) {
                    Expr expr1, expr2;
                    if (step > 0) {
                        for (unsigned int pFrom = 0; pFrom < instance.NumberOfProcessors(); pFrom++) {
                            if (!isMaxBsp_ || pFrom == processor) {
                                expr1
                                    += commProcessorToProcessorSuperstepNodeVar_[pFrom][processor][step - 1][static_cast<int>(node)];
                            } else if (step > 1) {
                                expr1
                                    += commProcessorToProcessorSuperstepNodeVar_[pFrom][processor][step - 2][static_cast<int>(node)];
                            }
                        }
                    }

                    expr1 += nodeToProcessorSuperstepVar_[node][processor][static_cast<int>(step)];

                    for (unsigned int pTo = 0; pTo < instance.NumberOfProcessors(); pTo++) {
                        expr2 += commProcessorToProcessorSuperstepNodeVar_[processor][pTo][step][static_cast<int>(node)];
                    }

                    model.AddConstr(instance.NumberOfProcessors() * (expr1) >= expr2);
                }
            }
        }

        // synchronization cost calculation & forcing continuous schedule in maxBsp
        if (isMaxBsp_) {
            for (unsigned int step = 0; step < maxNumberSupersteps_; step++) {
                Expr expr;
                for (const auto &node : instance.vertices()) {
                    for (unsigned int pFrom = 0; pFrom < instance.NumberOfProcessors(); pFrom++) {
                        for (unsigned int pTo = 0; pTo < instance.NumberOfProcessors(); pTo++) {
                            if (pFrom != pTo) {
                                expr += commProcessorToProcessorSuperstepNodeVar_[pFrom][pTo][step][static_cast<int>(node)];
                            }
                        }
                    }
                }
                model.AddConstr(static_cast<unsigned>(instance.NumberOfProcessors() * instance.NumberOfProcessors()
                                                      * instance.NumberOfVertices())
                                    * superstepHasComm[static_cast<int>(step)]
                                >= expr);
            }

            // if step i and (i+1) has no comm, and (i+2) has work, then (i+1) and (i+2) are mergeable -> penalize
            for (unsigned int step = 0; step < maxNumberSupersteps_ - 2; step++) {
                model.AddConstr(superstepUsedVar_[static_cast<int>(step + 2)] - superstepHasComm[static_cast<int>(step)]
                                    - superstepHasComm[static_cast<int>(step + 1)]
                                <= mergeableSuperstepPenalty[static_cast<int>(step)]);
            }
        }

        maxCommSuperstepVar_ = model.AddVars(static_cast<int>(maxNumberSupersteps_), COPT_INTEGER, "max_comm_superstep");
        // coptModel.AddVars(max_number_supersteps, 0, COPT_INFINITY, 0, COPT_INTEGER, "max_comm_superstep");

        maxWorkSuperstepVar_ = model.AddVars(static_cast<int>(maxNumberSupersteps_), COPT_INTEGER, "max_work_superstep");
        // coptModel.AddVars(max_number_supersteps, 0, COPT_INFINITY, 0, COPT_INTEGER, "max_work_superstep");

        for (unsigned int step = 0; step < maxNumberSupersteps_; step++) {
            for (unsigned int processor = 0; processor < instance.NumberOfProcessors(); processor++) {
                Expr expr;
                for (unsigned int node = 0; node < instance.NumberOfVertices(); node++) {
                    expr += instance.GetComputationalDag().VertexWorkWeight(node)
                            * nodeToProcessorSuperstepVar_[node][processor][static_cast<int>(step)];
                }

                model.AddConstr(maxWorkSuperstepVar_[static_cast<int>(step)] >= expr);
            }
        }

        for (unsigned int step = 0; step < maxNumberSupersteps_; step++) {
            for (unsigned int processor = 0; processor < instance.NumberOfProcessors(); processor++) {
                Expr expr;
                for (const auto &node : instance.vertices()) {
                    for (unsigned int pTo = 0; pTo < instance.NumberOfProcessors(); pTo++) {
                        if (processor != pTo) {
                            expr += instance.GetComputationalDag().VertexCommWeight(node) * instance.sendCosts(processor, pTo)
                                    * commProcessorToProcessorSuperstepNodeVar_[processor][pTo][step][static_cast<int>(node)];
                        }
                    }
                }

                model.AddConstr(maxCommSuperstepVar_[static_cast<int>(step)] >= expr);
            }
        }

        for (unsigned int step = 0; step < maxNumberSupersteps_; step++) {
            for (unsigned int processor = 0; processor < instance.NumberOfProcessors(); processor++) {
                Expr expr;
                for (const auto &node : instance.vertices()) {
                    for (unsigned int pFrom = 0; pFrom < instance.NumberOfProcessors(); pFrom++) {
                        if (processor != pFrom) {
                            expr += instance.GetComputationalDag().VertexCommWeight(node) * instance.sendCosts(pFrom, processor)
                                    * commProcessorToProcessorSuperstepNodeVar_[pFrom][processor][step][static_cast<int>(node)];
                        }
                    }
                }

                model.AddConstr(maxCommSuperstepVar_[static_cast<int>(step)] >= expr);
            }
        }

        // vertex type restrictions
        for (const vertex_idx_t<Graph_t> &node : instance.vertices()) {
            for (unsigned int processor = 0; processor < instance.NumberOfProcessors(); processor++) {
                if (!instance.isCompatible(node, processor)) {
                    for (unsigned int step = 0; step < max_number_supersteps; step++) {
                        model.AddConstr(node_to_processor_superstep_var[node][processor][static_cast<int>(step)] == 0);
                    }
                }
            }
        }

        /*
        Objective function
          */
        Expr expr;

        if (isMaxBsp_) {
            VarArray maxSuperstepVar = model.AddVars(static_cast<int>(maxNumberSupersteps_), COPT_INTEGER, "max_superstep");
            for (unsigned int step = 0; step < maxNumberSupersteps_; step++) {
                model.AddConstr(maxSuperstepVar[static_cast<int>(step)] >= maxWorkSuperstepVar_[static_cast<int>(step)]);
                if (step > 0) {
                    model.AddConstr(maxSuperstepVar[static_cast<int>(step)]
                                    >= instance.CommunicationCosts() * maxCommSuperstepVar_[static_cast<int>(step - 1)]);
                }
                expr += maxSuperstepVar[static_cast<int>(step)];
                expr += instance.SynchronisationCosts() * superstepHasComm[static_cast<int>(step)];
                expr += instance.SynchronisationCosts() * mergeableSuperstepPenalty[static_cast<int>(step)];
            }
        } else {
            for (unsigned int step = 0; step < maxNumberSupersteps_; step++) {
                expr += maxWorkSuperstepVar_[static_cast<int>(step)]
                        + instance.CommunicationCosts() * maxCommSuperstepVar_[static_cast<int>(step)]
                        + instance.SynchronisationCosts() * superstepUsedVar_[static_cast<int>(step)];
            }
            expr -= instance.SynchronisationCosts();
        }

        model.SetObjective(expr, COPT_MINIMIZE);
    }

    RETURN_STATUS RunScheduler(BspScheduleCS<GraphT> &schedule) {
        auto &instance = schedule.GetInstance();
        Envr env;
        Model model = env.CreateModel("bsp_schedule");

        SetupVariablesConstraintsObjective(instance, model);

        if (useInitialSchedule_) {
            LoadInitialSchedule(model, instance);
        }

        ComputeScheduleBase(schedule, model);

        if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_OPTIMAL) {
            ConstructBspScheduleFromSolution(schedule, true);
            return RETURN_STATUS::OSP_SUCCESS;

        } else if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {
            return RETURN_STATUS::ERROR;

        } else {
            if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {
                ConstructBspScheduleFromSolution(schedule, true);
                return RETURN_STATUS::BEST_FOUND;

            } else {
                return RETURN_STATUS::TIMEOUT;
            }
        }
    }

  public:
    CoptFullScheduler(unsigned steps = 5)
        : allowRecomputation_(false),
          useMemoryConstraint_(false),
          useInitialSchedule_(false),
          writeSolutionsFound_(false),
          initialSchedule_(0),
          maxNumberSupersteps_(steps) {
        // solution_callback.comm_processor_to_processor_superstep_node_var_ptr =
        //     &comm_processor_to_processor_superstep_node_var;
        // solution_callback.node_to_processor_superstep_var_ptr = &node_to_processor_superstep_var;
    }

    CoptFullScheduler(const BspScheduleCS<GraphT> &schedule)
        : allowRecomputation_(false),
          useMemoryConstraint_(false),
          useInitialSchedule_(true),
          writeSolutionsFound_(false),
          initialSchedule_(&schedule),
          maxNumberSupersteps_(schedule.NumberOfSupersteps()) {
        // solution_callback.comm_processor_to_processor_superstep_node_var_ptr =
        //     &comm_processor_to_processor_superstep_node_var;
        // solution_callback.node_to_processor_superstep_var_ptr = &node_to_processor_superstep_var;
    }

    CoptFullScheduler(const BspScheduleRecomp<GraphT> &schedule)
        : allowRecomputation_(true),
          useMemoryConstraint_(false),
          useInitialScheduleRecomp_(true),
          writeSolutionsFound_(false),
          initialScheduleRecomp_(&schedule),
          maxNumberSupersteps_(schedule.NumberOfSupersteps()) {}

    virtual ~CoptFullScheduler() = default;

    /**
     * @brief Compute the schedule for the given BspInstance using the COPT solver.
     *
     * @param instance the BspInstance for which to compute the schedule
     *
     * @return a pair containing the return status and the computed BspSchedule
     *
     * @throws std::invalid_argument if the instance parameters do not
     *         agree with those of the initial schedule's instance
     */
    virtual RETURN_STATUS computeSchedule(BspSchedule<GraphT> &schedule) override {
        BspScheduleCS<GraphT> scheduleCs(schedule.GetInstance());
        RETURN_STATUS status = computeScheduleCS(schedule_cs);
        if (status == RETURN_STATUS::OSP_SUCCESS || status == RETURN_STATUS::BEST_FOUND) {
            schedule = std::move(scheduleCs);
            return status;
        } else {
            return status;
        }
    }

    virtual RETURN_STATUS ComputeScheduleWithTimeLimit(BspSchedule<GraphT> &schedule, unsigned timeLimit) {
        timeLimitSeconds_ = timeLimit;
        return computeSchedule(schedule);
    }

    virtual RETURN_STATUS ComputeMaxBspSchedule(MaxBspSchedule<GraphT> &schedule) {
        MaxBspScheduleCS<GraphT> scheduleCs(schedule.GetInstance());
        RETURN_STATUS status = computeMaxBspScheduleCS(schedule_cs);
        if (status == RETURN_STATUS::OSP_SUCCESS || status == RETURN_STATUS::BEST_FOUND) {
            schedule = std::move(scheduleCs);
            return status;
        } else {
            return status;
        }
    }

    virtual RETURN_STATUS ComputeMaxBspScheduleCs(MaxBspScheduleCS<GraphT> &schedule) {
        allowRecomputation_ = false;
        isMaxBsp_ = true;
        return run_scheduler(schedule);
    }

    virtual RETURN_STATUS computeScheduleCS(BspScheduleCS<GraphT> &schedule) override {
        allowRecomputation_ = false;
        isMaxBsp_ = false;
        return run_scheduler(schedule);
    }

    virtual RETURN_STATUS ComputeScheduleRecomp(BspScheduleRecomp<GraphT> &schedule) {
        allowRecomputation_ = true;
        isMaxBsp_ = false;

        Envr env;
        Model model = env.CreateModel("bsp_schedule");

        SetupVariablesConstraintsObjective(schedule.GetInstance(), model);

        if (useInitialSchedule_ || useInitialScheduleRecomp_) {
            LoadInitialSchedule(model, schedule.GetInstance());
        }

        ComputeScheduleBase(schedule, model);

        if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_OPTIMAL) {
            ConstructBspScheduleRecompFromSolution(schedule, true);
            return RETURN_STATUS::OSP_SUCCESS;

        } else if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {
            return RETURN_STATUS::ERROR;

        } else {
            if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {
                ConstructBspScheduleRecompFromSolution(schedule, true);
                return RETURN_STATUS::BEST_FOUND;

            } else {
                return RETURN_STATUS::TIMEOUT;
            }
        }
    };

    virtual void ComputeScheduleBase(const BspScheduleRecomp<GraphT> &schedule, Model &model) {
        if (timeLimitSeconds_ > 0) {
            model.SetDblParam(COPT_DBLPARAM_TIMELIMIT, timeLimitSeconds_);
        }
        model.SetIntParam(COPT_INTPARAM_THREADS, 128);

        model.SetIntParam(COPT_INTPARAM_STRONGBRANCHING, 1);
        model.SetIntParam(COPT_INTPARAM_LPMETHOD, 1);
        model.SetIntParam(COPT_INTPARAM_ROUNDINGHEURLEVEL, 1);

        model.SetIntParam(COPT_INTPARAM_SUBMIPHEURLEVEL, 1);
        // model.SetIntParam(COPT_INTPARAM_PRESOLVE, 1);
        // model.SetIntParam(COPT_INTPARAM_CUTLEVEL, 0);
        model.SetIntParam(COPT_INTPARAM_TREECUTLEVEL, 2);
        // model.SetIntParam(COPT_INTPARAM_DIVINGHEURLEVEL, 2);

        if (writeSolutionsFound_) {
            WriteSolutionCallback solutionCallback;
            solutionCallback.instancePtr_ = &schedule.GetInstance();
            solutionCallback.commProcessorToProcessorSuperstepNodeVarPtr_ = &commProcessorToProcessorSuperstepNodeVar_;
            solutionCallback.nodeToProcessorSuperstepVarPtr_ = &nodeToProcessorSuperstepVar_;
            solutionCallback.solutionFilePrefixCb_ = solutionFilePrefix_;
            solutionCallback.writeSolutionsPathCb_ = writeSolutionsPath_;
            solutionCallback.allowRecomputationCb_ = allowRecomputation_;
            std::cout << "setting up callback with recomputation: " << allowRecomputation_ << std::endl;
            model.SetCallback(&solutionCallback, COPT_CBCONTEXT_MIPSOL);
        }

        model.Solve();
    }

    /**
     * @brief Sets the provided schedule as the initial solution for the ILP.
     *
     * This function sets the provided schedule as the initial solution for the ILP.
     * The maximum number of allowed supersteps is set to the number of supersteps
     * in the provided schedule.
     *
     * @param schedule The provided schedule.
     */
    inline void SetInitialSolutionFromBspSchedule(const BspScheduleCS<GraphT> &schedule) {
        initialSchedule_ = &schedule;

        maxNumberSupersteps_ = schedule.NumberOfSupersteps();

        useInitialSchedule_ = true;
    }

    /**
     * @brief Sets the maximum number of supersteps allowed.
     *
     * This function sets the maximum number of supersteps allowed
     * for the computation of the BSP schedule. If an initial
     * solution is used, the maximum number of supersteps must be
     * greater or equal to the number of supersteps in the initial
     * solution.
     *
     * @param max The maximum number of supersteps allowed.
     *
     * @throws std::invalid_argument If the maximum number of
     *         supersteps is less than the number of supersteps in
     *         the initial solution.
     */
    void SetMaxNumberOfSupersteps(unsigned max) {
        if (useInitialSchedule_ && max < initialSchedule_->NumberOfSupersteps()) {
            throw std::invalid_argument("Invalid Argument while setting "
                                        "max number of supersteps to a value "
                                        "which is less than the number of "
                                        "supersteps of the initial schedule!");
        }
        maxNumberSupersteps_ = max;
    }

    /**
     * @brief Enables writing intermediate solutions.
     *
     * This function enables the writing of intermediate solutions. The
     * `path` parameter specifies the path where the solutions will be
     * written, and the `file_prefix` parameter specifies the prefix
     * that will be used for the solution files.
     *
     * @param path The path where the solutions will be written.
     * @param file_prefix The prefix that will be used for the solution files.
     */
    inline void EnableWriteIntermediateSol(std::string path, std::string filePrefix) {
        writeSolutionsFound_ = true;
        writeSolutionsPath_ = path;
        solutionFilePrefix_ = filePrefix;
    }

    /**
     * Disables writing intermediate solutions.
     *
     * This function disables the writing of intermediate solutions. After
     * calling this function, the `enableWriteIntermediateSol` function needs
     * to be called again in order to enable writing of intermediate solutions.
     */
    inline void DisableWriteIntermediateSol() { writeSolutionsFound_ = false; }

    /**
     * @brief Set the use of memory constraint.
     *
     * This function sets the use of memory constraint. If the memory
     * constraint is enabled, the solver will use a memory constraint
     * to limit total memory of nodes assigend to a processor in a superstep.
     *
     * @param use True if the memory constraint should be used, false otherwise.
     */
    inline void SetUseMemoryConstraint(bool use) { useMemoryConstraint_ = use; }

    /**
     * @brief Get the maximum number of supersteps.
     *
     * @return The maximum number of supersteps.
     */
    inline unsigned GetMaxNumberOfSupersteps() const { return maxNumberSupersteps_; }

    /**
     * @brief Sets the time limit for the ILP solving.
     *
     * @param time_limit_seconds_ The time limit in seconds.
     */
    inline void SetTimeLimitSeconds(unsigned timeLimitSeconds) { timeLimitSeconds_ = timeLimitSeconds; }

    /**
     * @brief Get the name of the schedule.
     *
     * @return The name of the schedule.
     */
    virtual std::string getScheduleName() const override { return "FullIlp"; }
};

}    // namespace osp
