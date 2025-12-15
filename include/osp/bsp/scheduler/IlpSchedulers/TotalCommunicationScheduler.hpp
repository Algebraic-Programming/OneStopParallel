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
#include "osp/bsp/scheduler/LocalSearch/KernighanLin/kl_total_comm.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/graph_algorithms/directed_graph_edge_view.hpp"

namespace osp {

template <typename GraphT>
class TotalCommunicationScheduler : public Scheduler<GraphT> {
  private:
    Envr env_;
    Model model_;

    bool useMemoryConstraint_;
    bool ignoreWorkloadBalance_;

    bool useInitialSchedule_;
    const BspSchedule<GraphT> *initialSchedule_;

    bool writeSolutionsFound_;
    bool useLkHeuristicCallback_;

    class WriteSolutionCallback : public CallbackBase {
      private:
        unsigned counter_;
        unsigned maxNumberSolution_;

        double bestObj_;

      public:
        WriteSolutionCallback()
            : counter_(0),
              maxNumberSolution_(100),
              bestObj_(COPT_INFINITY),
              writeSolutionsPathCb_(""),
              solutionFilePrefixCb_(""),
              instancePtr_(0),
              nodeToProcessorSuperstepVarPtr_() {}

        std::string writeSolutionsPathCb_;
        std::string solutionFilePrefixCb_;
        const BspInstance<GraphT> *instancePtr_;

        std::vector<std::vector<VarArray>> *nodeToProcessorSuperstepVarPtr_;

        void Callback() override {
            if (Where() == COPT_CBCONTEXT_MIPSOL && counter < maxNumberSolution_ && GetIntInfo(COPT_CBINFO_HASINCUMBENT)) {
                try {
                    if (GetDblInfo(COPT_CBINFO_BESTOBJ) < bestObj_ && 0.0 < GetDblInfo(COPT_CBINFO_BESTBND)) {
                        bestObj_ = GetDblInfo(COPT_CBINFO_BESTOBJ);

                        auto sched = ConstructBspScheduleFromCallback();
                        DotFileWriter schedWriter;
                        schedWriter.WriteSchedule(writeSolutionsPathCb_ + "intmed_sol_" + solutionFilePrefixCb_ + "_"
                                                       + std::to_string(counter_) + "_schedule.dot",
                                                   sched);
                        counter_++;
                    }

                } catch (const std::exception &e) {}
            }
        }

        BspSchedule<GraphT> ConstructBspScheduleFromCallback() {
            BspSchedule<GraphT> schedule(*instancePtr_);

            for (const auto &node : instancePtr_->Vertices()) {
                for (unsigned processor = 0; processor < instancePtr_->NumberOfProcessors(); processor++) {
                    for (unsigned step = 0; step < static_cast<unsigned>((*nodeToProcessorSuperstepVarPtr_)[0][0].Size());
                         step++) {
                        assert(size < std::numeric_limits<int>::max());
                        if (GetSolution((*nodeToProcessorSuperstepVarPtr_)[node][processor][static_cast<int>(step)]) >= .99) {
                            schedule.SetAssignedProcessor(node, processor);
                            schedule.SetAssignedSuperstep(node, step);
                        }
                    }
                }
            }

            return schedule;
        }
    };

    class LKHeuristicCallback : public CallbackBase {
      private:
        kl_total_comm<GraphT> lkHeuristic_;

        double bestObj_;

      public:
        LKHeuristicCallback()
            : lk_heuristic(),
              bestObj_(COPT_INFINITY),
              numStep_(0),
              instancePtr_(0),
              maxWorkSuperstepVarPtr_(0),
              superstepUsedVarPtr_(0),
              nodeToProcessorSuperstepVarPtr_(0),
              edgeVarsPtr_(0) {}

        unsigned numStep_;
        const BspInstance<GraphT> *instancePtr_;

        VarArray *maxWorkSuperstepVarPtr_;
        VarArray *superstepUsedVarPtr_;
        std::vector<std::vector<VarArray>> *nodeToProcessorSuperstepVarPtr_;
        std::vector<std::vector<VarArray>> *edgeVarsPtr_;

        void Callback() override {
            if (Where() == COPT_CBCONTEXT_MIPSOL && GetIntInfo(COPT_CBINFO_HASINCUMBENT)) {
                try {
                    if (0.0 < GetDblInfo(COPT_CBINFO_BESTBND) && 1.0 < GetDblInfo(COPT_CBINFO_BESTOBJ) &&
                        // GetDblInfo(COPT_CBINFO_BESTOBJ) < bestObj_ &&
                        0.1 < (GetDblInfo(COPT_CBINFO_BESTOBJ) - GetDblInfo(COPT_CBINFO_BESTBND))
                                  / GetDblInfo(COPT_CBINFO_BESTOBJ)) {
                        // bestObj_ = GetDblInfo(COPT_CBINFO_BESTOBJ);

                        auto sched = ConstructBspScheduleFromCallback();

                        if (sched.NumberOfSupersteps() > 2) {
                            auto status = lk_heuristic.ImproveSchedule(sched);

                            if (status == ReturnStatus::OSP_SUCCESS) {
                                FeedImprovedSchedule(sched);
                            }
                        }
                    }

                } catch (const std::exception &e) {}
            }
        }

        BspSchedule<GraphT> ConstructBspScheduleFromCallback() {
            BspSchedule schedule(*instancePtr_);

            for (const auto &node : instancePtr_->Vertices()) {
                for (unsigned processor = 0; processor < instancePtr_->NumberOfProcessors(); processor++) {
                    for (unsigned step = 0; step < static_cast<unsigned>((*nodeToProcessorSuperstepVarPtr_)[0][0].Size());
                         step++) {
                        assert(step <= std::numeric_limits<int>::max());
                        if (GetSolution((*nodeToProcessorSuperstepVarPtr_)[node][processor][static_cast<int>(step)]) >= .99) {
                            schedule.SetAssignedProcessor(node, processor);
                            schedule.SetAssignedSuperstep(node, step);
                        }
                    }
                }
            }

            return schedule;
        };

        void FeedImprovedSchedule(const BspSchedule<GraphT> &schedule) {
            for (unsigned step = 0; step < numStep_; step++) {
                if (step < schedule.NumberOfSupersteps()) {
                    assert(step <= std::numeric_limits<int>::max());
                    SetSolution((*superstepUsedVarPtr_)[static_cast<int>(step)], 1.0);
                } else {
                    assert(step <= std::numeric_limits<int>::max());
                    SetSolution((*superstepUsedVarPtr_)[static_cast<int>(step)], 0.0);
                }
            }

            for (const auto &node : instancePtr_->Vertices()) {
                for (unsigned processor = 0; processor < instancePtr_->NumberOfProcessors(); processor++) {
                    for (unsigned step = 0; step < static_cast<unsigned>((*nodeToProcessorSuperstepVarPtr_)[0][0].Size());
                         step++) {
                        if (schedule.AssignedProcessor(node) == processor && schedule.AssignedSuperstep(node) == step) {
                            assert(step <= std::numeric_limits<int>::max());
                            SetSolution((*nodeToProcessorSuperstepVarPtr_)[node][processor][static_cast<int>(step)], 1.0);
                        } else {
                            assert(step <= std::numeric_limits<int>::max());
                            SetSolution((*nodeToProcessorSuperstepVarPtr_)[node][processor][static_cast<int>(step)], 0.0);
                        }
                    }
                }
            }

            std::vector<std::vector<VWorkwT<GraphT>>> work(num_step,
                                                           std::vector<VWorkwT<GraphT>>(instance_ptr->NumberOfProcessors(), 0));

            for (const auto &node : instancePtr_->Vertices()) {
                work[schedule.AssignedSuperstep(node)][schedule.AssignedProcessor(node)]
                    += instancePtr_->GetComputationalDag().VertexWorkWeight(node);
            }

            for (unsigned step = 0; step < numStep_; step++) {
                VWorkwT<GraphT> maxWork = 0;
                for (unsigned proc = 0; proc < instancePtr_->NumberOfProcessors(); proc++) {
                    if (maxWork < work[step][proc]) {
                        maxWork = work[step][proc];
                    }
                }

                assert(step <= std::numeric_limits<int>::max());
                SetSolution((*maxWorkSuperstepVarPtr_)[static_cast<int>(step)], maxWork);
            }

            if (instancePtr_->GetArchitecture().IsNumaArchitecture()) {
                for (unsigned p1 = 0; p1 < instancePtr_->NumberOfProcessors(); p1++) {
                    for (unsigned p2 = 0; p2 < instancePtr_->NumberOfProcessors(); p2++) {
                        if (p1 != p2) {
                            int edgeId = 0;
                            for (const auto &ep : EdgeView(instancePtr_->GetComputationalDag())) {
                                if (schedule.AssignedProcessor(ep.source) == p1 && schedule.AssignedProcessor(ep.target) == p2) {
                                    SetSolution((*edgeVarsPtr_)[p1][p2][edgeId], 1.0);
                                } else {
                                    SetSolution((*edgeVarsPtr_)[p1][p2][edgeId], 0.0);
                                }

                                edgeId++;
                            }
                        }
                    }
                }

            } else {
                int edgeId = 0;
                for (const auto &ep : EdgeView(instancePtr_->GetComputationalDag())) {
                    if (schedule.AssignedProcessor(ep.source) != schedule.AssignedProcessor(ep.target)) {
                        SetSolution((*edgeVarsPtr_)[0][0][edgeId], 1.0);
                    } else {
                        SetSolution((*edgeVarsPtr_)[0][0][edgeId], 0.0);
                    }

                    edgeId++;
                }
            }

            LoadSolution();
        }
    };

    WriteSolutionCallback solutionCallback_;
    LKHeuristicCallback heuristicCallback_;

  protected:
    unsigned int maxNumberSupersteps_;

    unsigned timeLimitSeconds_;

    VarArray superstepUsedVar_;
    std::vector<std::vector<VarArray>> nodeToProcessorSuperstepVar_;
    std::vector<std::vector<VarArray>> edgeVars_;
    VarArray maxWorkSuperstepVar_;

    void ConstructBspScheduleFromSolution(BspSchedule<GraphT> &schedule, bool cleanup = false) {
        const auto &instance = schedule.GetInstance();

        for (const auto &node : instance.Vertices()) {
            for (unsigned processor = 0; processor < instance.NumberOfProcessors(); processor++) {
                for (unsigned step = 0; step < maxNumberSupersteps_; step++) {
                    if (nodeToProcessorSuperstepVar_[node][processor][step].Get(COPT_DBLINFO_VALUE) >= .99) {
                        schedule.SetAssignedProcessor(node, processor);
                        schedule.SetAssignedSuperstep(node, step);
                    }
                }
            }
        }

        if (cleanup) {
            nodeToProcessorSuperstepVar_.clear();
        }
    }

    void LoadInitialSchedule() {
        for (unsigned step = 0; step < maxNumberSupersteps_; step++) {
            if (step < initialSchedule_->NumberOfSupersteps()) {
                assert(step <= std::numeric_limits<int>::max());
                model.SetMipStart(superstepUsedVar_[static_cast<int>(step)], 1);

            } else {
                assert(step <= std::numeric_limits<int>::max());
                model.SetMipStart(superstepUsedVar_[static_cast<int>(step)], 0);
            }
        }

        for (const auto &node : initialSchedule_->GetInstance().Vertices()) {
            for (unsigned proc = 0; proc < initialSchedule_->GetInstance().NumberOfProcessors(); proc++) {
                for (unsigned step = 0; step < maxNumberSupersteps_; step++) {
                    if (proc == initialSchedule_->AssignedProcessor(node) && step == initialSchedule_->AssignedSuperstep(node)) {
                        assert(step <= std::numeric_limits<int>::max());
                        model.SetMipStart(nodeToProcessorSuperstepVar_[node][proc][static_cast<int>(step)], 1);

                    } else {
                        assert(step <= std::numeric_limits<int>::max());
                        model.SetMipStart(nodeToProcessorSuperstepVar_[node][proc][static_cast<int>(step)], 0);
                    }
                }
            }
        }

        std::vector<std::vector<VWorkwT<GraphT>>> work(
            maxNumberSupersteps_, std::vector<VWorkwT<GraphT>>(initial_schedule->GetInstance().NumberOfProcessors(), 0));

        for (const auto &node : initialSchedule_->GetInstance().Vertices()) {
            work[initialSchedule_->AssignedSuperstep(node)][initialSchedule_->AssignedProcessor(node)]
                += initialSchedule_->GetInstance().GetComputationalDag().VertexWorkWeight(node);
        }

        for (unsigned step = 0; step < maxNumberSupersteps_; step++) {
            VWorkwT<GraphT> maxWork = 0;
            for (unsigned i = 0; i < initialSchedule_->GetInstance().NumberOfProcessors(); i++) {
                if (maxWork < work[step][i]) {
                    maxWork = work[step][i];
                }
            }

            assert(step <= std::numeric_limits<int>::max());
            model.SetMipStart(maxWorkSuperstepVar_[static_cast<int>(step)], maxWork);
        }

        model.LoadMipStart();
        model.SetIntParam(COPT_INTPARAM_MIPSTARTMODE, 2);
    }

    void SetupVariablesConstraintsObjective(const BspInstance<GraphT> &instance) {
        /*
        Variables
        */

        // variables indicating if superstep is used at all
        superstepUsedVar_ = model.AddVars(static_cast<int>(maxNumberSupersteps_), COPT_BINARY, "superstep_used");

        nodeToProcessorSuperstepVar_ = std::vector<std::vector<VarArray>>(
            instance.NumberOfVertices(), std::vector<VarArray>(instance.NumberOfProcessors()));
        assert(maxNumberSupersteps_ <= std::numeric_limits<int>::max());
        // variables for assigments of nodes to processor and superstep
        for (const auto &node : instance.Vertices()) {
            for (unsigned int processor = 0; processor < instance.NumberOfProcessors(); processor++) {
                nodeToProcessorSuperstepVar_[node][processor]
                    = model.AddVars(static_cast<int>(maxNumberSupersteps_), COPT_BINARY, "node_to_processor_superstep");
            }
        }

        /*
        Constraints
          */

        /*
        Constraints
          */
        if (useMemoryConstraint_) {
            for (unsigned int processor = 0; processor < instance.NumberOfProcessors(); processor++) {
                for (unsigned step = 0; step < maxNumberSupersteps_; step++) {
                    Expr expr;
                    for (unsigned int node = 0; node < instance.NumberOfVertices(); node++) {
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
            for (const auto &node : instance.Vertices()) {
                for (unsigned int processor = 0; processor < instance.NumberOfProcessors(); processor++) {
                    expr += nodeToProcessorSuperstepVar_[node][processor][static_cast<int>(step)];
                }
            }
            model.AddConstr(expr <= static_cast<double>(instance.NumberOfVertices() * instance.NumberOfProcessors())
                                        * superstepUsedVar_.GetVar(static_cast<int>(step)));
        }

        // nodes are assigend depending on whether recomputation is allowed or not
        for (const auto &node : instance.Vertices()) {
            Expr expr;
            for (unsigned int processor = 0; processor < instance.NumberOfProcessors(); processor++) {
                assert(maxNumberSupersteps_ <= std::numeric_limits<int>::max());
                for (unsigned int step = 0; step < maxNumberSupersteps_; step++) {
                    expr += nodeToProcessorSuperstepVar_[node][processor].GetVar(static_cast<int>(step));
                }
            }

            model.AddConstr(expr == 1);
            // model.AddConstr(instance.allowRecomputation() ? expr >= .99 : expr == 1);
        }

        for (const auto &node : instance.Vertices()) {
            for (unsigned processor = 0; processor < instance.NumberOfProcessors(); processor++) {
                assert(maxNumberSupersteps_ <= std::numeric_limits<int>::max());
                for (unsigned step = 0; step < maxNumberSupersteps_; step++) {
                    for (const auto &source : instance.GetComputationalDag().Parents(node)) {
                        Expr expr1;

                        for (unsigned p2 = 0; p2 < instance.NumberOfProcessors(); p2++) {
                            for (unsigned stepPrime = 0; stepPrime < step; stepPrime++) {
                                expr1 += nodeToProcessorSuperstepVar_[source][p2][static_cast<int>(stepPrime)];
                            }
                        }

                        expr1 += nodeToProcessorSuperstepVar_[source][processor][static_cast<int>(step)];

                        model.AddConstr(nodeToProcessorSuperstepVar_[node][processor][static_cast<int>(step)] <= expr1);
                    }
                }
            }
        }

        Expr totalEdgesCut;

        if (instance.GetArchitecture().IsNumaArchitecture()) {
            edgeVars_ = std::vector<std::vector<VarArray>>(instance.NumberOfProcessors(),
                                                           std::vector<VarArray>(instance.NumberOfProcessors()));

            for (unsigned int p1 = 0; p1 < instance.NumberOfProcessors(); p1++) {
                for (unsigned int p2 = 0; p2 < instance.NumberOfProcessors(); p2++) {
                    if (p1 != p2) {
                        assert(instance.GetComputationalDag().NumEdges() <= std::numeric_limits<int>::max());
                        edgeVars_[p1][p2]
                            = model.AddVars(static_cast<int>(instance.GetComputationalDag().NumEdges()), COPT_BINARY, "edge");

                        int edgeId = 0;
                        for (const auto &ep : EdgeView(instance.GetComputationalDag())) {
                            Expr expr1, expr2;
                            assert(maxNumberSupersteps_ <= std::numeric_limits<int>::max());
                            for (unsigned step = 0; step < maxNumberSupersteps_; step++) {
                                expr1 += nodeToProcessorSuperstepVar_[ep.source][p1][static_cast<int>(step)];
                                expr2 += nodeToProcessorSuperstepVar_[ep.target][p2][static_cast<int>(step)];
                            }
                            model.AddConstr(edgeVars_[p1][p2][edgeId] >= expr1 + expr2 - 1.001);

                            totalEdgesCut += edgeVars_[p1][p2][edgeId]
                                               * instance.GetComputationalDag().VertexCommWeight(ep.source)
                                               * instance.SendCosts(p1, p2);

                            edgeId++;
                        }
                    }
                }
            }

        } else {
            edgeVars_ = std::vector<std::vector<VarArray>>(1, std::vector<VarArray>(1));
            assert(instance.GetComputationalDag().NumEdges() <= std::numeric_limits<int>::max());
            edgeVars_[0][0] = model.AddVars(static_cast<int>(instance.GetComputationalDag().NumEdges()), COPT_BINARY, "edge");

            int edgeId = 0;
            for (const auto &ep : EdgeView(instance.GetComputationalDag())) {
                for (unsigned p1 = 0; p1 < instance.NumberOfProcessors(); p1++) {
                    Expr expr1, expr2;
                    for (unsigned step = 0; step < maxNumberSupersteps_; step++) {
                        expr1 += nodeToProcessorSuperstepVar_[ep.source][p1][static_cast<int>(step)];
                    }

                    for (unsigned p2 = 0; p2 < instance.NumberOfProcessors(); p2++) {
                        if (p1 != p2) {
                            for (unsigned step = 0; step < maxNumberSupersteps_; step++) {
                                expr2 += nodeToProcessorSuperstepVar_[ep.target][p2][static_cast<int>(step)];
                            }
                        }
                    }
                    model.AddConstr(edgeVars_[0][0][edgeId] >= expr1 + expr2 - 1.001);
                }

                totalEdgesCut += instance.GetComputationalDag().VertexCommWeight(ep.source) * edgeVars_[0][0][edgeId];

                edgeId++;
            }
        }

        Expr expr;

        if (ignoreWorkloadBalance_) {
            for (unsigned step = 0; step < maxNumberSupersteps_; step++) {
                assert(step <= std::numeric_limits<int>::max());
                expr += instance.SynchronisationCosts() * superstepUsedVar_[static_cast<int>(step)];
            }

        } else {
            assert(maxNumberSupersteps_ <= std::numeric_limits<int>::max());
            maxWorkSuperstepVar_ = model.AddVars(static_cast<int>(maxNumberSupersteps_), COPT_CONTINUOUS, "max_work_superstep");
            // coptModel.AddVars(maxNumberSupersteps_, 0, COPT_INFINITY, 0, COPT_INTEGER, "max_work_superstep");

            for (unsigned int step = 0; step < maxNumberSupersteps_; step++) {
                assert(step <= std::numeric_limits<int>::max());
                for (unsigned int processor = 0; processor < instance.NumberOfProcessors(); processor++) {
                    Expr exprWork;
                    for (const auto &node : instance.Vertices()) {
                        expr_work += instance.GetComputationalDag().VertexWorkWeight(node)
                                     * nodeToProcessorSuperstepVar_[node][processor][static_cast<int>(step)];
                    }

                    model.AddConstr(maxWorkSuperstepVar_[static_cast<int>(step)] >= expr_work);
                }
            }

            for (unsigned step = 0; step < maxNumberSupersteps_; step++) {
                assert(step <= std::numeric_limits<int>::max());
                expr += maxWorkSuperstepVar_[static_cast<int>(step)]
                        + instance.SynchronisationCosts() * superstepUsedVar_[static_cast<int>(step)];
            }
        }

        /*
        Objective function
          */

        double commCost = static_cast<double>(instance.CommunicationCosts()) / instance.NumberOfProcessors();
        model.SetObjective(comm_cost * totalEdgesCut + expr - instance.SynchronisationCosts(), COPT_MINIMIZE);
    }

  public:
    TotalCommunicationScheduler(unsigned steps = 5)
        : Scheduler<GraphT>(),
          env(),
          model(env.CreateModel("TotalCommScheduler")),
          useMemoryConstraint_(false),
          ignoreWorkloadBalance_(false),
          useInitialSchedule_(false),
          initialSchedule_(0),
          writeSolutionsFound_(false),
          useLkHeuristicCallback_(true),
          solutionCallback_(),
          heuristicCallback_(),
          maxNumberSupersteps_(steps) {
        heuristicCallback_.maxWorkSuperstepVarPtr_ = &maxWorkSuperstepVar_;
        heuristicCallback_.superstepUsedVarPtr_ = &superstepUsedVar_;
        heuristicCallback_.nodeToProcessorSuperstepVarPtr_ = &nodeToProcessorSuperstepVar_;
        heuristicCallback_.edgeVarsPtr_ = &edgeVars_;

        solution_callback.nodeToProcessorSuperstepVarPtr_ = &nodeToProcessorSuperstepVar_;
    }

    TotalCommunicationScheduler(const BspSchedule<GraphT> &schedule)
        : Scheduler<GraphT>(),
          env(),
          model(env.CreateModel("TotalCommScheduler")),
          useMemoryConstraint_(false),
          ignoreWorkloadBalance_(false),
          useInitialSchedule_(true),
          initialSchedule_(&schedule),
          writeSolutionsFound_(false),
          useLkHeuristicCallback_(true),
          solutionCallback_(),
          heuristicCallback_(),
          maxNumberSupersteps_(schedule.NumberOfSupersteps()) {
        heuristicCallback_.maxWorkSuperstepVarPtr_ = &maxWorkSuperstepVar_;
        heuristicCallback_.superstepUsedVarPtr_ = &superstepUsedVar_;
        heuristicCallback_.nodeToProcessorSuperstepVarPtr_ = &nodeToProcessorSuperstepVar_;
        heuristicCallback_.edgeVarsPtr_ = &edgeVars_;

        solution_callback.nodeToProcessorSuperstepVarPtr_ = &nodeToProcessorSuperstepVar_;
    }

    virtual ~TotalCommunicationScheduler() = default;

    virtual ReturnStatus ComputeScheduleWithTimeLimit(BspSchedule<GraphT> &schedule, unsigned timeout) {
        model.SetDblParam(COPT_DBLPARAM_TIMELIMIT, timeout);
        return ComputeSchedule(schedule);
    }

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
    virtual ReturnStatus ComputeSchedule(BspSchedule<GraphT> &schedule) override {
        auto &instance = schedule.GetInstance();

        assert(!ignoreWorkloadBalance_ || !useLkHeuristicCallback_);

        if (useInitialSchedule_
            && (maxNumberSupersteps_ < initialSchedule_->NumberOfSupersteps()
                || instance.NumberOfProcessors() != initialSchedule_->GetInstance().NumberOfProcessors()
                || instance.NumberOfVertices() != initialSchedule_->GetInstance().NumberOfVertices())) {
            throw std::invalid_argument("Invalid Argument while ComputeSchedule(instance): instance parameters do not "
                                        "agree with those of the initial schedule's instance!");
        }

        SetupVariablesConstraintsObjective(instance);

        if (useInitialSchedule_) {
            LoadInitialSchedule();
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
            solutionCallback_.instancePtr_ = &instance;
            model.SetCallback(&solution_callback, COPT_CBCONTEXT_MIPSOL);
        }
        if (useLkHeuristicCallback_) {
            heuristicCallback_.instancePtr_ = &instance;
            heuristicCallback_.numStep_ = maxNumberSupersteps_;
            model.SetCallback(&heuristicCallback_, COPT_CBCONTEXT_MIPSOL);
        }

        model.Solve();

        if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_OPTIMAL) {
            return ReturnStatus::OSP_SUCCESS;    //, constructBspScheduleFromSolution(instance, true)};

        } else if (model.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {
            return ReturnStatus::ERROR;

        } else {
            if (model.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {
                return ReturnStatus::BEST_FOUND;    //, constructBspScheduleFromSolution(instance, true)};

            } else {
                return ReturnStatus::TIMEOUT;
            }
        }
    };

    /**
     * @brief Sets the provided schedule as the initial solution for the ILP.
     *
     * This function sets the provided schedule as the initial solution for the ILP.
     * The maximum number of allowed supersteps is set to the number of supersteps
     * in the provided schedule.
     *
     * @param schedule The provided schedule.
     */
    inline void SetInitialSolutionFromBspSchedule(const BspSchedule<GraphT> &schedule) {
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
        solutionCallback_.writeSolutionsPathCb_ = path;
        solutionCallback_.solutionFilePrefixCb_ = filePrefix;
    }

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
     * @brief Set the use of workload balance constraint.
     *
     * This function sets the use of workload balance constraint. If the
     * workload balance constraint is enabled, the solver will use a workload
     * balance constraint to limit the difference of total work of nodes
     * assigned to a processor in a superstep.
     *
     * @param use True if the workload balance constraint should be used, false otherwise.
     */
    inline void SetIgnoreWorkloadBalance(bool use) { ignoreWorkloadBalance_ = use; }

    /**
     * @brief Set the use of LK heuristic callback.
     *
     * This function sets the use of LK heuristic callback. If the LK heuristic
     * callback is enabled, the solver will use the LK heuristic on any feasible solution
     * that is found to improve it.
     *
     *
     * @param use True if the LK heuristic callback should be used, false otherwise.
     */
    inline void SetUseLkHeuristicCallback(bool use) { useLkHeuristicCallback_ = use; }

    /**
     * Disables writing intermediate solutions.
     *
     * This function disables the writing of intermediate solutions. After
     * calling this function, the `enableWriteIntermediateSol` function needs
     * to be called again in order to enable writing of intermediate solutions.
     */
    inline void DisableWriteIntermediateSol() { writeSolutionsFound_ = false; }

    /**
     * @brief Get the maximum number of supersteps.
     *
     * @return The maximum number of supersteps.
     */
    inline unsigned GetMaxNumberOfSupersteps() const { return maxNumberSupersteps_; }

    /**
     * @brief Get the best gap found by the solver.
     *
     * @return The best gap found by the solver.
     */
    inline double BestGap() { return model.GetDblAttr(COPT_DBLATTR_BESTGAP); }

    /**
     * @brief Get the best objective value found by the solver.
     *
     * @return The best objective value found by the solver.
     */
    inline double BestObjective() { return model.GetDblAttr(COPT_DBLATTR_BESTOBJ); }

    /**
     * @brief Get the best bound found by the solver.
     *
     * @return The best bound found by the solver.
     */
    inline double BestBound() { return model.GetDblAttr(COPT_DBLATTR_BESTBND); }

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
    virtual std::string GetScheduleName() const override { return "TotalCommIlp"; }
};

}    // namespace osp
