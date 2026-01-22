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

#include "osp/bsp/model/BspInstance.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/pebbling/pebblers/pebblingILP/COPTEnv.hpp"

namespace osp {

template <typename GraphT>
class AcyclicPartitioningILP {
    static_assert(isComputationalDagV<GraphT>, "PebblingSchedule can only be used with computational DAGs.");

  private:
    using VertexIdx = VertexIdxT<GraphT>;
    using commweight_type = VCommwT<GraphT>;

    Model model_;

    bool writeSolutionsFound_;
    bool ignoreSourcesForConstraint_ = true;

    class WriteSolutionCallback : public CallbackBase {
      private:
        unsigned counter_;
        unsigned maxNumberSolution_;

        double bestObj_;

      public:
        WriteSolutionCallback()
            : counter_(0), maxNumberSolution_(500), bestObj_(COPT_INFINITY), writeSolutionsPathCb_(""), solutionFilePrefixCb_("") {}

        std::string writeSolutionsPathCb_;
        std::string solutionFilePrefixCb_;

        void callback() override;
    };

    WriteSolutionCallback solutionCallback_;

    unsigned numberOfParts_ = 0;

    std::vector<bool> isOriginalSource_;

    unsigned timeLimitSeconds_;

  protected:
    std::vector<VarArray> nodeInPartition_;
    std::vector<VarArray> hyperedgeIntersectsPartition_;

    unsigned minPartitionSize_ = 500, maxPartitionSize_ = 1400;

    std::vector<unsigned> ReturnAssignment(const BspInstance<GraphT> &instance);

    void SetupVariablesConstraintsObjective(const BspInstance<GraphT> &instance);

    void SolveIlp();

  public:
    AcyclicPartitioningILP() : model_(COPTEnv::GetInstance().CreateModel("AsyncPart")), writeSolutionsFound_(false) {}

    virtual ~AcyclicPartitioningILP() = default;

    ReturnStatus ComputePartitioning(const BspInstance<GraphT> &instance, std::vector<unsigned> &partitioning);

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
     * Disables writing intermediate solutions.
     *
     * This function disables the writing of intermediate solutions. After
     * calling this function, the `enableWriteIntermediateSol` function needs
     * to be called again in order to enable writing of intermediate solutions.
     */
    inline void DisableWriteIntermediateSol() { writeSolutionsFound_ = false; }

    /**
     * @brief Get the best gap found by the solver.
     *
     * @return The best gap found by the solver.
     */
    inline double BestGap() { return model_.GetDblAttr(COPT_DBLATTR_BESTGAP); }

    /**
     * @brief Get the best objective value found by the solver.
     *
     * @return The best objective value found by the solver.
     */
    inline double BestObjective() { return model_.GetDblAttr(COPT_DBLATTR_BESTOBJ); }

    /**
     * @brief Get the best bound found by the solver.
     *
     * @return The best bound found by the solver.
     */
    inline double BestBound() { return model_.GetDblAttr(COPT_DBLATTR_BESTBND); }

    /**
     * @brief Get the name of the schedule.
     *
     * @return The name of the schedule.
     */
    virtual std::string GetScheduleName() const { return "AcyclicPartitioningILP"; }

    // getters and setters for problem parameters
    inline std::pair<unsigned, unsigned> GetMinAndMaxSize() const { return std::make_pair(minPartitionSize_, maxPartitionSize_); }

    inline void SetMinAndMaxSize(const std::pair<unsigned, unsigned> minAndMax) {
        minPartitionSize_ = minAndMax.first;
        maxPartitionSize_ = minAndMax.second;
    }

    inline unsigned GetNumberOfParts() const { return numberOfParts_; }

    inline void SetNumberOfParts(const unsigned numberOfParts) { numberOfParts_ = numberOfParts; }

    inline void SetIgnoreSourceForConstraint(const bool ignore) { ignoreSourcesForConstraint_ = ignore; }

    inline void SetIsOriginalSource(const std::vector<bool> &isOriginalSource) { isOriginalSource_ = isOriginalSource; }

    void SetTimeLimitSeconds(unsigned timeLimitSeconds) { timeLimitSeconds_ = timeLimitSeconds; }
};

template <typename GraphT>
void AcyclicPartitioningILP<GraphT>::SolveIlp() {
    // model_.SetIntParam(COPT_INTPARAM_LOGTOCONSOLE, 0);

    model_.SetDblParam(COPT_DBLPARAM_TIMELIMIT, timeLimitSeconds_);
    model_.SetIntParam(COPT_INTPARAM_THREADS, 128);

    model_.SetIntParam(COPT_INTPARAM_STRONGBRANCHING, 1);
    model_.SetIntParam(COPT_INTPARAM_LPMETHOD, 1);
    model_.SetIntParam(COPT_INTPARAM_ROUNDINGHEURLEVEL, 1);

    model_.SetIntParam(COPT_INTPARAM_SUBMIPHEURLEVEL, 1);
    // model.SetIntParam(COPT_INTPARAM_PRESOLVE, 1);
    // model.SetIntParam(COPT_INTPARAM_CUTLEVEL, 0);
    model_.SetIntParam(COPT_INTPARAM_TREECUTLEVEL, 2);
    // model.SetIntParam(COPT_INTPARAM_DIVINGHEURLEVEL, 2);

    model_.Solve();
}

template <typename GraphT>
ReturnStatus AcyclicPartitioningILP<GraphT>::ComputePartitioning(const BspInstance<GraphT> &instance,
                                                                 std::vector<unsigned> &partitioning) {
    partitioning.clear();

    if (numberOfParts_ == 0) {
        numberOfParts_ = static_cast<unsigned>(
            std::floor(static_cast<double>(instance.NumberOfVertices()) / static_cast<double>(minPartitionSize_)));
        std::cout << "ILP nr parts: " << numberOfParts_ << std::endl;
    }

    SetupVariablesConstraintsObjective(instance);

    SolveIlp();

    if (model_.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_OPTIMAL) {
        partitioning = ReturnAssignment(instance);
        return ReturnStatus::OSP_SUCCESS;

    } else if (model_.GetIntAttr(COPT_INTATTR_MIPSTATUS) == COPT_MIPSTATUS_INF_OR_UNB) {
        partitioning.resize(instance.NumberOfVertices(), UINT_MAX);
        return ReturnStatus::ERROR;

    } else {
        if (model_.GetIntAttr(COPT_INTATTR_HASMIPSOL)) {
            partitioning = ReturnAssignment(instance);
            return ReturnStatus::OSP_SUCCESS;

        } else {
            partitioning.resize(instance.NumberOfVertices(), UINT_MAX);
            return ReturnStatus::ERROR;
        }
    }
}

template <typename GraphT>
void AcyclicPartitioningILP<GraphT>::SetupVariablesConstraintsObjective(const BspInstance<GraphT> &instance) {
    // Variables

    nodeInPartition_ = std::vector<VarArray>(instance.NumberOfVertices());

    for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
        nodeInPartition_[node] = model_.AddVars(static_cast<int>(numberOfParts_), COPT_BINARY, "node_in_partition");
    }

    std::map<VertexIdx, unsigned> nodeToHyperedgeIndex;
    unsigned numberOfHyperedges = 0;
    for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
        if (instance.GetComputationalDag().OutDegree(node) > 0) {
            nodeToHyperedgeIndex[node] = numberOfHyperedges;
            ++numberOfHyperedges;
        }
    }

    hyperedgeIntersectsPartition_ = std::vector<VarArray>(numberOfHyperedges);

    for (unsigned hyperedge = 0; hyperedge < numberOfHyperedges; hyperedge++) {
        hyperedgeIntersectsPartition_[hyperedge]
            = model_.AddVars(static_cast<int>(numberOfParts_), COPT_BINARY, "hyperedge_intersects_partition");
    }

    // Constraints

    // each node assigned to exactly one partition
    for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
        Expr expr;
        for (unsigned part = 0; part < numberOfParts_; part++) {
            expr += nodeInPartition_[node][static_cast<int>(part)];
        }
        model_.AddConstr(expr == 1);
    }

    // hyperedge indicators match node variables
    for (unsigned part = 0; part < numberOfParts_; part++) {
        for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
            if (instance.GetComputationalDag().OutDegree(node) == 0) {
                continue;
            }

            model_.AddConstr(hyperedgeIntersectsPartition_[nodeToHyperedgeIndex[node]][static_cast<int>(part)]
                             >= nodeInPartition_[node][static_cast<int>(part)]);
            for (const auto &succ : instance.GetComputationalDag().Children(node)) {
                model_.AddConstr(hyperedgeIntersectsPartition_[nodeToHyperedgeIndex[node]][static_cast<int>(part)]
                                 >= nodeInPartition_[succ][static_cast<int>(part)]);
            }
        }
    }

    // partition size constraints
    for (unsigned part = 0; part < numberOfParts_; part++) {
        Expr expr;
        for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
            if (!ignoreSourcesForConstraint_ || isOriginalSource_.empty() || !isOriginalSource_[node]) {
                expr += nodeInPartition_[node][static_cast<int>(part)];
            }
        }

        model_.AddConstr(expr <= maxPartitionSize_);
        model_.AddConstr(expr >= minPartitionSize_);
    }

    // acyclicity constraints
    for (unsigned fromPart = 0; fromPart < numberOfParts_; fromPart++) {
        for (unsigned toPart = 0; toPart < fromPart; toPart++) {
            for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
                for (const auto &succ : instance.GetComputationalDag().Children(node)) {
                    model_.AddConstr(
                        nodeInPartition_[node][static_cast<int>(fromPart)] + nodeInPartition_[succ][static_cast<int>(toPart)] <= 1);
                }
            }
        }
    }

    // set objective
    Expr expr;
    for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
        if (instance.GetComputationalDag().OutDegree(node) > 0) {
            expr -= instance.GetComputationalDag().VertexCommWeight(node);
            for (unsigned part = 0; part < numberOfParts_; part++) {
                expr += instance.GetComputationalDag().VertexCommWeight(node)
                        * hyperedgeIntersectsPartition_[nodeToHyperedgeIndex[node]][static_cast<int>(part)];
            }
        }
    }

    model_.SetObjective(expr, COPT_MINIMIZE);
}

template <typename GraphT>
void AcyclicPartitioningILP<GraphT>::WriteSolutionCallback::callback() {
    if (Where() == COPT_CBCONTEXT_MIPSOL && counter_ < maxNumberSolution_ && GetIntInfo(COPT_CBINFO_HASINCUMBENT)) {
        try {
            if (GetDblInfo(COPT_CBINFO_BESTOBJ) < bestObj_ && 0.0 < GetDblInfo(COPT_CBINFO_BESTBND)) {
                bestObj_ = GetDblInfo(COPT_CBINFO_BESTOBJ);
                counter_++;
            }

        } catch (const std::exception &e) {}
    }
}

template <typename GraphT>
std::vector<unsigned> AcyclicPartitioningILP<GraphT>::ReturnAssignment(const BspInstance<GraphT> &instance) {
    std::vector<unsigned> nodeToPartition(instance.NumberOfVertices(), UINT_MAX);

    std::set<unsigned> nonemptyPartitionIds;
    for (unsigned node = 0; node < instance.NumberOfVertices(); node++) {
        for (unsigned part = 0; part < numberOfParts_; part++) {
            if (nodeInPartition_[node][static_cast<int>(part)].Get(COPT_DBLINFO_VALUE) >= .99) {
                nodeToPartition[node] = part;
                nonemptyPartitionIds.insert(part);
            }
        }
    }

    for (unsigned chosenPartition : nodeToPartition) {
        if (chosenPartition == UINT_MAX) {
            std::cout << "Error: partitioning returned by ILP seems incomplete!" << std::endl;
        }
    }

    unsigned currentIndex = 0;
    std::map<unsigned, unsigned> newIndex;
    for (unsigned partIndex : nonemptyPartitionIds) {
        newIndex[partIndex] = currentIndex;
        ++currentIndex;
    }

    for (VertexIdx node = 0; node < instance.NumberOfVertices(); node++) {
        nodeToPartition[node] = newIndex[nodeToPartition[node]];
    }

    std::cout << "Acyclic partitioning ILP best solution value: " << model_.GetDblAttr(COPT_DBLATTR_BESTOBJ)
              << ", best lower bound: " << model_.GetDblAttr(COPT_DBLATTR_BESTBND) << std::endl;

    return nodeToPartition;
}

}    // namespace osp
