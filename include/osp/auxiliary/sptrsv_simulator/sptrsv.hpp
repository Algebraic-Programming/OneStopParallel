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

@author Toni Boehnlein, Christos Matzoros, Pal Andras Papp, Raphael S. Steiner
*/

#pragma once

#ifdef EIGEN_FOUND

#    include <omp.h>

#    include <Eigen/Core>
#    include <algorithm>
#    include <atomic>
#    include <cassert>
#    include <chrono>
#    include <iostream>
#    include <list>
#    include <map>
#    include <memory>
#    include <random>
#    include <set>
#    include <stdexcept>
#    include <thread>
#    include <vector>

#    include "osp/auxiliary/sptrsv_simulator/WeakBarriers/flat_checkpoint_counter_barrier.hpp"
#    include "osp/bsp/model/BspInstance.hpp"
#    include "osp/bsp/model/BspSchedule.hpp"
#    include "osp/graph_implementations/eigen_matrix_adapter/sparse_matrix.hpp"

namespace osp {

template <typename IndexType>
class Sptrsv {
    using UVertType = typename SparseMatrixImp<IndexType>::VertexIdx;
    using StorageIdx = IndexType;
    using StepProcessorVertices = std::vector<std::vector<std::vector<IndexType>>>;

    static StepProcessorVertices CreateStepProcessorStorage(size_t supersteps, size_t processors) {
        return StepProcessorVertices(supersteps, std::vector<std::vector<IndexType>>(processors));
    }

  private:
    const BspInstance<SparseMatrixImp<IndexType>> *instance_;
    bool baseStorageReady_ = false;
    std::vector<double> baseVal_;
    std::vector<double> baseCscVal_;
    std::vector<StorageIdx> baseColIdx_;
    std::vector<StorageIdx> baseRowPtr_;
    std::vector<StorageIdx> baseRowIdx_;
    std::vector<StorageIdx> baseColPtr_;

    template <typename SparseStorageType>
    void CopySparseStorage(const SparseStorageType *matrix,
                           std::vector<double> &values,
                           std::vector<StorageIdx> &innerIndices,
                           std::vector<StorageIdx> &outerPointers) {
        const size_t nnz = static_cast<size_t>(matrix->nonZeros());
        const size_t nrows = instance_->NumberOfVertices();

        values.assign(matrix->valuePtr(), matrix->valuePtr() + nnz);

        innerIndices.clear();
        innerIndices.reserve(nnz);
        const StorageIdx *inner = matrix->innerIndexPtr();
        for (size_t idx = 0; idx < nnz; ++idx) {
            innerIndices.push_back(inner[idx]);
        }

        outerPointers.clear();
        outerPointers.reserve(nrows + 1);
        const StorageIdx *outer = matrix->outerIndexPtr();
        for (size_t row = 0; row <= nrows; ++row) {
            outerPointers.push_back(outer[row]);
        }
    }

    void BuildBaseStorageIfNeeded() {
        if (baseStorageReady_) {
            return;
        }

        CopySparseStorage(instance_->GetComputationalDag().GetCSR(), baseVal_, baseColIdx_, baseRowPtr_);
        CopySparseStorage(instance_->GetComputationalDag().GetCSC(), baseCscVal_, baseRowIdx_, baseColPtr_);
        baseStorageReady_ = true;
    }

    static void AppendIncreasingBounds(const std::vector<IndexType> &vertices, std::vector<IndexType> &bounds) {
        if (vertices.empty()) {
            return;
        }
        IndexType start = vertices.front();
        IndexType prev = vertices.front();
        for (size_t i = 1; i < vertices.size(); ++i) {
            if (vertices[i] != prev + 1) {
                bounds.push_back(start);
                bounds.push_back(prev);
                start = vertices[i];
            }
            prev = vertices[i];
        }
        bounds.push_back(start);
        bounds.push_back(prev);
    }

    static void AppendDecreasingBounds(const std::vector<IndexType> &vertices, std::vector<IndexType> &bounds) {
        if (vertices.empty()) {
            return;
        }
        IndexType start = vertices.front();
        IndexType prev = vertices.front();
        for (size_t i = 1; i < vertices.size(); ++i) {
            if (vertices[i] != prev - 1) {
                bounds.push_back(start);
                bounds.push_back(prev);
                start = vertices[i];
            }
            prev = vertices[i];
        }
        bounds.push_back(start);
        bounds.push_back(prev);
    }

    static inline void SolveLowerTriangularCsrInPlace(const StorageIdx n,
                                                      const StorageIdx *rowPtr,
                                                      const StorageIdx *colIdx,
                                                      const double *values,
                                                      double *solution) {
        for (StorageIdx i = 0; i < n; i++) {
            const StorageIdx rowBegin = rowPtr[i];
            const StorageIdx diagIndex = rowPtr[i + 1] - 1;
            double accumulator = solution[i];
            for (StorageIdx j = rowBegin; j < diagIndex; j++) {
                accumulator -= values[j] * solution[colIdx[j]];
            }
            solution[i] = accumulator / values[diagIndex];
        }
    }

    static inline void SolveLowerTriangularCsr(const StorageIdx n,
                                               const StorageIdx *rowPtr,
                                               const StorageIdx *colIdx,
                                               const double *values,
                                               const double *rhs,
                                               double *solution) {
        for (StorageIdx i = 0; i < n; i++) {
            const StorageIdx rowBegin = rowPtr[i];
            const StorageIdx diagIndex = rowPtr[i + 1] - 1;
            double accumulator = rhs[i];
            for (StorageIdx j = rowBegin; j < diagIndex; j++) {
                accumulator -= values[j] * solution[colIdx[j]];
            }
            solution[i] = accumulator / values[diagIndex];
        }
    }

    static inline void SolveLowerRowInPlace(const StorageIdx row,
                                            const StorageIdx *rowPtr,
                                            const StorageIdx *colIdx,
                                            const double *values,
                                            double *solution) {
        const StorageIdx rowBegin = rowPtr[row];
        const StorageIdx diagIndex = rowPtr[row + 1] - 1;
        double accumulator = solution[row];
        for (StorageIdx j = rowBegin; j < diagIndex; ++j) {
            accumulator -= values[j] * solution[colIdx[j]];
        }
        solution[row] = accumulator / values[diagIndex];
    }

    static inline void SolveLowerRow(const StorageIdx row,
                                     const StorageIdx *rowPtr,
                                     const StorageIdx *colIdx,
                                     const double *values,
                                     const double *rhs,
                                     double *solution) {
        const StorageIdx rowBegin = rowPtr[row];
        const StorageIdx diagIndex = rowPtr[row + 1] - 1;
        double accumulator = rhs[row];
        for (StorageIdx j = rowBegin; j < diagIndex; ++j) {
            accumulator -= values[j] * solution[colIdx[j]];
        }
        solution[row] = accumulator / values[diagIndex];
    }

    static inline void SolveUpperColumnInPlace(const StorageIdx col,
                                               const StorageIdx *colPtr,
                                               const StorageIdx *rowIdx,
                                               const double *values,
                                               double *solution) {
        const StorageIdx diagIndex = colPtr[col];
        const StorageIdx colEnd = colPtr[col + 1];
        double accumulator = solution[col];
        for (StorageIdx j = diagIndex + 1; j < colEnd; ++j) {
            accumulator -= values[j] * solution[rowIdx[j]];
        }
        solution[col] = accumulator / values[diagIndex];
    }

    static inline void SolveUpperColumn(const StorageIdx col,
                                        const StorageIdx *colPtr,
                                        const StorageIdx *rowIdx,
                                        const double *values,
                                        const double *rhs,
                                        double *solution) {
        const StorageIdx diagIndex = colPtr[col];
        const StorageIdx colEnd = colPtr[col + 1];
        double accumulator = rhs[col];
        for (StorageIdx j = diagIndex + 1; j < colEnd; ++j) {
            accumulator -= values[j] * solution[rowIdx[j]];
        }
        solution[col] = accumulator / values[diagIndex];
    }

    static inline void SolveUpperTriangularCscInPlace(const StorageIdx n,
                                                      const StorageIdx *colPtr,
                                                      const StorageIdx *rowIdx,
                                                      const double *values,
                                                      double *solution) {
        StorageIdx col = n;
        do {
            col--;
            const StorageIdx diagIndex = colPtr[col];
            const StorageIdx colEnd = colPtr[col + 1];
            double accumulator = solution[col];
            for (StorageIdx j = diagIndex + 1; j < colEnd; ++j) {
                accumulator -= values[j] * solution[rowIdx[j]];
            }
            solution[col] = accumulator / values[diagIndex];
        } while (col != 0);
    }

    static inline void SolveUpperTriangularCsc(const StorageIdx n,
                                               const StorageIdx *colPtr,
                                               const StorageIdx *rowIdx,
                                               const double *values,
                                               const double *rhs,
                                               double *solution) {
        StorageIdx col = n;
        do {
            col--;
            const StorageIdx diagIndex = colPtr[col];
            const StorageIdx colEnd = colPtr[col + 1];
            double accumulator = rhs[col];
            for (StorageIdx j = diagIndex + 1; j < colEnd; ++j) {
                accumulator -= values[j] * solution[rowIdx[j]];
            }
            solution[col] = accumulator / values[diagIndex];
        } while (col != 0);
    }

  public:
    std::vector<double> val_;
    std::vector<double> cscVal_;
    std::vector<StorageIdx> colIdx_;
    std::vector<StorageIdx> rowPtr_;
    std::vector<StorageIdx> rowIdx_;
    std::vector<StorageIdx> colPtr_;
    std::vector<std::vector<unsigned>> stepProcPtr_;
    std::vector<std::vector<unsigned>> stepProcNum_;
    double *x_;
    const double *b_;
    unsigned numSupersteps_;
    StepProcessorVertices vectorStepProcessorVertices_;
    StepProcessorVertices vectorStepProcessorVerticesU_;
    std::vector<int> ready_;
    StepProcessorVertices boundsArrayL_;
    StepProcessorVertices boundsArrayU_;

    Sptrsv() = default;
    Sptrsv(BspInstance<SparseMatrixImp<IndexType>> &inst) : instance_(&inst) { BuildBaseStorageIfNeeded(); };

    void SetupCsrNoPermutation(const BspSchedule<SparseMatrixImp<IndexType>> &schedule) {
        BuildBaseStorageIfNeeded();
        numSupersteps_ = schedule.NumberOfSupersteps();
        const size_t processors = schedule.GetInstance().NumberOfProcessors();
        const size_t numberOfVertices = instance_->GetComputationalDag().NumVertices();

        vectorStepProcessorVertices_ = CreateStepProcessorStorage(numSupersteps_, processors);
        vectorStepProcessorVerticesU_ = CreateStepProcessorStorage(numSupersteps_, processors);
        boundsArrayL_ = CreateStepProcessorStorage(numSupersteps_, processors);
        boundsArrayU_ = CreateStepProcessorStorage(numSupersteps_, processors);

#    pragma omp parallel num_threads(2)
        {
            if (omp_get_thread_num() == 0) {
                for (size_t node = 0; node < numberOfVertices; ++node) {
                    vectorStepProcessorVertices_[schedule.AssignedSuperstep(node)][schedule.AssignedProcessor(node)].push_back(
                        static_cast<IndexType>(node));
                }
                for (unsigned int step = 0; step < numSupersteps_; ++step) {
                    for (size_t proc = 0; proc < processors; ++proc) {
                        AppendIncreasingBounds(vectorStepProcessorVertices_[step][proc], boundsArrayL_[step][proc]);
                    }
                }
            } else {
                size_t node = numberOfVertices;
                do {
                    node--;
                    vectorStepProcessorVerticesU_[schedule.AssignedSuperstep(node)][schedule.AssignedProcessor(node)].push_back(
                        static_cast<IndexType>(node));
                } while (node > 0);
                for (unsigned int step = 0; step < numSupersteps_; ++step) {
                    for (size_t proc = 0; proc < processors; ++proc) {
                        AppendDecreasingBounds(vectorStepProcessorVerticesU_[step][proc], boundsArrayU_[step][proc]);
                    }
                }
            }
        }
    }

    void SetupCsrWithPermutation(const BspSchedule<SparseMatrixImp<IndexType>> &schedule, std::vector<size_t> &perm) {
        BuildBaseStorageIfNeeded();
        std::vector<size_t> permInv(perm.size());
        for (size_t i = 0; i < perm.size(); i++) {
            permInv[perm[i]] = i;
        }

        numSupersteps_ = schedule.NumberOfSupersteps();
        val_.clear();
        val_.reserve(static_cast<size_t>(instance_->GetComputationalDag().GetCSR()->nonZeros()));
        colIdx_.clear();
        colIdx_.reserve(static_cast<size_t>(instance_->GetComputationalDag().GetCSR()->nonZeros()));
        rowPtr_.clear();
        rowPtr_.reserve(instance_->NumberOfVertices() + 1);
        stepProcPtr_ = std::vector<std::vector<unsigned>>(numSupersteps_, std::vector<unsigned>(instance_->NumberOfProcessors(), 0));
        stepProcNum_ = schedule.NumAssignedNodesPerSuperstepProcessor();

        unsigned currentStep = 0;
        unsigned currentProcessor = 0;
        stepProcPtr_[currentStep][currentProcessor] = 0;

        for (const UVertType &node : permInv) {
            if (schedule.AssignedProcessor(node) != currentProcessor || schedule.AssignedSuperstep(node) != currentStep) {
                while (schedule.AssignedProcessor(node) != currentProcessor || schedule.AssignedSuperstep(node) != currentStep) {
                    if (currentProcessor < instance_->NumberOfProcessors() - 1) {
                        currentProcessor++;
                    } else {
                        currentProcessor = 0;
                        currentStep++;
                    }
                }
                stepProcPtr_[currentStep][currentProcessor] = static_cast<unsigned>(rowPtr_.size());
            }

            rowPtr_.push_back(static_cast<StorageIdx>(colIdx_.size()));
            std::set<UVertType> parents;
            for (UVertType par : instance_->GetComputationalDag().Parents(node)) {
                parents.insert(perm[par]);
            }
            for (const UVertType &par : parents) {
                colIdx_.push_back(static_cast<StorageIdx>(par));
                unsigned found = 0;
                const StorageIdx originalParent = static_cast<StorageIdx>(permInv[par]);
                const size_t rowBegin = static_cast<size_t>(baseRowPtr_[node]);
                const size_t rowEnd = static_cast<size_t>(baseRowPtr_[node + 1] - 1);
                for (size_t parInd = rowBegin; parInd < rowEnd; ++parInd) {
                    if (baseColIdx_[parInd] == originalParent) {
                        val_.push_back(baseVal_[parInd]);
                        found++;
                    }
                }
                assert(found == 1);
            }

            colIdx_.push_back(static_cast<StorageIdx>(perm[node]));
            const size_t diagonalIndex = static_cast<size_t>(baseRowPtr_[node + 1] - 1);
            val_.push_back(baseVal_[diagonalIndex]);
        }

        rowPtr_.push_back(static_cast<StorageIdx>(colIdx_.size()));
    }

    void LsolveSerial() {
        const StorageIdx *outer = baseRowPtr_.data();
        const StorageIdx *inner = baseColIdx_.data();
        const double *valPtr = baseVal_.data();
        SolveLowerTriangularCsr(static_cast<StorageIdx>(instance_->NumberOfVertices()), outer, inner, valPtr, b_, x_);
    }

    void UsolveSerial() {
        const StorageIdx *outer = baseColPtr_.data();
        const StorageIdx *inner = baseRowIdx_.data();
        const double *valPtr = baseCscVal_.data();
        SolveUpperTriangularCsc(static_cast<StorageIdx>(instance_->NumberOfVertices()), outer, inner, valPtr, b_, x_);
    }

    void LsolveNoPermutationInPlace() {
        const StorageIdx *outer = baseRowPtr_.data();
        const StorageIdx *inner = baseColIdx_.data();
        const double *valPtr = baseVal_.data();
#    pragma omp parallel num_threads(instance_->NumberOfProcessors())
        {
            const size_t proc = static_cast<size_t>(omp_get_thread_num());
            for (unsigned step = 0; step < numSupersteps_; ++step) {
                const size_t boundsStrSize = boundsArrayL_[step][proc].size();
                for (size_t index = 0; index < boundsStrSize; index += 2) {
                    IndexType lowerB = boundsArrayL_[step][proc][index];
                    const IndexType upperB = boundsArrayL_[step][proc][index + 1];
                    for (IndexType node = lowerB; node <= upperB; ++node) {
                        SolveLowerRowInPlace(static_cast<StorageIdx>(node), outer, inner, valPtr, x_);
                    }
                }
#    pragma omp barrier
            }
        }
    }

    void UsolveNoPermutationInPlace() {
        const StorageIdx *outer = baseColPtr_.data();
        const StorageIdx *inner = baseRowIdx_.data();
        const double *valPtr = baseCscVal_.data();
#    pragma omp parallel num_threads(instance_->NumberOfProcessors())
        {
            const size_t proc = static_cast<size_t>(omp_get_thread_num());
            unsigned step = numSupersteps_;
            do {
                step--;
                const size_t boundsStrSize = boundsArrayU_[step][proc].size();
                for (size_t index = 0; index < boundsStrSize; index += 2) {
                    IndexType node = boundsArrayU_[step][proc][index] + 1;
                    const IndexType lowerB = boundsArrayU_[step][proc][index + 1];
                    do {
                        node--;
                        SolveUpperColumnInPlace(static_cast<StorageIdx>(node), outer, inner, valPtr, x_);
                    } while (node != lowerB);
                }
#    pragma omp barrier
            } while (step != 0);
        }
    }

    void LsolveNoPermutation() {
        const StorageIdx *outer = baseRowPtr_.data();
        const StorageIdx *inner = baseColIdx_.data();
        const double *valPtr = baseVal_.data();
#    pragma omp parallel num_threads(instance_->NumberOfProcessors())
        {
            const size_t proc = static_cast<size_t>(omp_get_thread_num());
            for (unsigned step = 0; step < numSupersteps_; ++step) {
                const size_t boundsStrSize = boundsArrayL_[step][proc].size();
                for (size_t index = 0; index < boundsStrSize; index += 2) {
                    IndexType lowerB = boundsArrayL_[step][proc][index];
                    const IndexType upperB = boundsArrayL_[step][proc][index + 1];
                    for (IndexType node = lowerB; node <= upperB; ++node) {
                        SolveLowerRow(static_cast<StorageIdx>(node), outer, inner, valPtr, b_, x_);
                    }
                }
#    pragma omp barrier
            }
        }
    }

    void UsolveNoPermutation() {
        const StorageIdx *outer = baseColPtr_.data();
        const StorageIdx *inner = baseRowIdx_.data();
        const double *valPtr = baseCscVal_.data();
#    pragma omp parallel num_threads(instance_->NumberOfProcessors())
        {
            const size_t proc = static_cast<size_t>(omp_get_thread_num());
            unsigned step = numSupersteps_;
            do {
                step--;
                const size_t boundsStrSize = boundsArrayU_[step][proc].size();
                for (size_t index = 0; index < boundsStrSize; index += 2) {
                    IndexType node = boundsArrayU_[step][proc][index] + 1;
                    const IndexType lowerB = boundsArrayU_[step][proc][index + 1];
                    do {
                        node--;
                        SolveUpperColumn(static_cast<StorageIdx>(node), outer, inner, valPtr, b_, x_);
                    } while (node != lowerB);
                }
#    pragma omp barrier
            } while (step != 0);
        }
    }

    void LsolveSerialInPlace() {
        const StorageIdx *outer = baseRowPtr_.data();
        const StorageIdx *inner = baseColIdx_.data();
        const double *valPtr = baseVal_.data();
        SolveLowerTriangularCsrInPlace(static_cast<StorageIdx>(instance_->NumberOfVertices()), outer, inner, valPtr, x_);
    }

    void UsolveSerialInPlace() {
        const StorageIdx *outer = baseColPtr_.data();
        const StorageIdx *inner = baseRowIdx_.data();
        const double *valPtr = baseCscVal_.data();
        SolveUpperTriangularCscInPlace(static_cast<StorageIdx>(instance_->NumberOfVertices()), outer, inner, valPtr, x_);
    }

    void LsolveWithPermutationInPlace() {
#    pragma omp parallel num_threads(instance_->NumberOfProcessors())
        {
            for (unsigned step = 0; step < numSupersteps_; step++) {
                const size_t proc = static_cast<size_t>(omp_get_thread_num());
                const StorageIdx upperLimit = static_cast<StorageIdx>(stepProcPtr_[step][proc] + stepProcNum_[step][proc]);
                const StorageIdx *outer = rowPtr_.data();
                const StorageIdx *inner = colIdx_.data();
                const double *vals = val_.data();
                for (StorageIdx rowIdx = static_cast<StorageIdx>(stepProcPtr_[step][proc]); rowIdx < upperLimit; rowIdx++) {
                    SolveLowerRowInPlace(rowIdx, outer, inner, vals, x_);
                }
#    pragma omp barrier
            }
        }
    }

    void LsolveWithPermutation() {
#    pragma omp parallel num_threads(instance_->NumberOfProcessors())
        {
            for (unsigned step = 0; step < numSupersteps_; step++) {
                const size_t proc = static_cast<size_t>(omp_get_thread_num());
                const StorageIdx upperLimit = static_cast<StorageIdx>(stepProcPtr_[step][proc] + stepProcNum_[step][proc]);
                const StorageIdx *outer = rowPtr_.data();
                const StorageIdx *inner = colIdx_.data();
                const double *vals = val_.data();
                for (StorageIdx rowIdx = static_cast<StorageIdx>(stepProcPtr_[step][proc]); rowIdx < upperLimit; rowIdx++) {
                    SolveLowerRow(rowIdx, outer, inner, vals, b_, x_);
                }
#    pragma omp barrier
            }
        }
    }

    void ResetX() {
        IndexType numberOfVertices = static_cast<IndexType>(instance_->NumberOfVertices());
        for (IndexType i = 0; i < numberOfVertices; i++) {
            x_[i] = 1.0;
        }
    }

    void PermuteXVector(const std::vector<size_t> &perm) {
        std::vector<double> vecPerm(perm.size());
        for (size_t i = 0; i < perm.size(); i++) {
            vecPerm[i] = x_[perm[i]];
        }
        for (size_t i = 0; i < perm.size(); i++) {
            x_[i] = vecPerm[i];
        }
    }

    void PermuteXVectorInverse(const std::vector<size_t> &perm) {
        std::vector<double> vecUnperm(perm.size());
        for (size_t i = 0; i < perm.size(); i++) {
            vecUnperm[perm[i]] = x_[i];
        }
        for (size_t i = 0; i < perm.size(); i++) {
            x_[i] = vecUnperm[i];
        }
    }

    std::size_t GetNumberOfVertices() { return instance_->NumberOfVertices(); }

    template <unsigned staleness = 2U>
    void SspLsolveStaleness() {
        const unsigned nthreads = instance_->NumberOfProcessors();
        FlatCheckpointCounterBarrier barrier(nthreads);
        const StorageIdx *outer = baseRowPtr_.data();
        const StorageIdx *inner = baseColIdx_.data();
        const double *vals = baseVal_.data();
#    pragma omp parallel num_threads(nthreads)
        {
            const std::size_t proc = static_cast<std::size_t>(omp_get_thread_num());
            for (unsigned step = 0; step < numSupersteps_; ++step) {
                const size_t boundsStrSize = boundsArrayL_[step][proc].size();
                if (boundsStrSize > 0U) {
                    barrier.Wait(proc, staleness - 1U);
                }
                for (size_t index = 0; index < boundsStrSize; index += 2) {
                    IndexType lowerB = boundsArrayL_[step][proc][index];
                    const IndexType upperB = boundsArrayL_[step][proc][index + 1];
                    for (IndexType node = lowerB; node <= upperB; ++node) {
                        SolveLowerRow(static_cast<StorageIdx>(node), outer, inner, vals, b_, x_);
                    }
                }
                barrier.Arrive(proc);
            }
        }
    }

    template <unsigned staleness = 2U>
    void SspLsolveStalenessInPlace() {
        const unsigned nthreads = instance_->NumberOfProcessors();
        FlatCheckpointCounterBarrier barrier(nthreads);
        const StorageIdx *outer = baseRowPtr_.data();
        const StorageIdx *inner = baseColIdx_.data();
        const double *vals = baseVal_.data();
#    pragma omp parallel num_threads(nthreads)
        {
            const std::size_t proc = static_cast<std::size_t>(omp_get_thread_num());
            for (unsigned step = 0; step < numSupersteps_; ++step) {
                const size_t boundsStrSize = boundsArrayL_[step][proc].size();
                if (boundsStrSize > 0U) {
                    barrier.Wait(proc, staleness - 1U);
                }
                for (size_t index = 0; index < boundsStrSize; index += 2) {
                    IndexType lowerB = boundsArrayL_[step][proc][index];
                    const IndexType upperB = boundsArrayL_[step][proc][index + 1];
                    for (IndexType node = lowerB; node <= upperB; ++node) {
                        SolveLowerRowInPlace(static_cast<StorageIdx>(node), outer, inner, vals, x_);
                    }
                }
                barrier.Arrive(proc);
            }
        }
    }

    template <unsigned staleness = 2U>
    void SspUsolveStaleness() {
        const unsigned nthreads = instance_->NumberOfProcessors();
        FlatCheckpointCounterBarrier barrier(nthreads);
        const StorageIdx *outer = baseColPtr_.data();
        const StorageIdx *inner = baseRowIdx_.data();
        const double *vals = baseCscVal_.data();
#    pragma omp parallel num_threads(nthreads)
        {
            const std::size_t proc = static_cast<std::size_t>(omp_get_thread_num());
            unsigned step = numSupersteps_;
            do {
                step--;
                const size_t boundsStrSize = boundsArrayU_[step][proc].size();
                if (boundsStrSize > 0U) {
                    barrier.Wait(proc, staleness - 1U);
                }
                for (size_t index = 0; index < boundsStrSize; index += 2) {
                    IndexType node = boundsArrayU_[step][proc][index] + 1;
                    const IndexType lowerB = boundsArrayU_[step][proc][index + 1];
                    do {
                        node--;
                        SolveUpperColumn(static_cast<StorageIdx>(node), outer, inner, vals, b_, x_);
                    } while (node != lowerB);
                }
                barrier.Arrive(proc);
            } while (step != 0);
        }
    }

    virtual ~Sptrsv() = default;
};

}    // namespace osp

#endif
