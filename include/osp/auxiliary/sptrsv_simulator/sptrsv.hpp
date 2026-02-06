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
#    include <chrono>
#    include <iostream>
#    include <list>
#    include <map>
#    include <memory>
#    include <random>
#    include <stdexcept>
#    include <thread>
#    include <vector>

#    include "osp/auxiliary/sptrsv_simulator/WeakBarriers/flat_checkpoint_counter_barrier.hpp"
#    include "osp/auxiliary/sptrsv_simulator/WeakBarriers/flat_checkpoint_counter_barrier_cached.hpp"
#    include "osp/bsp/model/BspInstance.hpp"
#    include "osp/bsp/model/BspSchedule.hpp"
#    include "osp/graph_implementations/eigen_matrix_adapter/sparse_matrix.hpp"

namespace osp {

template <typename EigenIdxType>
class Sptrsv {
    using UVertType = typename SparseMatrixImp<EigenIdxType>::VertexIdx;

  private:
    const BspInstance<SparseMatrixImp<EigenIdxType>> *instance_;

  public:
    struct BarrierOps {
        void *ctx;
        void (*arrive)(void *ctx, std::size_t threadId);
        void (*wait)(void *ctx, std::size_t threadId, std::size_t diff);
    };

    template <typename BarrierT>
    static BarrierOps MakeBarrierOps(BarrierT &barrier) {
        return BarrierOps{
            static_cast<void *>(&barrier),
            [](void *ctx, std::size_t threadId) {
                static_cast<BarrierT *>(ctx)->Arrive(threadId);
            },
            [](void *ctx, std::size_t threadId, std::size_t diff) {
                static_cast<BarrierT *>(ctx)->Wait(threadId, diff);
            }};
    }
    std::vector<double> val_;
    std::vector<double> cscVal_;

    std::vector<UVertType> colIdx_;
    std::vector<UVertType> rowPtr_;

    std::vector<UVertType> rowIdx_;
    std::vector<UVertType> colPtr_;

    std::vector<std::vector<unsigned>> stepProcPtr_;
    std::vector<std::vector<unsigned>> stepProcNum_;

    double *x_;
    const double *b_;

    unsigned numSupersteps_;

    std::vector<std::vector<std::vector<EigenIdxType>>> vectorStepProcessorVertices_;
    std::vector<std::vector<std::vector<EigenIdxType>>> vectorStepProcessorVerticesU_;
    std::vector<int> ready_;

    std::vector<std::vector<std::vector<EigenIdxType>>> boundsArrayL_;
    std::vector<std::vector<std::vector<EigenIdxType>>> boundsArrayU_;

    Sptrsv() = default;

    Sptrsv(BspInstance<SparseMatrixImp<EigenIdxType>> &inst) : instance_(&inst) {};

    void SetupCsrNoPermutation(const BspSchedule<SparseMatrixImp<EigenIdxType>> &schedule) {
        vectorStepProcessorVertices_ = std::vector<std::vector<std::vector<EigenIdxType>>>(
            schedule.NumberOfSupersteps(), std::vector<std::vector<EigenIdxType>>(schedule.GetInstance().NumberOfProcessors()));

        vectorStepProcessorVerticesU_ = std::vector<std::vector<std::vector<EigenIdxType>>>(
            schedule.NumberOfSupersteps(), std::vector<std::vector<EigenIdxType>>(schedule.GetInstance().NumberOfProcessors()));

        boundsArrayL_ = std::vector<std::vector<std::vector<EigenIdxType>>>(
            schedule.NumberOfSupersteps(), std::vector<std::vector<EigenIdxType>>(schedule.GetInstance().NumberOfProcessors()));
        boundsArrayU_ = std::vector<std::vector<std::vector<EigenIdxType>>>(
            schedule.NumberOfSupersteps(), std::vector<std::vector<EigenIdxType>>(schedule.GetInstance().NumberOfProcessors()));

        numSupersteps_ = schedule.NumberOfSupersteps();
        size_t numberOfVertices = instance_->GetComputationalDag().NumVertices();

#    pragma omp parallel num_threads(2)
        {
            int id = omp_get_thread_num();
            switch (id) {
                case 0: {
                    for (size_t node = 0; node < numberOfVertices; ++node) {
                        vectorStepProcessorVertices_[schedule.AssignedSuperstep(node)][schedule.AssignedProcessor(node)].push_back(
                            static_cast<EigenIdxType>(node));
                    }

                    for (unsigned int step = 0; step < schedule.NumberOfSupersteps(); ++step) {
                        for (unsigned int proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
                            if (!vectorStepProcessorVertices_[step][proc].empty()) {
                                EigenIdxType start = vectorStepProcessorVertices_[step][proc][0];
                                EigenIdxType prev = vectorStepProcessorVertices_[step][proc][0];

                                for (size_t i = 1; i < vectorStepProcessorVertices_[step][proc].size(); ++i) {
                                    if (vectorStepProcessorVertices_[step][proc][i] != prev + 1) {
                                        boundsArrayL_[step][proc].push_back(start);
                                        boundsArrayL_[step][proc].push_back(prev);
                                        start = vectorStepProcessorVertices_[step][proc][i];
                                    }
                                    prev = vectorStepProcessorVertices_[step][proc][i];
                                }

                                boundsArrayL_[step][proc].push_back(start);
                                boundsArrayL_[step][proc].push_back(prev);
                            }
                        }
                    }

                    break;
                }
                case 1: {
                    size_t node = numberOfVertices;
                    do {
                        node--;
                        vectorStepProcessorVerticesU_[schedule.AssignedSuperstep(node)][schedule.AssignedProcessor(node)].push_back(
                // --- SSP SpTRSV kernel integration from BspSptrsvCSR.hpp/cpp ---

                            static_cast<EigenIdxType>(node));
                    } while (node > 0);

                    for (unsigned int step = 0; step < schedule.NumberOfSupersteps(); ++step) {
                        for (unsigned int proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
                            if (!vectorStepProcessorVerticesU_[step][proc].empty()) {
                                EigenIdxType startU = static_cast<EigenIdxType>(vectorStepProcessorVerticesU_[step][proc][0]);
                                EigenIdxType prevU = static_cast<EigenIdxType>(vectorStepProcessorVerticesU_[step][proc][0]);

                                for (size_t i = 1; i < vectorStepProcessorVerticesU_[step][proc].size(); ++i) {
                                    if (static_cast<EigenIdxType>(vectorStepProcessorVerticesU_[step][proc][i]) != prevU - 1) {
                                        boundsArrayU_[step][proc].push_back(startU);
                                        boundsArrayU_[step][proc].push_back(prevU);
                                        startU = static_cast<EigenIdxType>(vectorStepProcessorVerticesU_[step][proc][i]);
                                    }
                                    prevU = static_cast<EigenIdxType>(vectorStepProcessorVerticesU_[step][proc][i]);
                                }

                                boundsArrayU_[step][proc].push_back(startU);
                                boundsArrayU_[step][proc].push_back(prevU);
                            }
                        }
                    }

                    break;
                }
                default: {
                    std::cout << "Unexpected Behaviour" << std::endl;
                }
            }
        }
    }

    void SetupCsrWithPermutation(const BspSchedule<SparseMatrixImp<EigenIdxType>> &schedule, std::vector<size_t> &perm) {
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

        stepProcPtr_
            = std::vector<std::vector<unsigned>>(numSupersteps_, std::vector<unsigned>(instance_->NumberOfProcessors(), 0));

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

            rowPtr_.push_back(colIdx_.size());

            std::set<UVertType> parents;

            for (UVertType par : instance_->GetComputationalDag().Parents(node)) {
                parents.insert(perm[par]);
            }

            for (const UVertType &par : parents) {
                colIdx_.push_back(par);
                unsigned found = 0;

                const auto *outer = instance_->GetComputationalDag().GetCSR()->outerIndexPtr();
                for (UVertType parInd = static_cast<UVertType>(outer[node]); parInd < static_cast<UVertType>(outer[node + 1] - 1);
                     ++parInd) {
                    if (static_cast<size_t>(instance_->GetComputationalDag().GetCSR()->innerIndexPtr()[parInd]) == permInv[par]) {
                        val_.push_back(instance_->GetComputationalDag().GetCSR()->valuePtr()[parInd]);
                        found++;
                    }
                }
                assert(found == 1);
            }

            colIdx_.push_back(perm[node]);
            val_.push_back(instance_->GetComputationalDag()
                               .GetCSR()
                               ->valuePtr()[instance_->GetComputationalDag().GetCSR()->outerIndexPtr()[node + 1] - 1]);
        }

        rowPtr_.push_back(colIdx_.size());
    }

    void LsolveSerial() {
        EigenIdxType numberOfVertices = static_cast<EigenIdxType>(instance_->NumberOfVertices());
        for (EigenIdxType i = 0; i < numberOfVertices; ++i) {
            x_[i] = b_[i];
            for (EigenIdxType j = (*(instance_->GetComputationalDag().GetCSR())).outerIndexPtr()[i];
                 j < (*(instance_->GetComputationalDag().GetCSR())).outerIndexPtr()[i + 1] - 1;
                 ++j) {
                x_[i] -= (*(instance_->GetComputationalDag().GetCSR())).valuePtr()[j]
                         * x_[(*(instance_->GetComputationalDag().GetCSR())).innerIndexPtr()[j]];
            }
            x_[i] /= (*(instance_->GetComputationalDag().GetCSR()))
                         .valuePtr()[(*(instance_->GetComputationalDag().GetCSR())).outerIndexPtr()[i + 1] - 1];
        }
    }

    void UsolveSerial() {
        EigenIdxType numberOfVertices = static_cast<EigenIdxType>(instance_->NumberOfVertices());

        EigenIdxType i = numberOfVertices;
        do {
            i--;
            x_[i] = b_[i];
            for (EigenIdxType j = (*(instance_->GetComputationalDag().GetCSC())).outerIndexPtr()[i] + 1;
                 j < (*(instance_->GetComputationalDag().GetCSC())).outerIndexPtr()[i + 1];
                 ++j) {
                x_[i] -= (*(instance_->GetComputationalDag().GetCSC())).valuePtr()[j]
                         * x_[(*(instance_->GetComputationalDag().GetCSC())).innerIndexPtr()[j]];
            }
            x_[i] /= (*(instance_->GetComputationalDag().GetCSC()))
                         .valuePtr()[(*(instance_->GetComputationalDag().GetCSC())).outerIndexPtr()[i]];
        } while (i != 0);
    }

    void LsolveNoPermutationInPlace() {
#    pragma omp parallel num_threads(instance_->NumberOfProcessors())
        {
            const size_t proc = static_cast<size_t>(omp_get_thread_num());
            for (unsigned step = 0; step < numSupersteps_; ++step) {
                const size_t boundsStrSize = boundsArrayL_[step][proc].size();

                for (size_t index = 0; index < boundsStrSize; index += 2) {
                    EigenIdxType lowerB = boundsArrayL_[step][proc][index];
                    const EigenIdxType upperB = boundsArrayL_[step][proc][index + 1];

                    for (EigenIdxType node = lowerB; node <= upperB; ++node) {
                        for (EigenIdxType i = (*(instance_->GetComputationalDag().GetCSR())).outerIndexPtr()[node];
                             i < (*(instance_->GetComputationalDag().GetCSR())).outerIndexPtr()[node + 1] - 1;
                             ++i) {
                            x_[node] -= (*(instance_->GetComputationalDag().GetCSR())).valuePtr()[i]
                                        * x_[(*(instance_->GetComputationalDag().GetCSR())).innerIndexPtr()[i]];
                        }
                        x_[node] /= (*(instance_->GetComputationalDag().GetCSR()))
                                        .valuePtr()[(*(instance_->GetComputationalDag().GetCSR())).outerIndexPtr()[node + 1] - 1];
                    }
                }
#    pragma omp barrier
            }
        }
    }

    void UsolveNoPermutationInPlace() {
#    pragma omp parallel num_threads(instance_->NumberOfProcessors())
        {
            // Process each superstep starting from the last one (opposite of lsolve)
            const size_t proc = static_cast<size_t>(omp_get_thread_num());
            unsigned step = numSupersteps_;
            do {
                step--;
                const size_t boundsStrSize = boundsArrayU_[step][proc].size();
                for (size_t index = 0; index < boundsStrSize; index += 2) {
                    EigenIdxType node = boundsArrayU_[step][proc][index] + 1;
                    const EigenIdxType lowerB = boundsArrayU_[step][proc][index + 1];

                    do {
                        node--;
                        for (EigenIdxType i = (*(instance_->GetComputationalDag().GetCSC())).outerIndexPtr()[node] + 1;
                             i < (*(instance_->GetComputationalDag().GetCSC())).outerIndexPtr()[node + 1];
                             ++i) {
                            x_[node] -= (*(instance_->GetComputationalDag().GetCSC())).valuePtr()[i]
                                        * x_[(*(instance_->GetComputationalDag().GetCSC())).innerIndexPtr()[i]];
                        }
                        x_[node] /= (*(instance_->GetComputationalDag().GetCSC()))
                                        .valuePtr()[(*(instance_->GetComputationalDag().GetCSC())).outerIndexPtr()[node]];
                    } while (node != lowerB);
                }
#    pragma omp barrier
            } while (step != 0);
        }
    }

    void LsolveNoPermutation() {
#    pragma omp parallel num_threads(instance_->NumberOfProcessors())
        {
            const size_t proc = static_cast<size_t>(omp_get_thread_num());
            for (unsigned step = 0; step < numSupersteps_; ++step) {
                const size_t boundsStrSize = boundsArrayL_[step][proc].size();

                for (size_t index = 0; index < boundsStrSize; index += 2) {
                    EigenIdxType lowerB = boundsArrayL_[step][proc][index];
                    const EigenIdxType upperB = boundsArrayL_[step][proc][index + 1];

                    for (EigenIdxType node = lowerB; node <= upperB; ++node) {
                        x_[node] = b_[node];
                        for (EigenIdxType i = (*(instance_->GetComputationalDag().GetCSR())).outerIndexPtr()[node];
                             i < (*(instance_->GetComputationalDag().GetCSR())).outerIndexPtr()[node + 1] - 1;
                             ++i) {
                            x_[node] -= (*(instance_->GetComputationalDag().GetCSR())).valuePtr()[i]
                                        * x_[(*(instance_->GetComputationalDag().GetCSR())).innerIndexPtr()[i]];
                        }
                        x_[node] /= (*(instance_->GetComputationalDag().GetCSR()))
                                        .valuePtr()[(*(instance_->GetComputationalDag().GetCSR())).outerIndexPtr()[node + 1] - 1];
                    }
                }
#    pragma omp barrier
            }
        }
    }

    void UsolveNoPermutation() {
#    pragma omp parallel num_threads(instance_->NumberOfProcessors())
        {
            // Process each superstep starting from the last one (opposite of lsolve)
            const size_t proc = static_cast<size_t>(omp_get_thread_num());
            unsigned step = numSupersteps_;
            do {
                step--;
                const size_t boundsStrSize = boundsArrayU_[step][proc].size();
                for (size_t index = 0; index < boundsStrSize; index += 2) {
                    EigenIdxType node = boundsArrayU_[step][proc][index] + 1;
                    const EigenIdxType lowerB = boundsArrayU_[step][proc][index + 1];

                    do {
                        node--;
                        x_[node] = b_[node];
                        for (EigenIdxType i = (*(instance_->GetComputationalDag().GetCSC())).outerIndexPtr()[node] + 1;
                             i < (*(instance_->GetComputationalDag().GetCSC())).outerIndexPtr()[node + 1];
                             ++i) {
                            x_[node] -= (*(instance_->GetComputationalDag().GetCSC())).valuePtr()[i]
                                        * x_[(*(instance_->GetComputationalDag().GetCSC())).innerIndexPtr()[i]];
                        }
                        x_[node] /= (*(instance_->GetComputationalDag().GetCSC()))
                                        .valuePtr()[(*(instance_->GetComputationalDag().GetCSC())).outerIndexPtr()[node]];
                    } while (node != lowerB);
                }
#    pragma omp barrier
            } while (step != 0);
        }
    }

    void LsolveSerialInPlace() {
        EigenIdxType numberOfVertices = static_cast<EigenIdxType>(instance_->NumberOfVertices());
        for (EigenIdxType i = 0; i < numberOfVertices; ++i) {
            for (EigenIdxType j = (*(instance_->GetComputationalDag().GetCSR())).outerIndexPtr()[i];
                 j < (*(instance_->GetComputationalDag().GetCSR())).outerIndexPtr()[i + 1] - 1;
                 ++j) {
                x_[i] -= (*(instance_->GetComputationalDag().GetCSR())).valuePtr()[j]
                         * x_[(*(instance_->GetComputationalDag().GetCSR())).innerIndexPtr()[j]];
            }
            x_[i] /= (*(instance_->GetComputationalDag().GetCSR()))
                         .valuePtr()[(*(instance_->GetComputationalDag().GetCSR())).outerIndexPtr()[i + 1] - 1];
        }
    }

    void UsolveSerialInPlace() {
        EigenIdxType numberOfVertices = static_cast<EigenIdxType>(instance_->NumberOfVertices());
        EigenIdxType i = numberOfVertices;
        do {
            i--;
            for (EigenIdxType j = (*(instance_->GetComputationalDag().GetCSC())).outerIndexPtr()[i] + 1;
                 j < (*(instance_->GetComputationalDag().GetCSC())).outerIndexPtr()[i + 1];
                 ++j) {
                x_[i] -= (*(instance_->GetComputationalDag().GetCSC())).valuePtr()[j]
                         * x_[(*(instance_->GetComputationalDag().GetCSC())).innerIndexPtr()[j]];
            }
            x_[i] /= (*(instance_->GetComputationalDag().GetCSC()))
                         .valuePtr()[(*(instance_->GetComputationalDag().GetCSC())).outerIndexPtr()[i]];
        } while (i != 0);
    }

    void LsolveWithPermutationInPlace() {
#    pragma omp parallel num_threads(instance_->NumberOfProcessors())
        {
            for (unsigned step = 0; step < numSupersteps_; step++) {
                const size_t proc = static_cast<size_t>(omp_get_thread_num());
                const UVertType upperLimit = stepProcPtr_[step][proc] + stepProcNum_[step][proc];
                for (UVertType rowIdx = stepProcPtr_[step][proc]; rowIdx < upperLimit; rowIdx++) {
                    for (UVertType i = rowPtr_[rowIdx]; i < rowPtr_[rowIdx + 1] - 1; i++) {
                        x_[rowIdx] -= val_[i] * x_[colIdx_[i]];
                    }

                    x_[rowIdx] /= val_[rowPtr_[rowIdx + 1] - 1];
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
                const UVertType upperLimit = stepProcPtr_[step][proc] + stepProcNum_[step][proc];
                for (UVertType rowIdx = stepProcPtr_[step][proc]; rowIdx < upperLimit; rowIdx++) {
                    x_[rowIdx] = b_[rowIdx];
                    for (UVertType i = rowPtr_[rowIdx]; i < rowPtr_[rowIdx + 1] - 1; i++) {
                        x_[rowIdx] -= val_[i] * x_[colIdx_[i]];
                    }

                    x_[rowIdx] /= val_[rowPtr_[rowIdx + 1] - 1];
                }

#    pragma omp barrier
            }
        }
    }

    void ResetX() {
        EigenIdxType numberOfVertices = static_cast<EigenIdxType>(instance_->NumberOfVertices());
        for (EigenIdxType i = 0; i < numberOfVertices; i++) {
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

    // SSP Lsolve with staleness=2 (allowing at most one superstep of lag).
    // Barrier operations are injected via function pointers.
    void SspLsolveStaleness2(const BarrierOps &barrierOps) {
        constexpr std::size_t staleness = 2U; // Maximum allowed superstep difference
        const unsigned nthreads = instance_->NumberOfProcessors();

        auto *csr = instance_->GetComputationalDag().GetCSR();
        const auto *outer = csr->outerIndexPtr();
        const auto *inner = csr->innerIndexPtr();
        const auto *vals = csr->valuePtr();

        #pragma omp parallel num_threads(nthreads)
        {
            const std::size_t proc = static_cast<std::size_t>(omp_get_thread_num());
            for (unsigned step = 0; step < numSupersteps_; ++step) {
                // Enforce staleness window before starting this superstep.
                barrierOps.wait(barrierOps.ctx, proc, staleness - 1U);
                // Process nodes assigned to this (step, proc) pair.
                const size_t boundsStrSize = boundsArrayL_[step][proc].size();
                for (size_t index = 0; index < boundsStrSize; index += 2) {
                    EigenIdxType lowerB = boundsArrayL_[step][proc][index];
                    const EigenIdxType upperB = boundsArrayL_[step][proc][index + 1];
                    for (EigenIdxType node = lowerB; node <= upperB; ++node) {
                        // Initialize solution for this node
                        x_[node] = b_[node];
                        // Perform lower-triangular solve for this node
                        for (EigenIdxType i = outer[node]; i < outer[node + 1] - 1; ++i) {
                            // Subtract contributions from previously solved nodes
                            x_[node] -= vals[i] * x_[inner[i]];
                        }
                        // Divide by diagonal element to complete solve for this node
                        x_[node] /= vals[outer[node + 1] - 1];
                    }
                }
                // Signal completion of this superstep.
                barrierOps.arrive(barrierOps.ctx, proc);
            }
        }
    }

    // Default SSP Lsolve uses the cached flat checkpoint counter barrier.
    void SspLsolveStaleness2() {
        const unsigned nthreads = instance_->NumberOfProcessors();
        FlatCheckpointCounterBarrierCached barrier(nthreads);
        const BarrierOps ops = MakeBarrierOps(barrier);
        SspLsolveStaleness2(ops);
    }

    virtual ~Sptrsv() = default;
};

}    // namespace osp

#endif
