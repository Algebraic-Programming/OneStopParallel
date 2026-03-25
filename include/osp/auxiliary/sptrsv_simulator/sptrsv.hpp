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
#    include <limits>
#    include <list>
#    include <map>
#    include <memory>
#    include <random>
#    include <stdexcept>
#    include <thread>
#    include <type_traits>
#    include <vector>

#    include "osp/auxiliary/sptrsv_simulator/WeakBarriers/flat_checkpoint_counter_barrier.hpp"
#    include "osp/bsp/model/BspInstance.hpp"
#    include "osp/bsp/model/BspSchedule.hpp"
#    include "osp/graph_implementations/eigen_matrix_adapter/sparse_matrix.hpp"
#    include "osp/auxiliary/sptrsv_simulator/sptrsv_kernels.hpp"

namespace osp {

template <typename EigenIdxType>
class Sptrsv {
    using UVertType = typename SparseMatrixImp<EigenIdxType>::VertexIdx;

  private:
    const BspInstance<SparseMatrixImp<EigenIdxType>> *instance_;

  public:
    std::vector<double> val_;
    std::vector<double> cscVal_;

    std::vector<UVertType> colIdx_;
    std::vector<UVertType> rowPtr_;

    std::vector<UVertType> rowIdx_;
    std::vector<UVertType> colPtr_;

    std::vector<std::vector<UVertType>> procStepPtr_;
    std::vector<std::vector<UVertType>> procStepNum_;

    std::vector<UVertType> procFirstStepPtr_;

    std::vector<std::vector<UVertType>> stepProcPtr_;
    std::vector<std::vector<UVertType>> stepProcNum_;

    double *x_;
    const double *b_;

    unsigned numSupersteps_;

    std::vector<std::vector<std::vector<EigenIdxType>>> vectorProcessorStepVerticesL_;
    std::vector<std::vector<std::vector<EigenIdxType>>> vectorProcessorStepVerticesU_;
    std::vector<int> ready_;

    std::vector<std::vector<std::vector<EigenIdxType>>> boundsArrayL_;
    std::vector<std::vector<std::vector<EigenIdxType>>> boundsArrayU_;

    Sptrsv() = default;

    Sptrsv(BspInstance<SparseMatrixImp<EigenIdxType>> &inst) : instance_(&inst) {};

    void SetupCsrNoPermutation(const BspSchedule<SparseMatrixImp<EigenIdxType>> &schedule) {
        vectorProcessorStepVerticesL_ = std::vector<std::vector<std::vector<EigenIdxType>>>(
            schedule.GetInstance().NumberOfProcessors(), std::vector<std::vector<EigenIdxType>>(schedule.NumberOfSupersteps()));

        vectorProcessorStepVerticesU_ = std::vector<std::vector<std::vector<EigenIdxType>>>(
            schedule.GetInstance().NumberOfProcessors(), std::vector<std::vector<EigenIdxType>>(schedule.NumberOfSupersteps()));

        boundsArrayL_ = std::vector<std::vector<std::vector<EigenIdxType>>>(
            schedule.GetInstance().NumberOfProcessors(), std::vector<std::vector<EigenIdxType>>(schedule.NumberOfSupersteps()));
        boundsArrayU_ = std::vector<std::vector<std::vector<EigenIdxType>>>(
            schedule.GetInstance().NumberOfProcessors(), std::vector<std::vector<EigenIdxType>>(schedule.NumberOfSupersteps()));

        numSupersteps_ = schedule.NumberOfSupersteps();
        UVertType numberOfVertices = instance_->GetComputationalDag().NumVertices();

#    pragma omp parallel num_threads(2)
        {
            int id = omp_get_thread_num();
            switch (id) {
                case 0: {
                    for (UVertType node = 0; node < numberOfVertices; ++node) {
                        vectorProcessorStepVerticesL_[schedule.AssignedProcessor(node)][schedule.AssignedSuperstep(node)].push_back(
                            static_cast<EigenIdxType>(node));
                    }

                    for (unsigned int proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
                        for (unsigned int step = 0; step < schedule.NumberOfSupersteps(); ++step) {
                            const auto &vectorVerticesL = vectorProcessorStepVerticesL_[proc][step];
                            auto &localBoundsArrayL_ = boundsArrayL_[proc][step];

                            if (!vectorVerticesL.empty()) {
                                EigenIdxType start = vectorVerticesL[0];
                                EigenIdxType prev = vectorVerticesL[0];

                                for (UVertType i = 1; i < vectorVerticesL.size(); ++i) {
                                    if (vectorVerticesL[i] != prev + 1) {
                                        localBoundsArrayL_.push_back(start);
                                        localBoundsArrayL_.push_back(prev);
                                        start = vectorVerticesL[i];
                                    }
                                    prev = vectorVerticesL[i];
                                }

                                localBoundsArrayL_.push_back(start);
                                localBoundsArrayL_.push_back(prev);
                            }
                        }
                    }

                    break;
                }
                case 1: {
                    UVertType node = numberOfVertices;
                    do {
                        node--;
                        vectorProcessorStepVerticesU_[schedule.AssignedProcessor(node)][schedule.AssignedSuperstep(node)].push_back(
                            // --- SSP SpTRSV kernel integration from BspSptrsvCSR.hpp/cpp ---

                            static_cast<EigenIdxType>(node));
                    } while (node > 0);

                    for (unsigned int proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
                        for (unsigned int step = 0; step < schedule.NumberOfSupersteps(); ++step) {
                            const auto &vectorVerticesU = vectorProcessorStepVerticesU_[proc][step];
                            auto &localBoundsArrayU = boundsArrayU_[proc][step];

                            if (!vectorVerticesU.empty()) {
                                EigenIdxType startU = static_cast<EigenIdxType>(vectorVerticesU[0]);
                                EigenIdxType prevU = static_cast<EigenIdxType>(vectorVerticesU[0]);

                                for (UVertType i = 1; i < vectorVerticesU.size(); ++i) {
                                    if (static_cast<EigenIdxType>(vectorVerticesU[i]) != prevU - 1) {
                                        localBoundsArrayU.push_back(startU);
                                        localBoundsArrayU.push_back(prevU);
                                        startU = static_cast<EigenIdxType>(vectorVerticesU[i]);
                                    }
                                    prevU = static_cast<EigenIdxType>(vectorVerticesU[i]);
                                }

                                localBoundsArrayU.push_back(startU);
                                localBoundsArrayU.push_back(prevU);
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

    void SetupCsrWithPermutationLoopProcessors(const BspSchedule<SparseMatrixImp<EigenIdxType>> &schedule, std::vector<UVertType> &perm) {
        const auto *const csr = instance_->GetComputationalDag().GetCSR();
        const EigenIdxType *const outer = csr->outerIndexPtr();
        const EigenIdxType *const inner = csr->innerIndexPtr();
        const double *const values = csr->valuePtr();

        const SparseMatrixImp<EigenIdxType> &graph = instance_->GetComputationalDag();
        assert(static_cast<std::size_t>(graph.NumVertices()) + static_cast<std::size_t>(graph.NumEdges()) <= static_cast<std::size_t>(std::numeric_limits<UVertType>::max()));
        const UVertType numVert = static_cast<UVertType>(graph.NumVertices());
        numSupersteps_ = schedule.NumberOfSupersteps();
        const unsigned numProcs = instance_->NumberOfProcessors();

        perm = std::vector<UVertType>(numVert, 0U);

        val_ = std::vector<double>(static_cast<std::size_t>(csr->nonZeros()));
        colIdx_ = std::vector<UVertType>(static_cast<std::size_t>(csr->nonZeros()));
        rowPtr_ = std::vector<UVertType>(numVert + 1U, 0U);

        procStepPtr_ = std::vector<std::vector<UVertType>>(numProcs, std::vector<UVertType>(numSupersteps_, 0U));
        procStepNum_ = std::vector<std::vector<UVertType>>(numProcs, std::vector<UVertType>(numSupersteps_, 0U));

        for (const auto vert : graph.Vertices()) {
            const unsigned whichStep = schedule.AssignedSuperstep(vert);
            const unsigned whichProc = schedule.AssignedProcessor(vert);

            perm[vert] = procStepNum_[whichProc][whichStep]++; // offsets
        }

        UVertType accNode = 0U;
        for (unsigned step = 0U; step < numSupersteps_; ++step) {
            for (unsigned proc = 0U; proc < numProcs; ++proc) {
                procStepPtr_[proc][step] = accNode;
                accNode += procStepNum_[proc][step];
            }
        }

        for (const auto vert : graph.Vertices()) {
            perm[vert] += procStepPtr_[schedule.AssignedProcessor(vert)][schedule.AssignedSuperstep(vert)];
        }

        std::vector<std::vector<UVertType>> entryAccumulation = std::vector<std::vector<UVertType>>(numProcs, std::vector<UVertType>(numSupersteps_, 0U));

        for (const auto vert : graph.Vertices()) {
            const unsigned whichStep = schedule.AssignedSuperstep(vert);
            const unsigned whichProc = schedule.AssignedProcessor(vert);

            rowPtr_[perm[vert]] = entryAccumulation[whichProc][whichStep];
            entryAccumulation[whichProc][whichStep] += static_cast<UVertType>(graph.InDegree(vert)) + 1;
        }

        UVertType accEntry = 0U;
        for (unsigned step = 0U; step < numSupersteps_; ++step) {
            for (unsigned proc = 0U; proc < numProcs; ++proc) {
                UVertType temp = entryAccumulation[proc][step];
                entryAccumulation[proc][step] = accEntry;
                accEntry += temp;
            }
        }
        rowPtr_[numVert] = accEntry;
        assert(static_cast<std::size_t>(accEntry) == static_cast<std::size_t>(graph.NumVertices()) + static_cast<std::size_t>(graph.NumEdges()) );

        for (const auto vert : graph.Vertices()) {
            rowPtr_[perm[vert]] += entryAccumulation[schedule.AssignedProcessor(vert)][schedule.AssignedSuperstep(vert)];
        }

        for (const auto vert : graph.Vertices()) {
            std::vector<std::pair<UVertType, UVertType>> parents;
            parents.reserve(graph.InDegree(vert));
            for (EigenIdxType edge = outer[vert]; edge < outer[vert + 1] - 1; ++edge) {
                parents.emplace_back(perm[static_cast<UVertType>(inner[edge])], static_cast<UVertType>(edge));
            }
            std::sort(parents.begin(), parents.end());

            const UVertType permVert = perm[vert];
            UVertType location = rowPtr_[permVert];
            for (const auto &[permPar, edgeIdx] : parents) {
                colIdx_[location] = permPar;
                val_[location] = values[edgeIdx];
                ++location;
            }
            colIdx_[location] = permVert;
            val_[location] = values[outer[vert + 1] - 1];
        }
    }

    void SetupCsrWithPermutationProcessorsFirst(const BspSchedule<SparseMatrixImp<EigenIdxType>> &schedule, std::vector<UVertType> &perm) {
        const auto *const csr = instance_->GetComputationalDag().GetCSR();
        const EigenIdxType *const outer = csr->outerIndexPtr();
        const EigenIdxType *const inner = csr->innerIndexPtr();
        const double *const values = csr->valuePtr();

        const SparseMatrixImp<EigenIdxType> &graph = instance_->GetComputationalDag();
        assert(static_cast<std::size_t>(graph.NumVertices()) + static_cast<std::size_t>(graph.NumEdges()) <= static_cast<std::size_t>(std::numeric_limits<UVertType>::max()));
        const UVertType numVert = static_cast<unsigned>(graph.NumVertices());
        numSupersteps_ = schedule.NumberOfSupersteps();
        const unsigned numProcs = instance_->NumberOfProcessors();

        perm = std::vector<UVertType>(numVert, 0U);

        val_ = std::vector<double>(static_cast<std::size_t>(csr->nonZeros()));
        colIdx_ = std::vector<UVertType>(static_cast<std::size_t>(csr->nonZeros()));
        rowPtr_ = std::vector<UVertType>(numVert + 1U, 0U);

        procFirstStepPtr_ = std::vector<UVertType>(0U);
        procFirstStepPtr_.reserve(numProcs + numSupersteps_ + 1U);

        procStepNum_ = std::vector<std::vector<UVertType>>(numProcs, std::vector<UVertType>(numSupersteps_, 0U));

        for (const auto vert : graph.Vertices()) {
            const unsigned whichStep = schedule.AssignedSuperstep(vert);
            const unsigned whichProc = schedule.AssignedProcessor(vert);

            perm[vert] = procStepNum_[whichProc][whichStep]++; // offsets
        }

        UVertType accNode = 0U;
        for (unsigned proc = 0U; proc < numProcs; ++proc) {
            for (unsigned step = 0U; step < numSupersteps_; ++step) {
                procFirstStepPtr_.emplace_back(accNode);
                accNode += procStepNum_[proc][step];
            }
        }
        procFirstStepPtr_.emplace_back(accNode);


        for (const auto vert : graph.Vertices()) {
            perm[vert] += procFirstStepPtr_[schedule.AssignedProcessor(vert) * numSupersteps_ + schedule.AssignedSuperstep(vert)];
        }

        std::vector<std::vector<UVertType>> entryAccumulation = std::vector<std::vector<UVertType>>(numProcs, std::vector<UVertType>(numSupersteps_, 0U));

        for (const auto vert : graph.Vertices()) {
            const unsigned whichStep = schedule.AssignedSuperstep(vert);
            const unsigned whichProc = schedule.AssignedProcessor(vert);

            rowPtr_[perm[vert]] = entryAccumulation[whichProc][whichStep];
            entryAccumulation[whichProc][whichStep] += static_cast<UVertType>(graph.InDegree(vert)) + 1;
        }

        UVertType accEntry = 0U;
        for (unsigned proc = 0U; proc < numProcs; ++proc) {
            for (unsigned step = 0U; step < numSupersteps_; ++step) {
                UVertType temp = entryAccumulation[proc][step];
                entryAccumulation[proc][step] = accEntry;
                accEntry += temp;
            }
        }
        rowPtr_[numVert] = accEntry;
        assert(static_cast<std::size_t>(accEntry) == static_cast<std::size_t>(graph.NumVertices()) + static_cast<std::size_t>(graph.NumEdges()) );

        for (const auto vert : graph.Vertices()) {
            rowPtr_[perm[vert]] += entryAccumulation[schedule.AssignedProcessor(vert)][schedule.AssignedSuperstep(vert)];
        }

        for (const auto vert : graph.Vertices()) {
            std::vector<std::pair<UVertType, UVertType>> parents;
            parents.reserve(graph.InDegree(vert));
            for (EigenIdxType edge = outer[vert]; edge < outer[vert + 1] - 1; ++edge) {
                parents.emplace_back(perm[static_cast<UVertType>(inner[edge])], static_cast<UVertType>(edge));
            }
            std::sort(parents.begin(), parents.end());

            const UVertType permVert = perm[vert];
            UVertType location = rowPtr_[permVert];
            for (const auto &[permPar, edgeIdx] : parents) {
                colIdx_[location] = permPar;
                val_[location] = values[edgeIdx];
                ++location;
            }
            colIdx_[location] = permVert;
            val_[location] = values[outer[vert + 1] - 1];
        }
    }

    void SetupCsrWithPermutation(const BspSchedule<SparseMatrixImp<EigenIdxType>> &schedule, std::vector<UVertType> &perm) {
        std::vector<UVertType> permInv(perm.size());
        for (UVertType i = 0; i < perm.size(); i++) {
            permInv[perm[i]] = i;
        }

        numSupersteps_ = schedule.NumberOfSupersteps();

        val_.clear();
        val_.reserve(static_cast<std::size_t>(instance_->GetComputationalDag().GetCSR()->nonZeros()));

        colIdx_.clear();
        colIdx_.reserve(static_cast<std::size_t>(instance_->GetComputationalDag().GetCSR()->nonZeros()));

        rowPtr_.clear();
        rowPtr_.reserve(instance_->NumberOfVertices() + 1);

        stepProcPtr_
            = std::vector<std::vector<UVertType>>(numSupersteps_, std::vector<UVertType>(instance_->NumberOfProcessors(), 0));

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

                stepProcPtr_[currentStep][currentProcessor] = static_cast<UVertType>(rowPtr_.size());
            }

            rowPtr_.push_back(static_cast<UVertType>(colIdx_.size()));

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
                    if (static_cast<UVertType>(instance_->GetComputationalDag().GetCSR()->innerIndexPtr()[parInd]) == permInv[par]) {
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

        rowPtr_.push_back(static_cast<UVertType>(colIdx_.size()));
    }

    void LsolveSerial() const {
        const EigenIdxType *const outer = (*(instance_->GetComputationalDag().GetCSR())).outerIndexPtr();
        const EigenIdxType *const inner = (*(instance_->GetComputationalDag().GetCSR())).innerIndexPtr();
        const double *const valPtr = (*(instance_->GetComputationalDag().GetCSR())).valuePtr();
        double *const x = x_;
        const double *const b = b_;
        const EigenIdxType numberOfVertices = static_cast<EigenIdxType>(instance_->NumberOfVertices());

        SpLTrSvSerial(numberOfVertices, x, b, outer, inner, valPtr);
    }

    void UsolveSerial() const {
        const EigenIdxType *const outer = (*(instance_->GetComputationalDag().GetCSC())).outerIndexPtr();
        const EigenIdxType *const inner = (*(instance_->GetComputationalDag().GetCSC())).innerIndexPtr();
        const double *const valPtr = (*(instance_->GetComputationalDag().GetCSC())).valuePtr();
        double *const x = x_;
        const double *const b = b_;

        const EigenIdxType numberOfVertices = static_cast<EigenIdxType>(instance_->NumberOfVertices());

        EigenIdxType i = numberOfVertices;
        do {
            i--;
            double acc = b[i];
            for (EigenIdxType j = outer[i] + 1; j < outer[i + 1]; ++j) {
                acc -= valPtr[j] * x[inner[j]];
            }
            x[i] = acc / valPtr[outer[i]];
        } while (i != 0);
    }

    void LsolveNoPermutationInPlace() const {
        const EigenIdxType *const outer = (*(instance_->GetComputationalDag().GetCSR())).outerIndexPtr();
        const EigenIdxType *const inner = (*(instance_->GetComputationalDag().GetCSR())).innerIndexPtr();
        const double *const valPtr = (*(instance_->GetComputationalDag().GetCSR())).valuePtr();
        double *const x = x_;

        SpLTrSvBSPParallelInPlace(x, outer, inner, valPtr, boundsArrayL_);
    }

    void UsolveNoPermutationInPlace() const {
        const EigenIdxType *const outer = (*(instance_->GetComputationalDag().GetCSC())).outerIndexPtr();
        const EigenIdxType *const inner = (*(instance_->GetComputationalDag().GetCSC())).innerIndexPtr();
        const double *const valPtr = (*(instance_->GetComputationalDag().GetCSC())).valuePtr();
        double *const x = x_;

#    pragma omp parallel num_threads(instance_->NumberOfProcessors())
        {
            // Process each superstep starting from the last one (opposite of lsolve)
            const std::size_t proc = static_cast<std::size_t>(omp_get_thread_num());
            const auto& procLocalBoundsArrayU = boundsArrayU_[proc];
            unsigned step = numSupersteps_;
            do {
                step--;
                const auto &localBoundsArrayU = procLocalBoundsArrayU[step];
                const std::size_t boundsStrSize = localBoundsArrayU.size();
                for (std::size_t index = 0; index < boundsStrSize; ++index) {
                    EigenIdxType node = localBoundsArrayU[index] + 1;
                    const EigenIdxType lowerB = localBoundsArrayU[++index];

                    do {
                        node--;
                        double acc = x[node];
                        for (EigenIdxType i = outer[node] + 1; i < outer[node + 1]; ++i) {
                            acc -= valPtr[i] * x[inner[i]];
                        }
                        x[node] = acc / valPtr[outer[node]];
                    } while (node != lowerB);
                }
#    pragma omp barrier
            } while (step != 0);
        }
    }

    void LsolveNoPermutation() const {
        const EigenIdxType *const outer = (*(instance_->GetComputationalDag().GetCSR())).outerIndexPtr();
        const EigenIdxType *const inner = (*(instance_->GetComputationalDag().GetCSR())).innerIndexPtr();
        const double *const valPtr = (*(instance_->GetComputationalDag().GetCSR())).valuePtr();
        double *const x = x_;
        const double *const b = b_;

        SpLTrSvBSPParallel(x, b, outer, inner, valPtr, boundsArrayL_);
    }

    void UsolveNoPermutation() const {
        const EigenIdxType *const outer = (*(instance_->GetComputationalDag().GetCSC())).outerIndexPtr();
        const EigenIdxType *const inner = (*(instance_->GetComputationalDag().GetCSC())).innerIndexPtr();
        const double *const valPtr = (*(instance_->GetComputationalDag().GetCSC())).valuePtr();
        double *const x = x_;
        const double *const b = b_;

#    pragma omp parallel num_threads(instance_->NumberOfProcessors())
        {
            // Process each superstep starting from the last one (opposite of lsolve)
            const std::size_t proc = static_cast<std::size_t>(omp_get_thread_num());
            const auto &procLocalBoundsArrayU = boundsArrayU_[proc];
            unsigned step = numSupersteps_;
            do {
                step--;
                const auto &localBoundsArrayU = procLocalBoundsArrayU[step];
                const std::size_t boundsStrSize = localBoundsArrayU.size();
                for (std::size_t index = 0; index < boundsStrSize; ++index) {
                    EigenIdxType node = localBoundsArrayU[index] + 1;
                    const EigenIdxType lowerB = localBoundsArrayU[++index];

                    do {
                        node--;
                        double acc = b[node];
                        for (EigenIdxType i = outer[node] + 1; i < outer[node + 1]; ++i) {
                            acc -= valPtr[i] * x[inner[i]];
                        }
                        x[node] = acc / valPtr[outer[node]];
                    } while (node != lowerB);
                }
#    pragma omp barrier
            } while (step != 0);
        }
    }

    void LsolveSerialInPlace() const {
        const EigenIdxType *const outer = (*(instance_->GetComputationalDag().GetCSR())).outerIndexPtr();
        const EigenIdxType *const inner = (*(instance_->GetComputationalDag().GetCSR())).innerIndexPtr();
        const double *const valPtr = (*(instance_->GetComputationalDag().GetCSR())).valuePtr();
        double *const x = x_;
        const EigenIdxType numberOfVertices = static_cast<EigenIdxType>(instance_->NumberOfVertices());

        SpLTrSvSerialInPlace(numberOfVertices, x, outer, inner, valPtr);
    }

    void UsolveSerialInPlace() const {
        const EigenIdxType *const outer = (*(instance_->GetComputationalDag().GetCSC())).outerIndexPtr();
        const EigenIdxType *const inner = (*(instance_->GetComputationalDag().GetCSC())).innerIndexPtr();
        const double *const valPtr = (*(instance_->GetComputationalDag().GetCSC())).valuePtr();
        double *const x = x_;

        const EigenIdxType numberOfVertices = static_cast<EigenIdxType>(instance_->NumberOfVertices());
        EigenIdxType i = numberOfVertices;
        do {
            i--;
            double acc = x[i];
            for (EigenIdxType j = outer[i] + 1; j < outer[i + 1]; ++j) {
                acc -= valPtr[j] * x[inner[j]];
            }
            x[i] = acc / valPtr[outer[i]];
        } while (i != 0);
    }

    void LsolveWithPermutationInPlace() const {
        double *const x = x_;

#    pragma omp parallel num_threads(instance_->NumberOfProcessors())
        {
            const std::size_t proc = static_cast<std::size_t>(omp_get_thread_num());
            for (unsigned step = 0; step < numSupersteps_; step++) {
                const UVertType upperLimit = procStepPtr_[proc][step] + procStepNum_[proc][step];
                for (UVertType rowIdx = procStepPtr_[proc][step]; rowIdx < upperLimit; rowIdx++) {
                    double acc = x[rowIdx];
                    for (UVertType i = rowPtr_[rowIdx]; i < rowPtr_[rowIdx + 1] - 1; i++) {
                        acc -= val_[i] * x[colIdx_[i]];
                    }

                    x[rowIdx] = acc / val_[rowPtr_[rowIdx + 1] - 1];
                }

#    pragma omp barrier
            }
        }
    }

    void LsolveWithProcFirstPermutationInPlace() const {
        double *const x = x_;

        SpLTrSvProcPermBSPParallelInPlace(x, rowPtr_.data(), colIdx_.data(), val_.data(), instance_->NumberOfProcessors(), numSupersteps_, procFirstStepPtr_.data());
    }

    void LsolveWithPermutation() const {
        double *const x = x_;
        const double *const b = b_;

#    pragma omp parallel num_threads(instance_->NumberOfProcessors())
        {
            for (unsigned step = 0; step < numSupersteps_; step++) {
                const std::size_t proc = static_cast<std::size_t>(omp_get_thread_num());
                const UVertType upperLimit = procStepPtr_[proc][step] + procStepNum_[proc][step];
                for (UVertType rowIdx = procStepPtr_[proc][step]; rowIdx < upperLimit; rowIdx++) {
                    double acc = b[rowIdx];
                    for (UVertType i = rowPtr_[rowIdx]; i < rowPtr_[rowIdx + 1] - 1; i++) {
                        acc -= val_[i] * x[colIdx_[i]];
                    }

                    x[rowIdx] = acc / val_[rowPtr_[rowIdx + 1] - 1];
                }

#    pragma omp barrier
            }
        }
    }

    template <unsigned staleness = 2U>
    void SspLsolveStalenessWithPermutationInPlace() const {
        const unsigned nthreads = instance_->NumberOfProcessors();
        FlatCheckpointCounterBarrier barrier(nthreads);

        const auto *const csr = instance_->GetComputationalDag().GetCSR();
        const EigenIdxType *const outer = csr->outerIndexPtr();
        const EigenIdxType *const inner = csr->innerIndexPtr();
        const double *const vals = csr->valuePtr();
        double *const x = x_;

#    pragma omp parallel num_threads(nthreads)
        {
            const std::size_t proc = static_cast<std::size_t>(omp_get_thread_num());
            for (unsigned step = 0; step < numSupersteps_; ++step) {
                if (procStepNum_[proc][step] > 0U) {
                    barrier.Wait(proc, staleness - 1U);
                }

                const UVertType upperLimit = procStepPtr_[proc][step] + procStepNum_[proc][step];
                for (UVertType rowIdx = procStepPtr_[proc][step]; rowIdx < upperLimit; rowIdx++) {
                    double acc = x[rowIdx];
                    for (UVertType i = rowPtr_[rowIdx]; i < rowPtr_[rowIdx + 1] - 1; i++) {
                        acc -= val_[i] * x[colIdx_[i]];
                    }

                    x[rowIdx] = acc / val_[rowPtr_[rowIdx + 1] - 1];
                }
                // Signal completion of this superstep.
                barrier.Arrive(proc);
            }
        }
    }

    template <unsigned staleness = 2U>
    void SspLsolveStalenessWithProcFirstPermutationInPlace() const {
        double *const x = x_;

        SpLTrSvProcPermSSPParallelInPlace<UVertType, staleness>(x, rowPtr_.data(), colIdx_.data(), val_.data(), instance_->NumberOfProcessors(), numSupersteps_, procFirstStepPtr_.data());
    }

    void ResetX() {
        const EigenIdxType numberOfVertices = static_cast<EigenIdxType>(instance_->NumberOfVertices());
        for (EigenIdxType i = 0; i < numberOfVertices; i++) {
            x_[i] = 1.0;
        }
    }

    template<typename IntegralType>
    void PermuteXVector(const std::vector<IntegralType> &perm) {
        static_assert(std::is_integral_v<IntegralType>);
        std::vector<double> vecPerm(perm.size());
        for (IntegralType i = 0; i < perm.size(); i++) {
            vecPerm[i] = x_[perm[i]];
        }
        for (IntegralType i = 0; i < perm.size(); i++) {
            x_[i] = vecPerm[i];
        }
    }

    void PermuteXVectorInverse(const std::vector<UVertType> &perm) {
        std::vector<double> vecUnperm(perm.size());
        for (UVertType i = 0; i < perm.size(); i++) {
            vecUnperm[perm[i]] = x_[i];
        }
        for (UVertType i = 0; i < perm.size(); i++) {
            x_[i] = vecUnperm[i];
        }
    }

    UVertType GetNumberOfVertices() const { return instance_->NumberOfVertices(); }

    // SSP Lsolve with staleness=2 (allowing at most one superstep of lag).
    // Uses FlatCheckpointCounterBarrier created internally.
    template <unsigned staleness = 2U>
    void SspLsolveStaleness() const {
        const EigenIdxType *const outer = (*(instance_->GetComputationalDag().GetCSR())).outerIndexPtr();
        const EigenIdxType *const inner = (*(instance_->GetComputationalDag().GetCSR())).innerIndexPtr();
        const double *const valPtr = (*(instance_->GetComputationalDag().GetCSR())).valuePtr();
        double *const x = x_;
        const double *const b = b_;

        SpLTrSvSSPParallel<EigenIdxType, staleness>(x, b, outer, inner, valPtr, boundsArrayL_);
    }

    // SSP Lsolve in-place with staleness=2 (allowing at most one superstep of lag).
    // Uses FlatCheckpointCounterBarrier created internally.
    template <unsigned staleness = 2U>
    void SspLsolveStalenessInPlace() const {
        const EigenIdxType *const outer = (*(instance_->GetComputationalDag().GetCSR())).outerIndexPtr();
        const EigenIdxType *const inner = (*(instance_->GetComputationalDag().GetCSR())).innerIndexPtr();
        const double *const valPtr = (*(instance_->GetComputationalDag().GetCSR())).valuePtr();
        double *const x = x_;

        SpLTrSvSSPParallelInPlace<EigenIdxType, staleness>(x, outer, inner, valPtr, boundsArrayL_);
    }

    // SSP Usolve with configurable staleness.
    // Uses FlatCheckpointCounterBarrier created internally.
    template <unsigned staleness = 2U>
    void SspUsolveStaleness() const {
        const unsigned nthreads = instance_->NumberOfProcessors();
        FlatCheckpointCounterBarrier barrier(nthreads);

        const auto *const csc = instance_->GetComputationalDag().GetCSC();
        const EigenIdxType *const outer = csc->outerIndexPtr();
        const EigenIdxType *const inner = csc->innerIndexPtr();
        const double *const vals = csc->valuePtr();
        double *const x = x_;
        const double *const b = b_;

#    pragma omp parallel num_threads(nthreads)
        {
            const std::size_t proc = static_cast<std::size_t>(omp_get_thread_num());
            const auto &procLocalBoundsArrayU = boundsArrayU_[proc];
            unsigned step = numSupersteps_;
            do {
                step--;
                const auto &localBoundsArrayU = procLocalBoundsArrayU[step];
                const std::size_t boundsStrSize = localBoundsArrayU.size();
                if (boundsStrSize > 0U) {
                    barrier.Wait(proc, staleness - 1U);
                }

                for (std::size_t index = 0; index < boundsStrSize; ++index) {
                    EigenIdxType node = localBoundsArrayU[index] + 1;
                    const EigenIdxType lowerB = localBoundsArrayU[++index];

                    do {
                        node--;
                        double acc = b[node];
                        for (EigenIdxType i = outer[node] + 1; i < outer[node + 1]; ++i) {
                            acc -= vals[i] * x[inner[i]];
                        }
                        x[node] = acc / vals[outer[node]];
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
