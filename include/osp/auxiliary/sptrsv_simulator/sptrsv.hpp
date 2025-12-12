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
#    include <iostream>
#    include <list>
#    include <map>
#    include <random>
#    include <stdexcept>
#    include <vector>

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
    std::vector<double> val;
    std::vector<double> cscVal;

    std::vector<UVertType> colIdx;
    std::vector<UVertType> rowPtr;

    std::vector<UVertType> rowIdx;
    std::vector<UVertType> colPtr;

    std::vector<std::vector<unsigned>> stepProcPtr;
    std::vector<std::vector<unsigned>> stepProcNum;

    double *x;
    const double *b;

    unsigned numSupersteps;

    std::vector<std::vector<std::vector<EigenIdxType>>> vectorStepProcessorVertices;
    std::vector<std::vector<std::vector<EigenIdxType>>> vectorStepProcessorVerticesU;
    std::vector<int> ready;

    std::vector<std::vector<std::vector<EigenIdxType>>> boundsArrayL;
    std::vector<std::vector<std::vector<EigenIdxType>>> boundsArrayU;

    Sptrsv() = default;

    Sptrsv(BspInstance<SparseMatrixImp<EigenIdxType>> &inst) : instance_(&inst) {};

    void SetupCsrNoPermutation(const BspSchedule<SparseMatrixImp<EigenIdxType>> &schedule) {
        vectorStepProcessorVertices = std::vector<std::vector<std::vector<EigenIdxType>>>(
            schedule.NumberOfSupersteps(), std::vector<std::vector<EigenIdxType>>(schedule.GetInstance().NumberOfProcessors()));

        vectorStepProcessorVerticesU = std::vector<std::vector<std::vector<EigenIdxType>>>(
            schedule.NumberOfSupersteps(), std::vector<std::vector<EigenIdxType>>(schedule.GetInstance().NumberOfProcessors()));

        boundsArrayL = std::vector<std::vector<std::vector<EigenIdxType>>>(
            schedule.NumberOfSupersteps(), std::vector<std::vector<EigenIdxType>>(schedule.GetInstance().NumberOfProcessors()));
        boundsArrayU = std::vector<std::vector<std::vector<EigenIdxType>>>(
            schedule.NumberOfSupersteps(), std::vector<std::vector<EigenIdxType>>(schedule.GetInstance().NumberOfProcessors()));

        numSupersteps = schedule.NumberOfSupersteps();
        size_t numberOfVertices = instance_->GetComputationalDag().NumVertices();

#    pragma omp parallel num_threads(2)
        {
            int id = omp_get_thread_num();
            switch (id) {
                case 0: {
                    for (size_t node = 0; node < numberOfVertices; ++node) {
                        vectorStepProcessorVertices[schedule.AssignedSuperstep(node)][schedule.AssignedProcessor(node)].push_back(
                            static_cast<EigenIdxType>(node));
                    }

                    for (unsigned int step = 0; step < schedule.NumberOfSupersteps(); ++step) {
                        for (unsigned int proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
                            if (!vectorStepProcessorVertices[step][proc].empty()) {
                                EigenIdxType start = vectorStepProcessorVertices[step][proc][0];
                                EigenIdxType prev = vectorStepProcessorVertices[step][proc][0];

                                for (size_t i = 1; i < vectorStepProcessorVertices[step][proc].size(); ++i) {
                                    if (vectorStepProcessorVertices[step][proc][i] != prev + 1) {
                                        boundsArrayL[step][proc].push_back(start);
                                        boundsArrayL[step][proc].push_back(prev);
                                        start = vectorStepProcessorVertices[step][proc][i];
                                    }
                                    prev = vectorStepProcessorVertices[step][proc][i];
                                }

                                boundsArrayL[step][proc].push_back(start);
                                boundsArrayL[step][proc].push_back(prev);
                            }
                        }
                    }

                    break;
                }
                case 1: {
                    size_t node = numberOfVertices;
                    do {
                        node--;
                        vectorStepProcessorVerticesU[schedule.AssignedSuperstep(node)][schedule.AssignedProcessor(node)].push_back(
                            static_cast<EigenIdxType>(node));
                    } while (node > 0);

                    for (unsigned int step = 0; step < schedule.NumberOfSupersteps(); ++step) {
                        for (unsigned int proc = 0; proc < instance_->NumberOfProcessors(); ++proc) {
                            if (!vectorStepProcessorVerticesU[step][proc].empty()) {
                                EigenIdxType startU = static_cast<EigenIdxType>(vectorStepProcessorVerticesU[step][proc][0]);
                                EigenIdxType prevU = static_cast<EigenIdxType>(vectorStepProcessorVerticesU[step][proc][0]);

                                for (size_t i = 1; i < vectorStepProcessorVerticesU[step][proc].size(); ++i) {
                                    if (static_cast<EigenIdxType>(vectorStepProcessorVerticesU[step][proc][i]) != prevU - 1) {
                                        boundsArrayU[step][proc].push_back(startU);
                                        boundsArrayU[step][proc].push_back(prevU);
                                        startU = static_cast<EigenIdxType>(vectorStepProcessorVerticesU[step][proc][i]);
                                    }
                                    prevU = static_cast<EigenIdxType>(vectorStepProcessorVerticesU[step][proc][i]);
                                }

                                boundsArrayU[step][proc].push_back(startU);
                                boundsArrayU[step][proc].push_back(prevU);
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

        numSupersteps = schedule.NumberOfSupersteps();

        val.clear();
        val.reserve(static_cast<size_t>(instance_->GetComputationalDag().GetCsr()->nonZeros()));

        colIdx.clear();
        colIdx.reserve(static_cast<size_t>(instance_->GetComputationalDag().GetCsr()->nonZeros()));

        rowPtr.clear();
        rowPtr.reserve(instance_->NumberOfVertices() + 1);

        stepProcPtr = std::vector<std::vector<unsigned>>(numSupersteps, std::vector<unsigned>(instance_->NumberOfProcessors(), 0));

        stepProcNum = schedule.NumAssignedNodesPerSuperstepProcessor();

        unsigned currentStep = 0;
        unsigned currentProcessor = 0;

        stepProcPtr[currentStep][currentProcessor] = 0;

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

                stepProcPtr[currentStep][currentProcessor] = static_cast<unsigned>(rowPtr.size());
            }

            rowPtr.push_back(colIdx.size());

            std::set<UVertType> parents;

            for (UVertType par : instance_->GetComputationalDag().Parents(node)) {
                parents.insert(perm[par]);
            }

            for (const UVertType &par : parents) {
                colIdx.push_back(par);
                unsigned found = 0;

                const auto *outer = instance_->GetComputationalDag().GetCsr()->outerIndexPtr();
                for (UVertType parInd = static_cast<UVertType>(outer[node]); parInd < static_cast<UVertType>(outer[node + 1] - 1);
                     ++parInd) {
                    if (static_cast<size_t>(instance_->GetComputationalDag().GetCsr()->innerIndexPtr()[parInd]) == permInv[par]) {
                        val.push_back(instance_->GetComputationalDag().GetCsr()->valuePtr()[parInd]);
                        found++;
                    }
                }
                assert(found == 1);
            }

            colIdx.push_back(perm[node]);
            val.push_back(instance_->GetComputationalDag()
                              .GetCsr()
                              ->valuePtr()[instance_->GetComputationalDag().GetCsr()->outerIndexPtr()[node + 1] - 1]);
        }

        rowPtr.push_back(colIdx.size());
    }

    void LsolveSerial() {
        EigenIdxType numberOfVertices = static_cast<EigenIdxType>(instance_->NumberOfVertices());
        for (EigenIdxType i = 0; i < numberOfVertices; ++i) {
            x[i] = b[i];
            for (EigenIdxType j = (*(instance_->GetComputationalDag().GetCsr())).outerIndexPtr()[i];
                 j < (*(instance_->GetComputationalDag().GetCsr())).outerIndexPtr()[i + 1] - 1;
                 ++j) {
                x[i] -= (*(instance_->GetComputationalDag().GetCsr())).valuePtr()[j]
                        * x[(*(instance_->GetComputationalDag().GetCsr())).innerIndexPtr()[j]];
            }
            x[i] /= (*(instance_->GetComputationalDag().GetCsr()))
                        .valuePtr()[(*(instance_->GetComputationalDag().GetCsr())).outerIndexPtr()[i + 1] - 1];
        }
    }

    void UsolveSerial() {
        EigenIdxType numberOfVertices = static_cast<EigenIdxType>(instance_->NumberOfVertices());

        EigenIdxType i = numberOfVertices;
        do {
            i--;
            x[i] = b[i];
            for (EigenIdxType j = (*(instance_->GetComputationalDag().GetCsc())).outerIndexPtr()[i] + 1;
                 j < (*(instance_->GetComputationalDag().GetCsc())).outerIndexPtr()[i + 1];
                 ++j) {
                x[i] -= (*(instance_->GetComputationalDag().GetCsc())).valuePtr()[j]
                        * x[(*(instance_->GetComputationalDag().GetCsc())).innerIndexPtr()[j]];
            }
            x[i] /= (*(instance_->GetComputationalDag().GetCsc()))
                        .valuePtr()[(*(instance_->GetComputationalDag().GetCsc())).outerIndexPtr()[i]];
        } while (i != 0);
    }

    void LsolveNoPermutationInPlace() {
#    pragma omp parallel num_threads(instance_->NumberOfProcessors())
        {
            const size_t proc = static_cast<size_t>(omp_get_thread_num());
            for (unsigned step = 0; step < numSupersteps; ++step) {
                const size_t boundsStrSize = boundsArrayL[step][proc].size();

                for (size_t index = 0; index < boundsStrSize; index += 2) {
                    EigenIdxType lowerB = boundsArrayL[step][proc][index];
                    const EigenIdxType upperB = boundsArrayL[step][proc][index + 1];

                    for (EigenIdxType node = lowerB; node <= upperB; ++node) {
                        for (EigenIdxType i = (*(instance_->GetComputationalDag().GetCsr())).outerIndexPtr()[node];
                             i < (*(instance_->GetComputationalDag().GetCsr())).outerIndexPtr()[node + 1] - 1;
                             ++i) {
                            x[node] -= (*(instance_->GetComputationalDag().GetCsr())).valuePtr()[i]
                                       * x[(*(instance_->GetComputationalDag().GetCsr())).innerIndexPtr()[i]];
                        }
                        x[node] /= (*(instance_->GetComputationalDag().GetCsr()))
                                       .valuePtr()[(*(instance_->GetComputationalDag().GetCsr())).outerIndexPtr()[node + 1] - 1];
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
            unsigned step = numSupersteps;
            do {
                step--;
                const size_t boundsStrSize = boundsArrayU[step][proc].size();
                for (size_t index = 0; index < boundsStrSize; index += 2) {
                    EigenIdxType node = boundsArrayU[step][proc][index] + 1;
                    const EigenIdxType lowerB = boundsArrayU[step][proc][index + 1];

                    do {
                        node--;
                        for (EigenIdxType i = (*(instance_->GetComputationalDag().GetCsc())).outerIndexPtr()[node] + 1;
                             i < (*(instance_->GetComputationalDag().GetCsc())).outerIndexPtr()[node + 1];
                             ++i) {
                            x[node] -= (*(instance_->GetComputationalDag().GetCsc())).valuePtr()[i]
                                       * x[(*(instance_->GetComputationalDag().GetCsc())).innerIndexPtr()[i]];
                        }
                        x[node] /= (*(instance_->GetComputationalDag().GetCsc()))
                                       .valuePtr()[(*(instance_->GetComputationalDag().GetCsc())).outerIndexPtr()[node]];
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
            for (unsigned step = 0; step < numSupersteps; ++step) {
                const size_t boundsStrSize = boundsArrayL[step][proc].size();

                for (size_t index = 0; index < boundsStrSize; index += 2) {
                    EigenIdxType lowerB = boundsArrayL[step][proc][index];
                    const EigenIdxType upperB = boundsArrayL[step][proc][index + 1];

                    for (EigenIdxType node = lowerB; node <= upperB; ++node) {
                        x[node] = b[node];
                        for (EigenIdxType i = (*(instance_->GetComputationalDag().GetCsr())).outerIndexPtr()[node];
                             i < (*(instance_->GetComputationalDag().GetCsr())).outerIndexPtr()[node + 1] - 1;
                             ++i) {
                            x[node] -= (*(instance_->GetComputationalDag().GetCsr())).valuePtr()[i]
                                       * x[(*(instance_->GetComputationalDag().GetCsr())).innerIndexPtr()[i]];
                        }
                        x[node] /= (*(instance_->GetComputationalDag().GetCsr()))
                                       .valuePtr()[(*(instance_->GetComputationalDag().GetCsr())).outerIndexPtr()[node + 1] - 1];
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
            unsigned step = numSupersteps;
            do {
                step--;
                const size_t boundsStrSize = boundsArrayU[step][proc].size();
                for (size_t index = 0; index < boundsStrSize; index += 2) {
                    EigenIdxType node = boundsArrayU[step][proc][index] + 1;
                    const EigenIdxType lowerB = boundsArrayU[step][proc][index + 1];

                    do {
                        node--;
                        x[node] = b[node];
                        for (EigenIdxType i = (*(instance_->GetComputationalDag().GetCsc())).outerIndexPtr()[node] + 1;
                             i < (*(instance_->GetComputationalDag().GetCsc())).outerIndexPtr()[node + 1];
                             ++i) {
                            x[node] -= (*(instance_->GetComputationalDag().GetCsc())).valuePtr()[i]
                                       * x[(*(instance_->GetComputationalDag().GetCsc())).innerIndexPtr()[i]];
                        }
                        x[node] /= (*(instance_->GetComputationalDag().GetCsc()))
                                       .valuePtr()[(*(instance_->GetComputationalDag().GetCsc())).outerIndexPtr()[node]];
                    } while (node != lowerB);
                }
#    pragma omp barrier
            } while (step != 0);
        }
    }

    void LsolveSerialInPlace() {
        EigenIdxType numberOfVertices = static_cast<EigenIdxType>(instance_->NumberOfVertices());
        for (EigenIdxType i = 0; i < numberOfVertices; ++i) {
            for (EigenIdxType j = (*(instance_->GetComputationalDag().GetCsr())).outerIndexPtr()[i];
                 j < (*(instance_->GetComputationalDag().GetCsr())).outerIndexPtr()[i + 1] - 1;
                 ++j) {
                x[i] -= (*(instance_->GetComputationalDag().GetCsr())).valuePtr()[j]
                        * x[(*(instance_->GetComputationalDag().GetCsr())).innerIndexPtr()[j]];
            }
            x[i] /= (*(instance_->GetComputationalDag().GetCsr()))
                        .valuePtr()[(*(instance_->GetComputationalDag().GetCsr())).outerIndexPtr()[i + 1] - 1];
        }
    }

    void UsolveSerialInPlace() {
        EigenIdxType numberOfVertices = static_cast<EigenIdxType>(instance_->NumberOfVertices());
        EigenIdxType i = numberOfVertices;
        do {
            i--;
            for (EigenIdxType j = (*(instance_->GetComputationalDag().GetCsc())).outerIndexPtr()[i] + 1;
                 j < (*(instance_->GetComputationalDag().GetCsc())).outerIndexPtr()[i + 1];
                 ++j) {
                x[i] -= (*(instance_->GetComputationalDag().GetCsc())).valuePtr()[j]
                        * x[(*(instance_->GetComputationalDag().GetCsc())).innerIndexPtr()[j]];
            }
            x[i] /= (*(instance_->GetComputationalDag().GetCsc()))
                        .valuePtr()[(*(instance_->GetComputationalDag().GetCsc())).outerIndexPtr()[i]];
        } while (i != 0);
    }

    void LsolveWithPermutationInPlace() {
#    pragma omp parallel num_threads(instance_->NumberOfProcessors())
        {
            for (unsigned step = 0; step < numSupersteps; step++) {
                const size_t proc = static_cast<size_t>(omp_get_thread_num());
                const UVertType upperLimit = stepProcPtr[step][proc] + stepProcNum[step][proc];
                for (UVertType rowIdx_ = stepProcPtr[step][proc]; rowIdx_ < upperLimit; rowIdx_++) {
                    for (UVertType i = rowPtr[rowIdx_]; i < rowPtr[rowIdx_ + 1] - 1; i++) {
                        x[rowIdx_] -= val[i] * x[colIdx[i]];
                    }

                    x[rowIdx_] /= val[rowPtr[rowIdx_ + 1] - 1];
                }

#    pragma omp barrier
            }
        }
    }

    void LsolveWithPermutation() {
#    pragma omp parallel num_threads(instance_->NumberOfProcessors())
        {
            for (unsigned step = 0; step < numSupersteps; step++) {
                const size_t proc = static_cast<size_t>(omp_get_thread_num());
                const UVertType upperLimit = stepProcPtr[step][proc] + stepProcNum[step][proc];
                for (UVertType rowIdx_ = stepProcPtr[step][proc]; rowIdx_ < upperLimit; rowIdx_++) {
                    x[rowIdx_] = b[rowIdx_];
                    for (UVertType i = rowPtr[rowIdx_]; i < rowPtr[rowIdx_ + 1] - 1; i++) {
                        x[rowIdx_] -= val[i] * x[colIdx[i]];
                    }

                    x[rowIdx_] /= val[rowPtr[rowIdx_ + 1] - 1];
                }

#    pragma omp barrier
            }
        }
    }

    void ResetX() {
        EigenIdxType numberOfVertices = static_cast<EigenIdxType>(instance_->numberOfVertices());
        for (EigenIdxType i = 0; i < numberOfVertices; i++) {
            x[i] = 1.0;
        }
    }

    void PermuteXVector(const std::vector<size_t> &perm) {
        std::vector<double> vecPerm(perm.size());
        for (size_t i = 0; i < perm.size(); i++) {
            vecPerm[i] = x[perm[i]];
        }
        for (size_t i = 0; i < perm.size(); i++) {
            x[i] = vecPerm[i];
        }
    }

    void PermuteXVectorInverse(const std::vector<size_t> &perm) {
        std::vector<double> vecUnperm(perm.size());
        for (size_t i = 0; i < perm.size(); i++) {
            vecUnperm[perm[i]] = x[i];
        }
        for (size_t i = 0; i < perm.size(); i++) {
            x[i] = vecUnperm[i];
        }
    }

    std::size_t GetNumberOfVertices() { return instance_->numberOfVertices(); }

    virtual ~Sptrsv() = default;
};

}    // namespace osp

#endif
