/*
Copyright 2026 Huawei Technologies Co., Ltd.

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

#include <omp.h>

#include <cstddef>
#include <type_traits>
#include <vector>

#include "osp/auxiliary/sptrsv_simulator/WeakBarriers/flat_checkpoint_counter_barrier.hpp"

namespace osp {

template <typename IdxType>
void SpLTrSvSerial(const IdxType N,
                   double *__restrict__ const x,
                   const double *__restrict__ const b,
                   const IdxType *__restrict__ const outer,
                   const IdxType *__restrict__ const inner,
                   const double *__restrict__ const val) {
    static_assert(std::is_integral_v<IdxType>);

    for (IdxType row = 0; row < N; ++row) {
        double acc = b[row];
        for (IdxType entryIdx = outer[row]; entryIdx < outer[row + 1] - 1; ++entryIdx) {
            acc -= val[entryIdx] * x[inner[entryIdx]];
        }
        x[row] = acc / val[outer[row + 1] - 1];
    }
}

template <typename IdxType>
void SpLTrSvSerialInPlace(const IdxType N,
                          double *__restrict__ const x,
                          const IdxType *__restrict__ const outer,
                          const IdxType *__restrict__ const inner,
                          const double *__restrict__ const val) {
    static_assert(std::is_integral_v<IdxType>);

    for (IdxType row = 0; row < N; ++row) {
        double acc = x[row];
        for (IdxType entryIdx = outer[row]; entryIdx < outer[row + 1] - 1; ++entryIdx) {
            acc -= val[entryIdx] * x[inner[entryIdx]];
        }
        x[row] = acc / val[outer[row + 1] - 1];
    }
}

template <typename IdxType>
void SpLTrSvBSPParallel(double *__restrict__ const x,
                        const double *__restrict__ const b,
                        const IdxType *__restrict__ const outer,
                        const IdxType *__restrict__ const inner,
                        const double *__restrict__ const val,
                        const std::vector<std::vector<std::vector<IdxType>>> &BoundsProcStepIdx) {
    static_assert(std::is_integral_v<IdxType>);

#pragma omp parallel num_threads(BoundsProcStepIdx.size())
    {
        const std::size_t proc = static_cast<std::size_t>(omp_get_thread_num());
        const std::vector<std::vector<IdxType>> &BoundsStepIdx = BoundsProcStepIdx[proc];
        const std::size_t numSuperSteps = BoundsStepIdx.size();

        for (std::size_t step = 0U; step < numSuperSteps; ++step) {
            const std::vector<IdxType> &BoundsIdx = BoundsStepIdx[step];
            const auto idxItEnd = BoundsIdx.cend();
            for (auto idxIt = BoundsIdx.cbegin(); idxIt != idxItEnd; ++idxIt) {
                IdxType row = *idxIt;
                const IdxType ubRow = *(++idxIt);
                for (; row <= ubRow; ++row) {
                    double acc = b[row];
                    for (IdxType entryIdx = outer[row]; entryIdx < outer[row + 1] - 1; ++entryIdx) {
                        acc -= val[entryIdx] * x[inner[entryIdx]];
                    }
                    x[row] = acc / val[outer[row + 1] - 1];
                }
            }
#pragma omp barrier
        }
    }
}

template <typename IdxType>
void SpLTrSvBSPParallelInPlace(double *__restrict__ const x,
                               const IdxType *__restrict__ const outer,
                               const IdxType *__restrict__ const inner,
                               const double *__restrict__ const val,
                               const std::vector<std::vector<std::vector<IdxType>>> &BoundsProcStepIdx) {
    static_assert(std::is_integral_v<IdxType>);

#pragma omp parallel num_threads(BoundsProcStepIdx.size())
    {
        const std::size_t proc = static_cast<std::size_t>(omp_get_thread_num());
        const std::vector<std::vector<IdxType>> &BoundsStepIdx = BoundsProcStepIdx[proc];
        const std::size_t numSuperSteps = BoundsStepIdx.size();

        for (std::size_t step = 0U; step < numSuperSteps; ++step) {
            const std::vector<IdxType> &BoundsIdx = BoundsStepIdx[step];
            const auto idxItEnd = BoundsIdx.cend();
            for (auto idxIt = BoundsIdx.cbegin(); idxIt != idxItEnd; ++idxIt) {
                IdxType row = *idxIt;
                const IdxType ubRow = *(++idxIt);
                for (; row <= ubRow; ++row) {
                    double acc = x[row];
                    for (IdxType entryIdx = outer[row]; entryIdx < outer[row + 1] - 1; ++entryIdx) {
                        acc -= val[entryIdx] * x[inner[entryIdx]];
                    }
                    x[row] = acc / val[outer[row + 1] - 1];
                }
            }
#pragma omp barrier
        }
    }
}

template <typename IdxType, unsigned staleness = 2U>
void SpLTrSvSSPParallel(double *__restrict__ const x,
                        const double *__restrict__ const b,
                        const IdxType *__restrict__ const outer,
                        const IdxType *__restrict__ const inner,
                        const double *__restrict__ const val,
                        const std::vector<std::vector<std::vector<IdxType>>> &BoundsProcStepIdx) {
    static_assert(std::is_integral_v<IdxType>);

    const std::size_t nthreads = BoundsProcStepIdx.size();
    FlatCheckpointCounterBarrier barrier(nthreads);

#pragma omp parallel num_threads(nthreads)
    {
        const std::size_t proc = static_cast<std::size_t>(omp_get_thread_num());
        const std::vector<std::vector<IdxType>> &BoundsStepIdx = BoundsProcStepIdx[proc];
        for (std::size_t step = 0; step < BoundsStepIdx.size(); ) {
            const std::vector<IdxType> &BoundsIdx = BoundsStepIdx[step];
            auto idxIt = BoundsIdx.cbegin();
            const auto idxItEnd = BoundsIdx.cend();

            if (idxIt != idxItEnd) {
                constexpr std::size_t diff = staleness - 1U;
                const std::size_t minStep = std::max(step, diff) - diff;
                barrier.Wait(proc, minStep);
            }

            for (; idxIt != idxItEnd; ++idxIt) {
                IdxType row = *idxIt;
                const IdxType ubRow = *(++idxIt);
                for (; row <= ubRow; ++row) {
                    double acc = b[row];
                    for (IdxType entryIdx = outer[row]; entryIdx < outer[row + 1] - 1; ++entryIdx) {
                        acc -= val[entryIdx] * x[inner[entryIdx]];
                    }
                    x[row] = acc / val[outer[row + 1] - 1];
                }
            }
            barrier.Arrive(proc, ++step);
        }
    }
}

template <typename IdxType, unsigned staleness = 2U>
void SpLTrSvSSPParallelInPlace(double *__restrict__ const x,
                               const IdxType *__restrict__ const outer,
                               const IdxType *__restrict__ const inner,
                               const double *__restrict__ const val,
                               const std::vector<std::vector<std::vector<IdxType>>> &BoundsProcStepIdx) {
    static_assert(std::is_integral_v<IdxType>);

    const std::size_t nthreads = BoundsProcStepIdx.size();
    FlatCheckpointCounterBarrier barrier(nthreads);

#pragma omp parallel num_threads(nthreads)
    {
        const std::size_t proc = static_cast<std::size_t>(omp_get_thread_num());
        const std::vector<std::vector<IdxType>> &BoundsStepIdx = BoundsProcStepIdx[proc];
        for (std::size_t step = 0; step < BoundsStepIdx.size();) {
            const std::vector<IdxType> &BoundsIdx = BoundsStepIdx[step];
            auto idxIt = BoundsIdx.cbegin();
            const auto idxItEnd = BoundsIdx.cend();

            if (idxIt != idxItEnd) {
                constexpr std::size_t diff = staleness - 1U;
                const std::size_t minStep = std::max(step, diff) - diff;
                barrier.Wait(proc, minStep);
            }

            for (; idxIt != idxItEnd; ++idxIt) {
                IdxType row = *idxIt;
                const IdxType ubRow = *(++idxIt);
                for (; row <= ubRow; ++row) {
                    double acc = x[row];
                    for (IdxType entryIdx = outer[row]; entryIdx < outer[row + 1] - 1; ++entryIdx) {
                        acc -= val[entryIdx] * x[inner[entryIdx]];
                    }
                    x[row] = acc / val[outer[row + 1] - 1];
                }
            }
            barrier.Arrive(proc, ++step);
        }
    }
}

template <typename IdxType>
void SpLTrSvProcPermBSPParallel(double *__restrict__ const x,
                                const double *__restrict__ const b,
                                const IdxType *__restrict__ const outer,
                                const IdxType *__restrict__ const inner,
                                const double *__restrict__ const val,
                                const unsigned numProcs,
                                const unsigned numSuperSteps,
                                const IdxType *__restrict__ const procStepPtr) {
    static_assert(std::is_integral_v<IdxType>);

#pragma omp parallel num_threads(numProcs)
    {
        const unsigned proc = static_cast<unsigned>(omp_get_thread_num());
        const IdxType *const endStepPtr = procStepPtr + ((proc + 1U) * numSuperSteps);
        for (const IdxType *stepPtr = procStepPtr + (proc * numSuperSteps); stepPtr != endStepPtr;) {
            IdxType row = *stepPtr;
            const IdxType endRow = *(++stepPtr);
            for (; row != endRow; ++row) {
                double acc = b[row];
                for (IdxType entryIdx = outer[row]; entryIdx < outer[row + 1] - 1; ++entryIdx) {
                    acc -= val[entryIdx] * x[inner[entryIdx]];
                }

                x[row] = acc / val[outer[row + 1] - 1];
            }

#pragma omp barrier
        }
    }
}

template <typename IdxType>
void SpLTrSvProcPermBSPParallelInPlace(double *__restrict__ const x,
                                       const IdxType *__restrict__ const outer,
                                       const IdxType *__restrict__ const inner,
                                       const double *__restrict__ const val,
                                       const unsigned numProcs,
                                       const unsigned numSuperSteps,
                                       const IdxType *__restrict__ const procStepPtr) {
    static_assert(std::is_integral_v<IdxType>);

#pragma omp parallel num_threads(numProcs)
    {
        const unsigned proc = static_cast<unsigned>(omp_get_thread_num());
        const IdxType *const endStepPtr = procStepPtr + ((proc + 1U) * numSuperSteps);
        for (const IdxType *stepPtr = procStepPtr + (proc * numSuperSteps); stepPtr != endStepPtr;) {
            IdxType row = *stepPtr;
            const IdxType endRow = *(++stepPtr);
            for (; row != endRow; ++row) {
                double acc = x[row];
                for (IdxType entryIdx = outer[row]; entryIdx < outer[row + 1] - 1; ++entryIdx) {
                    acc -= val[entryIdx] * x[inner[entryIdx]];
                }

                x[row] = acc / val[outer[row + 1] - 1];
            }

#pragma omp barrier
        }
    }
}

template <typename IdxType, unsigned staleness = 2U>
void SpLTrSvProcPermSSPParallel(double *__restrict__ const x,
                                const double *__restrict__ const b,
                                const IdxType *__restrict__ const outer,
                                const IdxType *__restrict__ const inner,
                                const double *__restrict__ const val,
                                const unsigned numProcs,
                                const unsigned numSuperSteps,
                                const IdxType *__restrict__ const procStepPtr) {
    static_assert(std::is_integral_v<IdxType>);

    FlatCheckpointCounterBarrier barrier(numProcs);
#pragma omp parallel num_threads(numProcs)
    {
        const unsigned proc = static_cast<unsigned>(omp_get_thread_num());
        const IdxType *const endStepPtr = procStepPtr + ((proc + 1U) * numSuperSteps);
        std::size_t step = 0U;
        for (const IdxType *stepPtr = procStepPtr + (proc * numSuperSteps); stepPtr != endStepPtr;) {
            IdxType row = *stepPtr;
            const IdxType endRow = *(++stepPtr);

            if (row != endRow) {
                constexpr std::size_t diff = staleness - 1U;
                const std::size_t minStep = std::max(step, diff) - diff;
                barrier.Wait(proc, minStep);
            }

            for (; row != endRow; ++row) {
                double acc = b[row];
                for (IdxType entryIdx = outer[row]; entryIdx < outer[row + 1] - 1; entryIdx++) {
                    acc -= val[entryIdx] * x[inner[entryIdx]];
                }

                x[row] = acc / val[outer[row + 1] - 1];
            }
            barrier.Arrive(proc, ++step);
        }
    }
}

template <typename IdxType, unsigned staleness = 2U>
void SpLTrSvProcPermSSPParallelInPlace(double *__restrict__ const x,
                                       const IdxType *__restrict__ const outer,
                                       const IdxType *__restrict__ const inner,
                                       const double *__restrict__ const val,
                                       const unsigned numProcs,
                                       const unsigned numSuperSteps,
                                       const IdxType *__restrict__ const procStepPtr) {
    static_assert(std::is_integral_v<IdxType>);

    FlatCheckpointCounterBarrier barrier(numProcs);
#pragma omp parallel num_threads(numProcs)
    {
        const unsigned proc = static_cast<unsigned>(omp_get_thread_num());
        const IdxType *const endStepPtr = procStepPtr + ((proc + 1U) * numSuperSteps);
        std::size_t step = 0U;
        for (const IdxType *stepPtr = procStepPtr + (proc * numSuperSteps); stepPtr != endStepPtr;) {
            IdxType row = *stepPtr;
            const IdxType endRow = *(++stepPtr);

            if (row != endRow) {
                constexpr std::size_t diff = staleness - 1U;
                const std::size_t minStep = std::max(step, diff) - diff;
                barrier.Wait(proc, minStep);
            }

            for (; row != endRow; ++row) {
                double acc = x[row];
                for (IdxType entryIdx = outer[row]; entryIdx < outer[row + 1] - 1; entryIdx++) {
                    acc -= val[entryIdx] * x[inner[entryIdx]];
                }

                x[row] = acc / val[outer[row + 1] - 1];
            }
            barrier.Arrive(proc, ++step);
        }
    }
}

}    // end namespace osp
