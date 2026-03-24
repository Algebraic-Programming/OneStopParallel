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
                        const std::vector<std::vector<std::vector<IdxType>>> &BoundsStepProcIdx) {
    static_assert(std::is_integral_v<IdxType>);

#pragma omp parallel num_threads(BoundsStepProcIdx[0U].size())
    {
        const std::size_t proc = static_cast<std::size_t>(omp_get_thread_num());
        const std::size_t numSuperSteps = BoundsStepProcIdx.size();

        for (std::size_t step = 0U; step < numSuperSteps; ++step) {
            const std::size_t ubIdx = BoundsStepProcIdx[step][proc].size();
            for (std::size_t idx = 0U; idx < ubIdx; ++idx) {
                IdxType row = BoundsStepProcIdx[step][proc][idx];
                const IdxType ubRow = BoundsStepProcIdx[step][proc][++idx];
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
                               const std::vector<std::vector<std::vector<IdxType>>> &BoundsStepProcIdx) {
    static_assert(std::is_integral_v<IdxType>);

#pragma omp parallel num_threads(BoundsStepProcIdx[0U].size())
    {
        const std::size_t proc = static_cast<std::size_t>(omp_get_thread_num());
        const std::size_t numSuperSteps = BoundsStepProcIdx.size();

        for (std::size_t step = 0U; step < numSuperSteps; ++step) {
            const std::size_t ubIdx = BoundsStepProcIdx[step][proc].size();
            for (std::size_t idx = 0U; idx < ubIdx; ++idx) {
                IdxType row = BoundsStepProcIdx[step][proc][idx];
                const IdxType ubRow = BoundsStepProcIdx[step][proc][++idx];
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
                        const std::vector<std::vector<std::vector<IdxType>>> &BoundsStepProcIdx) {
    static_assert(std::is_integral_v<IdxType>);

    const std::size_t nthreads = BoundsStepProcIdx[0U].size();
    FlatCheckpointCounterBarrier barrier(nthreads);

#pragma omp parallel num_threads(nthreads)
    {
        const std::size_t proc = static_cast<std::size_t>(omp_get_thread_num());
        for (std::size_t step = 0; step < BoundsStepProcIdx.size(); ++step) {
            const std::size_t ubIdx = BoundsStepProcIdx[step][proc].size();
            if (ubIdx > 0U) {
                barrier.Wait(proc, staleness - 1U);
            }
            for (std::size_t idx = 0; idx < ubIdx; ++idx) {
                IdxType row = BoundsStepProcIdx[step][proc][idx];
                const IdxType ubRow = BoundsStepProcIdx[step][proc][++idx];
                for (; row <= ubRow; ++row) {
                    double acc = b[row];
                    for (IdxType entryIdx = outer[row]; entryIdx < outer[row + 1] - 1; ++entryIdx) {
                        acc -= val[entryIdx] * x[inner[entryIdx]];
                    }
                    x[row] = acc / val[outer[row + 1] - 1];
                }
            }
            barrier.Arrive(proc);
        }
    }
}

template <typename IdxType, unsigned staleness = 2U>
void SpLTrSvSSPParallelInPlace(double *__restrict__ const x,
                               const IdxType *__restrict__ const outer,
                               const IdxType *__restrict__ const inner,
                               const double *__restrict__ const val,
                               const std::vector<std::vector<std::vector<IdxType>>> &BoundsStepProcIdx) {
    static_assert(std::is_integral_v<IdxType>);

    const std::size_t nthreads = BoundsStepProcIdx[0U].size();
    FlatCheckpointCounterBarrier barrier(nthreads);

#pragma omp parallel num_threads(nthreads)
    {
        const std::size_t proc = static_cast<std::size_t>(omp_get_thread_num());
        for (std::size_t step = 0; step < BoundsStepProcIdx.size(); ++step) {
            const std::size_t ubIdx = BoundsStepProcIdx[step][proc].size();
            if (ubIdx > 0U) {
                barrier.Wait(proc, staleness - 1U);
            }
            for (std::size_t idx = 0; idx < ubIdx; ++idx) {
                IdxType row = BoundsStepProcIdx[step][proc][idx];
                const IdxType ubRow = BoundsStepProcIdx[step][proc][++idx];
                for (; row <= ubRow; ++row) {
                    double acc = x[row];
                    for (IdxType entryIdx = outer[row]; entryIdx < outer[row + 1] - 1; ++entryIdx) {
                        acc -= val[entryIdx] * x[inner[entryIdx]];
                    }
                    x[row] = acc / val[outer[row + 1] - 1];
                }
            }
            barrier.Arrive(proc);
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
                                const std::vector<IdxType> &procStepPtr) {
    static_assert(std::is_integral_v<IdxType>);

#pragma omp parallel num_threads(numProcs)
    {
        const unsigned proc = static_cast<unsigned>(omp_get_thread_num());
        const auto endStepPtr = std::next(procStepPtr.cbegin(), (proc + 1U) * numSuperSteps);
        for (auto stepPtr = std::next(procStepPtr.cbegin(), proc * numSuperSteps); stepPtr != endStepPtr;) {
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
                                       const std::vector<IdxType> &procStepPtr) {
    static_assert(std::is_integral_v<IdxType>);

#pragma omp parallel num_threads(numProcs)
    {
        const unsigned proc = static_cast<unsigned>(omp_get_thread_num());
        const auto endStepPtr = std::next(procStepPtr.cbegin(), (proc + 1U) * numSuperSteps);
        for (auto stepPtr = std::next(procStepPtr.cbegin(), proc * numSuperSteps); stepPtr != endStepPtr;) {
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
                                const std::vector<IdxType> &procStepPtr) {
    static_assert(std::is_integral_v<IdxType>);

    FlatCheckpointCounterBarrier barrier(numProcs);
#pragma omp parallel num_threads(numProcs)
    {
        const unsigned proc = static_cast<unsigned>(omp_get_thread_num());
        const auto endStepPtr = std::next(procStepPtr.cbegin(), (proc + 1U) * numSuperSteps);
        for (auto stepPtr = std::next(procStepPtr.cbegin(), proc * numSuperSteps); stepPtr != endStepPtr;) {
            IdxType row = *stepPtr;
            const IdxType endRow = *(++stepPtr);

            if (row != endRow) {
                barrier.Wait(proc, staleness - 1U);
            }

            for (; row != endRow; ++row) {
                double acc = b[row];
                for (IdxType entryIdx = outer[row]; entryIdx < outer[row + 1] - 1; entryIdx++) {
                    acc -= val[entryIdx] * x[inner[entryIdx]];
                }

                x[row] = acc / val[outer[row + 1] - 1];
            }
            barrier.Arrive(proc);
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
                                       const std::vector<IdxType> &procStepPtr) {
    static_assert(std::is_integral_v<IdxType>);

    FlatCheckpointCounterBarrier barrier(numProcs);
#pragma omp parallel num_threads(numProcs)
    {
        const unsigned proc = static_cast<unsigned>(omp_get_thread_num());
        const auto endStepPtr = std::next(procStepPtr.cbegin(), (proc + 1U) * numSuperSteps);
        for (auto stepPtr = std::next(procStepPtr.cbegin(), proc * numSuperSteps); stepPtr != endStepPtr;) {
            IdxType row = *stepPtr;
            const IdxType endRow = *(++stepPtr);

            if (row != endRow) {
                barrier.Wait(proc, staleness - 1U);
            }

            for (; row != endRow; ++row) {
                double acc = x[row];
                for (IdxType entryIdx = outer[row]; entryIdx < outer[row + 1] - 1; entryIdx++) {
                    acc -= val[entryIdx] * x[inner[entryIdx]];
                }

                x[row] = acc / val[outer[row + 1] - 1];
            }
            barrier.Arrive(proc);
        }
    }
}

}    // end namespace osp
