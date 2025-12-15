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
#ifdef EIGEN_FOUND

#    include <cxxabi.h>

#    include <Eigen/Core>
#    include <algorithm>
#    include <cmath>
#    include <map>
#    include <numeric>
#    include <sstream>
#    include <string>
#    include <typeinfo>
#    include <vector>

#    include "IStatsModule.hpp"
#    include "osp/auxiliary/sptrsv_simulator/ScheduleNodePermuter.hpp"
#    include "osp/auxiliary/sptrsv_simulator/sptrsv.hpp"
#    include "osp/bsp/model/BspSchedule.hpp"
#    include "osp/graph_implementations/boost_graphs/boost_graph.hpp"    // For graph_t
#    include "osp/graph_implementations/eigen_matrix_adapter/sparse_matrix.hpp"

namespace osp {

// Turn permutation mode into a human-readable prefix used in metric names
inline const char *ModeTag(ScheduleNodePermutationModes m) {
    switch (m) {
        case NO_PERMUTE:
            return "NoPermute_";
        case LOOP_PROCESSORS:
            return "LoopProc_";
        case SNAKE_PROCESSORS:
            return "SnakeProc_";
        default:
            return "Unknown_";
    }
}

bool CompareVectors(Eigen::VectorXd &v1, Eigen::VectorXd &v2) {
    std::cout << std::fixed;
    std::cout << std::setprecision(15);

    assert(v1.size() == v2.size());
    bool same = true;
    const double epsilon = 1e-10;
    for (long long int i = 0; i < v1.size(); ++i) {
        // std::cout << "Ind: " << i << ": | " << v1[i] << " - " << v2[i] << " | = " << abs(v1[i]-v2[i]) << "\n";
        if (std::abs(v1[i] - v2[i]) / (std::abs(v1[i]) + std::abs(v2[i]) + epsilon) > epsilon) {
            std::cout << "We have differences in the matrix in position: " << i << std::endl;
            std::cout << v1[i] << " , " << v2[i] << std::endl;
            same = false;
            break;
        }
    }
    return same;
}

template <typename TargetObjectType>
class BspSptrsvStatsModule : public IStatisticModule<TargetObjectType> {
  public:
    explicit BspSptrsvStatsModule(ScheduleNodePermutationModes mode = NO_PERMUTE) : mode_(mode) {}

    std::vector<std::string> GetMetricHeaders() const override {
        const std::string prefix = ModeTag(mode_);
        return {prefix + "SpTrSV_Runtime_Geomean(ns)",
                prefix + "SpTrSV_Runtime_Stddev",
                prefix + "SpTrSV_Runtime_Q25(ns)",
                prefix + "SpTrSV_Runtime_Q75(ns)"};
    }

    std::map<std::string, std::string> RecordStatistics(const TargetObjectType &schedule, std::ofstream &) const override {
        std::map<std::string, std::string> stats;

        if constexpr (std::is_same_v<TargetObjectType, osp::BspSchedule<osp::SparseMatrixImp<int32_t>>>
                      || std::is_same_v<TargetObjectType, osp::BspSchedule<osp::SparseMatrixImp<int64_t>>>) {
            using IndexT
                = std::conditional_t<std::is_same_v<TargetObjectType, osp::BspSchedule<osp::SparseMatrixImp<int32_t>>>, int32_t, int64_t>;

            auto instance = schedule.GetInstance();
            Sptrsv<IndexT> sim{instance};

            std::vector<size_t> perm;

            if (mode_ == NO_PERMUTE) {
                sim.SetupCsrNoPermutation(schedule);
            } else if (mode_ == LOOP_PROCESSORS) {
                perm = ScheduleNodePermuterBasic(schedule, LOOP_PROCESSORS);
                sim.SetupCsrWithPermutation(schedule, perm);
            } else if (mode_ == SNAKE_PROCESSORS) {
                perm = ScheduleNodePermuterBasic(schedule, SNAKE_PROCESSORS);
                sim.SetupCsrWithPermutation(schedule, perm);
            } else {
                std::cout << "Wrong type of permutation provided" << std::endl;
            }

            Eigen::VectorXd lBRef, lXRef;
            auto n = instance.GetComputationalDag().GetCsc()->cols();
            lXRef.resize(n);
            lBRef.resize(n);
            auto lView = (*instance.GetComputationalDag().GetCsr()).template triangularView<Eigen::Lower>();
            lBRef.setOnes();
            lXRef.setZero();
            lXRef = lView.solve(lBRef);

            std::vector<long long> timesNs;
            Eigen::VectorXd lXOsp = lXRef, lBOsp = lBRef;

            for (int i = 0; i < runs_; ++i) {
                lBOsp.setOnes();
                lXOsp.setZero();
                sim.x_ = &lXOsp[0];
                sim.b_ = &lBOsp[0];
                std::chrono::_V2::system_clock::time_point start, end;

                if (mode_ == NO_PERMUTE) {
                    start = std::chrono::high_resolution_clock::now();
                    sim.LsolveNoPermutation();
                    end = std::chrono::high_resolution_clock::now();
                } else {
                    start = std::chrono::high_resolution_clock::now();
                    sim.LsolveWithPermutation();
                    end = std::chrono::high_resolution_clock::now();
                }

                long long elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
                timesNs.push_back(elapsed);
            }

            // Geometric mean (requires conversion to double)
            double totalLog = std::accumulate(timesNs.begin(), timesNs.end(), 0.0, [](double sum, long long val) {
                return sum + std::log(static_cast<double>(val));
            });
            long long geomMean = static_cast<long long>(std::exp(totalLog / runs_));

            // Standard deviation
            double mean = std::accumulate(timesNs.begin(), timesNs.end(), 0.0) / runs_;
            double sqSum = std::accumulate(timesNs.begin(), timesNs.end(), 0.0, [mean](double acc, long long val) {
                double diff = static_cast<double>(val) - mean;
                return acc + diff * diff;
            });
            long long stddev = static_cast<long long>(std::sqrt(sqSum / runs_));

            // Quartiles
            std::sort(timesNs.begin(), timesNs.end());
            long long q25 = timesNs[runs_ / 4];
            long long q75 = timesNs[3 * runs_ / 4];

            auto toStr = [](long long value) {
                return std::to_string(value);    // no decimal points
            };

            // Permute back if needed
            if (mode_ != NO_PERMUTE) {
                sim.PermuteXVector(perm);
            }

            if (!CompareVectors(lXRef, lXOsp)) {
                std::cout << "Output is not equal" << std::endl;
            }

            const std::string prefix = ModeTag(mode_);
            stats[prefix + "SpTrSV_Runtime_Geomean(ns)"] = toStr(geomMean);
            stats[prefix + "SpTrSV_Runtime_Stddev"] = toStr(stddev);
            stats[prefix + "SpTrSV_Runtime_Q25(ns)"] = toStr(q25);
            stats[prefix + "SpTrSV_Runtime_Q75(ns)"] = toStr(q75);

        } else {
            std::cout << "Simulation is not available without the SparseMatrix type" << std::endl;
        }

        return stats;
    }

  private:
    ScheduleNodePermutationModes mode_;
    static constexpr int runs_ = 100;    // number of runs for benchmarking
};

}    // namespace osp

#endif
