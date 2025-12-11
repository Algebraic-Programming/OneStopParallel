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
inline const char *mode_tag(SCHEDULE_NODE_PERMUTATION_MODES m) {
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

bool compare_vectors(Eigen::VectorXd &v1, Eigen::VectorXd &v2) {
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
    explicit BspSptrsvStatsModule(SCHEDULE_NODE_PERMUTATION_MODES _mode = NO_PERMUTE) : mode(_mode) {}

    std::vector<std::string> get_metric_headers() const override {
        const std::string prefix = mode_tag(mode);
        return {prefix + "SpTrSV_Runtime_Geomean(ns)",
                prefix + "SpTrSV_Runtime_Stddev",
                prefix + "SpTrSV_Runtime_Q25(ns)",
                prefix + "SpTrSV_Runtime_Q75(ns)"};
    }

    std::map<std::string, std::string> record_statistics(const TargetObjectType &schedule, std::ofstream &) const override {
        std::map<std::string, std::string> stats;

        if constexpr (std::is_same_v<TargetObjectType, osp::BspSchedule<osp::SparseMatrixImp<int32_t>>>
                      || std::is_same_v<TargetObjectType, osp::BspSchedule<osp::SparseMatrixImp<int64_t>>>) {
            using index_t
                = std::conditional_t<std::is_same_v<TargetObjectType, osp::BspSchedule<osp::SparseMatrixImp<int32_t>>>, int32_t, int64_t>;

            auto instance = schedule.getInstance();
            Sptrsv<index_t> sim{instance};

            std::vector<size_t> perm;

            if (mode == NO_PERMUTE) {
                sim.setup_csr_no_permutation(schedule);
            } else if (mode == LOOP_PROCESSORS) {
                perm = schedule_node_permuter_basic(schedule, LOOP_PROCESSORS);
                sim.setup_csr_with_permutation(schedule, perm);
            } else if (mode == SNAKE_PROCESSORS) {
                perm = schedule_node_permuter_basic(schedule, SNAKE_PROCESSORS);
                sim.setup_csr_with_permutation(schedule, perm);
            } else {
                std::cout << "Wrong type of permutation provided" << std::endl;
            }

            Eigen::VectorXd L_b_ref, L_x_ref;
            auto n = instance.getComputationalDag().getCSC()->cols();
            L_x_ref.resize(n);
            L_b_ref.resize(n);
            auto L_view = (*instance.getComputationalDag().getCSR()).template triangularView<Eigen::Lower>();
            L_b_ref.setOnes();
            L_x_ref.setZero();
            L_x_ref = L_view.solve(L_b_ref);

            std::vector<long long> times_ns;
            Eigen::VectorXd L_x_osp = L_x_ref, L_b_osp = L_b_ref;

            for (int i = 0; i < runs; ++i) {
                L_b_osp.setOnes();
                L_x_osp.setZero();
                sim.x = &L_x_osp[0];
                sim.b = &L_b_osp[0];
                std::chrono::_V2::system_clock::time_point start, end;

                if (mode == NO_PERMUTE) {
                    start = std::chrono::high_resolution_clock::now();
                    sim.lsolve_no_permutation();
                    end = std::chrono::high_resolution_clock::now();
                } else {
                    start = std::chrono::high_resolution_clock::now();
                    sim.lsolve_with_permutation();
                    end = std::chrono::high_resolution_clock::now();
                }

                long long elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
                times_ns.push_back(elapsed);
            }

            // Geometric mean (requires conversion to double)
            double total_log = std::accumulate(times_ns.begin(), times_ns.end(), 0.0, [](double sum, long long val) {
                return sum + std::log(static_cast<double>(val));
            });
            long long geom_mean = static_cast<long long>(std::exp(total_log / runs));

            // Standard deviation
            double mean = std::accumulate(times_ns.begin(), times_ns.end(), 0.0) / runs;
            double sq_sum = std::accumulate(times_ns.begin(), times_ns.end(), 0.0, [mean](double acc, long long val) {
                double diff = static_cast<double>(val) - mean;
                return acc + diff * diff;
            });
            long long stddev = static_cast<long long>(std::sqrt(sq_sum / runs));

            // Quartiles
            std::sort(times_ns.begin(), times_ns.end());
            long long q25 = times_ns[runs / 4];
            long long q75 = times_ns[3 * runs / 4];

            auto to_str = [](long long value) {
                return std::to_string(value);    // no decimal points
            };

            // Permute back if needed
            if (mode != NO_PERMUTE) { sim.permute_x_vector(perm); }

            if (!compare_vectors(L_x_ref, L_x_osp)) { std::cout << "Output is not equal" << std::endl; }

            const std::string prefix = mode_tag(mode);
            stats[prefix + "SpTrSV_Runtime_Geomean(ns)"] = to_str(geom_mean);
            stats[prefix + "SpTrSV_Runtime_Stddev"] = to_str(stddev);
            stats[prefix + "SpTrSV_Runtime_Q25(ns)"] = to_str(q25);
            stats[prefix + "SpTrSV_Runtime_Q75(ns)"] = to_str(q75);

        } else {
            std::cout << "Simulation is not available without the SparseMatrix type" << std::endl;
        }

        return stats;
    }

  private:
    SCHEDULE_NODE_PERMUTATION_MODES mode;
    static constexpr int runs = 100;    // number of runs for benchmarking
};

}    // namespace osp

#endif
