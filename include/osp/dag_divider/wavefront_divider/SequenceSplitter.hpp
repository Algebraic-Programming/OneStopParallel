/*
Copyright 2024 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,.
See the License for the specific language governing permissions and
limitations under the License.

@author Toni Boehnlein, Benjamin Lozes, Pal Andras Papp, Raphael S. Steiner
*/
#pragma once
#include <algorithm>
#include <cmath>
#include <iterator>    // Required for std::distance and std::iterator_traits
#include <limits>
#include <numeric>
#include <vector>

namespace osp {

enum class SplitAlgorithm { LARGEST_STEP, VARIANCE, THRESHOLD_SCAN };

/**
 * @class SequenceSplitter
 * @brief Abstract base class for algorithms that split a sequence of numbers.
 */
class SequenceSplitter {
  public:
    virtual ~SequenceSplitter() = default;

    /**
     * @brief Splits a sequence and returns the indices of the split points.
     * @param seq The sequence of numbers to split.
     * @return A vector of indices where the sequence is split.
     */
    virtual std::vector<size_t> split(const std::vector<double> &seq) = 0;
};

/**
 * @class VarianceSplitter
 * @brief Splits a sequence recursively based on variance reduction.
 * A split is performed if it reduces the sum of variances of the two resulting
 * sub-sequences by a factor (var_mult_) and if the original variance is above a threshold.
 */
class VarianceSplitter : public SequenceSplitter {
  public:
    VarianceSplitter(double var_mult,
                     double var_threshold,
                     size_t min_subseq_len = 1,
                     size_t max_depth = std::numeric_limits<size_t>::max())
        : var_mult_(var_mult), var_threshold_(var_threshold), min_subseq_len_(min_subseq_len), max_depth_(max_depth) {}

    std::vector<size_t> split(const std::vector<double> &seq) override {
        if (seq.empty()) { return {}; }

        // Precompute prefix sums for the entire sequence
        prefix_sum_.assign(seq.size() + 1, 0.0);
        prefix_sq_sum_.assign(seq.size() + 1, 0.0);

        for (size_t i = 0; i < seq.size(); ++i) {
            prefix_sum_[i + 1] = prefix_sum_[i] + seq[i];
            prefix_sq_sum_[i + 1] = prefix_sq_sum_[i] + seq[i] * seq[i];
        }

        std::vector<size_t> splits;
        split_recursive(0, seq.size(), splits, 0);
        std::sort(splits.begin(), splits.end());
        return splits;
    }

  private:
    // Compute mean & variance in [l, r) in O(1)
    void compute_variance(size_t l, size_t r, double &mean, double &variance) const {
        size_t n = r - l;
        if (n <= 1) {
            mean = (n == 1) ? (prefix_sum_[r] - prefix_sum_[l]) : 0.0;
            variance = 0.0;
            return;
        }
        double sum = prefix_sum_[r] - prefix_sum_[l];
        double sq_sum = prefix_sq_sum_[r] - prefix_sq_sum_[l];
        mean = sum / static_cast<double>(n);
        variance = sq_sum / static_cast<double>(n) - mean * mean;
    }

    void split_recursive(size_t l, size_t r, std::vector<size_t> &splits, size_t depth) {
        if (depth >= max_depth_ || r - l < 2 * min_subseq_len_) { return; }

        double mean, variance;
        compute_variance(l, r, mean, variance);

        if (variance > var_threshold_) {
            size_t best_split = 0;
            if (compute_best_split(l, r, best_split, variance)) {
                // enforce minimum sub-sequence length
                if ((best_split - l) >= min_subseq_len_ && (r - best_split) >= min_subseq_len_) {
                    splits.push_back(best_split);
                    split_recursive(l, best_split, splits, depth + 1);
                    split_recursive(best_split, r, splits, depth + 1);
                }
            }
        }
    }

    bool compute_best_split(size_t l, size_t r, size_t &best_split, double original_variance) const {
        size_t n = r - l;
        if (n < 2) { return false; }

        double min_weighted_variance_sum = std::numeric_limits<double>::max();
        best_split = 0;

        for (size_t i = l + 1; i < r; ++i) {
            double left_mean, left_var, right_mean, right_var;
            compute_variance(l, i, left_mean, left_var);
            compute_variance(i, r, right_mean, right_var);

            double weighted_sum = static_cast<double>(i - l) * left_var + static_cast<double>(r - i) * right_var;

            if (weighted_sum < min_weighted_variance_sum) {
                min_weighted_variance_sum = weighted_sum;
                best_split = i;
            }
        }

        double total_original_variance = original_variance * static_cast<double>(n);
        return best_split > l && min_weighted_variance_sum < var_mult_ * total_original_variance;
    }

    double var_mult_;
    double var_threshold_;
    size_t min_subseq_len_;
    size_t max_depth_;
    std::vector<double> prefix_sum_;
    std::vector<double> prefix_sq_sum_;
};

/**
 * @class LargestStepSplitter
 * @brief Splits a monotonic sequence recursively at the point of the largest change.
 * A split is performed if the largest difference between two consecutive elements
 * exceeds a given threshold.
 */
class LargestStepSplitter : public SequenceSplitter {
  private:
    using ConstIterator = std::vector<double>::const_iterator;
    using difference_type = typename std::iterator_traits<ConstIterator>::difference_type;

  public:
    LargestStepSplitter(double diff_threshold, size_t min_subseq_len, size_t max_depth = std::numeric_limits<size_t>::max())
        : diff_threshold_(diff_threshold), min_subseq_len_(min_subseq_len), max_depth_(max_depth) {}

    std::vector<size_t> split(const std::vector<double> &seq) override {
        std::vector<size_t> splits;
        split_recursive(seq.begin(), seq.end(), splits, 0, 0);
        std::sort(splits.begin(), splits.end());
        return splits;
    }

  private:
    void split_recursive(ConstIterator begin, ConstIterator end, std::vector<size_t> &splits, size_t offset, size_t current_depth) {
        if (current_depth >= max_depth_) { return; }

        const difference_type size = std::distance(begin, end);
        if (static_cast<size_t>(size) < 2 * min_subseq_len_) { return; }

        double max_diff = 0.0;
        difference_type split_point_local = 0;

        difference_type current_local_idx = 0;
        for (auto it = begin; it != end - 1; ++it) {
            double diff = std::abs(*it - *(it + 1));
            if (diff > max_diff) {
                max_diff = diff;
                split_point_local = current_local_idx + 1;
            }
            current_local_idx++;
        }

        if (max_diff > diff_threshold_ && split_point_local > 0) {
            size_t split_point_global = static_cast<size_t>(split_point_local) + offset;

            if ((split_point_local >= static_cast<difference_type>(min_subseq_len_))
                && ((size - split_point_local) >= static_cast<difference_type>(min_subseq_len_))) {
                splits.push_back(split_point_global);

                ConstIterator split_it = begin + split_point_local;
                split_recursive(begin, split_it, splits, offset, current_depth + 1);
                split_recursive(split_it, end, splits, split_point_global, current_depth + 1);
            }
        }
    }

    double diff_threshold_;
    size_t min_subseq_len_;
    size_t max_depth_;
};

/**
 * @class ThresholdScanSplitter
 * @brief Splits a sequence by scanning for significant changes or crossing an absolute threshold.
 * This is a non-recursive splitter that performs a single pass.
 */
class ThresholdScanSplitter : public SequenceSplitter {
  public:
    ThresholdScanSplitter(double diff_threshold, double absolute_threshold, size_t min_subseq_len = 1)
        : diff_threshold_(diff_threshold), absolute_threshold_(absolute_threshold), min_subseq_len_(min_subseq_len) {}

    std::vector<size_t> split(const std::vector<double> &seq) override {
        std::vector<size_t> splits;
        if (seq.size() < 2) { return splits; }

        size_t last_cut = 0;
        for (size_t i = 0; i < seq.size() - 1; ++i) {
            bool should_cut = false;
            double current = seq[i];
            double next = seq[i + 1];

            // A split is triggered by a significant change OR by crossing the absolute threshold.
            if (current > next) {    // Dropping
                if ((current - next) > diff_threshold_ || (next < absolute_threshold_ && current >= absolute_threshold_)) {
                    should_cut = true;
                }
            } else if (current < next) {    // Rising
                if ((next - current) > diff_threshold_ || (next > absolute_threshold_ && current <= absolute_threshold_)) {
                    should_cut = true;
                }
            }

            if (should_cut) {
                if ((i + 1 - last_cut) >= min_subseq_len_ && (seq.size() - (i + 1)) >= min_subseq_len_) {
                    splits.push_back(i + 1);
                    last_cut = i + 1;
                }
            }
        }
        return splits;
    }

  private:
    double diff_threshold_;
    double absolute_threshold_;
    size_t min_subseq_len_;
};

}    // namespace osp
