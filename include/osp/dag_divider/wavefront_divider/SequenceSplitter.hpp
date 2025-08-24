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
#include <vector>
#include <limits>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iterator> // Required for std::distance and std::iterator_traits

namespace osp {

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
    virtual std::vector<size_t> split(const std::vector<double>& seq) = 0;
};

/**
 * @class VarianceSplitter
 * @brief Splits a sequence recursively based on variance reduction.
 * A split is performed if it reduces the sum of variances of the two resulting
 * sub-sequences by a certain factor and the original variance is above a threshold.
 */
class VarianceSplitter : public SequenceSplitter {
private:
    using ConstIterator = std::vector<double>::const_iterator;
    using difference_type = typename std::iterator_traits<ConstIterator>::difference_type;

public:
    VarianceSplitter(double var_mult, double var_threshold, size_t max_depth = std::numeric_limits<size_t>::max())
        : var_mult_(var_mult), var_threshold_(var_threshold), max_depth_(max_depth) {}

    std::vector<size_t> split(const std::vector<double>& seq) override {
        std::vector<size_t> splits;
        split_recursive(seq.begin(), seq.end(), splits, 0, 0);
        std::sort(splits.begin(), splits.end());
        return splits;
    }

private:
    void split_recursive(ConstIterator begin, ConstIterator end, std::vector<size_t>& splits, size_t offset, size_t current_depth) {
        if (current_depth >= max_depth_ || std::distance(begin, end) < 2) return;

        double mean, variance;
        compute_variance(begin, end, mean, variance);

        if (variance > var_threshold_) {
            difference_type split_point_local = 0;
            if (compute_best_split(begin, end, split_point_local, variance)) {
                size_t split_point_global = static_cast<size_t>(split_point_local) + offset;
                splits.push_back(split_point_global);
                
                ConstIterator split_it = begin + split_point_local;
                split_recursive(begin, split_it, splits, offset, current_depth + 1);
                split_recursive(split_it, end, splits, split_point_global, current_depth + 1);
            }
        }
    }

    bool compute_best_split(ConstIterator begin, ConstIterator end, difference_type& best_split, double original_variance) const {
        double min_weighted_variance_sum = std::numeric_limits<double>::max();
        best_split = 0;
        difference_type current_size = std::distance(begin, end);

        for (difference_type i = 1; i < current_size; ++i) {
            ConstIterator current_split_it = begin + i;
            
            double left_mean, left_variance, right_mean, right_variance;
            compute_variance(begin, current_split_it, left_mean, left_variance);
            compute_variance(current_split_it, end, right_mean, right_variance);

            double current_weighted_variance_sum = static_cast<double>(i) * left_variance + static_cast<double>(current_size - i) * right_variance;

            if (current_weighted_variance_sum < min_weighted_variance_sum) {
                min_weighted_variance_sum = current_weighted_variance_sum;
                best_split = i;
            }
        }

        // A split is justified if the total within-group variance is smaller than the parent's total variance.
        double total_original_variance = original_variance * static_cast<double>(current_size);
        return best_split > 0 && min_weighted_variance_sum < total_original_variance;
    }

    void compute_variance(ConstIterator begin, ConstIterator end, double& mean, double& variance) const {
        const difference_type size = std::distance(begin, end);
        if (size < 2) { // Variance is 0 for sequences with 0 or 1 elements.
            mean = (size == 1) ? *begin : 0;
            variance = 0;
            return;
        }
        double sum = std::accumulate(begin, end, 0.0);
        mean = sum / static_cast<double>(size);
        double sq_sum = std::inner_product(begin, end, begin, 0.0);
        variance = sq_sum / static_cast<double>(size) - mean * mean;
    }

    double var_mult_;
    double var_threshold_;
    size_t max_depth_;
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

    std::vector<size_t> split(const std::vector<double>& seq) override {
        std::vector<size_t> splits;
        split_recursive(seq.begin(), seq.end(), splits, 0, 0);
        std::sort(splits.begin(), splits.end());
        return splits;
    }

private:
    void split_recursive(ConstIterator begin, ConstIterator end, std::vector<size_t>& splits, size_t offset, size_t current_depth) {
        if (current_depth >= max_depth_) return;

        const difference_type size = std::distance(begin, end);
        if (static_cast<size_t>(size) < min_subseq_len_) return;

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
            splits.push_back(split_point_global);
            
            ConstIterator split_it = begin + split_point_local;
            split_recursive(begin, split_it, splits, offset, current_depth + 1);
            split_recursive(split_it, end, splits, split_point_global, current_depth + 1);
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
    ThresholdScanSplitter(double diff_threshold, double absolute_threshold)
        : diff_threshold_(diff_threshold), absolute_threshold_(absolute_threshold) {}

    std::vector<size_t> split(const std::vector<double>& seq) override {
        std::vector<size_t> splits;
        if (seq.size() < 2) return splits;

        for (size_t i = 0; i < seq.size() - 1; ++i) {
            bool should_cut = false;
            double current = seq[i];
            double next = seq[i+1];

            // A split is triggered by a significant change OR by crossing the absolute threshold.
            if (current > next) { // Dropping
                if ((current - next) > diff_threshold_ || (next < absolute_threshold_ && current >= absolute_threshold_)) {
                    should_cut = true;
                }
            } else if (current < next) { // Rising
                if ((next - current) > diff_threshold_ || (next > absolute_threshold_ && current <= absolute_threshold_)) {
                    should_cut = true;
                }
            }
            
            if (should_cut) {
                splits.push_back(i + 1);                
            }
        }
        return splits;
    }
private:
    double diff_threshold_;
    double absolute_threshold_;
};

}; // namespace osp
