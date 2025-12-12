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
    virtual std::vector<size_t> Split(const std::vector<double> &seq) = 0;
};

/**
 * @class VarianceSplitter
 * @brief Splits a sequence recursively based on variance reduction.
 * A split is performed if it reduces the sum of variances of the two resulting
 * sub-sequences by a factor (var_mult_) and if the original variance is above a threshold.
 */
class VarianceSplitter : public SequenceSplitter {
  public:
    VarianceSplitter(double varMult,
                     double varThreshold,
                     size_t minSubseqLen = 1,
                     size_t maxDepth = std::numeric_limits<size_t>::max())
        : varMult_(varMult), varThreshold_(varThreshold), minSubseqLen_(minSubseqLen), maxDepth_(maxDepth) {}

    std::vector<size_t> Split(const std::vector<double> &seq) override {
        if (seq.empty()) {
            return {};
        }

        // Precompute prefix sums for the entire sequence
        prefixSum_.assign(seq.size() + 1, 0.0);
        prefixSqSum_.assign(seq.size() + 1, 0.0);

        for (size_t i = 0; i < seq.size(); ++i) {
            prefixSum_[i + 1] = prefixSum_[i] + seq[i];
            prefixSqSum_[i + 1] = prefixSqSum_[i] + seq[i] * seq[i];
        }

        std::vector<size_t> splits;
        SplitRecursive(0, seq.size(), splits, 0);
        std::sort(splits.begin(), splits.end());
        return splits;
    }

  private:
    // Compute mean & variance in [l, r) in O(1)
    void ComputeVariance(size_t l, size_t r, double &mean, double &variance) const {
        size_t n = r - l;
        if (n <= 1) {
            mean = (n == 1) ? (prefixSum_[r] - prefixSum_[l]) : 0.0;
            variance = 0.0;
            return;
        }
        double sum = prefixSum_[r] - prefixSum_[l];
        double sqSum = prefixSqSum_[r] - prefixSqSum_[l];
        mean = sum / static_cast<double>(n);
        variance = sqSum / static_cast<double>(n) - mean * mean;
    }

    void SplitRecursive(size_t l, size_t r, std::vector<size_t> &splits, size_t depth) {
        if (depth >= maxDepth_ || r - l < 2 * minSubseqLen_) {
            return;
        }

        double mean, variance;
        ComputeVariance(l, r, mean, variance);

        if (variance > varThreshold_) {
            size_t bestSplit = 0;
            if (ComputeBestSplit(l, r, bestSplit, variance)) {
                // enforce minimum sub-sequence length
                if ((bestSplit - l) >= minSubseqLen_ && (r - bestSplit) >= minSubseqLen_) {
                    splits.push_back(bestSplit);
                    SplitRecursive(l, bestSplit, splits, depth + 1);
                    SplitRecursive(bestSplit, r, splits, depth + 1);
                }
            }
        }
    }

    bool ComputeBestSplit(size_t l, size_t r, size_t &bestSplit, double originalVariance) const {
        size_t n = r - l;
        if (n < 2) {
            return false;
        }

        double minWeightedVarianceSum = std::numeric_limits<double>::max();
        bestSplit = 0;

        for (size_t i = l + 1; i < r; ++i) {
            double leftMean, leftVar, rightMean, rightVar;
            ComputeVariance(l, i, leftMean, leftVar);
            ComputeVariance(i, r, rightMean, rightVar);

            double weightedSum = static_cast<double>(i - l) * leftVar + static_cast<double>(r - i) * rightVar;

            if (weightedSum < minWeightedVarianceSum) {
                minWeightedVarianceSum = weightedSum;
                bestSplit = i;
            }
        }

        double totalOriginalVariance = originalVariance * static_cast<double>(n);
        return bestSplit > l && minWeightedVarianceSum < varMult_ * totalOriginalVariance;
    }

    double varMult_;
    double varThreshold_;
    size_t minSubseqLen_;
    size_t maxDepth_;
    std::vector<double> prefixSum_;
    std::vector<double> prefixSqSum_;
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
    LargestStepSplitter(double diffThreshold, size_t minSubseqLen, size_t maxDepth = std::numeric_limits<size_t>::max())
        : diffThreshold_(diffThreshold), minSubseqLen_(minSubseqLen), maxDepth_(maxDepth) {}

    std::vector<size_t> Split(const std::vector<double> &seq) override {
        std::vector<size_t> splits;
        SplitRecursive(seq.begin(), seq.end(), splits, 0, 0);
        std::sort(splits.begin(), splits.end());
        return splits;
    }

  private:
    void SplitRecursive(ConstIterator begin, ConstIterator end, std::vector<size_t> &splits, size_t offset, size_t currentDepth) {
        if (currentDepth >= maxDepth_) {
            return;
        }

        const difference_type size = std::distance(begin, end);
        if (static_cast<size_t>(size) < 2 * minSubseqLen_) {
            return;
        }

        double maxDiff = 0.0;
        difference_type splitPointLocal = 0;

        difference_type currentLocalIdx = 0;
        for (auto it = begin; it != end - 1; ++it) {
            double diff = std::abs(*it - *(it + 1));
            if (diff > maxDiff) {
                maxDiff = diff;
                splitPointLocal = currentLocalIdx + 1;
            }
            currentLocalIdx++;
        }

        if (maxDiff > diffThreshold_ && splitPointLocal > 0) {
            size_t splitPointGlobal = static_cast<size_t>(splitPointLocal) + offset;

            if ((splitPointLocal >= static_cast<difference_type>(minSubseqLen_))
                && ((size - splitPointLocal) >= static_cast<difference_type>(minSubseqLen_))) {
                splits.push_back(splitPointGlobal);

                ConstIterator splitIt = begin + splitPointLocal;
                SplitRecursive(begin, splitIt, splits, offset, currentDepth + 1);
                SplitRecursive(splitIt, end, splits, splitPointGlobal, currentDepth + 1);
            }
        }
    }

    double diffThreshold_;
    size_t minSubseqLen_;
    size_t maxDepth_;
};

/**
 * @class ThresholdScanSplitter
 * @brief Splits a sequence by scanning for significant changes or crossing an absolute threshold.
 * This is a non-recursive splitter that performs a single pass.
 */
class ThresholdScanSplitter : public SequenceSplitter {
  public:
    ThresholdScanSplitter(double diffThreshold, double absoluteThreshold, size_t minSubseqLen = 1)
        : diffThreshold_(diffThreshold), absoluteThreshold_(absoluteThreshold), minSubseqLen_(minSubseqLen) {}

    std::vector<size_t> Split(const std::vector<double> &seq) override {
        std::vector<size_t> splits;
        if (seq.size() < 2) {
            return splits;
        }

        size_t lastCut = 0;
        for (size_t i = 0; i < seq.size() - 1; ++i) {
            bool shouldCut = false;
            double current = seq[i];
            double next = seq[i + 1];

            // A split is triggered by a significant change OR by crossing the absolute threshold.
            if (current > next) {    // Dropping
                if ((current - next) > diffThreshold_ || (next < absoluteThreshold_ && current >= absoluteThreshold_)) {
                    shouldCut = true;
                }
            } else if (current < next) {    // Rising
                if ((next - current) > diffThreshold_ || (next > absoluteThreshold_ && current <= absoluteThreshold_)) {
                    shouldCut = true;
                }
            }

            if (shouldCut) {
                if ((i + 1 - lastCut) >= minSubseqLen_ && (seq.size() - (i + 1)) >= minSubseqLen_) {
                    splits.push_back(i + 1);
                    lastCut = i + 1;
                }
            }
        }
        return splits;
    }

  private:
    double diffThreshold_;
    double absoluteThreshold_;
    size_t minSubseqLen_;
};

}    // namespace osp
