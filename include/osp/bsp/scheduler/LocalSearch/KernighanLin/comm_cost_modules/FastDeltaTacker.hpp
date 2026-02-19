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

#include <vector>

template <typename CommWeightT>
struct FastDeltaTracker {
    std::vector<CommWeightT> denseVals_;      // Size: num_procs
    std::vector<unsigned> dirtyProcs_;        // List of modified indices
    std::vector<unsigned> procDirtyIndex_;    // Map proc -> index in dirtyProcs_ (num_procs if not dirty)
    unsigned numProcs_ = 0;

    void Initialize(unsigned nProcs) {
        if (nProcs > numProcs_) {
            numProcs_ = nProcs;
            denseVals_.resize(numProcs_, 0);
            dirtyProcs_.reserve(numProcs_);
            procDirtyIndex_.resize(numProcs_, numProcs_);
        }
    }

    inline void Add(unsigned proc, CommWeightT val) {
        if (val == 0) {
            return;
        }

        // If currently 0, it is becoming dirty
        if (denseVals_[proc] == 0) {
            procDirtyIndex_[proc] = static_cast<unsigned>(dirtyProcs_.size());
            dirtyProcs_.push_back(proc);
        }

        denseVals_[proc] += val;

        // If it returns to 0, remove it from dirty list (Swap and Pop for O(1))
        if (denseVals_[proc] == 0) {
            unsigned idx = procDirtyIndex_[proc];
            unsigned lastProc = dirtyProcs_.back();

            // Move last element to the hole
            dirtyProcs_[idx] = lastProc;
            procDirtyIndex_[lastProc] = idx;

            // Remove last
            dirtyProcs_.pop_back();
            procDirtyIndex_[proc] = numProcs_;
        }
    }

    inline CommWeightT Get(unsigned proc) const {
        if (proc < denseVals_.size()) {
            return denseVals_[proc];
        }
        return 0;
    }

    /// Returns true if proc has a non-zero accumulated delta.
    inline bool IsDirty(unsigned proc) const { return procDirtyIndex_[proc] != numProcs_; }

    inline void Clear() {
        for (unsigned p : dirtyProcs_) {
            denseVals_[p] = 0;
            procDirtyIndex_[p] = numProcs_;
        }
        dirtyProcs_.clear();
    }
};
