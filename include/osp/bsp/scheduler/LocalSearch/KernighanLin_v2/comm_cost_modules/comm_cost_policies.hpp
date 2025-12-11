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

#include <algorithm>
#include <limits>
#include <vector>

namespace osp {

struct EagerCommCostPolicy {
    using ValueType = unsigned;

    template <typename DS, typename comm_weight_t>
    static inline void attribute_communication(DS &ds,
                                               const comm_weight_t &cost,
                                               const unsigned u_step,
                                               const unsigned u_proc,
                                               const unsigned v_proc,
                                               const unsigned v_step,
                                               const ValueType &val) {
        ds.step_proc_receive(u_step, v_proc) += cost;
        ds.step_proc_send(u_step, u_proc) += cost;
    }

    template <typename DS, typename comm_weight_t>
    static inline void unattribute_communication(DS &ds,
                                                 const comm_weight_t &cost,
                                                 const unsigned u_step,
                                                 const unsigned u_proc,
                                                 const unsigned v_proc,
                                                 const unsigned v_step,
                                                 const ValueType &val) {
        ds.step_proc_receive(u_step, v_proc) -= cost;
        ds.step_proc_send(u_step, u_proc) -= cost;
    }

    static inline bool add_child(ValueType &val, unsigned step) {
        val++;
        return val == 1;
    }

    static inline bool remove_child(ValueType &val, unsigned step) {
        val--;
        return val == 0;
    }

    static inline void reset(ValueType &val) { val = 0; }

    static inline bool has_entry(const ValueType &val) { return val > 0; }

    static inline bool is_single_entry(const ValueType &val) { return val == 1; }

    template <typename DeltaTracker, typename comm_weight_t>
    static inline void calculate_delta_remove(const ValueType &val,
                                              unsigned child_step,
                                              unsigned parent_step,
                                              unsigned parent_proc,
                                              unsigned child_proc,
                                              comm_weight_t cost,
                                              DeltaTracker &dt) {
        if (val == 1) {
            dt.add(true, parent_step, child_proc, -cost);
            dt.add(false, parent_step, parent_proc, -cost);
        }
    }

    template <typename DeltaTracker, typename comm_weight_t>
    static inline void calculate_delta_add(const ValueType &val,
                                           unsigned child_step,
                                           unsigned parent_step,
                                           unsigned parent_proc,
                                           unsigned child_proc,
                                           comm_weight_t cost,
                                           DeltaTracker &dt) {
        if (val == 0) {
            dt.add(true, parent_step, child_proc, cost);
            dt.add(false, parent_step, parent_proc, cost);
        }
    }

    template <typename DeltaTracker, typename comm_weight_t>
    static inline void calculate_delta_outgoing(
        const ValueType &val, unsigned node_step, unsigned node_proc, unsigned child_proc, comm_weight_t cost, DeltaTracker &dt) {
        if (val > 0) {
            comm_weight_t total_cost = cost * val;
            dt.add(true, node_step, child_proc, total_cost);
            dt.add(false, node_step, node_proc, total_cost);
        }
    }
};

struct LazyCommCostPolicy {
    using ValueType = std::vector<unsigned>;

    template <typename DS, typename comm_weight_t>
    static inline void attribute_communication(DS &ds,
                                               const comm_weight_t &cost,
                                               const unsigned u_step,
                                               const unsigned u_proc,
                                               const unsigned v_proc,
                                               const unsigned v_step,
                                               const ValueType &val) {
        // val contains v_step (already added).
        // Check if v_step is the new minimum.
        unsigned min_step = std::numeric_limits<unsigned>::max();
        for (unsigned s : val) {
            min_step = std::min(min_step, s);
        }

        if (min_step == v_step) {
            // Check if it was strictly smaller than previous min.
            unsigned prev_min = std::numeric_limits<unsigned>::max();
            for (size_t i = 0; i < val.size() - 1; ++i) {
                prev_min = std::min(prev_min, val[i]);
            }

            if (v_step < prev_min) {
                if (prev_min != std::numeric_limits<unsigned>::max() && prev_min > 0) {
                    ds.step_proc_receive(prev_min - 1, v_proc) -= cost;
                    ds.step_proc_send(prev_min - 1, u_proc) -= cost;
                }
                if (v_step > 0) {
                    ds.step_proc_receive(v_step - 1, v_proc) += cost;
                    ds.step_proc_send(v_step - 1, u_proc) += cost;
                }
            }
        }
    }

    template <typename DS, typename comm_weight_t>
    static inline void unattribute_communication(DS &ds,
                                                 const comm_weight_t &cost,
                                                 const unsigned u_step,
                                                 const unsigned u_proc,
                                                 const unsigned v_proc,
                                                 const unsigned v_step,
                                                 const ValueType &val) {
        // val is state AFTER removal.

        if (val.empty()) {
            // Removed the last child.
            if (v_step > 0) {
                ds.step_proc_receive(v_step - 1, v_proc) -= cost;
                ds.step_proc_send(v_step - 1, u_proc) -= cost;
            }
        } else {
            // Check if v_step was the unique minimum.
            unsigned new_min = val[0];
            for (unsigned s : val) {
                new_min = std::min(new_min, s);
            }

            if (v_step < new_min) {
                // v_step was the unique minimum.
                if (v_step > 0) {
                    ds.step_proc_receive(v_step - 1, v_proc) -= cost;
                    ds.step_proc_send(v_step - 1, u_proc) -= cost;
                }
                if (new_min > 0) {
                    ds.step_proc_receive(new_min - 1, v_proc) += cost;
                    ds.step_proc_send(new_min - 1, u_proc) += cost;
                }
            }
        }
    }

    static inline bool add_child(ValueType &val, unsigned step) {
        val.push_back(step);
        if (val.size() == 1) {
            return true;
        }
        unsigned min_s = val[0];
        for (unsigned s : val) {
            min_s = std::min(min_s, s);
        }
        return step == min_s;
    }

    static inline bool remove_child(ValueType &val, unsigned step) {
        auto it = std::find(val.begin(), val.end(), step);
        if (it != val.end()) {
            val.erase(it);
            if (val.empty()) {
                return true;
            }
            unsigned new_min = val[0];
            for (unsigned s : val) {
                new_min = std::min(new_min, s);
            }
            bool res = step < new_min;
            return res;
        }
        return false;
    }

    static inline void reset(ValueType &val) { val.clear(); }

    static inline bool has_entry(const ValueType &val) { return !val.empty(); }

    static inline bool is_single_entry(const ValueType &val) { return val.size() == 1; }

    template <typename DeltaTracker, typename comm_weight_t>
    static inline void calculate_delta_remove(const ValueType &val,
                                              unsigned child_step,
                                              unsigned parent_step,
                                              unsigned parent_proc,
                                              unsigned child_proc,
                                              comm_weight_t cost,
                                              DeltaTracker &dt) {
        if (val.empty()) {
            return;
        }
        unsigned min_s = val[0];
        for (unsigned s : val) {
            min_s = std::min(min_s, s);
        }

        if (child_step == min_s) {
            int count = 0;
            for (unsigned s : val) {
                if (s == min_s) {
                    count++;
                }
            }

            if (count == 1) {
                if (min_s > 0) {
                    dt.add(true, min_s - 1, child_proc, -cost);
                    dt.add(false, min_s - 1, parent_proc, -cost);
                }
                if (val.size() > 1) {
                    unsigned next_min = std::numeric_limits<unsigned>::max();
                    for (unsigned s : val) {
                        if (s != min_s) {
                            next_min = std::min(next_min, s);
                        }
                    }
                    if (next_min != std::numeric_limits<unsigned>::max() && next_min > 0) {
                        dt.add(true, next_min - 1, child_proc, cost);
                        dt.add(false, next_min - 1, parent_proc, cost);
                    }
                }
            }
        }
    }

    template <typename DeltaTracker, typename comm_weight_t>
    static inline void calculate_delta_add(const ValueType &val,
                                           unsigned child_step,
                                           unsigned parent_step,
                                           unsigned parent_proc,
                                           unsigned child_proc,
                                           comm_weight_t cost,
                                           DeltaTracker &dt) {
        if (val.empty()) {
            if (child_step > 0) {
                dt.add(true, child_step - 1, child_proc, cost);
                dt.add(false, child_step - 1, parent_proc, cost);
            }
        } else {
            unsigned min_s = val[0];
            for (unsigned s : val) {
                min_s = std::min(min_s, s);
            }

            if (child_step < min_s) {
                if (min_s > 0) {
                    dt.add(true, min_s - 1, child_proc, -cost);
                    dt.add(false, min_s - 1, parent_proc, -cost);
                }
                if (child_step > 0) {
                    dt.add(true, child_step - 1, child_proc, cost);
                    dt.add(false, child_step - 1, parent_proc, cost);
                }
            }
        }
    }

    template <typename DeltaTracker, typename comm_weight_t>
    static inline void calculate_delta_outgoing(
        const ValueType &val, unsigned node_step, unsigned node_proc, unsigned child_proc, comm_weight_t cost, DeltaTracker &dt) {
        for (unsigned s : val) {
            if (s > 0) {
                dt.add(true, s - 1, child_proc, cost);
                dt.add(false, s - 1, node_proc, cost);
            }
        }
    }
};

struct BufferedCommCostPolicy {
    using ValueType = std::vector<unsigned>;

    template <typename DS, typename comm_weight_t>
    static inline void attribute_communication(DS &ds,
                                               const comm_weight_t &cost,
                                               const unsigned u_step,
                                               const unsigned u_proc,
                                               const unsigned v_proc,
                                               const unsigned v_step,
                                               const ValueType &val) {
        // Buffered: Send at u_step, Receive at v_step - 1.

        unsigned min_step = std::numeric_limits<unsigned>::max();
        for (unsigned s : val) {
            min_step = std::min(min_step, s);
        }

        if (min_step == v_step) {
            unsigned prev_min = std::numeric_limits<unsigned>::max();
            for (size_t i = 0; i < val.size() - 1; ++i) {
                prev_min = std::min(prev_min, val[i]);
            }

            if (v_step < prev_min) {
                if (prev_min != std::numeric_limits<unsigned>::max() && prev_min > 0) {
                    ds.step_proc_receive(prev_min - 1, v_proc) -= cost;
                }
                if (v_step > 0) {
                    ds.step_proc_receive(v_step - 1, v_proc) += cost;
                }
            }
        }

        // Send side logic (u_step)
        // If this is the FIRST child on this proc, add send cost.
        if (val.size() == 1) {
            ds.step_proc_send(u_step, u_proc) += cost;
        }
    }

    template <typename DS, typename comm_weight_t>
    static inline void unattribute_communication(DS &ds,
                                                 const comm_weight_t &cost,
                                                 const unsigned u_step,
                                                 const unsigned u_proc,
                                                 const unsigned v_proc,
                                                 const unsigned v_step,
                                                 const ValueType &val) {
        // val is state AFTER removal.

        if (val.empty()) {
            // Removed last child.
            ds.step_proc_send(u_step, u_proc) -= cost;    // Send side
            if (v_step > 0) {
                ds.step_proc_receive(v_step - 1, v_proc) -= cost;    // Recv side
            }
        } else {
            // Check if v_step was unique minimum for Recv side.
            unsigned new_min = val[0];
            for (unsigned s : val) {
                new_min = std::min(new_min, s);
            }

            if (v_step < new_min) {
                if (v_step > 0) {
                    ds.step_proc_receive(v_step - 1, v_proc) -= cost;
                }
                if (new_min > 0) {
                    ds.step_proc_receive(new_min - 1, v_proc) += cost;
                }
            }
            // Send side remains (val not empty).
        }
    }

    static inline bool add_child(ValueType &val, unsigned step) {
        val.push_back(step);
        if (val.size() == 1) {
            return true;    // Need update for send side
        }
        unsigned min_s = val[0];
        for (unsigned s : val) {
            min_s = std::min(min_s, s);
        }
        return step == min_s;    // Need update for recv side
    }

    static inline bool remove_child(ValueType &val, unsigned step) {
        auto it = std::find(val.begin(), val.end(), step);
        if (it != val.end()) {
            val.erase(it);
            if (val.empty()) {
                return true;    // Need update for send side
            }
            unsigned new_min = val[0];
            for (unsigned s : val) {
                new_min = std::min(new_min, s);
            }
            return step < new_min;    // Need update for recv side
        }
        return false;
    }

    static inline void reset(ValueType &val) { val.clear(); }

    static inline bool has_entry(const ValueType &val) { return !val.empty(); }

    static inline bool is_single_entry(const ValueType &val) { return val.size() == 1; }

    template <typename DeltaTracker, typename comm_weight_t>
    static inline void calculate_delta_remove(const ValueType &val,
                                              unsigned child_step,
                                              unsigned parent_step,
                                              unsigned parent_proc,
                                              unsigned child_proc,
                                              comm_weight_t cost,
                                              DeltaTracker &dt) {
        // Lazy: Send and Recv are both at min(child_steps) - 1.

        if (val.empty()) {
            return;
        }

        unsigned min_s = val[0];
        for (unsigned s : val) {
            min_s = std::min(min_s, s);
        }

        if (child_step == min_s) {
            int count = 0;
            for (unsigned s : val) {
                if (s == min_s) {
                    count++;
                }
            }

            if (count == 1) {
                // Unique min being removed.
                if (min_s > 0) {
                    dt.add(true, min_s - 1, child_proc, -cost);      // Remove Recv
                    dt.add(false, min_s - 1, parent_proc, -cost);    // Remove Send
                }

                if (val.size() > 1) {
                    unsigned next_min = std::numeric_limits<unsigned>::max();
                    for (unsigned s : val) {
                        if (s != min_s) {
                            next_min = std::min(next_min, s);
                        }
                    }

                    if (next_min != std::numeric_limits<unsigned>::max() && next_min > 0) {
                        dt.add(true, next_min - 1, child_proc, cost);      // Add Recv at new min
                        dt.add(false, next_min - 1, parent_proc, cost);    // Add Send at new min
                    }
                }
            }
        }
    }

    template <typename DeltaTracker, typename comm_weight_t>
    static inline void calculate_delta_add(const ValueType &val,
                                           unsigned child_step,
                                           unsigned parent_step,
                                           unsigned parent_proc,
                                           unsigned child_proc,
                                           comm_weight_t cost,
                                           DeltaTracker &dt) {
        // Lazy: Send and Recv are both at min(child_steps) - 1.

        if (val.empty()) {
            // First child.
            if (child_step > 0) {
                dt.add(true, child_step - 1, child_proc, cost);
                dt.add(false, child_step - 1, parent_proc, cost);
            }
        } else {
            unsigned min_s = val[0];
            for (unsigned s : val) {
                min_s = std::min(min_s, s);
            }

            if (child_step < min_s) {
                // New global minimum.
                if (min_s > 0) {
                    dt.add(true, min_s - 1, child_proc, -cost);      // Remove old Recv
                    dt.add(false, min_s - 1, parent_proc, -cost);    // Remove old Send
                }
                if (child_step > 0) {
                    dt.add(true, child_step - 1, child_proc, cost);      // Add new Recv
                    dt.add(false, child_step - 1, parent_proc, cost);    // Add new Send
                }
            }
        }
    }

    template <typename DeltaTracker, typename comm_weight_t>
    static inline void calculate_delta_outgoing(
        const ValueType &val, unsigned node_step, unsigned node_proc, unsigned child_proc, comm_weight_t cost, DeltaTracker &dt) {
        // Buffered Outgoing (Node -> Children)
        // Node is parent (sender). Pays at node_step.
        // Children are receivers. Pay at child_step - 1.

        // Send side: node_step.
        // If val is not empty, we pay send cost ONCE.
        if (!val.empty()) {
            dt.add(false, node_step, node_proc, cost);
        }

        // Recv side: iterate steps in val (child steps).
        // But we only pay at min(val) - 1.
        if (!val.empty()) {
            unsigned min_s = val[0];
            for (unsigned s : val) {
                min_s = std::min(min_s, s);
            }

            if (min_s > 0) {
                dt.add(true, min_s - 1, child_proc, cost);
            }
        }
    }
};

}    // namespace osp
