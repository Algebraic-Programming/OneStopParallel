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

    template <typename DS, typename CommWeightT, typename MarkStepFn>
    static inline void AttributeCommunication(DS &ds,
                                              const CommWeightT &cost,
                                              const unsigned uStep,
                                              const unsigned uProc,
                                              const unsigned vProc,
                                              const unsigned vStep,
                                              const ValueType &val,
                                              MarkStepFn &&markStep) {
        ds.StepProcReceive(uStep, vProc) += cost;
        ds.StepProcSend(uStep, uProc) += cost;
        markStep(uStep);
    }

    template <typename DS, typename CommWeightT, typename MarkStepFn>
    static inline void UnattributeCommunication(DS &ds,
                                                const CommWeightT &cost,
                                                const unsigned uStep,
                                                const unsigned uProc,
                                                const unsigned vProc,
                                                const unsigned vStep,
                                                const ValueType &val,
                                                MarkStepFn &&markStep) {
        ds.StepProcReceive(uStep, vProc) -= cost;
        ds.StepProcSend(uStep, uProc) -= cost;
        markStep(uStep);
    }

    /// Remove outgoing communication when a parent node moves (val unchanged).
    /// For Eager, comm is at the parent's step.
    template <typename DS, typename CommWeightT, typename MarkStepFn>
    static inline void RemoveOutgoingComm(DS &ds,
                                          const CommWeightT &cost,
                                          unsigned parentStep,
                                          unsigned parentProc,
                                          unsigned childProc,
                                          const ValueType &val,
                                          MarkStepFn &&markStep) {
        ds.StepProcSend(parentStep, parentProc) -= cost;
        ds.StepProcReceive(parentStep, childProc) -= cost;
        markStep(parentStep);
    }

    /// Add outgoing communication when a parent node moves (val unchanged).
    template <typename DS, typename CommWeightT, typename MarkStepFn>
    static inline void AddOutgoingComm(DS &ds,
                                       const CommWeightT &cost,
                                       unsigned parentStep,
                                       unsigned parentProc,
                                       unsigned childProc,
                                       const ValueType &val,
                                       MarkStepFn &&markStep) {
        ds.StepProcSend(parentStep, parentProc) += cost;
        ds.StepProcReceive(parentStep, childProc) += cost;
        markStep(parentStep);
    }

    static inline bool AddChild(ValueType &val, unsigned step) {
        val++;
        return val == 1;
    }

    static inline bool RemoveChild(ValueType &val, unsigned step) {
        val--;
        return val == 0;
    }

    static inline void Reset(ValueType &val) { val = 0; }

    static inline bool HasEntry(const ValueType &val) { return val > 0; }

    static inline bool IsSingleEntry(const ValueType &val) { return val == 1; }

    // For outgoing comm (parent→children on proc), where is send/recv attributed?
    // Eager: both at parent step.
    static constexpr bool outgoing_send_at_parent_step = true;
    static constexpr bool outgoing_recv_at_parent_step = true;

    static inline int OutgoingSendStep(unsigned parentStep, const ValueType &val) {
        return val > 0 ? static_cast<int>(parentStep) : -1;
    }

    static inline int OutgoingRecvStep(unsigned parentStep, const ValueType &val) {
        return val > 0 ? static_cast<int>(parentStep) : -1;
    }

    template <typename DeltaTracker, typename CommWeightT>
    static inline void CalculateDeltaRemove(const ValueType &val,
                                            unsigned childStep,
                                            unsigned parentStep,
                                            unsigned parentProc,
                                            unsigned childProc,
                                            CommWeightT cost,
                                            DeltaTracker &dt) {
        if (val == 1) {
            dt.Add(true, parentStep, childProc, -cost);
            dt.Add(false, parentStep, parentProc, -cost);
        }
    }

    template <typename DeltaTracker, typename CommWeightT>
    static inline void CalculateDeltaAdd(const ValueType &val,
                                         unsigned childStep,
                                         unsigned parentStep,
                                         unsigned parentProc,
                                         unsigned childProc,
                                         CommWeightT cost,
                                         DeltaTracker &dt) {
        if (val == 0) {
            dt.Add(true, parentStep, childProc, cost);
            dt.Add(false, parentStep, parentProc, cost);
        }
    }

    template <typename DeltaTracker, typename CommWeightT>
    static inline void CalculateDeltaOutgoing(
        const ValueType &val, unsigned nodeStep, unsigned nodeProc, unsigned childProc, CommWeightT cost, DeltaTracker &dt) {
        if (val > 0) {
            CommWeightT totalCost = cost * val;
            dt.Add(true, nodeStep, childProc, totalCost);
            dt.Add(false, nodeStep, nodeProc, totalCost);
        }
    }
};

struct LazyCommCostPolicy {
    using ValueType = std::vector<unsigned>;

    template <typename DS, typename CommWeightT, typename MarkStepFn>
    static inline void AttributeCommunication(DS &ds,
                                              const CommWeightT &cost,
                                              const unsigned uStep,
                                              const unsigned uProc,
                                              const unsigned vProc,
                                              const unsigned vStep,
                                              const ValueType &val,
                                              MarkStepFn &&markStep) {
        // val contains v_step (already added).
        // Check if v_step is the new minimum.
        unsigned minStep = std::numeric_limits<unsigned>::max();
        for (unsigned s : val) {
            minStep = std::min(minStep, s);
        }

        if (minStep == vStep) {
            // Check if it was strictly smaller than previous min.
            unsigned prevMin = std::numeric_limits<unsigned>::max();
            for (size_t i = 0; i < val.size() - 1; ++i) {
                prevMin = std::min(prevMin, val[i]);
            }

            if (vStep < prevMin) {
                if (prevMin != std::numeric_limits<unsigned>::max() && prevMin > 0) {
                    ds.StepProcReceive(prevMin - 1, vProc) -= cost;
                    ds.StepProcSend(prevMin - 1, uProc) -= cost;
                    markStep(prevMin - 1);
                }
                if (vStep > 0) {
                    ds.StepProcReceive(vStep - 1, vProc) += cost;
                    ds.StepProcSend(vStep - 1, uProc) += cost;
                    markStep(vStep - 1);
                }
            }
        }
    }

    template <typename DS, typename CommWeightT, typename MarkStepFn>
    static inline void UnattributeCommunication(DS &ds,
                                                const CommWeightT &cost,
                                                const unsigned uStep,
                                                const unsigned uProc,
                                                const unsigned vProc,
                                                const unsigned vStep,
                                                const ValueType &val,
                                                MarkStepFn &&markStep) {
        // val is state AFTER removal.

        if (val.empty()) {
            // Removed the last child.
            if (vStep > 0) {
                ds.StepProcReceive(vStep - 1, vProc) -= cost;
                ds.StepProcSend(vStep - 1, uProc) -= cost;
                markStep(vStep - 1);
            }
        } else {
            // Check if v_step was the unique minimum.
            unsigned newMin = val[0];
            for (unsigned s : val) {
                newMin = std::min(newMin, s);
            }

            if (vStep < newMin) {
                // v_step was the unique minimum.
                if (vStep > 0) {
                    ds.StepProcReceive(vStep - 1, vProc) -= cost;
                    ds.StepProcSend(vStep - 1, uProc) -= cost;
                    markStep(vStep - 1);
                }
                if (newMin > 0) {
                    ds.StepProcReceive(newMin - 1, vProc) += cost;
                    ds.StepProcSend(newMin - 1, uProc) += cost;
                    markStep(newMin - 1);
                }
            }
        }
    }

    /// Remove outgoing communication when a parent node moves (val unchanged).
    /// For Lazy, both send and recv are at min(child_steps_on_proc) - 1.
    template <typename DS, typename CommWeightT, typename MarkStepFn>
    static inline void RemoveOutgoingComm(DS &ds,
                                          const CommWeightT &cost,
                                          unsigned parentStep,
                                          unsigned parentProc,
                                          unsigned childProc,
                                          const ValueType &val,
                                          MarkStepFn &&markStep) {
        if (val.empty()) {
            return;
        }
        unsigned minS = std::numeric_limits<unsigned>::max();
        for (unsigned s : val) {
            minS = std::min(minS, s);
        }
        if (minS > 0) {
            ds.StepProcSend(minS - 1, parentProc) -= cost;
            ds.StepProcReceive(minS - 1, childProc) -= cost;
            markStep(minS - 1);
        }
    }

    template <typename DS, typename CommWeightT, typename MarkStepFn>
    static inline void AddOutgoingComm(DS &ds,
                                       const CommWeightT &cost,
                                       unsigned parentStep,
                                       unsigned parentProc,
                                       unsigned childProc,
                                       const ValueType &val,
                                       MarkStepFn &&markStep) {
        if (val.empty()) {
            return;
        }
        unsigned minS = std::numeric_limits<unsigned>::max();
        for (unsigned s : val) {
            minS = std::min(minS, s);
        }
        if (minS > 0) {
            ds.StepProcSend(minS - 1, parentProc) += cost;
            ds.StepProcReceive(minS - 1, childProc) += cost;
            markStep(minS - 1);
        }
    }

    static inline bool AddChild(ValueType &val, unsigned step) {
        val.push_back(step);
        if (val.size() == 1) {
            return true;
        }
        unsigned minS = val[0];
        for (unsigned s : val) {
            minS = std::min(minS, s);
        }
        return step == minS;
    }

    static inline bool RemoveChild(ValueType &val, unsigned step) {
        auto it = std::find(val.begin(), val.end(), step);
        if (it != val.end()) {
            val.erase(it);
            if (val.empty()) {
                return true;
            }
            unsigned newMin = val[0];
            for (unsigned s : val) {
                newMin = std::min(newMin, s);
            }
            bool res = step < newMin;
            return res;
        }
        return false;
    }

    static inline void Reset(ValueType &val) { val.clear(); }

    static inline bool HasEntry(const ValueType &val) { return !val.empty(); }

    static inline bool IsSingleEntry(const ValueType &val) { return val.size() == 1; }

    // For outgoing comm (parent→children on proc), where is send/recv attributed?
    // Lazy: both at min(child_steps) - 1.
    static constexpr bool outgoing_send_at_parent_step = false;
    static constexpr bool outgoing_recv_at_parent_step = false;

    static inline int OutgoingSendStep(unsigned /*parentStep*/, const ValueType &val) {
        if (val.empty()) {
            return -1;
        }
        unsigned minS = std::numeric_limits<unsigned>::max();
        for (unsigned s : val) {
            minS = std::min(minS, s);
        }
        return minS > 0 ? static_cast<int>(minS - 1) : -1;
    }

    static inline int OutgoingRecvStep(unsigned parentStep, const ValueType &val) { return OutgoingSendStep(parentStep, val); }

    template <typename DeltaTracker, typename CommWeightT>
    static inline void CalculateDeltaRemove(const ValueType &val,
                                            unsigned childStep,
                                            unsigned parentStep,
                                            unsigned parentProc,
                                            unsigned childProc,
                                            CommWeightT cost,
                                            DeltaTracker &dt) {
        if (val.empty()) {
            return;
        }
        unsigned minS = val[0];
        for (unsigned s : val) {
            minS = std::min(minS, s);
        }

        if (childStep == minS) {
            int count = 0;
            for (unsigned s : val) {
                if (s == minS) {
                    count++;
                }
            }

            if (count == 1) {
                if (minS > 0) {
                    dt.Add(true, minS - 1, childProc, -cost);
                    dt.Add(false, minS - 1, parentProc, -cost);
                }
                if (val.size() > 1) {
                    unsigned nextMin = std::numeric_limits<unsigned>::max();
                    for (unsigned s : val) {
                        if (s != minS) {
                            nextMin = std::min(nextMin, s);
                        }
                    }
                    if (nextMin != std::numeric_limits<unsigned>::max() && nextMin > 0) {
                        dt.Add(true, nextMin - 1, childProc, cost);
                        dt.Add(false, nextMin - 1, parentProc, cost);
                    }
                }
            }
        }
    }

    template <typename DeltaTracker, typename CommWeightT>
    static inline void CalculateDeltaAdd(const ValueType &val,
                                         unsigned childStep,
                                         unsigned parentStep,
                                         unsigned parentProc,
                                         unsigned childProc,
                                         CommWeightT cost,
                                         DeltaTracker &dt) {
        if (val.empty()) {
            if (childStep > 0) {
                dt.Add(true, childStep - 1, childProc, cost);
                dt.Add(false, childStep - 1, parentProc, cost);
            }
        } else {
            unsigned minS = val[0];
            for (unsigned s : val) {
                minS = std::min(minS, s);
            }

            if (childStep < minS) {
                if (minS > 0) {
                    dt.Add(true, minS - 1, childProc, -cost);
                    dt.Add(false, minS - 1, parentProc, -cost);
                }
                if (childStep > 0) {
                    dt.Add(true, childStep - 1, childProc, cost);
                    dt.Add(false, childStep - 1, parentProc, cost);
                }
            }
        }
    }

    template <typename DeltaTracker, typename CommWeightT>
    static inline void CalculateDeltaOutgoing(
        const ValueType &val, unsigned nodeStep, unsigned nodeProc, unsigned childProc, CommWeightT cost, DeltaTracker &dt) {
        // Lazy places ALL comm at min(val)-1, not at each child step.
        if (!val.empty()) {
            unsigned minS = std::numeric_limits<unsigned>::max();
            for (unsigned s : val) {
                minS = std::min(minS, s);
            }
            if (minS > 0) {
                dt.Add(true, minS - 1, childProc, cost);
                dt.Add(false, minS - 1, nodeProc, cost);
            }
        }
    }
};

struct BufferedCommCostPolicy {
    using ValueType = std::vector<unsigned>;

    template <typename DS, typename CommWeightT, typename MarkStepFn>
    static inline void AttributeCommunication(DS &ds,
                                              const CommWeightT &cost,
                                              const unsigned uStep,
                                              const unsigned uProc,
                                              const unsigned vProc,
                                              const unsigned vStep,
                                              const ValueType &val,
                                              MarkStepFn &&markStep) {
        // Buffered: Send at u_step, Receive at min(child_steps) - 1.

        unsigned minStep = std::numeric_limits<unsigned>::max();
        for (unsigned s : val) {
            minStep = std::min(minStep, s);
        }

        if (minStep == vStep) {
            unsigned prevMin = std::numeric_limits<unsigned>::max();
            for (size_t i = 0; i < val.size() - 1; ++i) {
                prevMin = std::min(prevMin, val[i]);
            }

            if (vStep < prevMin) {
                if (prevMin != std::numeric_limits<unsigned>::max() && prevMin > 0) {
                    ds.StepProcReceive(prevMin - 1, vProc) -= cost;
                    markStep(prevMin - 1);
                }
                if (vStep > 0) {
                    ds.StepProcReceive(vStep - 1, vProc) += cost;
                    markStep(vStep - 1);
                }
            }
        }

        // Send side logic (u_step)
        // If this is the FIRST child on this proc, add send cost.
        if (val.size() == 1) {
            ds.StepProcSend(uStep, uProc) += cost;
            markStep(uStep);
        }
    }

    template <typename DS, typename CommWeightT, typename MarkStepFn>
    static inline void UnattributeCommunication(DS &ds,
                                                const CommWeightT &cost,
                                                const unsigned uStep,
                                                const unsigned uProc,
                                                const unsigned vProc,
                                                const unsigned vStep,
                                                const ValueType &val,
                                                MarkStepFn &&markStep) {
        // val is state AFTER removal.

        if (val.empty()) {
            // Removed last child.
            ds.StepProcSend(uStep, uProc) -= cost;    // Send side
            markStep(uStep);
            if (vStep > 0) {
                ds.StepProcReceive(vStep - 1, vProc) -= cost;    // Recv side
                markStep(vStep - 1);
            }
        } else {
            // Check if v_step was unique minimum for Recv side.
            unsigned newMin = val[0];
            for (unsigned s : val) {
                newMin = std::min(newMin, s);
            }

            if (vStep < newMin) {
                if (vStep > 0) {
                    ds.StepProcReceive(vStep - 1, vProc) -= cost;
                    markStep(vStep - 1);
                }
                if (newMin > 0) {
                    ds.StepProcReceive(newMin - 1, vProc) += cost;
                    markStep(newMin - 1);
                }
            }
            // Send side remains (val not empty).
        }
    }

    /// Remove outgoing communication when a parent node moves (val unchanged).
    /// For Buffered: send at parentStep, recv at min(child_steps_on_proc) - 1.
    template <typename DS, typename CommWeightT, typename MarkStepFn>
    static inline void RemoveOutgoingComm(DS &ds,
                                          const CommWeightT &cost,
                                          unsigned parentStep,
                                          unsigned parentProc,
                                          unsigned childProc,
                                          const ValueType &val,
                                          MarkStepFn &&markStep) {
        if (val.empty()) {
            return;
        }
        ds.StepProcSend(parentStep, parentProc) -= cost;
        markStep(parentStep);
        unsigned minS = std::numeric_limits<unsigned>::max();
        for (unsigned s : val) {
            minS = std::min(minS, s);
        }
        if (minS > 0) {
            ds.StepProcReceive(minS - 1, childProc) -= cost;
            markStep(minS - 1);
        }
    }

    template <typename DS, typename CommWeightT, typename MarkStepFn>
    static inline void AddOutgoingComm(DS &ds,
                                       const CommWeightT &cost,
                                       unsigned parentStep,
                                       unsigned parentProc,
                                       unsigned childProc,
                                       const ValueType &val,
                                       MarkStepFn &&markStep) {
        if (val.empty()) {
            return;
        }
        ds.StepProcSend(parentStep, parentProc) += cost;
        markStep(parentStep);
        unsigned minS = std::numeric_limits<unsigned>::max();
        for (unsigned s : val) {
            minS = std::min(minS, s);
        }
        if (minS > 0) {
            ds.StepProcReceive(minS - 1, childProc) += cost;
            markStep(minS - 1);
        }
    }

    static inline bool AddChild(ValueType &val, unsigned step) {
        val.push_back(step);
        if (val.size() == 1) {
            return true;    // Need update for send side
        }
        unsigned minS = val[0];
        for (unsigned s : val) {
            minS = std::min(minS, s);
        }
        return step == minS;    // Need update for recv side
    }

    static inline bool RemoveChild(ValueType &val, unsigned step) {
        auto it = std::find(val.begin(), val.end(), step);
        if (it != val.end()) {
            val.erase(it);
            if (val.empty()) {
                return true;    // Need update for send side
            }
            unsigned newMin = val[0];
            for (unsigned s : val) {
                newMin = std::min(newMin, s);
            }
            return step < newMin;    // Need update for recv side
        }
        return false;
    }

    static inline void Reset(ValueType &val) { val.clear(); }

    static inline bool HasEntry(const ValueType &val) { return !val.empty(); }

    static inline bool IsSingleEntry(const ValueType &val) { return val.size() == 1; }

    // For outgoing comm (parent→children on proc), where is send/recv attributed?
    // Buffered: send at parent step, recv at min(child_steps) - 1.
    static constexpr bool outgoing_send_at_parent_step = true;
    static constexpr bool outgoing_recv_at_parent_step = false;

    static inline int OutgoingSendStep(unsigned parentStep, const ValueType &val) {
        return !val.empty() ? static_cast<int>(parentStep) : -1;
    }

    static inline int OutgoingRecvStep(unsigned /*parentStep*/, const ValueType &val) {
        if (val.empty()) {
            return -1;
        }
        unsigned minS = std::numeric_limits<unsigned>::max();
        for (unsigned s : val) {
            minS = std::min(minS, s);
        }
        return minS > 0 ? static_cast<int>(minS - 1) : -1;
    }

    template <typename DeltaTracker, typename CommWeightT>
    static inline void CalculateDeltaRemove(const ValueType &val,
                                            unsigned childStep,
                                            unsigned parentStep,
                                            unsigned parentProc,
                                            unsigned childProc,
                                            CommWeightT cost,
                                            DeltaTracker &dt) {
        // Buffered: Send at parentStep, Recv at min(child_steps) - 1.
        // Removing a child only affects recv step (shifts min) and possibly
        // removes send entirely (if last child on this proc).

        if (val.empty()) {
            return;
        }

        unsigned minS = val[0];
        for (unsigned s : val) {
            minS = std::min(minS, s);
        }

        if (childStep == minS) {
            int count = 0;
            for (unsigned s : val) {
                if (s == minS) {
                    count++;
                }
            }

            if (count == 1) {
                // Unique min being removed.
                // Recv: remove from old min step.
                if (minS > 0) {
                    dt.Add(true, minS - 1, childProc, -cost);
                }

                if (val.size() == 1) {
                    // Last child on this proc: also remove send.
                    dt.Add(false, parentStep, parentProc, -cost);
                } else {
                    // Not last: recv shifts to nextMin. Send stays at parentStep (no delta).
                    unsigned nextMin = std::numeric_limits<unsigned>::max();
                    for (unsigned s : val) {
                        if (s != minS) {
                            nextMin = std::min(nextMin, s);
                        }
                    }

                    if (nextMin != std::numeric_limits<unsigned>::max() && nextMin > 0) {
                        dt.Add(true, nextMin - 1, childProc, cost);
                    }
                }
            }
        }
    }

    template <typename DeltaTracker, typename CommWeightT>
    static inline void CalculateDeltaAdd(const ValueType &val,
                                         unsigned childStep,
                                         unsigned parentStep,
                                         unsigned parentProc,
                                         unsigned childProc,
                                         CommWeightT cost,
                                         DeltaTracker &dt) {
        // Buffered: Send at parentStep, Recv at min(child_steps) - 1.

        if (val.empty()) {
            // First child on this proc: add send at parentStep, recv at childStep - 1.
            dt.Add(false, parentStep, parentProc, cost);
            if (childStep > 0) {
                dt.Add(true, childStep - 1, childProc, cost);
            }
        } else {
            unsigned minS = val[0];
            for (unsigned s : val) {
                minS = std::min(minS, s);
            }

            if (childStep < minS) {
                // New global minimum: recv shifts. Send stays at parentStep (no delta).
                if (minS > 0) {
                    dt.Add(true, minS - 1, childProc, -cost);
                }
                if (childStep > 0) {
                    dt.Add(true, childStep - 1, childProc, cost);
                }
            }
        }
    }

    template <typename DeltaTracker, typename CommWeightT>
    static inline void CalculateDeltaOutgoing(
        const ValueType &val, unsigned nodeStep, unsigned nodeProc, unsigned childProc, CommWeightT cost, DeltaTracker &dt) {
        // Buffered Outgoing (Node -> Children)
        // Node is parent (sender). Pays at node_step.
        // Children are receivers. Pay at child_step - 1.

        // Send side: node_step.
        // If val is not empty, we pay send cost ONCE.
        if (!val.empty()) {
            dt.Add(false, nodeStep, nodeProc, cost);
        }

        // Recv side: iterate steps in val (child steps).
        // But we only pay at min(val) - 1.
        if (!val.empty()) {
            unsigned minS = val[0];
            for (unsigned s : val) {
                minS = std::min(minS, s);
            }

            if (minS > 0) {
                dt.Add(true, minS - 1, childProc, cost);
            }
        }
    }
};

}    // namespace osp
