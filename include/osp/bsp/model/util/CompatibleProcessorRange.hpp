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

#include "osp/bsp/model/BspInstance.hpp"

namespace osp {

/**
 * @class CompatibleProcessorRange
 * @brief Helper class to efficiently iterate over compatible processors for a given node or node type.
 *
 * This class precomputes and stores the list of compatible processors for each node type.
 *
 * @tparam Graph_t The type of the computational DAG.
 */
template <typename GraphT>
class CompatibleProcessorRange {
    std::vector<std::vector<unsigned>> typeProcessorIdx_;
    const BspInstance<GraphT> *instance_ = nullptr;

  public:
    /**
     * @brief Default constructor.
     */
    CompatibleProcessorRange() = default;

    /**
     * @brief Constructs a CompatibleProcessorRange for the given BspInstance.
     *
     * @param inst The BspInstance.
     */
    CompatibleProcessorRange(const BspInstance<GraphT> &inst) { Initialize(inst); }

    /**
     * @brief Initializes the CompatibleProcessorRange with a BspInstance.
     *
     * @param inst The BspInstance.
     */
    void Initialize(const BspInstance<GraphT> &inst) {
        instance_ = &inst;

        if constexpr (HasTypedVerticesV<GraphT>) {
            typeProcessorIdx_.resize(inst.GetComputationalDag().NumVertexTypes());

            for (VTypeT<GraphT> vType = 0; v_type < inst.GetComputationalDag().NumVertexTypes(); v_type++) {
                for (unsigned proc = 0; proc < inst.NumberOfProcessors(); proc++) {
                    if (inst.IsCompatibleType(v_type, inst.ProcessorType(proc))) {
                        typeProcessorIdx_[v_type].push_back(proc);
                    }
                }
            }
        }
    }

    /**
     * @brief Returns a range of compatible processors for a given node type.
     *
     * @param type The node type.
     * @return A const reference to a vector of compatible processor indices.
     */
    [[nodiscard]] const auto &CompatibleProcessorsType(const VTypeT<GraphT> type) const {
        assert(instance_ != nullptr);
        if constexpr (HasTypedVerticesV<GraphT>) {
            return typeProcessorIdx_[type];
        } else {
            return instance_->Processors();
        }
    }

    /**
     * @brief Returns a range of compatible processors for a given vertex.
     *
     * @param vertex The vertex index.
     * @return A const reference to a vector of compatible processor indices.
     */
    [[nodiscard]] const auto &CompatibleProcessorsVertex(const VertexIdxT<GraphT> vertex) const {
        assert(instance_ != nullptr);
        return CompatibleProcessorsType(instance_->GetComputationalDag().VertexType(vertex));
    }
};

}    // namespace osp
