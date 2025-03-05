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

#include "hdagg_coarser.hpp"

class hdagg_coarser_variant : public hdagg_coarser {

  public:
    hdagg_coarser_variant() {};

    virtual ~hdagg_coarser_variant() = default;

    virtual std::string getCoarserName() const override { return "hdagg_order_coarser_variant"; };

    virtual RETURN_STATUS coarseDag(const ComputationalDag &dag_in, ComputationalDag &dag_out,
                                    std::vector<std::vector<VertexType>> &vertex_map) override;
};