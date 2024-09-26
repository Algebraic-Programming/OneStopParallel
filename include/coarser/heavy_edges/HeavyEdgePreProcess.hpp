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

#include "model/ComputationalDag.hpp"
#include "structures/union_find.hpp"

std::vector<std::vector<VertexType>> heavy_edge_preprocess(const ComputationalDag &graph, const float heavy_is_x_times_median, const float min_percent_components_retained, const float bound_component_weight_percent);

