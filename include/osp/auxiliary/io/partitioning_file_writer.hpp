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

#include <fstream>
#include <iostream>

#include "osp/partitioning/model/partitioning.hpp"
#include "osp/partitioning/model/partitioning_replication.hpp"

namespace osp {
namespace file_writer {

template <typename hypergraph_t>
void write_txt(std::ostream &os, const Partitioning<hypergraph_t> &partition) {
    using index_type = typename hypergraph_t::vertex_idx;

    os << "%% Partitioning for " << partition.getInstance().getNumberOfPartitions() << " parts." << std::endl;

    for (index_type node = 0; node < partition.getInstance().getHypergraph().num_vertices(); ++node) {
        os << node << " " << partition.assignedPartition(node) << std::endl;
    }
}

template <typename hypergraph_t>
void write_txt(const std::string &filename, const Partitioning<hypergraph_t> &partition) {
    std::ofstream os(filename);
    write_txt(os, partition);
}

template <typename hypergraph_t>
void write_txt(std::ostream &os, const PartitioningWithReplication<hypergraph_t> &partition) {
    using index_type = typename hypergraph_t::vertex_idx;

    os << "%% Partitioning for " << partition.getInstance().getNumberOfPartitions() << " parts with replication." << std::endl;

    for (index_type node = 0; node < partition.getInstance().getHypergraph().num_vertices(); ++node) {
        os << node;
        for (unsigned part : partition.assignedPartitions(node)) { os << " " << part; }
        os << std::endl;
    }
}

template <typename hypergraph_t>
void write_txt(const std::string &filename, const PartitioningWithReplication<hypergraph_t> &partition) {
    std::ofstream os(filename);
    write_txt(os, partition);
}

}    // namespace file_writer
}    // namespace osp
