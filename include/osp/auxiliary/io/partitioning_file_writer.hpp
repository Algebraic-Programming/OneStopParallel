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

#include "osp/partitioning/model/partitioning.hpp"
#include "osp/partitioning/model/partitioning_replication.hpp"
#include <fstream>
#include <iostream>

namespace osp { namespace file_writer {

template<typename index_type, typename workw_type, typename memw_type, typename commw_type>
void write_txt(std::ostream &os, const Partitioning<index_type, workw_type, memw_type, commw_type> &partition) {

    os << "\%\% Partitioning for " << partition.getInstance().getNumberOfPartitions() << " parts." << std::endl;

    for(index_type node = 0; node < partition.getInstance().getHypergraph().num_vertices(); ++node)
        os << node << " " << partition.assignedPartition(node) << std::endl;
}

template<typename index_type, typename workw_type, typename memw_type, typename commw_type>
void write_txt(const std::string &filename, const Partitioning<index_type, workw_type, memw_type, commw_type> &partition) {
    std::ofstream os(filename);
    write_txt(os, partition);
}

template<typename index_type, typename workw_type, typename memw_type, typename commw_type>
void write_txt(std::ostream &os, const PartitioningWithReplication<index_type, workw_type, memw_type, commw_type> &partition) {

    os << "\%\% Partitioning for " << partition.getInstance().getNumberOfPartitions() << " parts with replication." << std::endl;

    for(index_type node = 0; node < partition.getInstance().getHypergraph().num_vertices(); ++node)
    {
        os << node;
        for(unsigned part : partition.assignedPartitions(node))
            os << " " << part;
        os << std::endl;
    }
}

template<typename index_type, typename workw_type, typename memw_type, typename commw_type>
void write_txt(const std::string &filename, const PartitioningWithReplication<index_type, workw_type, memw_type, commw_type> &partition) {
    std::ofstream os(filename);
    write_txt(os, partition);
}

}} // namespace osp::file_writer