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

#include "model/DAGPartition.hpp"
#include "model/ComputationalDag.hpp"
#include <boost/graph/graphviz.hpp>

#include <fstream>
#include <string>

class DAGPartitionWriter {
  private:
    const DAGPartition &partition;

  public:
    struct GraphPropertiesWriter_DOT {
        GraphPropertiesWriter_DOT() {};
        void operator()(std::ostream &out) const {
            out << "node[colorscheme=paired12;style=filled;];" << std::endl;
        }
    };

    struct EdgeWriterSchedule_DOT {
        const DAGPartition &partition;

        EdgeWriterSchedule_DOT(const DAGPartition &partition_) : partition(partition_) {}

        template<class VertexOrEdge>
        void operator()(std::ostream &out, const VertexOrEdge &i) const {
            out << "["
                << "comm_weight=\"" << partition.getInstance().getComputationalDag().getGraph()[i].communicationWeight << "\";";
            if (partition.assignedProcessor(i.m_source) == partition.assignedProcessor(i.m_target)) {
                out << "style=dotted;";
            }
            out << "]";
        }
    };

    struct VertexWriterSchedule_DOT {
        const DAGPartition &partition;

        VertexWriterSchedule_DOT(const DAGPartition &partition_) : partition(partition_) {}

        template<class VertexOrEdge>
        void operator()(std::ostream &out, const VertexOrEdge &i) const {
            out << "["
                << "work_weight=\"" << partition.getInstance().getComputationalDag().nodeWorkWeight(i) << "\";"
                << "comm_weight=\"" << partition.getInstance().getComputationalDag().nodeCommunicationWeight(i) << "\";"
                << "mem_weight=\"" << partition.getInstance().getComputationalDag().nodeMemoryWeight(i) << "\";"
                << "proc=\"" << partition.assignedProcessor(i) << "\";"
                << "color=" << partition.assignedProcessor(i)+1 << ";";

            out << "]";
        }
    };

    /**
     * Constructs a DAGPartitionWriter object with the given DAGPartition.
     *
     * @param partition_
     */
    DAGPartitionWriter(const DAGPartition &partition_) : partition(partition_) {}

    /**
     * Writes the DAGPartition to the specified output stream in DOT format.
     *
     * @tparam VertexWriterType The type of the vertex writer to be used.
     *         Default is VertexWriter_DOT.
     * @tparam EdgeWriterType The type of the edge writer to be used.
     *         Default is EdgeWriter_DOT.
     *
     * @param os The output stream to write the DOT representation of the computational DAG.
     */
    template<class VertexWriterType = VertexWriterSchedule_DOT, class EdgeWriterType = EdgeWriterSchedule_DOT, class GraphWriterType = GraphPropertiesWriter_DOT>
    void write_dot(std::ostream &os) const;

    /**
     * Writes the DAGPartition to the specified file in DOT format.
     *
     * @note The file will be overwritten if it already exists.
     * @note The file will be created if it does not exist.
     * @note This function is an alias for `write_dot(std::ostream &os)`
     *
     * @tparam VertexWriterType The type of the vertex writer to be used.
     *         Default is VertexWriter_DOT.
     * @tparam EdgeWriterType The type of the edge writer to be used.
     *         Default is EdgeWriter_DOT.
     *
     * @param filename The name of the file to write the DOT representation of the computational DAG.
     */
    template<class VertexWriterType = VertexWriterSchedule_DOT, class EdgeWriterType = EdgeWriterSchedule_DOT, class GraphWriterType = GraphPropertiesWriter_DOT>
    void write_dot(const std::string &filename) const {
        std::ofstream os(filename);
        write_dot(os);
    }

    void write_txt(const std::string &filename) const {
        std::ofstream os(filename);
        write_txt(os);
    }

    void write_txt(std::ostream &os) const;

};

template<class VertexWriterType, class EdgeWriterType, class GraphWriterType>
void DAGPartitionWriter::write_dot(std::ostream &os) const {
    const auto &g = partition.getInstance().getComputationalDag().getGraph();
    boost::write_graphviz(os, g, VertexWriterType(partition), EdgeWriterType(partition), GraphWriterType());
}