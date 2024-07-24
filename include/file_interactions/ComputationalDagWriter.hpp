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
#include <boost/graph/graphviz.hpp>

#include <fstream>
#include <string>

/**
 * The ComputationalDagWriter class is responsible for writing the computational DAG to a file or output stream.
 */
class ComputationalDagWriter {
  private:
    const ComputationalDag &dag;

  public:
    struct EdgeWriter_DOT {
        const GraphType &graph;

        EdgeWriter_DOT(const GraphType &graph_) : graph(graph_) {}

        template<class VertexOrEdge>
        void operator()(std::ostream &out, const VertexOrEdge &i) const {
            out << "["
                << "comm_weight=\"" << graph[i].communicationWeight << "\";"
                << "]";
        }
    };

    struct VertexWriter_DOT {
        const GraphType &graph;

        VertexWriter_DOT(const GraphType &graph_) : graph(graph_) {}

        template<class VertexOrEdge>
        void operator()(std::ostream &out, const VertexOrEdge &i) const {
            out << "["
                << "work_weight=\"" << graph[i].workWeight << "\";"
                << "comm_weight=\"" << graph[i].communicationWeight << "\";"
                << "mem_weight=\"" << graph[i].memoryWeight << "\";"
                << "]";
        }
    };

    /**
     * Constructs a ComputationalDagWriter object with the given computational DAG.
     *
     * @param dag The computational DAG to be written.
     */
    ComputationalDagWriter(const ComputationalDag &dag) : dag(dag) {}

    /**
     * Writes the computational DAG to the specified output stream in DOT format.
     *
     * @tparam VertexWriterType The type of the vertex writer to be used.
     *         Default is VertexWriter_DOT.
     * @tparam EdgeWriterType The type of the edge writer to be used.
     *         Default is EdgeWriter_DOT.
     *
     * @param os The output stream to write the DOT representation of the computational DAG.
     */
    template<class VertexWriterType = VertexWriter_DOT, class EdgeWriterType = EdgeWriter_DOT>
    void write_dot(std::ostream &os) const;

    /**
     * Writes the computational DAG to the specified file in DOT format.
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
    template<class VertexWriterType = VertexWriter_DOT, class EdgeWriterType = EdgeWriter_DOT>
    void write_dot(const std::string &filename) const {
        std::ofstream os(filename);
        write_dot(os);
    }
};

template<class VertexWriterType, class EdgeWriterType>
void ComputationalDagWriter::write_dot(std::ostream &os) const {
    const auto &g = dag.getGraph();
    boost::write_graphviz(os, g, VertexWriterType(g), EdgeWriterType(g));
}