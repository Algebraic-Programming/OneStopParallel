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

#include "bsp/model/BspSchedule.hpp"
#include "concepts/computational_dag_concept.hpp"

#include <fstream>
#include <string>

namespace osp {

class DotFileWriter {
  private:
    template<typename Graph_t>

    struct EdgeWriter_DOT {
        const Graph_t &graph;

        EdgeWriter_DOT(const Graph_t &graph_) : graph(graph_) {}

        void operator()(std::ostream &out, const edge_desc_t<Graph_t> &i) const {
            out << source(i, graph) << "->" << target(i, graph) << " ["
                << "comm_weight=\"" << graph.edge_comm_weight(i) << "\";"
                << "]";
        }
    };

    template<typename Graph_t>
    struct VertexWriterSchedule_DOT {

        const BspSchedule<Graph_t> &schedule;

        VertexWriterSchedule_DOT(const BspSchedule<Graph_t> &schedule_) : schedule(schedule_) {}

        void operator()(std::ostream &out, const vertex_idx_t<Graph_t> &i) const {
            out << i << " ["
                << "work_weight=\"" << schedule.getInstance().getComputationalDag().vertex_work_weight(i) << "\";"
                << "comm_weight=\"" << schedule.getInstance().getComputationalDag().vertex_comm_weight(i) << "\";"
                << "mem_weight=\"" << schedule.getInstance().getComputationalDag().vertex_mem_weight(i) << "\";";

            if constexpr (has_typed_vertices_v<Graph_t>) {

                out << "type=\"" << schedule.getInstance().getComputationalDag().vertex_type(i) << "\";";
            }

            out << "proc=\"" << schedule.assignedProcessor(i) << "\";" << "superstep=\""
                << schedule.assignedSuperstep(i) << "\";";

            out << "]";
        }
    };

    template<typename Graph_t>
    struct VertexWriterGraph_DOT {

        const Graph_t &graph;

        VertexWriterGraph_DOT(const Graph_t &graph_) : graph(graph_) {}

        void operator()(std::ostream &out, const vertex_idx_t<Graph_t> &i) const {
            out << i << " ["
                << "work_weight=\"" << graph.vertex_work_weight(i) << "\";"
                << "comm_weight=\"" << graph.vertex_comm_weight(i) << "\";"
                << "mem_weight=\"" << graph.vertex_mem_weight(i) << "\";";

            if constexpr (has_typed_vertices_v<Graph_t>) {

                out << "type=\"" << graph.vertex_type(i) << "\";";
            }

            out << "]";
        }
    };

  public:
    /**
     * Constructs a DotFileWriter object with the given BspSchdule.
     *
     * @param schdule_
     */
    DotFileWriter() {}

    /**
     * Writes the BspSchedule to the specified output stream in DOT format.
     *
     * @tparam VertexWriterType The type of the vertex writer to be used.
     *         Default is VertexWriter_DOT.
     * @tparam EdgeWriterType The type of the edge writer to be used.
     *         Default is EdgeWriter_DOT.
     *
     * @param os The output stream to write the DOT representation of the computational DAG.
     */
    template<typename Graph_t>
    void write_dot(std::ostream &os, const BspSchedule<Graph_t> &schedule) const {

        VertexWriterSchedule_DOT<Graph_t> vertex_writer(schedule);

        os << "digraph G {\n";
        for (const auto &v : schedule.getInstance().vertices()) {
            vertex_writer(os, v);
            os << "\n";
        }

        if constexpr (is_directed_graph_edge_desc_v<Graph_t>) {
            EdgeWriter_DOT<Graph_t> edge_writer(schedule.getInstance().getComputationalDag());

            for (const auto &e : schedule.getInstance().getComputationalDag().edges()) {
                edge_writer(os, e);
                os << "\n";
            }

        } else {

            const auto &graph = schedule.getInstance().getComputationalDag();
            for (const auto &v : graph.vertices()) {
                for (const auto &child : graph.children(v)) {
                    os << v << "->" << child << "\n";
                }
            }
        }
        os << "}\n";
    }

    /**
     * Writes the BspSchedule to the specified file in DOT format.
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
    template<typename Graph_t>
    void write_dot(const std::string &filename, const BspSchedule<Graph_t> &schedule) const {
        std::ofstream os(filename);
        write_dot(os, schedule);
    }

    template<typename Graph_t>
    void write_dot(std::ostream &os, const Graph_t &graph) const {

        static_assert(is_computational_dag_v<Graph_t>, "Graph_t must be a computational DAG");

        VertexWriterGraph_DOT<Graph_t> vertex_writer(graph);

        os << "digraph G {\n";
        for (const auto &v : graph.vertices()) {
            vertex_writer(os, v);
            os << "\n";
        }

        if constexpr (is_directed_graph_edge_desc_v<Graph_t>) {
            EdgeWriter_DOT<Graph_t> edge_writer(graph);

            for (const auto &e : graph.edges()) {
                edge_writer(os, e);
                os << "\n";
            }

        } else {

            for (const auto &v : graph.vertices()) {
                for (const auto &child : graph.children(v)) {
                    os << v << "->" << child << "\n";
                }
            }
        }
        os << "}\n";
    }

    template<typename Graph_t>
    void write_dot(const std::string &filename, const Graph_t &graph) const {

        static_assert(is_computational_dag_v<Graph_t>, "Graph_t must be a computational DAG");

        std::ofstream os(filename);
        write_dot(os, graph);
    }

    // void write_txt(const std::string &filename) const {
    //     std::ofstream os(filename);
    //     write_txt(os);
    // }

    // void write_txt(std::ostream &os) const;

    // void write_sankey(const std::string &filename) const {
    //     std::ofstream os(filename);
    //     write_sankey(os);
    // }

    // void write_sankey(std::ostream &os) const;
};

} // namespace osp