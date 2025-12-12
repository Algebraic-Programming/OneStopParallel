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
#include <string>

#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/model/BspScheduleRecomp.hpp"
#include "osp/concepts/computational_dag_concept.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"

namespace osp {

class DotFileWriter {
  private:
    template <typename GraphT>

    struct EdgeWriterDot {
        const GraphT &graph_;

        EdgeWriterDot(const GraphT &graph) : graph_(graph) {}

        void operator()(std::ostream &out, const edge_desc_t<Graph_t> &i) const {
            out << source(i, graph_) << "->" << target(i, graph_) << " ["
                << "comm_weight=\"" << graph_.edge_comm_weight(i) << "\";"
                << "]";
        }
    };

    template <typename GraphT>
    struct VertexWriterScheduleDot {
        const BspSchedule<GraphT> &schedule_;

        VertexWriterScheduleDot(const BspSchedule<GraphT> &schedule) : schedule_(schedule) {}

        void operator()(std::ostream &out, const vertex_idx_t<Graph_t> &i) const {
            out << i << " ["
                << "work_weight=\"" << schedule_.getInstance().getComputationalDag().vertex_work_weight(i) << "\";"
                << "comm_weight=\"" << schedule_.getInstance().getComputationalDag().vertex_comm_weight(i) << "\";"
                << "mem_weight=\"" << schedule_.getInstance().getComputationalDag().vertex_mem_weight(i) << "\";";

            if constexpr (has_typed_vertices_v<Graph_t>) {
                out << "type=\"" << schedule_.getInstance().getComputationalDag().vertex_type(i) << "\";";
            }

            out << "proc=\"" << schedule_.assignedProcessor(i) << "\";" << "superstep=\"" << schedule_.assignedSuperstep(i)
                << "\";";

            out << "]";
        }
    };

    template <typename GraphT>
    struct VertexWriterScheduleRecompDot {
        const BspScheduleRecomp<GraphT> &schedule_;

        VertexWriterScheduleRecompDot(const BspScheduleRecomp<GraphT> &schedule) : schedule_(schedule) {}

        void operator()(std::ostream &out, const vertex_idx_t<Graph_t> &i) const {
            out << i << " ["
                << "work_weight=\"" << schedule_.getInstance().getComputationalDag().vertex_work_weight(i) << "\";"
                << "comm_weight=\"" << schedule_.getInstance().getComputationalDag().vertex_comm_weight(i) << "\";"
                << "mem_weight=\"" << schedule_.getInstance().getComputationalDag().vertex_mem_weight(i) << "\";";

            if constexpr (has_typed_vertices_v<Graph_t>) {
                out << "type=\"" << schedule_.getInstance().getComputationalDag().vertex_type(i) << "\";";
            }

            out << "proc=\"(";
            for (size_t j = 0; j < schedule_.assignments(i).size() - 1; ++j) {
                out << schedule_.assignments(i)[j].first << ",";
            }
            out << schedule_.assignments(i)[schedule_.assignments(i).size() - 1].first << ")\";"
                << "superstep=\"(";
            for (size_t j = 0; j < schedule_.assignments(i).size() - 1; ++j) {
                out << schedule_.assignments(i)[j].second << ",";
            }
            out << schedule_.assignments(i)[schedule_.assignments(i).size() - 1].second << ")\";";

            bool found = false;

            for (const auto &[key, val] : schedule_.getCommunicationSchedule()) {
                if (std::get<0>(key) == i) {
                    if (!found) {
                        out << "cs=\"[";
                        found = true;
                    } else {
                        out << ";";
                    }

                    out << "(" << std::get<1>(key) << "," << std::get<2>(key) << "," << val << ")";
                }
            }

            if (found) {
                out << "]\";";
            }

            out << "]";
        }
    };

    template <typename GraphT>
    struct VertexWriterDuplicateRecompScheduleDot {
        const GraphT &graph_;
        const std::vector<std::string> name_;
        const std::vector<unsigned> nodeToProc_;
        const std::vector<unsigned> nodeToSuperstep_;

        VertexWriterDuplicateRecompScheduleDot(const GraphT &graph,
                                               const std::vector<std::string> &name,
                                               std::vector<unsigned> &nodeToProc,
                                               std::vector<unsigned> &nodeToSuperstep)
            : graph_(graph), name_(name), nodeToProc_(nodeToProc), nodeToSuperstep_(nodeToSuperstep) {}

        template <class VertexOrEdge>
        void operator()(std::ostream &out, const VertexOrEdge &i) const {
            out << i << " [" << "label=\"" << name_[i] << "\";" << "work_weight=\"" << graph_.vertex_work_weight(i) << "\";"
                << "comm_weight=\"" << graph_.vertex_comm_weight(i) << "\";" << "mem_weight=\"" << graph_.vertex_mem_weight(i)
                << "\";" << "proc=\"" << nodeToProc_[i] << "\";" << "superstep=\"" << nodeToSuperstep_[i] << "\";";

            out << "]";
        }
    };

    template <typename GraphT>
    struct VertexWriterScheduleCsDot {
        const BspScheduleCS<GraphT> &schedule_;

        VertexWriterScheduleCsDot(const BspScheduleCS<GraphT> &schedule) : schedule_(schedule) {}

        void operator()(std::ostream &out, const vertex_idx_t<Graph_t> &i) const {
            out << i << " ["
                << "work_weight=\"" << schedule_.getInstance().getComputationalDag().vertex_work_weight(i) << "\";"
                << "comm_weight=\"" << schedule_.getInstance().getComputationalDag().vertex_comm_weight(i) << "\";"
                << "mem_weight=\"" << schedule_.getInstance().getComputationalDag().vertex_mem_weight(i) << "\";";

            if constexpr (has_typed_vertices_v<Graph_t>) {
                out << "type=\"" << schedule_.getInstance().getComputationalDag().vertex_type(i) << "\";";
            }

            out << "proc=\"" << schedule_.assignedProcessor(i) << "\";" << "superstep=\"" << schedule_.assignedSuperstep(i)
                << "\";";

            bool found = false;

            for (const auto &[key, val] : schedule_.getCommunicationSchedule()) {
                if (std::get<0>(key) == i) {
                    if (!found) {
                        out << "cs=\"[";
                        found = true;
                    } else {
                        out << ";";
                    }

                    out << "(" << std::get<1>(key) << "," << std::get<2>(key) << "," << val << ")";
                }
            }

            if (found) {
                out << "]\";";
            }

            out << "]";
        }
    };

    template <typename GraphT>
    struct VertexWriterGraphDot {
        const GraphT &graph_;

        VertexWriterGraphDot(const GraphT &graph) : graph_(graph) {}

        void operator()(std::ostream &out, const vertex_idx_t<Graph_t> &i) const {
            out << i << " ["
                << "work_weight=\"" << graph_.vertex_work_weight(i) << "\";"
                << "comm_weight=\"" << graph_.vertex_comm_weight(i) << "\";"
                << "mem_weight=\"" << graph_.vertex_mem_weight(i) << "\";";

            if constexpr (has_typed_vertices_v<Graph_t>) {
                out << "type=\"" << graph_.vertex_type(i) << "\";";
            }

            out << "]";
        }
    };

    template <typename GraphT, typename ColorContainerT>
    struct ColoredVertexWriterGraphDot {
        const GraphT &graph_;
        const ColorContainerT &colors_;
        std::vector<std::string> colorStrings_;
        std::vector<std::string> shapeStrings_;

        ColoredVertexWriterGraphDot(const GraphT &graph, const ColorContainerT &colors) : graph_(graph), colors_(colors) {
            colorStrings_ = {"lightcoral",      "palegreen",   "lightblue",     "gold",
                             "orchid",          "sandybrown",  "aquamarine",    "burlywood",
                             "hotpink",         "yellowgreen", "skyblue",       "khaki",
                             "violet",          "salmon",      "turquoise",     "tan",
                             "deeppink",        "chartreuse",  "deepskyblue",   "lemonchiffon",
                             "magenta",         "orangered",   "cyan",          "wheat",
                             "mediumvioletred", "limegreen",   "dodgerblue",    "lightyellow",
                             "darkviolet",      "tomato",      "paleturquoise", "bisque",
                             "crimson",         "lime",        "steelblue",     "papayawhip",
                             "purple",          "darkorange",  "cadetblue",     "peachpuff",
                             "indianred",       "springgreen", "powderblue",    "cornsilk",
                             "mediumorchid",    "chocolate",   "darkturquoise", "navajowhite",
                             "firebrick",       "seagreen",    "royalblue",     "lightgoldenrodyellow",
                             "darkmagenta",     "coral",       "teal",          "moccasin",
                             "maroon",          "forestgreen", "blue",          "yellow",
                             "darkorchid",      "red",         "green",         "navy",
                             "darkred",         "darkgreen",   "mediumblue",    "ivory",
                             "indigo",          "orange",      "darkcyan",      "antiquewhite"};

            shapeStrings_ = {"oval", "rect", "hexagon", "parallelogram"};
        }

        void operator()(std::ostream &out, const vertex_idx_t<Graph_t> &i) const {
            if (i >= static_cast<vertex_idx_t<Graph_t>>(colors_.size())) {
                // Fallback for safety: print without color if colors vector is mismatched or palette is empty.
                out << i << " [";
            } else {
                // Use modulo operator to cycle through the fixed palette if there are more color
                // groups than available colors.
                const std::string &color = colorStrings_[colors_[i] % colorStrings_.size()];
                out << i << " [style=filled;fillcolor=" << color << ";";
            }

            out << "work_weight=\"" << graph_.vertex_work_weight(i) << "\";"
                << "comm_weight=\"" << graph_.vertex_comm_weight(i) << "\";"
                << "mem_weight=\"" << graph_.vertex_mem_weight(i) << "\";";

            if constexpr (has_typed_vertices_v<Graph_t>) {
                out << "type=\"" << graph_.vertex_type(i) << "\";shape=\""
                    << shapeStrings_[graph_.vertex_type(i) % shapeStrings_.size()] << "\";";
            }

            out << "]";
        }
    };

    template <typename GraphT, typename VertexWriterT>
    void WriteGraphStructure(std::ostream &os, const GraphT &graph, const VertexWriterT &vertexWriter) const {
        os << "digraph G {\n";
        for (const auto &v : graph.vertices()) {
            vertexWriter(os, v);
            os << "\n";
        }

        if constexpr (has_edge_weights_v<Graph_t>) {
            EdgeWriterDot<GraphT> edgeWriter(graph);

            for (const auto &e : edges(graph)) {
                edgeWriter(os, e);
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
    template <typename GraphT>
    void WriteSchedule(std::ostream &os, const BspSchedule<GraphT> &schedule) const {
        write_graph_structure(os, schedule.getInstance().getComputationalDag(), VertexWriterScheduleDot<GraphT>(schedule));
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
    template <typename GraphT>
    void WriteSchedule(const std::string &filename, const BspSchedule<GraphT> &schedule) const {
        std::ofstream os(filename);
        write_schedule(os, schedule);
    }

    template <typename GraphT>
    void WriteScheduleCs(std::ostream &os, const BspScheduleCS<GraphT> &schedule) const {
        write_graph_structure(os, schedule.getInstance().getComputationalDag(), VertexWriterScheduleCsDot<GraphT>(schedule));
    }

    template <typename GraphT>
    void WriteScheduleCs(const std::string &filename, const BspScheduleCS<GraphT> &schedule) const {
        std::ofstream os(filename);
        write_schedule_cs(os, schedule);
    }

    template <typename GraphT>
    void WriteScheduleRecomp(std::ostream &os, const BspScheduleRecomp<GraphT> &schedule) const {
        write_graph_structure(os, schedule.getInstance().getComputationalDag(), VertexWriterScheduleRecompDot<GraphT>(schedule));
    }

    template <typename GraphT>
    void WriteScheduleRecomp(const std::string &filename, const BspScheduleRecomp<GraphT> &schedule) const {
        std::ofstream os(filename);
        write_schedule_recomp(os, schedule);
    }

    template <typename GraphT>
    void WriteScheduleRecompDuplicate(std::ostream &os, const BspScheduleRecomp<GraphT> &schedule) const {
        const auto &g = schedule.getInstance().getComputationalDag();

        using VertexType = vertex_idx_t<Graph_t>;

        std::vector<std::string> names(schedule.getTotalAssignments());
        std::vector<unsigned> nodeToProc(schedule.getTotalAssignments());
        std::vector<unsigned> nodeToSuperstep(schedule.getTotalAssignments());

        std::unordered_map<VertexType, std::vector<size_t>> vertex_to_idx;

        using vertex_type_t_or_default
            = std::conditional_t<is_computational_dag_typed_vertices_v<Graph_t>, v_type_t<Graph_t>, unsigned>;
        using edge_commw_t_or_default = std::conditional_t<has_edge_weights_v<Graph_t>, e_commw_t<Graph_t>, v_commw_t<Graph_t>>;

        using cdag_vertex_impl_t
            = cdag_vertex_impl<vertex_idx_t<Graph_t>, v_workw_t<Graph_t>, v_commw_t<Graph_t>, v_memw_t<Graph_t>, vertex_type_t_or_default>;
        using cdag_edge_impl_t = cdag_edge_impl<edge_commw_t_or_default>;

        using graph_t = computational_dag_edge_idx_vector_impl<cdag_vertex_impl_t, cdag_edge_impl_t>;

        graph_t g2;

        size_t idxNew = 0;

        for (const auto &node : g.vertices()) {
            if (schedule.assignments(node).size() == 1) {
                g2.add_vertex(
                    g.vertex_work_weight(node), g.vertex_comm_weight(node), g.vertex_mem_weight(node), g.vertex_type(node));

                names[idxNew] = std::to_string(node);
                nodeToProc[idxNew] = schedule.assignments(node)[0].first;
                nodeToSuperstep[idxNew] = schedule.assignments(node)[0].second;

                vertex_to_idx.insert({node, {idx_new}});
                idxNew++;

            } else {
                std::vector<size_t> idxs;
                for (unsigned i = 0; i < schedule.assignments(node).size(); ++i) {
                    g2.add_vertex(
                        g.vertex_work_weight(node), g.vertex_comm_weight(node), g.vertex_mem_weight(node), g.vertex_type(node));

                    names[idxNew] = std::to_string(node).append("_").append(std::to_string(i));
                    nodeToProc[idxNew] = schedule.assignments(node)[i].first;
                    nodeToSuperstep[idxNew] = schedule.assignments(node)[i].second;

                    idxs.push_back(idxNew++);
                }
                vertex_to_idx.insert({node, idxs});
            }
        }

        for (const auto &[key, val] : vertex_to_idx) {
            if (val.size() == 1) {
                for (const auto &target : g.children(key)) {
                    for (const auto &new_node_target : vertex_to_idx[target]) {
                        g2.add_edge(val[0], new_node_target);
                    }
                }

            } else {
                std::unordered_set<unsigned> assigned_processors;

                for (const auto &assignment : schedule.assignments(key)) {
                    assigned_processors.insert(assignment.first);
                }

                for (unsigned i = 0; i < val.size(); i++) {
                    for (const auto &target : g.children(key)) {
                        for (size_t j = 0; j < vertex_to_idx[target].size(); j++) {
                            if (assigned_processors.find(node_to_proc[vertex_to_idx[target][j]]) == assigned_processors.end()
                                || node_to_proc[val[i]] == node_to_proc[vertex_to_idx[target][j]]) {
                                g2.add_edge(val[i], vertex_to_idx[target][j]);
                            }
                        }
                    }
                }
            }
        }

        write_graph_structure(os, g2, VertexWriterDuplicateRecompSchedule_DOT<graph_t>(g2, names, node_to_proc, node_to_superstep));
    }

    template <typename GraphT>
    void WriteScheduleRecompDuplicate(const std::string &filename, const BspScheduleRecomp<GraphT> &schedule) const {
        std::ofstream os(filename);
        write_schedule_recomp_duplicate(os, schedule);
    }

    template <typename GraphT, typename ColorContainerT>
    void WriteColoredGraph(std::ostream &os, const GraphT &graph, const ColorContainerT &colors) const {
        static_assert(is_computational_dag_v<Graph_t>, "Graph_t must be a computational DAG");

        write_graph_structure(os, graph, ColoredVertexWriterGraphDot<GraphT, ColorContainerT>(graph, colors));
    }

    template <typename GraphT, typename ColorContainerT>
    void WriteColoredGraph(const std::string &filename, const GraphT &graph, const ColorContainerT &colors) const {
        static_assert(is_computational_dag_v<Graph_t>, "Graph_t must be a computational DAG");

        std::ofstream os(filename);
        write_colored_graph(os, graph, colors);
    }

    template <typename GraphT>
    void WriteGraph(std::ostream &os, const GraphT &graph) const {
        static_assert(is_computational_dag_v<Graph_t>, "Graph_t must be a computational DAG");

        write_graph_structure(os, graph, VertexWriterGraphDot<GraphT>(graph));
    }

    template <typename GraphT>
    void WriteGraph(const std::string &filename, const GraphT &graph) const {
        static_assert(is_computational_dag_v<Graph_t>, "Graph_t must be a computational DAG");

        std::ofstream os(filename);
        write_graph(os, graph);
    }
};

}    // namespace osp
