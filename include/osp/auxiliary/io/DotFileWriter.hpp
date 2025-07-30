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

#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/model/BspScheduleRecomp.hpp"
#include "osp/concepts/computational_dag_concept.hpp"
#include "osp/graph_implementations/adj_list_impl/computational_dag_edge_idx_vector_impl.hpp"

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
    struct VertexWriterScheduleRecomp_DOT {

        const BspScheduleRecomp<Graph_t> &schedule;

        VertexWriterScheduleRecomp_DOT(const BspScheduleRecomp<Graph_t> &schedule_) : schedule(schedule_) {}

        void operator()(std::ostream &out, const vertex_idx_t<Graph_t> &i) const {
            out << i << " ["
                << "work_weight=\"" << schedule.getInstance().getComputationalDag().vertex_work_weight(i) << "\";"
                << "comm_weight=\"" << schedule.getInstance().getComputationalDag().vertex_comm_weight(i) << "\";"
                << "mem_weight=\"" << schedule.getInstance().getComputationalDag().vertex_mem_weight(i) << "\";";

            if constexpr (has_typed_vertices_v<Graph_t>) {

                out << "type=\"" << schedule.getInstance().getComputationalDag().vertex_type(i) << "\";";
            }

            out << "proc=\"(";
            for (size_t j = 0; j < schedule.assignments(i).size() - 1; ++j) {
                out << schedule.assignments(i)[j].first << ",";
            }
            out << schedule.assignments(i)[schedule.assignments(i).size() - 1].first << ")\";"
                << "superstep=\"(";
            for (size_t j = 0; j < schedule.assignments(i).size() - 1; ++j) {
                out << schedule.assignments(i)[j].second << ",";
            }
            out << schedule.assignments(i)[schedule.assignments(i).size() - 1].second << ")\";";

            bool found = false;

            for (const auto &[key, val] : schedule.getCommunicationSchedule()) {

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

    template<typename Graph_t>
    struct VertexWriterDuplicateRecompSchedule_DOT {
        const Graph_t &graph;
        const std::vector<std::string> name;
        const std::vector<unsigned> node_to_proc;
        const std::vector<unsigned> node_to_superstep;

        VertexWriterDuplicateRecompSchedule_DOT(const Graph_t &graph_,
                                                const std::vector<std::string> &name_,
                                                std::vector<unsigned> &node_to_proc_,
                                                std::vector<unsigned> &node_to_superstep_)
            : graph(graph_), name(name_), node_to_proc(node_to_proc_), node_to_superstep(node_to_superstep_) {}

        template<class VertexOrEdge>
        void operator()(std::ostream &out, const VertexOrEdge &i) const {
            out << i << " [" << "label=\"" << name[i] << "\";" << "work_weight=\""
                << graph.vertex_work_weight(i) << "\";" << "comm_weight=\""
                << graph.vertex_comm_weight(i) << "\";" << "mem_weight=\""
                << graph.vertex_mem_weight(i) << "\";" << "proc=\""
                << node_to_proc[i] << "\";" << "superstep=\"" << node_to_superstep[i] << "\";";

            out << "]";
        }
    };

    template<typename Graph_t>
    struct VertexWriterScheduleCS_DOT {

        const BspScheduleCS<Graph_t> &schedule;

        VertexWriterScheduleCS_DOT(const BspScheduleCS<Graph_t> &schedule_) : schedule(schedule_) {}

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

            bool found = false;

            for (const auto &[key, val] : schedule.getCommunicationSchedule()) {

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

    template<typename Graph_t>
    struct ColoredVertexWriterGraph_DOT {

        const Graph_t &graph;
        const std::vector<unsigned> &colors;
        std::vector<std::string> color_strings;
        std::vector<std::string> shape_strings;

        ColoredVertexWriterGraph_DOT(const Graph_t &graph_, const std::vector<unsigned> &colors_) : graph(graph_), colors(colors_) {

           color_strings = {
                "lightcoral", "palegreen", "lightblue", "gold", "orchid", "sandybrown", "aquamarine", "burlywood",
                "hotpink", "yellowgreen", "skyblue", "khaki", "violet", "salmon", "turquoise", "tan",
                "deeppink", "chartreuse", "deepskyblue", "lemonchiffon", "magenta", "orangered", "cyan", "wheat",
                "mediumvioletred", "limegreen", "dodgerblue", "lightyellow", "darkviolet", "tomato", "paleturquoise", "bisque",
                "crimson", "lime", "steelblue", "papayawhip", "purple", "darkorange", "cadetblue", "peachpuff",
                "indianred", "springgreen", "powderblue", "cornsilk", "mediumorchid", "chocolate", "darkturquoise", "navajowhite",
                "firebrick", "seagreen", "royalblue", "lightgoldenrodyellow", "darkmagenta", "coral", "teal", "moccasin",
                "maroon", "forestgreen", "blue", "yellow", "darkorchid", "red", "green", "navy",
                "darkred", "darkgreen", "mediumblue", "ivory", "indigo", "orange", "darkcyan", "antiquewhite"
            };
                
            shape_strings = {
                 "oval", "rect", "hexagon", "parallelogram"
            };
        }

        void operator()(std::ostream &out, const vertex_idx_t<Graph_t> &i) const {

            if (i >= colors.size()) {
                 // Fallback for safety: print without color if colors vector is mismatched or palette is empty.
                 out << i << " [";
            } else {
                 // Use modulo operator to cycle through the fixed palette if there are more color
                 // groups than available colors.
                 const std::string& color = color_strings[colors[i] % color_strings.size()];
                 out << i << " [style=filled;fillcolor=" << color << ";";
            }
          
            out << "work_weight=\"" << graph.vertex_work_weight(i) << "\";"
                << "comm_weight=\"" << graph.vertex_comm_weight(i) << "\";"
                << "mem_weight=\"" << graph.vertex_mem_weight(i) << "\";";

            if constexpr (has_typed_vertices_v<Graph_t>) {
                out << "type=\"" << graph.vertex_type(i) << "\";shape=\"" << shape_strings[graph.vertex_type(i) % shape_strings.size()] << "\";"; 
            }

            out << "]";
        }
    };

    template<typename Graph_t, typename vertex_writer_t>
    void write_graph_structure(std::ostream &os, const Graph_t &graph, const vertex_writer_t &vertex_writer) const {

        os << "digraph G {\n";
        for (const auto &v : graph.vertices()) {
            vertex_writer(os, v);
            os << "\n";
        }

        if constexpr (has_edge_weights_v<Graph_t>) {
            EdgeWriter_DOT<Graph_t> edge_writer(graph);

            for (const auto &e : edges(graph)) {
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
    void write_schedule(std::ostream &os, const BspSchedule<Graph_t> &schedule) const {

        write_graph_structure(os, schedule.getInstance().getComputationalDag(),
                              VertexWriterSchedule_DOT<Graph_t>(schedule));
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
    void write_schedule(const std::string &filename, const BspSchedule<Graph_t> &schedule) const {
        std::ofstream os(filename);
        write_schedule(os, schedule);
    }

    template<typename Graph_t>
    void write_schedule_cs(std::ostream &os, const BspScheduleCS<Graph_t> &schedule) const {

        write_graph_structure(os, schedule.getInstance().getComputationalDag(),
                              VertexWriterScheduleCS_DOT<Graph_t>(schedule));
    }

    template<typename Graph_t>
    void write_schedule_cs(const std::string &filename, const BspScheduleCS<Graph_t> &schedule) const {
        std::ofstream os(filename);
        write_schedule_cs(os, schedule);
    }

    template<typename Graph_t>
    void write_schedule_recomp(std::ostream &os, const BspScheduleRecomp<Graph_t> &schedule) const {

        write_graph_structure(os, schedule.getInstance().getComputationalDag(),
                              VertexWriterScheduleRecomp_DOT<Graph_t>(schedule));
    }

    template<typename Graph_t>
    void write_schedule_recomp(const std::string &filename, const BspScheduleRecomp<Graph_t> &schedule) const {
        std::ofstream os(filename);
        write_schedule_recomp(os, schedule);
    }

    template<typename Graph_t>
    void write_schedule_recomp_duplicate(std::ostream &os, const BspScheduleRecomp<Graph_t> &schedule) const {

        const auto &g = schedule.getInstance().getComputationalDag();

        using VertexType = vertex_idx_t<Graph_t>;

        std::vector<std::string> names(schedule.getTotalAssignments());
        std::vector<unsigned> node_to_proc(schedule.getTotalAssignments());
        std::vector<unsigned> node_to_superstep(schedule.getTotalAssignments());

        std::unordered_map<VertexType, std::vector<size_t>> vertex_to_idx;

        using vertex_type_t_or_default = std::conditional_t<is_computational_dag_typed_vertices_v<Graph_t>, v_type_t<Graph_t>, unsigned>;
        using edge_commw_t_or_default = std::conditional_t<has_edge_weights_v<Graph_t>, e_commw_t<Graph_t>, v_commw_t<Graph_t>>;

        using cdag_vertex_impl_t = cdag_vertex_impl<vertex_idx_t<Graph_t>, v_workw_t<Graph_t>, v_commw_t<Graph_t>,
                                                    v_memw_t<Graph_t>, vertex_type_t_or_default>;
        using cdag_edge_impl_t = cdag_edge_impl<edge_commw_t_or_default>;

        using graph_t = computational_dag_edge_idx_vector_impl<cdag_vertex_impl_t, cdag_edge_impl_t>;

        graph_t g2;

        size_t idx_new = 0;

        for (const auto &node : g.vertices()) {

            if (schedule.assignments(node).size() == 1) {

                g2.add_vertex(g.vertex_work_weight(node), g.vertex_comm_weight(node), g.vertex_mem_weight(node),
                              g.vertex_type(node));

                names[idx_new] = std::to_string(node);
                node_to_proc[idx_new] = schedule.assignments(node)[0].first;
                node_to_superstep[idx_new] = schedule.assignments(node)[0].second;

                vertex_to_idx.insert({node, {idx_new}});
                idx_new++;

            } else {

                std::vector<size_t> idxs;
                for (unsigned i = 0; i < schedule.assignments(node).size(); ++i) {

                    g2.add_vertex(g.vertex_work_weight(node), g.vertex_comm_weight(node), g.vertex_mem_weight(node),
                                  g.vertex_type(node));

                    names[idx_new] = std::to_string(node).append("_").append(std::to_string(i));
                    node_to_proc[idx_new] = schedule.assignments(node)[i].first;
                    node_to_superstep[idx_new] = schedule.assignments(node)[i].second;

                    idxs.push_back(idx_new++);
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

                            if (assigned_processors.find(node_to_proc[vertex_to_idx[target][j]]) ==
                                    assigned_processors.end() ||
                                node_to_proc[val[i]] == node_to_proc[vertex_to_idx[target][j]]) {
                                g2.add_edge(val[i], vertex_to_idx[target][j]);
                            }
                        }
                    }
                }
            }
        }

        write_graph_structure(
            os, g2, VertexWriterDuplicateRecompSchedule_DOT<Graph_t>(g2, names, node_to_proc, node_to_superstep));
    }

    template<typename Graph_t>
    void write_schedule_recomp_duplicate(const std::string &filename,
                                         const BspScheduleRecomp<Graph_t> &schedule) const {
        std::ofstream os(filename);
        write_schedule_recomp_duplicate(os, schedule);
    }

    template<typename Graph_t>
    void write_colored_graph(std::ostream &os, const Graph_t &graph, std::vector<unsigned> &colors) const {

        static_assert(is_computational_dag_v<Graph_t>, "Graph_t must be a computational DAG");

        write_graph_structure(os, graph, ColoredVertexWriterGraph_DOT<Graph_t>(graph, colors));
    }

    template<typename Graph_t>
    void write_colored_graph(const std::string &filename, const Graph_t &graph, std::vector<unsigned> &colors) const {

        static_assert(is_computational_dag_v<Graph_t>, "Graph_t must be a computational DAG");

        std::ofstream os(filename);
        write_colored_graph(os, graph, colors);
    }

    template<typename Graph_t>
    void write_graph(std::ostream &os, const Graph_t &graph) const {

        static_assert(is_computational_dag_v<Graph_t>, "Graph_t must be a computational DAG");

        write_graph_structure(os, graph, VertexWriterGraph_DOT<Graph_t>(graph));
    }

    template<typename Graph_t>
    void write_graph(const std::string &filename, const Graph_t &graph) const {

        static_assert(is_computational_dag_v<Graph_t>, "Graph_t must be a computational DAG");

        std::ofstream os(filename);
        write_graph(os, graph);
    }
};

} // namespace osp