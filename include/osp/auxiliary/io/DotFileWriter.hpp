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

        void operator()(std::ostream &out, const EdgeDescT<GraphT> &i) const {
            out << Source(i, graph_) << "->" << Target(i, graph_) << " ["
                << "comm_weight=\"" << graph_.EdgeCommWeight(i) << "\";"
                << "]";
        }
    };

    template <typename GraphT>
    struct VertexWriterScheduleDot {
        const BspSchedule<GraphT> &schedule_;

        VertexWriterScheduleDot(const BspSchedule<GraphT> &schedule) : schedule_(schedule) {}

        void operator()(std::ostream &out, const VertexIdxT<GraphT> &i) const {
            out << i << " ["
                << "work_weight=\"" << schedule_.GetInstance().GetComputationalDag().VertexWorkWeight(i) << "\";"
                << "comm_weight=\"" << schedule_.GetInstance().GetComputationalDag().VertexCommWeight(i) << "\";"
                << "mem_weight=\"" << schedule_.GetInstance().GetComputationalDag().VertexMemWeight(i) << "\";";

            if constexpr (HasTypedVerticesV<GraphT>) {
                out << "type=\"" << schedule_.GetInstance().GetComputationalDag().VertexType(i) << "\";";
            }

            out << "proc=\"" << schedule_.AssignedProcessor(i) << "\";" << "superstep=\"" << schedule_.AssignedSuperstep(i)
                << "\";";

            out << "]";
        }
    };

    template <typename GraphT>
    struct VertexWriterScheduleRecompDot {
        const BspScheduleRecomp<GraphT> &schedule_;

        VertexWriterScheduleRecompDot(const BspScheduleRecomp<GraphT> &schedule) : schedule_(schedule) {}

        void operator()(std::ostream &out, const VertexIdxT<GraphT> &i) const {
            out << i << " ["
                << "work_weight=\"" << schedule_.GetInstance().GetComputationalDag().VertexWorkWeight(i) << "\";"
                << "comm_weight=\"" << schedule_.GetInstance().GetComputationalDag().VertexCommWeight(i) << "\";"
                << "mem_weight=\"" << schedule_.GetInstance().GetComputationalDag().VertexMemWeight(i) << "\";";

            if constexpr (HasTypedVerticesV<GraphT>) {
                out << "type=\"" << schedule_.GetInstance().GetComputationalDag().VertexType(i) << "\";";
            }

            out << "proc=\"(";
            for (size_t j = 0; j < schedule_.Assignments(i).size() - 1; ++j) {
                out << schedule_.Assignments(i)[j].first << ",";
            }
            out << schedule_.Assignments(i)[schedule_.Assignments(i).size() - 1].first << ")\";"
                << "superstep=\"(";
            for (size_t j = 0; j < schedule_.Assignments(i).size() - 1; ++j) {
                out << schedule_.Assignments(i)[j].second << ",";
            }
            out << schedule_.Assignments(i)[schedule_.Assignments(i).size() - 1].second << ")\";";

            bool found = false;

            for (const auto &[key, val] : schedule_.GetCommunicationSchedule()) {
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
            out << i << " [" << "label=\"" << name_[i] << "\";" << "work_weight=\"" << graph_.VertexWorkWeight(i) << "\";"
                << "comm_weight=\"" << graph_.VertexCommWeight(i) << "\";" << "mem_weight=\"" << graph_.VertexMemWeight(i)
                << "\";" << "proc=\"" << nodeToProc_[i] << "\";" << "superstep=\"" << nodeToSuperstep_[i] << "\";";

            out << "]";
        }
    };

    template <typename GraphT>
    struct VertexWriterScheduleCsDot {
        const BspScheduleCS<GraphT> &schedule_;

        VertexWriterScheduleCsDot(const BspScheduleCS<GraphT> &schedule) : schedule_(schedule) {}

        void operator()(std::ostream &out, const VertexIdxT<GraphT> &i) const {
            out << i << " ["
                << "work_weight=\"" << schedule_.GetInstance().GetComputationalDag().VertexWorkWeight(i) << "\";"
                << "comm_weight=\"" << schedule_.GetInstance().GetComputationalDag().VertexCommWeight(i) << "\";"
                << "mem_weight=\"" << schedule_.GetInstance().GetComputationalDag().VertexMemWeight(i) << "\";";

            if constexpr (HasTypedVerticesV<GraphT>) {
                out << "type=\"" << schedule_.GetInstance().GetComputationalDag().VertexType(i) << "\";";
            }

            out << "proc=\"" << schedule_.AssignedProcessor(i) << "\";" << "superstep=\"" << schedule_.AssignedSuperstep(i)
                << "\";";

            bool found = false;

            for (const auto &[key, val] : schedule_.GetCommunicationSchedule()) {
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

        void operator()(std::ostream &out, const VertexIdxT<GraphT> &i) const {
            out << i << " ["
                << "work_weight=\"" << graph_.VertexWorkWeight(i) << "\";"
                << "comm_weight=\"" << graph_.VertexCommWeight(i) << "\";"
                << "mem_weight=\"" << graph_.VertexMemWeight(i) << "\";";

            if constexpr (HasTypedVerticesV<GraphT>) {
                out << "type=\"" << graph_.VertexType(i) << "\";";
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

        void operator()(std::ostream &out, const VertexIdxT<GraphT> &i) const {
            if (i >= static_cast<VertexIdxT<GraphT>>(colors_.size())) {
                // Fallback for safety: print without color if colors vector is mismatched or palette is empty.
                out << i << " [";
            } else {
                // Use modulo operator to cycle through the fixed palette if there are more color
                // groups than available colors.
                const std::string &color = colorStrings_[colors_[i] % colorStrings_.size()];
                out << i << " [style=filled;fillcolor=" << color << ";";
            }

            out << "work_weight=\"" << graph_.VertexWorkWeight(i) << "\";"
                << "comm_weight=\"" << graph_.VertexCommWeight(i) << "\";"
                << "mem_weight=\"" << graph_.VertexMemWeight(i) << "\";";

            if constexpr (HasTypedVerticesV<GraphT>) {
                out << "type=\"" << graph_.VertexType(i) << "\";shape=\""
                    << shapeStrings_[graph_.VertexType(i) % shapeStrings_.size()] << "\";";
            }

            out << "]";
        }
    };

    template <typename GraphT, typename VertexWriterT>
    void WriteGraphStructure(std::ostream &os, const GraphT &graph, const VertexWriterT &vertexWriter) const {
        os << "digraph G {\n";
        for (const auto &v : graph.Vertices()) {
            vertexWriter(os, v);
            os << "\n";
        }

        if constexpr (HasEdgeWeightsV<GraphT>) {
            EdgeWriterDot<GraphT> edgeWriter(graph);

            for (const auto &e : Edges(graph)) {
                edgeWriter(os, e);
                os << "\n";
            }

        } else {
            for (const auto &v : graph.Vertices()) {
                for (const auto &child : graph.Children(v)) {
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
        WriteGraphStructure(os, schedule.GetInstance().GetComputationalDag(), VertexWriterScheduleDot<GraphT>(schedule));
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
        WriteSchedule(os, schedule);
    }

    template <typename GraphT>
    void WriteScheduleCs(std::ostream &os, const BspScheduleCS<GraphT> &schedule) const {
        WriteGraphStructure(os, schedule.GetInstance().GetComputationalDag(), VertexWriterScheduleCsDot<GraphT>(schedule));
    }

    template <typename GraphT>
    void WriteScheduleCs(const std::string &filename, const BspScheduleCS<GraphT> &schedule) const {
        std::ofstream os(filename);
        WriteScheduleCs(os, schedule);
    }

    template <typename GraphT>
    void WriteScheduleRecomp(std::ostream &os, const BspScheduleRecomp<GraphT> &schedule) const {
        WriteGraphStructure(os, schedule.GetInstance().GetComputationalDag(), VertexWriterScheduleRecompDot<GraphT>(schedule));
    }

    template <typename GraphT>
    void WriteScheduleRecomp(const std::string &filename, const BspScheduleRecomp<GraphT> &schedule) const {
        std::ofstream os(filename);
        WriteScheduleRecomp(os, schedule);
    }

    template <typename GraphT>
    void WriteScheduleRecompDuplicate(std::ostream &os, const BspScheduleRecomp<GraphT> &schedule) const {
        const auto &g = schedule.GetInstance().GetComputationalDag();

        using VertexType = VertexIdxT<GraphT>;

        std::vector<std::string> names(schedule.GetTotalAssignments());
        std::vector<unsigned> nodeToProc(schedule.GetTotalAssignments());
        std::vector<unsigned> nodeToSuperstep(schedule.GetTotalAssignments());

        std::unordered_map<VertexType, std::vector<size_t>> vertexToIdx;

        using VertexTypeTOrDefault = std::conditional_t<IsComputationalDagTypedVerticesV<GraphT>, VTypeT<GraphT>, unsigned>;
        using EdgeCommwTOrDefault = std::conditional_t<HasEdgeWeightsV<GraphT>, ECommwT<GraphT>, VCommwT<GraphT>>;

        using CdagVertexImplT
            = CdagVertexImpl<VertexIdxT<GraphT>, VWorkwT<GraphT>, VCommwT<GraphT>, VMemwT<GraphT>, VertexTypeTOrDefault>;
        using CdagEdgeImplT = CdagEdgeImpl<EdgeCommwTOrDefault>;

        using GraphT2 = ComputationalDagEdgeIdxVectorImpl<CdagVertexImplT, CdagEdgeImplT>;

        GraphT2 g2;

        size_t idxNew = 0;

        for (const auto &node : g.Vertices()) {
            if (schedule.Assignments(node).size() == 1) {
                g2.AddVertex(g.VertexWorkWeight(node), g.VertexCommWeight(node), g.VertexMemWeight(node), g.VertexType(node));

                names[idxNew] = std::to_string(node);
                nodeToProc[idxNew] = schedule.Assignments(node)[0].first;
                nodeToSuperstep[idxNew] = schedule.Assignments(node)[0].second;

                vertexToIdx.insert({node, {idxNew}});
                idxNew++;

            } else {
                std::vector<size_t> idxs;
                for (unsigned i = 0; i < schedule.Assignments(node).size(); ++i) {
                    g2.add_vertex(g.VertexWorkWeight(node), g.VertexCommWeight(node), g.VertexMemWeight(node), g.VertexType(node));

                    names[idxNew] = std::to_string(node).append("_").append(std::to_string(i));
                    nodeToProc[idxNew] = schedule.Assignments(node)[i].first;
                    nodeToSuperstep[idxNew] = schedule.Assignments(node)[i].second;

                    idxs.push_back(idxNew++);
                }
                vertexToIdx.insert({node, idxs});
            }
        }

        for (const auto &[key, val] : vertexToIdx) {
            if (val.size() == 1) {
                for (const auto &target : g.Children(key)) {
                    for (const auto &newNodeTarget : vertexToIdx[target]) {
                        g2.AddEdge(val[0], newNodeTarget);
                    }
                }

            } else {
                std::unordered_set<unsigned> assignedProcessors;

                for (const auto &assignment : schedule.Assignments(key)) {
                    assignedProcessors.insert(assignment.first);
                }

                for (unsigned i = 0; i < val.size(); i++) {
                    for (const auto &target : g.Children(key)) {
                        for (size_t j = 0; j < vertexToIdx[target].size(); j++) {
                            if (assignedProcessors.find(nodeToProc[vertexToIdx[target][j]]) == assignedProcessors.end()
                                || nodeToProc[val[i]] == nodeToProc[vertexToIdx[target][j]]) {
                                g2.AddEdge(val[i], vertexToIdx[target][j]);
                            }
                        }
                    }
                }
            }
        }

        WriteGraphStructure(os, g2, VertexWriterDuplicateRecompScheduleDot<GraphT2>(g2, names, nodeToProc, nodeToSuperstep));
    }

    template <typename GraphT>
    void WriteScheduleRecompDuplicate(const std::string &filename, const BspScheduleRecomp<GraphT> &schedule) const {
        std::ofstream os(filename);
        write_schedule_recomp_duplicate(os, schedule);
    }

    template <typename GraphT, typename ColorContainerT>
    void WriteColoredGraph(std::ostream &os, const GraphT &graph, const ColorContainerT &colors) const {
        static_assert(IsComputationalDagV<GraphT>, "GraphT must be a computational DAG");

        WriteGraphStructure(os, graph, ColoredVertexWriterGraphDot<GraphT, ColorContainerT>(graph, colors));
    }

    template <typename GraphT, typename ColorContainerT>
    void WriteColoredGraph(const std::string &filename, const GraphT &graph, const ColorContainerT &colors) const {
        static_assert(IsComputationalDagV<GraphT>, "GraphT must be a computational DAG");

        std::ofstream os(filename);
        WriteColoredGraph(os, graph, colors);
    }

    template <typename GraphT>
    void WriteGraph(std::ostream &os, const GraphT &graph) const {
        static_assert(IsComputationalDagV<GraphT>, "GraphT must be a computational DAG");

        WriteGraphStructure(os, graph, VertexWriterGraphDot<GraphT>(graph));
    }

    template <typename GraphT>
    void WriteGraph(const std::string &filename, const GraphT &graph) const {
        static_assert(IsComputationalDagV<GraphT>, "GraphT must be a computational DAG");

        std::ofstream os(filename);
        WriteGraph(os, graph);
    }
};

}    // namespace osp
