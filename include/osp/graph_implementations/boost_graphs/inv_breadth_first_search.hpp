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

#include <boost/config.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/reverse_graph.hpp>
#include <boost/graph/visitors.hpp>
#include <iostream>
#include <vector>

namespace boost::extensions {

template <class IncidenceGraph, class Buffer, class BFSVisitor, class ColorMap, class SourceIterator>
void inv_breadth_first_visit(
    const IncidenceGraph &g, SourceIterator sources_begin, SourceIterator sources_end, Buffer &Q, BFSVisitor vis, ColorMap color) {
    BOOST_CONCEPT_ASSERT((IncidenceGraphConcept<IncidenceGraph>) );
    typedef graph_traits<IncidenceGraph> GTraits;
    typedef typename graph_traits<IncidenceGraph>::vertex_descriptor Vertex;
    BOOST_CONCEPT_ASSERT((BFSVisitorConcept<BFSVisitor, IncidenceGraph>) );
    BOOST_CONCEPT_ASSERT((ReadWritePropertyMapConcept<ColorMap, Vertex>) );
    typedef typename property_traits<ColorMap>::value_type ColorValue;
    typedef color_traits<ColorValue> Color;
    typename GTraits::in_edge_iterator ei, ei_end;

    for (; sources_begin != sources_end; ++sources_begin) {
        Vertex s = *sources_begin;
        put(color, s, Color::gray());
        vis.discover_vertex(s, g);
        Q.push(s);
    }
    while (!Q.empty()) {
        Vertex u = Q.top();
        Q.pop();
        vis.examine_vertex(u, g);
        for (boost::tie(ei, ei_end) = in_edges(u, g); ei != ei_end; ++ei) {
            Vertex v = target(*ei, g);
            vis.examine_edge(*ei, g);
            ColorValue v_color = get(color, v);
            if (v_color == Color::white()) {
                vis.tree_edge(*ei, g);
                put(color, v, Color::gray());
                vis.discover_vertex(v, g);
                Q.push(v);
            } else {
                vis.non_tree_edge(*ei, g);
                if (v_color == Color::gray()) {
                    vis.gray_target(*ei, g);
                } else {
                    vis.black_target(*ei, g);
                }
            }
        }    // end for
        put(color, u, Color::black());
        vis.finish_vertex(u, g);
    }    // end while
}

template <typename IncidenceGraph, class SourceVertex, class BFSVisitor>
void inv_breadth_first_search(const IncidenceGraph &graph, SourceVertex source, BFSVisitor vis) {
    const std::array sources = {source};
    typedef typename graph_traits<IncidenceGraph>::vertex_descriptor VertexT;
    boost::queue<VertexT> q;
    std::unordered_map<VertexT, default_color_type> color_map;
    inv_breadth_first_visit(graph, sources.begin(), sources.end(), q, vis, boost::associative_property_map(color_map));
}

}    // namespace boost::extensions
