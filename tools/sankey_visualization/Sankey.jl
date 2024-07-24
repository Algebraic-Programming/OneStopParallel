#=
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
=#

import Pkg
Pkg.activate("SankeyPlots_version.jl")

import SankeyPlots_version
import PlotUtils
import PlotUtils: palette

using SankeyPlots_version, Plots

function main()
    if length(ARGS) !== 1
        print("Arguments are: <sankey-file>\n")
        return
    end

    sankey_from_file(ARGS[1])
    savefig( chop(ARGS[1], tail=6) * "png")
    # savefig( chop(ARGS[1], tail=6) * "pdf")
end

function read_sankey_file( file )
    n_proc = 0
    n_step = 0
    workloads = Vector{Float64}()
    link_src = Vector{Int}()
    link_dtns = Vector{Int}()
    commloads = Vector{Float64}()

    line_counter = 1
    for line in eachline(file)
        if line_counter == 2
            n_proc, n_step = parse.(Int, split(line, ","))
        elseif 3 < line_counter && line_counter <= 3 + n_step
            append!(workloads, parse.(Float64, split(line, ",")))
        elseif 4 + n_step < line_counter
            step, send_proc, rec_proc, amount = parse.(Int, split(line, ","))
            amount = convert(Float64, amount)
            push!(link_src, (step-1)*n_proc + send_proc)
            push!(link_dtns, step*n_proc + rec_proc)
            push!(commloads, amount)
        end
        line_counter += 1
    end

    return n_proc, n_step, workloads, link_src, link_dtns, commloads
end


function sankey_from_file(sankey_file)
    n_proc, n_step, workloads, link_src, link_dtns, link_loads = read_sankey_file( sankey_file )

    max_width_ = 0.55
    max_height_ = 0.2

    energy_colors = palette(:cyclic_mygbm_30_95_c78_n256_s25)[1:div(255+n_proc, n_proc):256]

    names = Vector{String}()
    force_layer_ = Vector{Pair{Int,Int}}()
    force_order_ = Vector{Pair{Int,Int}}()
    force_equal_layers_ = Vector{Pair{Int,Int}}()
    for step = 1:n_step
        for proc = 1:n_proc
            s = string("S", step, "_P", proc)
            push!(names, s)
            push!(force_layer_, Pair( (step-1)*n_proc+proc ,step) )
            if proc != n_proc
                push!(force_order_, Pair( (step-1)*n_proc + proc, (step-1)*n_proc + proc + 1))
            end
        end
        if step !=n_step
            push!(force_equal_layers_, Pair( step, step + 1 ))
        end
    end

    sankey( link_src, link_dtns, link_loads;
            node_labels=names,
            node_colors=energy_colors,
            edge_color=:gradient,
            label_position=:bottom,
            label_size=7,
            compact=false,
            force_layer=force_layer_,
            force_order=force_order_,
            force_equal_layers=force_equal_layers_,
            max_height=max_height_,
            max_width=max_width_,
            node_widths=workloads,
            bsp_position_force=n_proc => n_step)
end



main()
