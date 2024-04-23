## Description
This is a modification of the package https://github.com/daschw/SankeyPlots.jl as to fix the position of the graph elements in order to plot BSP-schedules.
Each row represents a processor and each column a superstep. The width of a block represents the workload said processor in said superstep and is to scale.
Inter-processor communication is displayed in Sankey-style but is not to scale! Note intra-processor communication is zero in the BSP-schedule.


## Installation

```
julia
julia> include("../../third/SankeyPlots/SankeyPlots_version.jl")
```

## Usage

```
julia Sankey.jl {file}.sankey
```

## Example

```
../../build/main --inputDag ../../examples/instances/instance_bicgstab.txt --inputMachine ../../examples/instances/p4_g3_l5.txt --sankey --BaldMixR
julia Sankey.jl instance_bicgstab_p4_g3_l5_BalDMixR_sankey.sankey
```

## Known issue
If the number of supersteps is too large, the png is just blank or cropped off. If that is the case,
make the size of the plot smaller in SankeyPlots_version.jl at the marked place.
