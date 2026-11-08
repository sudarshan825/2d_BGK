# Doubly periodic double shear layer on [0, 1]²:
#   ρ = 1, T = 0,  u1 = tanh(30(y - 1/4)) for y ≤ 1/2,  tanh(30(3/4 - y)) otherwise,
#   u2 = 0.05 sin(2πx).
# Runs ε ∈ {0.5, 0.1} × τ ∈ {0.01, 0.001} up to t = 1.5 on 100 × 100 points.
#
#   julia -t auto examples/shear_layer.jl                      # all four cases
#   julia -t auto examples/shear_layer.jl 0.5 0.01             # one case: ε τ
#   julia -t auto examples/shear_layer.jl 0.5 0.01 100 8       # ε τ Nx Nv
#   julia -t auto examples/shear_layer.jl 0.5 0.01 100 8 eul   # Eulerian (fixed points)
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)
using ALEBGK2D
if Base.find_package("Plots") !== nothing          # plot every run if Plots.jl is installed
    include(joinpath(@__DIR__, "plot_results.jl"))
    ALEBGK2D.PLOT_HOOK[] = file -> (plot_file(file); nothing)
end

outdir = joinpath(@__DIR__, "..", "results")

if isempty(ARGS)
    for ε in (0.5, 0.1), τ in (0.01, 0.001)
        run_case(Params(eps = ε, tau = τ, tfinal = 1.5, outdir = outdir))
    end
else
    ε   = parse(Float64, ARGS[1])
    τ   = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.01
    Nx  = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 100
    Nv  = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 8
    ale = !(length(ARGS) >= 5 && ARGS[5] == "eul")
    run_case(Params(eps = ε, tau = τ, Nx = Nx, Nv = Nv, ale = ale, tfinal = 1.5,
                    outdir = outdir))
end
