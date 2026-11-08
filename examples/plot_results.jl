# Plotting for ALEBGK2D results (requires Plots.jl).
#
# In the REPL:
#   include("examples/plot_results.jl")
#   res = run_case(Params(eps = 0.1, tau = 0.001, outdir = R))
#   plot_file(res.file)                        # vorticity + points of that run, saved as PNG
#   plot_file("…/results/bgk2d_eps0.1_tau0.001_eul_Nx100_uniform.dat")   # any saved run
#   plot_file(res.file; field = :u1)           # :omega, :rho, :u1, :u2, :T
#   compare_files(fileA, fileB)                # side by side, same contour levels
#   plot_all(R)                                # every run in a folder
#
# From a terminal:
#   julia examples/plot_results.jl file1_uniform.dat [file2_uniform.dat ...]
#   julia examples/plot_results.jl results_folder
using Plots, Printf

const FIELDS = Dict(:rho => (3, "ρ"), :u1 => (4, "u₁"), :u2 => (5, "u₂"),
                    :T => (6, "T"), :omega => (7, "ω"))

"Read a result file: (data matrix, header Dict)."
function read_result(file)
    isfile(file) || error("file not found: $file")
    hdr = Dict{String,String}(); rows = Vector{Vector{Float64}}()
    for l in eachline(file)
        s = strip(l)
        isempty(s) && continue
        if startswith(s, "#")
            for m in eachmatch(r"(\w+)=([^\s,()]+)", s)
                hdr[m[1]] = m[2]
            end
        else
            push!(rows, parse.(Float64, split(s)))
        end
    end
    return permutedims(reduce(hcat, rows)), hdr
end

# lattice data (x fastest) -> axes and matrix Z[iy, ix]
function lattice(D, col)
    n = round(Int, sqrt(size(D, 1)))
    n * n == size(D, 1) || error("not a lattice file (use the *_uniform.dat file)")
    return D[1:n, 1], D[1:n:end, 2], permutedims(reshape(D[:, col], n, n))
end

uniform_file(f)   = replace(f, r"_particles\.dat$" => "_uniform.dat")
particles_file(f) = replace(f, r"_uniform\.dat$" => "_particles.dat")

function title_of(hdr, name, lim)
    g(k) = get(hdr, k, "?")
    wn = haskey(hdr, "weno") ? ", WENO $(hdr["weno"])" : ""
    return "$name   ε=$(g("eps")), τ=$(g("tau")), t=$(g("t")), $(g("frame")) Nx=$(g("Nx"))$wn\n" *
           @sprintf("max|%s| = %.3g", name, lim)
end

function contour_panel(file, field, lim)
    col, name = FIELDS[field]
    D, hdr = read_result(uniform_file(file))
    xs, ys, Z = lattice(D, col)
    lim = lim === nothing ? maximum(abs, Z) : lim
    lev = field in (:omega, :u1, :u2) ? range(-lim, lim; length = 21) : 21
    p = contour(xs, ys, Z; levels = lev, aspect_ratio = 1, colorbar = true,
                xlims = (0, 1), ylims = (0, 1), titlefontsize = 9,
                title = title_of(hdr, name, maximum(abs, Z)))
    return p, hdr
end

"""
    plot_file(file; field = :omega, points = true, save = true)

Contours of `field` from a result file (the `_uniform.dat` or the
`_particles.dat` file of a run) and, if `points`, the point cloud.
Displays the figure and saves it next to the data as `<tag>_<field>.png`.
"""
function plot_file(file; field = :omega, points = true, save = true)
    file = uniform_file(abspath(expanduser(file)))
    p1, hdr = contour_panel(file, field, nothing)
    panels = Any[p1]
    fp = particles_file(file)
    if points && isfile(fp)
        P, _ = read_result(fp)
        push!(panels, scatter(P[:, 1], P[:, 2]; ms = 0.6, msw = 0, legend = false,
                              aspect_ratio = 1, xlims = (0, 1), ylims = (0, 1),
                              titlefontsize = 9, title = "points  N = $(size(P, 1))"))
    end
    plt = plot(panels...; layout = (1, length(panels)), size = (540 * length(panels), 500))
    display(plt)
    if save
        png = replace(file, r"_uniform\.dat$" => "_$(field).png")
        savefig(plt, png); println("wrote ", png)
    end
    return plt
end

"""
    compare_files(files...; field = :omega, save = nothing)

Several runs side by side with the same contour levels.
"""
function compare_files(files...; field = :omega, save = nothing)
    fs  = [uniform_file(abspath(expanduser(f))) for f in files]
    lim = maximum(maximum(abs, lattice(read_result(f)[1], FIELDS[field][1])[3]) for f in fs)
    panels = [contour_panel(f, field, lim)[1] for f in fs]
    plt = plot(panels...; layout = (1, length(panels)), size = (540 * length(panels), 500))
    display(plt)
    if save !== nothing
        savefig(plt, save); println("wrote ", save)
    end
    return plt
end

"Plot every run (`*_uniform.dat`) in a folder."
function plot_all(dir; field = :omega)
    isdir(dir) || error("folder not found: $dir")
    fs = filter(f -> endswith(f, "_uniform.dat"), readdir(dir; join = true))
    isempty(fs) && error("no *_uniform.dat files in $dir")
    for f in fs
        plot_file(f; field = field)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = isempty(ARGS) ? [joinpath(@__DIR__, "..", "results")] : ARGS
    for a in args
        isdir(a) ? plot_all(a) : plot_file(a)
    end
end
