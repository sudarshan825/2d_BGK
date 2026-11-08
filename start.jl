# One-line start for ALEBGK2D in the Julia REPL or VS Code:
#
#   include("/Users/tiwari/DFG_BGK2/Julia_code/2d_BGK/ALEBGK2D/start.jl")
#
# It activates this package, installs what is missing (Plots.jl once, into the
# default environment), loads the solver and the plot functions, and sets
#   R = <package>/results
# Then e.g.
#   res = run_case(Params(eps = 0.1, tau = 0.001))      # writes data + PNG to R
#   plot_file(res.file; field = :u1)                    # other fields
#   run_tests()
import Pkg

const PKGDIR = @__DIR__

Pkg.activate(PKGDIR; io = devnull)
isfile(joinpath(PKGDIR, "Manifest.toml")) || Pkg.instantiate(; io = devnull)

if Base.find_package("Plots") === nothing
    println("Installing Plots.jl into the default environment (only once, takes a few minutes) ...")
    Pkg.activate(; io = devnull)
    Pkg.add("Plots")
    Pkg.activate(PKGDIR; io = devnull)
end

using ALEBGK2D
include(joinpath(PKGDIR, "examples", "plot_results.jl"))

# every run_case now also plots: vorticity + points -> <results>/<tag>_omega.png
ALEBGK2D.PLOT_HOOK[] = file -> (plot_file(file); nothing)

R = joinpath(PKGDIR, "results")
mkpath(R)

"Run the package tests."
run_tests() = (Pkg.activate(PKGDIR; io = devnull); Pkg.test())

println("""
ALEBGK2D ready.  Results folder R = $R
Threads: $(Threads.nthreads())$(Threads.nthreads() == 1 ? "   (start Julia with  julia -t auto  to use all cores)" : "")

  res = run_case(Params(eps = 0.1, tau = 0.001))           # ALE, 100×100: data + PNG in R
  res = run_case(Params(eps = 0.1, tau = 0.001, ale = false))
  plot_file(res.file; field = :u1)                          # other fields: :rho :u1 :u2 :T
  compare_files(fileA, fileB)                               # same contour levels
  plot_all(R)                                               # every run in R
  run_tests()""")
