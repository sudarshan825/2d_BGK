# ALEBGK2D.jl

A meshfree **Arbitrary Lagrangian–Eulerian (ALE)** solver for the two-dimensional
BGK model of the Boltzmann equation in the low-Mach / small-Knudsen scaling on a
doubly periodic box, written in pure Julia (standard library only).

The method follows

> S. Tiwari, A. Klar, G. Russo, *A meshfree arbitrary Lagrangian–Eulerian method
> for the BGK model of the Boltzmann equation with moving boundaries*,
> J. Comput. Phys. **458** (2022) 111088.

## Model

On $[0,L]^2$ with periodic boundary conditions in $x$ and $y$ the code solves the
scaled BGK equation

$$
\partial_t F + \frac{1}{\varepsilon}\, v\cdot\nabla_x F = \frac{1}{\varepsilon^2\tau}\,(M_F - F),
\qquad
M_F = \frac{\rho_F}{2\pi T_F}\exp\!\Big(-\frac{|v-u_F|^2}{2T_F}\Big),
\qquad v\in\mathbb{R}^2,
$$

where $\varepsilon$ is the scaling parameter (Knudsen ∼ Mach number) and $\tau$ the
relaxation time. Macroscopic quantities are the moments
$\rho_F=\int F\,dv$, $\rho_F u_F=\int vF\,dv$, $\rho_F|u_F|^2+2\rho_F T_F=\int |v|^2F\,dv$.

Initial data and results are expressed as fluctuations around the global
equilibrium $(\rho_F,u_F,T_F)=(1,0,1)$:

$$
\rho_F = 1+\varepsilon\rho,\qquad \rho_F u_F = \varepsilon u,\qquad
T = \frac12\Big(\frac{1}{\varepsilon}\Big(\int |v|^2F\,dv-2\Big)-2\rho\Big),
$$

and the vorticity is $\omega = \partial_x u_2 - \partial_y u_1$.
As $\varepsilon\to 0$ the fluctuations satisfy the incompressible Navier–Stokes
equations with viscosity $\tau$.

## Method

| Ingredient | Choice |
|---|---|
| Frame | ALE with grid velocity $w=\theta\,u_F/\varepsilon$, θ = `grid_velocity`: $\dot x_i = w$, and along the paths $\dot F = -\frac{v-\theta u_F}{\varepsilon}\cdot\nabla F + \frac{1}{\varepsilon^2\tau}(M_F-F)$. θ = 1 is Lagrangian, θ = 0 (or `ale = false`) Eulerian |
| Neighbours | Periodic cell lists of size $h = \Delta x_0/0.35$; the radius is enlarged locally until every half plane holds at least `min_half` points |
| Derivatives | Weighted least squares (GFDM), Gaussian weight $e^{-\alpha d^2/h^2}$, $\alpha = 6$. Stencils on the full neighbourhood and on the half planes $dx\le0$, $dx\ge0$, $dy\le0$, $dy\ge0$ |
| Upwinding | `order = 1`: linear fits on the upwind half plane. `order = 2`: quadratic fits, WENO-type combination of the upwind half-plane and the central stencil (linear weights ½, ½), weight ∝ ½/(β+ε_w)². Smoothness indicator β per direction (`weno = :direction`) or with all derivatives (default), $\beta=(f_x^2+f_y^2)\Delta x^2+(f_{xx}^2+f_{xy}^2+f_{yy}^2)\Delta x^4$ (`weno = :full`) |
| Time | `order = 1`: first-order splitting; `order = 2`: ARS(2,2,1)-type splitting with mid-point transport. The relaxation step is solved exactly-implicitly, so $\Delta t$ does not depend on $\tau$ |
| Point management | A pair closer than `r_merge·Δx₀` is replaced by one point at its midpoint; nodes of the initial lattice with no point within `r_hole·Δx₀` receive a new point. Values of new points come from a moving-least-squares (MLS) fit, linear by default (`mls_order`) |
| Velocity space | Tensor Gauss–Hermite quadrature with `Nv²` points (default), or a uniform grid on `[-vmax, vmax]²` |

Loops over points are multithreaded.

## Installation

```bash
git clone https://github.com/<user>/ALEBGK2D.jl.git
cd ALEBGK2D.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
```

Only the Julia standard library is needed for the solver; the plotting example
uses [Plots.jl](https://github.com/JuliaPlots/Plots.jl).

## Quick start

In the Julia REPL (or VS Code) one line sets everything up:

```julia
include("path/to/ALEBGK2D/start.jl")
```

It activates the package, installs Plots.jl the first time, loads the solver and
the plot functions and sets `R` to the folder `results/` of the package. Every
`run_case` then also saves a vorticity plot `<tag>_omega.png` next to its data
(`plot = false` switches this off). Then

```julia
res = run_case(Params(eps = 0.5, tau = 0.01))
plot_file(res.file)
run_tests()
```

Start Julia with `julia -t auto` to use all cores.

`res` is a named tuple with the point positions `x, y`, the fluctuations
`rho, u1, u2, T`, the vorticity `omega`, the kinetic fields `rhoF, uxF, uyF, TF`,
the same fields resampled to the `Nx × Nx` lattice (`xu, yu, rho_u, u1_u, u2_u,
T_u, omega_u`, x fastest), the distribution `F`, the velocities `vx, vy` and the
path of the output file `file`.

### Your own initial data

Pass a function `(x, y, ε) -> (ρ, u1, u2, T)` returning the initial fluctuations:

```julia
vortex(x, y, ε) = (0.0, -sin(2π * y), sin(2π * x), 0.0)
run_case(Params(eps = 0.1, tau = 0.01, init = vortex))
```

### Example: doubly periodic double shear layer

`examples/shear_layer.jl` starts from $\rho = 1$, $T = 0$,

$$
u_1 = \begin{cases}\tanh\big(k(y-\tfrac14)\big), & y\le \tfrac12,\\[2pt] \tanh\big(k(\tfrac34-y)\big), & y>\tfrac12,\end{cases}
\qquad u_2 = \delta\sin(2\pi x),\qquad k = 30,\ \delta = 0.05,
$$

and runs $\varepsilon\in\{0.5,0.1\}$, $\tau\in\{0.01,0.001\}$ up to $t=1.5$ on
$100\times100$ points:

```bash
julia -t auto examples/shear_layer.jl                 # all four cases
julia -t auto examples/shear_layer.jl 0.5 0.01        # one case: ε τ [Nx Nv [eul]]
julia examples/plot_results.jl                        # plot every run in results/
```

### Plotting

`examples/plot_results.jl` works directly with the result files. `run_case`
returns the path of its output file as `res.file`:

```julia
include("examples/plot_results.jl")
res = run_case(Params(eps = 0.1, tau = 0.001))
plot_file(res.file)                    # vorticity + point cloud, saved as <tag>_omega.png
plot_file(res.file; field = :u1)       # :omega, :rho, :u1, :u2, :T
plot_file("results/bgk2d_eps0.1_tau0.001_eul_Nx100_uniform.dat")   # any saved run
compare_files(fileA, fileB)            # side by side with the same contour levels
plot_all("results")                    # every run in a folder
```

Other values of $k$ and $\delta$: `init = make_shear_layer(k, δ)`.

Double vortex sheet (u₁ = ±U with jumps at y = ¼ and y = ¾, perturbation
a·sin(2πx) in u₁ and u₂): `init = step_shear_layer` (U = 0.5, a = 0.01) or
`init = make_step_shear_layer(U, a)`.

To compare with another solver (e.g. the relaxation system), write its lattice
data in the same format (columns `x y rho u1 u2 T omega`, x fastest, file name
ending in `_uniform.dat`) and use `compare_files`.

## Parameters

| Field | Default | Meaning |
|---|---|---|
| `eps` | `0.1` | scaling parameter ε |
| `tau` | `0.01` | relaxation time τ |
| `L` | `1.0` | side of the periodic box |
| `Nx` | `100` | initial points per direction (`Nx²` points) |
| `Nv` | `8` | velocity points per direction (`Nv²` velocities) |
| `vgrid` | `:hermite` | `:hermite` (Gauss–Hermite) or `:uniform` |
| `vmax` | `4.5` | velocity box for `:uniform` |
| `cfl` | `0.5` | time step `Δt = cfl·Δx·ε/(√2·max|v|)` |
| `dt` | `0.0` | fixed time step if `> 0` |
| `tfinal` | `1.5` | final time |
| `order` | `2` | `1`: first order, `2`: second order |
| `ale` | `true` | `false` keeps the points fixed (Eulerian) |
| `grid_velocity` | `1.0` | ALE grid velocity w = `grid_velocity`·u_F (1: Lagrangian, 0: Eulerian, e.g. 0.5: partial) |
| `weno` | `:full` | smoothness indicator: `:full` (all first and second derivatives of the stencil) or `:direction` (∂x from f_x, f_xx; ∂y from f_y, f_yy) |
| `hratio` | `0.35` | `Δx₀ = hratio·h` |
| `alpha` | `6.0` | Gaussian weight parameter |
| `min_half` | `7` | minimum neighbours in each half plane (`stencil = :grow`) |
| `stencil` | `:grow` | `:grow`: radius enlarged until each half plane has `min_half` points (accurate). `:tiwari`: fixed radius h, fit with constant term, linear fit if a set has ≤ 8 points, central stencil only if a half plane has too few points (more dissipative, robust on moving points) |
| `weno_eps` | `1e-6` | regularisation in the smoothness indicators |
| `r_merge`, `r_hole` | `0.4`, `0.9` | merge / insert radii (× `Δx₀`) |
| `mls_order` | `1` | MLS for new and merged points: `1` linear, `2` quadratic |
| `init` | `shear_layer` | initial fluctuations `(x, y, ε) -> (ρ, u1, u2, T)` |
| `outdir` | `<package>/results` | output folder (created if missing) |
| `write_output` | `true` | write `.dat` files |
| `nprint` | `0` | progress interval in steps (`0`: about 20 messages) |

## Recommended settings

| Points | Settings | Comment |
|---|---|---|
| fixed (Eulerian) | `ale = false` (defaults otherwise) | clean, most accurate; enough for periodic flows without moving boundaries |
| moving (Lagrangian) | `stencil = :tiwari, alpha = 2.25, weno_eps = 1e-7, hratio = 0.4` | robust on distorted point clouds, somewhat more dissipative |

With moving points and the default `stencil = :grow`, the double shear layer at
ε = 0.1, τ = 0.001 develops small spurious vortices in the braids.

## Output

For each run two text files are written to `outdir`, with
`<tag> = bgk2d_eps<ε>_tau<τ>_<frame>[_wdir]_Nx<Nx>` (frame `eul`, `ale` or `ale<θ>`):

- `<tag>_uniform.dat` – values resampled to the `Nx × Nx` lattice (x fastest),
- `<tag>_particles.dat` – values at the points,

with columns `x  y  rho  u1  u2  T  omega`. The first line records ε, τ, t, Nx,
Nv, order, ALE/Eulerian and the number of points. Runs that differ in ε, τ,
frame or Nx therefore never overwrite each other.

## Cost

The explicit transport step requires $\Delta t \propto \varepsilon\,\Delta x / v_{\max}$,
so the run time grows like $1/\varepsilon$, like $N_x^3$ and like `Nv²`.
The Gauss–Hermite grid with `Nv = 8` (64 velocities) is accurate for these
low-Mach flows (`Nv = 12` gives the same shear-wave decay); a uniform grid needs
many more velocities. Start Julia with `-t auto` to use all cores; for quick tests
use e.g. `Nx = 32`.

## Tests

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

The tests check that the global equilibrium stays at rest, the decay of a shear
wave against the limit $e^{-4\pi^2\tau t}$, mass conservation, agreement of the ALE
and Eulerian versions, and the first-order scheme.

## Citation

If you use this code, please cite

```bibtex
@article{TiwariKlarRusso2022,
  author  = {Tiwari, Sudarshan and Klar, Axel and Russo, Giovanni},
  title   = {A meshfree arbitrary {L}agrangian--{E}ulerian method for the {BGK}
             model of the {B}oltzmann equation with moving boundaries},
  journal = {Journal of Computational Physics},
  volume  = {458},
  pages   = {111088},
  year    = {2022}
}
```

## License

MIT – see [LICENSE](LICENSE).
