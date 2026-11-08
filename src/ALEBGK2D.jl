# =============================================================================
#  ALEBGK2D.jl
#
#  Meshfree Arbitrary Lagrangian-Eulerian (ALE) solver for the 2D BGK model
#  in the low-Mach / small-Knudsen scaling, doubly periodic box [0,L]²:
#
#        ∂t F + (1/ε) v·∇F = 1/(ε² τ) (M_F - F),
#        M_F = ρ_F / (2π T_F) exp(-|v - u_F|² / (2 T_F)),     v ∈ ℝ²,
#
#  following the meshfree ALE method of
#     S. Tiwari, A. Klar, G. Russo, J. Comput. Phys. 458 (2022) 111088.
#
#  ALE form:  dx_i/dt = u_F(x_i)/ε,
#             dF/dt   = -(v - u_F)/ε · ∇F + 1/(ε² τ) (M_F - F).
#
#  Space : weighted least squares (GFDM), Gaussian weight exp(-α d²/h²).
#          order 1: linear fits on half-plane neighbour sets (upwind);
#          order 2: quadratic fits, WENO-type combination of the upwind
#                   half-plane and the central stencil, per direction.
#  Time  : order 1: first-order splitting; order 2: ARS(2,2,1)-type splitting.
#          The BGK relaxation is solved exactly-implicitly.
#  Velocity: tensor Gauss-Hermite quadrature (default) or uniform grid.
#  Points: periodic cell lists; pairs closer than r_merge·Δx₀ are merged,
#          lattice nodes without a point within r_hole·Δx₀ are filled;
#          new values by moving least squares (MLS).
# =============================================================================
module ALEBGK2D

using LinearAlgebra, Printf
using Base.Threads

export Params, run_case, shear_layer, make_shear_layer, step_shear_layer,
       make_step_shear_layer, shear_wave, write_results, file_tag

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
Base.@kwdef mutable struct Params
    eps::Float64     = 0.1       # scaling parameter ε
    tau::Float64     = 0.01      # relaxation time τ
    L::Float64       = 1.0       # periodic box [0, L]²
    Nx::Int          = 100       # initial points per direction (Nx² points)
    Nv::Int          = 8         # velocity points per direction (Nv² velocities)
    vgrid::Symbol    = :hermite  # :hermite (Gauss-Hermite) or :uniform
    vmax::Float64    = 4.5       # velocity box [-vmax,vmax]² for :uniform
    cfl::Float64     = 0.5       # Δt = cfl Δx ε / (√2 max|v|)
    dt::Float64      = 0.0       # > 0 overrides the CFL time step
    tfinal::Float64  = 1.5
    order::Int       = 2         # 1: first order, 2: second order
    ale::Bool        = true      # false -> Eulerian (fixed points)
    grid_velocity::Float64 = 1.0 # ALE grid velocity w = grid_velocity * u_F
                                 #   (1: Lagrangian, 0: Eulerian, in between: partial)
    weno::Symbol     = :full     # smoothness indicator: :full (all derivatives) or :direction
    hratio::Float64  = 0.35      # Δx0 = hratio * h
    alpha::Float64   = 6.0       # Gaussian weight parameter
    min_half::Int    = 7         # minimum neighbours in each half plane (stencil = :grow)
    stencil::Symbol  = :grow     # :grow   radius enlarged until each half plane has min_half points
                                 # :tiwari fixed radius h, fit with constant, linear fit if a set
                                 #         has ≤ 8 points, central only if a half plane is too thin
    weno_eps::Float64 = 1e-6     # ε in the smoothness indicators
    r_merge::Float64 = 0.4       # merge points closer than r_merge*Δx0
    r_hole::Float64  = 0.9       # fill lattice nodes with no point within r_hole*Δx0
    mls_order::Int   = 1         # MLS for new/merged points: 1 linear, 2 quadratic
    init::Function   = shear_layer # (x, y, ε) -> (ρ, u1, u2, T) initial fluctuations
    outdir::String   = joinpath(dirname(@__DIR__), "results")   # <package>/results
    nprint::Int      = 0         # progress every nprint steps (0: ~20 messages)
    write_output::Bool = true
    nsave::Int       = 0         # > 0: also save nsave equally spaced snapshots (t = tfinal/nsave, ...)
    save_times::Vector{Float64} = Float64[]   # and/or snapshots at these times
    plot::Bool       = true      # call the plot hook after the run (see start.jl)
end

# -----------------------------------------------------------------------------
# Velocity grid (tensor product), F[k, i] = F(x_i, v_k)
# -----------------------------------------------------------------------------
struct VGrid
    vx::Vector{Float64}
    vy::Vector{Float64}
    w::Vector{Float64}           # ∫ g dv ≈ Σ w_k g(v_k)
    vmax::Float64
    Nv::Int
end

"Gauss-Hermite nodes/weights for ∫ g(v) dv (Golub-Welsch, weight e^{-v²/2})."
function gauss_hermite(n)
    J = SymTridiagonal(zeros(n), sqrt.(collect(1.0:n-1)))
    E = eigen(J)
    v = E.values
    w = sqrt(2π) .* E.vectors[1, :] .^ 2 .* exp.(v .^ 2 ./ 2)
    return v, w
end

function VGrid(p::Params)
    if p.vgrid == :hermite
        v1, w1 = gauss_hermite(p.Nv)
    elseif p.vgrid == :uniform
        v1 = collect(range(-p.vmax, p.vmax; length = p.Nv))
        w1 = fill(v1[2] - v1[1], p.Nv)
    else
        error("unknown vgrid $(p.vgrid) (use :hermite or :uniform)")
    end
    n = p.Nv
    vx = [v1[a] for b in 1:n for a in 1:n]
    vy = [v1[b] for b in 1:n for a in 1:n]
    w  = [w1[a] * w1[b] for b in 1:n for a in 1:n]
    return VGrid(vx, vy, w, maximum(abs, v1), n * n)
end

# -----------------------------------------------------------------------------
# Moments, Maxwellian, relaxation
# -----------------------------------------------------------------------------
"ρ_F, u_F (2 comps), T_F and E = ∫|v|²F of column i."
@inline function moments(F, i, vg::VGrid)
    m0 = 0.0; mx = 0.0; my = 0.0; e = 0.0
    @inbounds @simd for k in 1:vg.Nv
        a = vg.w[k] * F[k, i]
        m0 += a; mx += a * vg.vx[k]; my += a * vg.vy[k]
        e  += a * (vg.vx[k]^2 + vg.vy[k]^2)
    end
    ux = mx / m0; uy = my / m0
    T  = 0.5 * (e / m0 - ux^2 - uy^2)
    return m0, ux, uy, T, e
end

function all_moments(F, vg::VGrid)
    N = size(F, 2)
    ρ = zeros(N); ux = zeros(N); uy = zeros(N); T = zeros(N); E = zeros(N)
    @threads for i in 1:N
        ρ[i], ux[i], uy[i], T[i], E[i] = moments(F, i, vg)
    end
    return ρ, ux, uy, T, E
end

@inline maxwell(ρ, ux, uy, T, vx, vy) =
    ρ / (2π * T) * exp(-((vx - ux)^2 + (vy - uy)^2) / (2T))

# F ← (F̃ + a M[F̃]) / (1 + a),   a = Δt / (ε² τ)   (moments of F̃ are conserved)
function relax!(F, vg::VGrid, a)
    c = 1 / (1 + a)
    @threads for i in axes(F, 2)
        ρ, ux, uy, T, _ = moments(F, i, vg)
        @inbounds for k in 1:vg.Nv
            F[k, i] = c * (F[k, i] + a * maxwell(ρ, ux, uy, T, vg.vx[k], vg.vy[k]))
        end
    end
    return F
end

# -----------------------------------------------------------------------------
# Periodic cell list and neighbour search
# -----------------------------------------------------------------------------
@inline pdist(d, L) = d - L * round(d / L)

struct CellList
    head::Matrix{Int}
    nxt::Vector{Int}
    nc::Int
    cs::Float64                  # cell size (>= search radius h)
    L::Float64
end

@inline cellidx(q, cs, nc) = clamp(floor(Int, q / cs) + 1, 1, nc)

function CellList(x, y, L, h)
    nc = max(1, floor(Int, L / h)); cs = L / nc; N = length(x)
    head = zeros(Int, nc, nc); nxt = zeros(Int, N)
    for i in 1:N
        a = cellidx(x[i], cs, nc); b = cellidx(y[i], cs, nc)
        nxt[i] = head[a, b]; head[a, b] = i
    end
    return CellList(head, nxt, nc, cs, L)
end

"""
Indices and periodic offsets (dx, dy) of all points within r of (qx, qy),
excluding `self`. Works for any r (cells are visited with periodic wrap).
"""
function query!(js, dxs, dys, cl::CellList, x, y, qx, qy, r, self = 0)
    empty!(js); empty!(dxs); empty!(dys)
    L = cl.L; nc = cl.nc; r2 = r * r
    m = ceil(Int, r / cl.cs)                             # cell layers to visit
    a = cellidx(qx, cl.cs, nc); b = cellidx(qy, cl.cs, nc)
    full = 2m + 1 >= nc                                  # then visit every cell once
    ra = full ? (1:nc) : (a-m:a+m)
    rb = full ? (1:nc) : (b-m:b+m)
    for aa in ra, bb in rb
        j = cl.head[mod1(aa, nc), mod1(bb, nc)]
        while j != 0
            if j != self
                dx = pdist(x[j] - qx, L); dy = pdist(y[j] - qy, L)
                if dx * dx + dy * dy < r2
                    push!(js, j); push!(dxs, dx); push!(dys, dy)
                end
            end
            j = cl.nxt[j]
        end
    end
    return js
end

# -----------------------------------------------------------------------------
# Small dense inverse (Gauss-Jordan, partial pivoting), no allocations
# -----------------------------------------------------------------------------
function inv_small!(Ai, A, n)
    @inbounds begin
        for r in 1:n, c in 1:n
            Ai[r, c] = r == c ? 1.0 : 0.0
        end
        scale = 0.0
        for r in 1:n; scale = max(scale, abs(A[r, r])); end
        for c in 1:n
            piv = c; best = abs(A[c, c])
            for r in c+1:n
                abs(A[r, c]) > best && (best = abs(A[r, c]); piv = r)
            end
            best < 1e-12 * scale && return false
            if piv != c
                for q in 1:n
                    A[c, q], A[piv, q] = A[piv, q], A[c, q]
                    Ai[c, q], Ai[piv, q] = Ai[piv, q], Ai[c, q]
                end
            end
            d = 1 / A[c, c]
            for q in 1:n; A[c, q] *= d; Ai[c, q] *= d; end
            for r in 1:n
                r == c && continue
                f = A[r, c]
                f == 0 && continue
                for q in 1:n
                    A[r, q] -= f * A[c, q]; Ai[r, q] -= f * Ai[c, q]
                end
            end
        end
    end
    return true
end

# -----------------------------------------------------------------------------
# Weighted least squares stencils
#   f(x_j) - f(x_i) ≈ Σ_s P_s(dx,dy) a_s,  P = (dx, dy[, dx²/2, dx dy, dy²/2])
#   a = A⁻¹ Σ_j w_j P(d_j) (f_j - f_i)   -> per-edge coefficients
# Sets  s: 1 C (all), 2 L (dx ≤ 0), 3 R (dx ≥ 0), 4 D (dy ≤ 0), 5 U (dy ≥ 0).
# Derivatives d: 1 ∂x, 2 ∂y, 3 ∂xx, 4 ∂xy, 5 ∂yy  (3-5 zero for order 1).
# W[(s-1)*5 + d, e] = coefficient of edge e (CSR over neighbours).
# -----------------------------------------------------------------------------
struct Stencils
    start::Vector{Int}
    idx::Vector{Int}
    W::Matrix{Float64}            # 25 × number of edges
    cx_only::Vector{Bool}         # use the central stencil only for ∂x (stencil = :tiwari)
    cy_only::Vector{Bool}         # ... for ∂y
end

@inline srow(s, d) = (s - 1) * 5 + d

@inline function basis!(P, dx, dy, nb)
    P[1] = dx; P[2] = dy
    if nb == 5
        P[3] = 0.5 * dx * dx; P[4] = dx * dy; P[5] = 0.5 * dy * dy
    end
    return P
end

# fit on the subset `sel` of the neighbour list; writes all nb derivatives of set s
function fit_set!(W, s, off, dxs, dys, ws, sel, nb, A, Ai, P)
    n = length(dxs)
    fill!(A, 0.0)
    cnt = 0
    @inbounds for e in 1:n
        sel(dxs[e], dys[e]) || continue
        cnt += 1
        basis!(P, dxs[e], dys[e], nb)
        for r in 1:nb, c in 1:nb
            A[r, c] += ws[e] * P[r] * P[c]
        end
    end
    ok = cnt >= nb && inv_small!(Ai, A, nb)
    @inbounds for e in 1:n
        use = ok && sel(dxs[e], dys[e])
        use && basis!(P, dxs[e], dys[e], nb)
        for d in 1:5
            v = 0.0
            if use && d <= nb
                for c in 1:nb
                    v += Ai[d, c] * P[c]
                end
                v *= ws[e]
            end
            W[srow(s, d), off + e] = v
        end
    end
    return ok
end

# fit with constant term, basis (1, dx, dy[, dx²/2, dx dy, dy²/2]) of size nbas (3 or 6);
# derivative d is row d+1 of the inverse. Used by stencil = :tiwari.
function fit_set_const!(W, s, off, dxs, dys, ws, sel, nbas, A, Ai, P)
    n = length(dxs)
    fill!(A, 0.0)
    @inbounds for e in 1:n
        sel(dxs[e], dys[e]) || continue
        dx = dxs[e]; dy = dys[e]
        P[1] = 1.0; P[2] = dx; P[3] = dy
        if nbas == 6
            P[4] = 0.5 * dx * dx; P[5] = dx * dy; P[6] = 0.5 * dy * dy
        end
        for r in 1:nbas, c in 1:nbas
            A[r, c] += ws[e] * P[r] * P[c]
        end
    end
    ok = inv_small!(Ai, A, nbas)
    @inbounds for e in 1:n
        use = ok && sel(dxs[e], dys[e])
        dx = dxs[e]; dy = dys[e]
        P[1] = 1.0; P[2] = dx; P[3] = dy
        if nbas == 6
            P[4] = 0.5 * dx * dx; P[5] = dx * dy; P[6] = 0.5 * dy * dy
        end
        for d in 1:5
            v = 0.0
            if use && d + 1 <= nbas
                for c in 1:nbas
                    v += Ai[d + 1, c] * P[c]
                end
                v *= ws[e]
            end
            W[srow(s, d), off + e] = v
        end
    end
    return ok
end

const SEL_C = (dx, dy) -> true
const SEL_L = (dx, dy) -> dx <= 0
const SEL_R = (dx, dy) -> dx >= 0
const SEL_D = (dx, dy) -> dy <= 0
const SEL_U = (dx, dy) -> dy >= 0

function build_stencils(x, y, h, p::Params)
    p.stencil == :tiwari && return build_stencils_tiwari(x, y, h, p)
    p.stencil == :grow || error("unknown stencil $(p.stencil) (use :grow or :tiwari)")
    N  = length(x); L = p.L
    cl = CellList(x, y, L, h)
    nb = p.order >= 2 ? 5 : 2               # basis size

    # pass 1: neighbour lists (radius enlarged until each half plane has min_half points)
    lists = Vector{Vector{Int}}(undef, N)
    dxl = Vector{Vector{Float64}}(undef, N); dyl = Vector{Vector{Float64}}(undef, N)
    hil = zeros(N)
    @threads for i in 1:N
        js = Int[]; dxs = Float64[]; dys = Float64[]
        r = h
        for _ in 1:20
            query!(js, dxs, dys, cl, x, y, x[i], y[i], r, i)
            nl = count(<=(0), dxs); nr = count(>=(0), dxs)
            nd = count(<=(0), dys); nu = count(>=(0), dys)
            min(nl, nr, nd, nu) >= p.min_half && break
            r *= 1.25
        end
        lists[i] = copy(js); dxl[i] = copy(dxs); dyl[i] = copy(dys)
        dmax = 0.0
        for e in eachindex(dxs); dmax = max(dmax, hypot(dxs[e], dys[e])); end
        hil[i] = max(h, 1.0001 * dmax)
    end
    start = Vector{Int}(undef, N + 1); start[1] = 1
    for i in 1:N; start[i + 1] = start[i] + length(lists[i]); end
    ne = start[end] - 1
    S = Stencils(start, Vector{Int}(undef, ne), zeros(25, ne), fill(false, N), fill(false, N))

    # pass 2: coefficients
    chunks = collect(Iterators.partition(1:N, cld(N, nthreads())))
    @threads for ch in chunks
        A = zeros(nb, nb); Ai = zeros(nb, nb); P = zeros(nb); ws = Float64[]
        for i in ch
            off = start[i] - 1
            dxs = dxl[i]; dys = dyl[i]; hi = hil[i]
            for e in eachindex(dxs); S.idx[off + e] = lists[i][e]; end
            resize!(ws, length(dxs))
            for e in eachindex(dxs)
                ws[e] = exp(-p.alpha * (dxs[e]^2 + dys[e]^2) / hi^2)
            end
            fit_set!(S.W, 1, off, dxs, dys, ws, SEL_C, nb, A, Ai, P)
            # one-sided sets; fall back to the central stencil if degenerate
            for (s, sel) in ((2, SEL_L), (3, SEL_R), (4, SEL_D), (5, SEL_U))
                if !fit_set!(S.W, s, off, dxs, dys, ws, sel, nb, A, Ai, P)
                    for e in eachindex(dxs), d in 1:5
                        S.W[srow(s, d), off + e] = S.W[srow(1, d), off + e]
                    end
                end
            end
        end
    end
    return S
end

"""
Stencils as in the Fortran code of S. Tiwari: fixed radius h, the point itself is
part of every set, weight exp(-α r²/h²), weighted least squares with a constant
term; quadratic if a set has more than 8 points, linear otherwise. If a half
plane has ≤ 7 points (≤ 4 for a linear central fit), only the central stencil
is used in that direction.
"""
function build_stencils_tiwari(x, y, h, p::Params)
    N  = length(x)
    cl = CellList(x, y, p.L, h)
    quadratic = p.order >= 2
    lists = Vector{Vector{Int}}(undef, N)
    dxl = Vector{Vector{Float64}}(undef, N); dyl = Vector{Vector{Float64}}(undef, N)
    @threads for i in 1:N
        js = Int[]; dxs = Float64[]; dys = Float64[]
        query!(js, dxs, dys, cl, x, y, x[i], y[i], h)      # includes i itself
        lists[i] = copy(js); dxl[i] = copy(dxs); dyl[i] = copy(dys)
    end
    start = Vector{Int}(undef, N + 1); start[1] = 1
    for i in 1:N; start[i + 1] = start[i] + length(lists[i]); end
    ne = start[end] - 1
    S = Stencils(start, Vector{Int}(undef, ne), zeros(25, ne), fill(false, N), fill(false, N))
    chunks = collect(Iterators.partition(1:N, cld(N, nthreads())))
    @threads for ch in chunks
        A = zeros(6, 6); Ai = zeros(6, 6); P = zeros(6); ws = Float64[]
        for i in ch
            off = start[i] - 1
            dxs = dxl[i]; dys = dyl[i]
            for e in eachindex(dxs); S.idx[off + e] = lists[i][e]; end
            resize!(ws, length(dxs))
            for e in eachindex(dxs)
                ws[e] = exp(-p.alpha * (dxs[e]^2 + dys[e]^2) / h^2)
            end
            cnt = (length(dxs), count(<=(0), dxs), count(>=(0), dxs),
                   count(<=(0), dys), count(>=(0), dys))
            for (s, sel) in ((1, SEL_C), (2, SEL_L), (3, SEL_R), (4, SEL_D), (5, SEL_U))
                nbas = quadratic && cnt[s] > 8 ? 6 : 3
                if cnt[s] < nbas || !fit_set_const!(S.W, s, off, dxs, dys, ws, sel, nbas, A, Ai, P)
                    for e in eachindex(dxs), d in 1:5
                        S.W[srow(s, d), off + e] = 0.0
                    end
                end
            end
            thr = quadratic && cnt[1] > 8 ? 7 : 4
            S.cx_only[i] = cnt[2] <= thr || cnt[3] <= thr
            S.cy_only[i] = cnt[4] <= thr || cnt[5] <= thr
        end
    end
    return S
end

# -----------------------------------------------------------------------------
# Transport:  E[k,i] = -(v_k - w_i)/ε · ∇F(x_i, v_k),   grid velocity w = θ u_F
# WENO-type combination of the upwind half-plane stencil and the central one
# (linear weights ½, ½). Smoothness indicator of a stencil:
#   :direction  β = (f_x² Δx² + f_xx² Δx⁴)                for ∂x (y analogous)
#   :full       β = (f_x² + f_y²) Δx² + (f_xx² + f_xy² + f_yy²) Δx⁴
# weight ∝ ½ / (β + weno_eps)².
# -----------------------------------------------------------------------------
"Grid velocity factor θ: points move with θ u_F (0: Eulerian, 1: Lagrangian)."
theta(p::Params) = p.ale ? p.grid_velocity : 0.0

@inline weno_w(β, we) = 0.5 / (β + we)^2

@inline function smooth_ind(B, k, s, dir, full, dx2, dx4)
    if full
        return (B[k, srow(s, 1)]^2 + B[k, srow(s, 2)]^2) * dx2 +
               (B[k, srow(s, 3)]^2 + B[k, srow(s, 4)]^2 + B[k, srow(s, 5)]^2) * dx4
    else
        return B[k, srow(s, dir)]^2 * dx2 + B[k, srow(s, dir == 1 ? 3 : 5)]^2 * dx4
    end
end

function transport!(E, F, ux, uy, S::Stencils, vg::VGrid, p::Params, dx0)
    N = size(F, 2); nv = vg.Nv
    dx2 = dx0^2; dx4 = dx0^4; we = p.weno_eps; ε = p.eps; θ = theta(p)
    second = p.order >= 2
    full = p.weno == :full
    p.weno in (:full, :direction) || error("unknown weno $(p.weno) (use :direction or :full)")
    # rows that are needed
    need = !second ? [srow(s, d) for (s, d) in ((2, 1), (3, 1), (4, 2), (5, 2))] :
           full    ? collect(1:25) :
                     [srow(1, 1), srow(1, 2), srow(1, 3), srow(1, 5),
                      srow(2, 1), srow(2, 3), srow(3, 1), srow(3, 3),
                      srow(4, 2), srow(4, 5), srow(5, 2), srow(5, 5)]
    chunks = collect(Iterators.partition(1:N, cld(N, nthreads())))
    @threads for ch in chunks
        B = zeros(nv, 25); dv = zeros(nv)
        for i in ch
            fill!(B, 0.0)
            @inbounds for e in S.start[i]:S.start[i+1]-1
                j = S.idx[e]
                @simd for k in 1:nv
                    dv[k] = F[k, j] - F[k, i]
                end
                for r in need
                    c = S.W[r, e]
                    c == 0.0 && continue
                    @simd for k in 1:nv
                        B[k, r] += c * dv[k]
                    end
                end
            end
            sx = θ * ux[i]; sy = θ * uy[i]
            @inbounds for k in 1:nv
                ax = (vg.vx[k] - sx) / ε; ay = (vg.vy[k] - sy) / ε
                sxu = ax > 0 ? 2 : 3          # upwind set for ∂x: L or R
                syu = ay > 0 ? 4 : 5          # upwind set for ∂y: D or U
                if second
                    wC = weno_w(smooth_ind(B, k, 1, 1, full, dx2, dx4), we)
                    wU = weno_w(smooth_ind(B, k, sxu, 1, full, dx2, dx4), we)
                    fx = S.cx_only[i] ? B[k, srow(1, 1)] :
                         (wU * B[k, srow(sxu, 1)] + wC * B[k, srow(1, 1)]) / (wU + wC)
                    if !full
                        wC = weno_w(smooth_ind(B, k, 1, 2, full, dx2, dx4), we)
                    end
                    wU = weno_w(smooth_ind(B, k, syu, 2, full, dx2, dx4), we)
                    fy = S.cy_only[i] ? B[k, srow(1, 2)] :
                         (wU * B[k, srow(syu, 2)] + wC * B[k, srow(1, 2)]) / (wU + wC)
                else
                    fx = B[k, srow(sxu, 1)]
                    fy = B[k, srow(syu, 2)]
                end
                E[k, i] = -(ax * fx + ay * fy)
            end
        end
    end
    return E
end

# -----------------------------------------------------------------------------
# Moving least squares (linear basis) at (qx, qy) -> weights on neighbours
# -----------------------------------------------------------------------------
@inline mls_basis(dx, dy, nb) = nb == 6 ?
    (1.0, dx, dy, 0.5 * dx * dx, dx * dy, 0.5 * dy * dy) : (1.0, dx, dy)

"""
Moving least squares weights at (qx, qy): f(q) ≈ Σ_m ws[m] f(x_js[m]).
`order = 2`: quadratic basis (exact for quadratic fields), falling back to a
linear basis; `order = 1`: linear basis. Shepard weights if degenerate.
`exclude` lists points that must not be used (e.g. a pair being merged).
"""
function mls_weights(cl::CellList, x, y, qx, qy, h, alpha; exclude = (), order = 2)
    js = Int[]; dxs = Float64[]; dys = Float64[]
    r = h
    for _ in 1:12
        query!(js, dxs, dys, cl, x, y, qx, qy, r)
        if !isempty(exclude)
            k = [m for m in eachindex(js) if !(js[m] in exclude)]
            js = js[k]; dxs = dxs[k]; dys = dys[k]
        end
        length(js) >= 12 && break
        r *= 1.25
    end
    ws0 = [exp(-alpha * (dxs[m]^2 + dys[m]^2) / r^2) for m in eachindex(js)]
    for nb in (order >= 2 ? (6, 3) : (3,))
        length(js) >= nb + 2 || continue
        A = zeros(nb, nb)
        for m in eachindex(js)
            Pm = mls_basis(dxs[m] / r, dys[m] / r, nb)
            for a in 1:nb, b in 1:nb
                A[a, b] += ws0[m] * Pm[a] * Pm[b]
            end
        end
        cond(A) < 1e10 || continue
        e1 = zeros(nb); e1[1] = 1.0
        z = A \ e1
        ws = similar(ws0)
        for m in eachindex(js)
            Pm = mls_basis(dxs[m] / r, dys[m] / r, nb)
            ws[m] = ws0[m] * sum(z[a] * Pm[a] for a in 1:nb)
        end
        return js, ws
    end
    return js, ws0 ./ sum(ws0)                # Shepard fall-back
end

"MLS value of all velocity components at (qx, qy); negative values are cut to 0."
function mls_value(cl, x, y, F, qx, qy, h, alpha; exclude = (), order = 2)
    jw, ww = mls_weights(cl, x, y, qx, qy, h, alpha; exclude = exclude, order = order)
    g = zeros(size(F, 1))
    for q in eachindex(jw)
        @views g .+= ww[q] .* F[:, jw[q]]
    end
    return max.(g, 0.0)
end

# -----------------------------------------------------------------------------
# Point management (merge clustered pairs, fill holes)
# -----------------------------------------------------------------------------
function manage_points(x, y, F, p::Params, dx0, h)
    L = p.L; N = length(x)
    cl = CellList(x, y, L, h)
    keep = trues(N)
    js = Int[]; dxs = Float64[]; dys = Float64[]
    # merge: a pair closer than r_merge·Δx0 is replaced by one point at its
    # midpoint; its distribution is the MLS value there from the other points
    mx = Float64[]; my = Float64[]; mF = Vector{Vector{Float64}}()
    for i in 1:N
        keep[i] || continue
        query!(js, dxs, dys, cl, x, y, x[i], y[i], p.r_merge * dx0, i)
        for m in eachindex(js)
            j = js[m]
            if j > i && keep[j]
                qx = mod(x[i] + 0.5 * dxs[m], L); qy = mod(y[i] + 0.5 * dys[m], L)
                push!(mx, qx); push!(my, qy)
                push!(mF, mls_value(cl, x, y, F, qx, qy, h, p.alpha; exclude = (i, j), order = p.mls_order))
                keep[i] = false; keep[j] = false
                break
            end
        end
    end
    nmerge = length(mx)
    if nmerge > 0
        k = findall(keep)
        x = vcat(x[k], mx); y = vcat(y[k], my); F = hcat(F[:, k], reduce(hcat, mF))
        cl = CellList(x, y, L, h)
    end
    # holes: lattice nodes (cell centres of the initial lattice)
    n = round(Int, L / dx0)
    newx = Float64[]; newy = Float64[]
    for a in 1:n, b in 1:n
        qx = (a - 0.5) * dx0; qy = (b - 0.5) * dx0
        query!(js, dxs, dys, cl, x, y, qx, qy, p.r_hole * dx0)
        isempty(js) && (push!(newx, qx); push!(newy, qy))
    end
    if !isempty(newx)
        G = zeros(size(F, 1), length(newx))
        for m in eachindex(newx)
            @views G[:, m] .= mls_value(cl, x, y, F, newx[m], newy[m], h, p.alpha; order = p.mls_order)
        end
        x = vcat(x, newx); y = vcat(y, newy); F = hcat(F, G)
    end
    return x, y, F, nmerge, length(newx)
end

# -----------------------------------------------------------------------------
# Time stepping
# -----------------------------------------------------------------------------
wrap(x, L) = mod.(x, L)
moving(p::Params) = theta(p) != 0
frame_name(p::Params) = !moving(p) ? "Eulerian" : theta(p) == 1 ? "ALE" : @sprintf("ALE(w=%g u)", theta(p))

"First order splitting: explicit transport, exact implicit relaxation."
function step1(x, y, F, dt, p, vg, dx0, h; S = nothing)
    _, ux, uy, _, _ = all_moments(F, vg)
    S === nothing && (S = build_stencils(x, y, h, p))
    E = similar(F); transport!(E, F, ux, uy, S, vg, p, dx0)
    Fn = F .+ dt .* E
    if moving(p)
        c = theta(p) * dt / p.eps
        x = wrap(x .+ c .* ux, p.L); y = wrap(y .+ c .* uy, p.L)
    end
    relax!(Fn, vg, dt / (p.eps^2 * p.tau))
    return x, y, Fn
end

"Second order ARS(2,2,1)-type splitting with mid-point transport."
function step2(x, y, F, dt, p, vg, dx0, h; S = nothing)
    aε = p.eps^2 * p.tau
    # stage 1: half step
    _, ux, uy, _, _ = all_moments(F, vg)
    S1 = S === nothing ? build_stencils(x, y, h, p) : S
    E  = similar(F); transport!(E, F, ux, uy, S1, vg, p, dx0)
    Fh = F .+ (dt / 2) .* E
    relax!(Fh, vg, dt / (2aε))
    if moving(p)
        c = theta(p) * dt / (2p.eps)
        xh = wrap(x .+ c .* ux, p.L); yh = wrap(y .+ c .* uy, p.L)
    else
        xh = x; yh = y
    end
    # stage 2: full step with mid-point transport
    _, uxh, uyh, _, _ = all_moments(Fh, vg)
    S2 = moving(p) ? build_stencils(xh, yh, h, p) : S1
    transport!(E, Fh, uxh, uyh, S2, vg, p, dx0)
    Fn = F .+ dt .* E
    relax!(Fn, vg, dt / aε)
    if moving(p)
        c = theta(p) * dt / p.eps
        x = wrap(x .+ c .* uxh, p.L); y = wrap(y .+ c .* uyh, p.L)
    end
    return x, y, Fn
end

# -----------------------------------------------------------------------------
# Initial data
# -----------------------------------------------------------------------------
"""
    shear_layer(x, y, ε) -> (ρ, u1, u2, T)

Doubly periodic double shear layer: ρ = 1, T = 0,
u1 = tanh(k(y - 1/4)) for y ≤ 1/2, tanh(k(3/4 - y)) otherwise,
u2 = δ sin(2πx), with k = 30, δ = 0.05.  Use `make_shear_layer(k, δ)` for
other values.
"""
shear_layer(x, y, ε) = make_shear_layer(30.0, 0.05)(x, y, ε)

make_shear_layer(k, δ) = (x, y, ε) -> (1.0,
    y <= 0.5 ? tanh(k * (y - 0.25)) : tanh(k * (0.75 - y)),
    δ * sin(2π * x), 0.0)

"""
    make_step_shear_layer(U = 0.5, a = 0.01) -> (x, y, ε) -> (ρ, u1, u2, T)

Double vortex sheet: u1 = U for y ≤ 1/4 and y > 3/4, u1 = -U in between,
plus the perturbation a·sin(2πx) in both u1 and u2; ρ = T = 0.
`step_shear_layer` uses the defaults U = 0.5, a = 0.01.
"""
make_step_shear_layer(U = 0.5, a = 0.01) = function (x, y, ε)
    aa = a * sin(2π * x)
    u1 = (y <= 0.25 || y > 0.75) ? U + aa : -U + aa
    return (0.0, u1, aa, 0.0)
end

step_shear_layer(x, y, ε) = make_step_shear_layer()(x, y, ε)

"Shear wave u1 = A sin(2πy): in the limit ε → 0 it decays like exp(-4π²τt)."
shear_wave(A = 1.0) = (x, y, ε) -> (0.0, A * sin(2π * y), 0.0, 0.0)

"""
Maxwellian initial data from the fluctuations (ρ, u, T) returned by `p.init`:
ρ_F = 1 + ερ,  ρ_F u_F = εu,  T_F = 1 + εT/ρ_F - ε²|u|²/(2ρ_F²).
"""
function init_F(x, y, vg, p)
    ε = p.eps; N = length(x)
    F = zeros(vg.Nv, N)
    for i in 1:N
        ρ, u1, u2, T = p.init(x[i], y[i], ε)
        ρF = 1 + ε * ρ
        TF = 1 + ε * T / ρF - ε^2 * (u1^2 + u2^2) / (2ρF^2)
        TF > 0 || error("initial temperature T_F <= 0 at ($(x[i]), $(y[i]))")
        for k in 1:vg.Nv
            F[k, i] = maxwell(ρF, ε * u1 / ρF, ε * u2 / ρF, TF, vg.vx[k], vg.vy[k])
        end
    end
    return F
end

"""
Fluctuations around (ρ_F, u_F, T_F) = (1, 0, 1):
ρ = (ρ_F-1)/ε,  u = ρ_F u_F/ε,  T = ((∫|v|²F dv - 2)/ε - 2ρ)/2.
"""
function fluctuations(F, vg, ε)
    ρF, uxF, uyF, TF, E = all_moments(F, vg)
    ρ  = (ρF .- 1) ./ ε
    u1 = ρF .* uxF ./ ε
    u2 = ρF .* uyF ./ ε
    T  = ((E .- 2) ./ ε .- 2 .* ρ) ./ 2
    return ρ, u1, u2, T, ρF, uxF, uyF, TF
end

"Vorticity ω = ∂x u2 - ∂y u1 at the points (central WLS stencil)."
function vorticity(x, y, u1, u2, S::Stencils)
    ω = zeros(length(x))
    for i in eachindex(x)
        s = 0.0
        for e in S.start[i]:S.start[i+1]-1
            j = S.idx[e]
            s += S.W[1, e] * (u2[j] - u2[i]) - S.W[2, e] * (u1[j] - u1[i])
        end
        ω[i] = s
    end
    return ω
end

"Resample the rows of G (fields × points) to the n×n lattice by MLS (x fastest)."
function resample(x, y, G::Matrix{Float64}, n, p, h)
    cl = CellList(x, y, p.L, h); dx0 = p.L / n
    out = zeros(size(G, 1), n * n)
    for b in 1:n, a in 1:n
        q = a + (b - 1) * n
        jw, ww = mls_weights(cl, x, y, (a - 0.5) * dx0, (b - 0.5) * dx0, h, p.alpha)
        for m in eachindex(jw)
            @views out[:, q] .+= ww[m] .* G[:, jw[m]]
        end
    end
    return out
end

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
"""
    run_case(p::Params; verbose = true) -> named tuple

Run one case. The result holds the point positions `x, y`, the fluctuations
`rho, u1, u2, T`, the vorticity `omega`, the kinetic fields `rhoF, uxF, uyF, TF`,
the same fields resampled to the Nx×Nx lattice (`xu, yu, rho_u, u1_u, u2_u,
T_u, omega_u`, ordered x fastest), the distribution `F`, the velocity grid
and, if files were written, `file` = path of the final lattice output file and
`snapshots` = paths of the snapshot files (`nsave`, `save_times`).
"""
function run_case(p::Params; verbose = true)
    vg  = VGrid(p)
    dx0 = p.L / p.Nx
    h   = dx0 / p.hratio
    x   = [(a - 0.5) * dx0 for b in 1:p.Nx for a in 1:p.Nx]
    y   = [(b - 0.5) * dx0 for b in 1:p.Nx for a in 1:p.Nx]
    F   = init_F(x, y, vg, p)
    dt  = p.dt > 0 ? p.dt : p.cfl * dx0 * p.eps / (sqrt(2) * vg.vmax)
    nst = ceil(Int, p.tfinal / dt - 1e-9); dt = p.tfinal / nst
    np  = p.nprint > 0 ? p.nprint : max(1, nst ÷ 20)
    mass(F) = sum(all_moments(F, vg)[1]) / size(F, 2)

    verbose && @printf("ε=%g τ=%g  Nx=%d  Nv=%d² (%s)  order=%d  dt=%.3e  steps=%d  threads=%d  %s\n",
                       p.eps, p.tau, p.Nx, p.Nv, p.vgrid, p.order, dt, nst, nthreads(),
                       frame_name(p))
    S = moving(p) ? nothing : build_stencils(x, y, h, p)  # fixed points: build once
    m0 = mass(F)
    # steps at which snapshots are written (the final state is always written)
    ts = vcat(p.save_times, p.nsave > 0 ? [k * p.tfinal / p.nsave for k in 1:p.nsave-1] : Float64[])
    savesteps = Set(clamp(round(Int, τs / dt), 1, nst - 1) for τs in ts if 0 < τs < p.tfinal)
    snaps = String[]
    t = 0.0; nm = 0; ni = 0; tstart = time()
    for n in 1:nst
        x, y, F = p.order == 1 ? step1(x, y, F, dt, p, vg, dx0, h; S = S) :
                                 step2(x, y, F, dt, p, vg, dx0, h; S = S)
        t += dt
        all(isfinite, F) || error("non-finite solution at step $n, t = $t")
        if moving(p)
            x, y, F, a, b = manage_points(x, y, F, p, dx0, h)
            nm += a; ni += b
        end
        if p.write_output && n in savesteps
            r = make_result(x, y, F, vg, p, h, dx0, t)
            push!(snaps, output(r, p, @sprintf("_t%.4g", t); verbose = verbose))
        end
        if verbose && (n % np == 0 || n == nst)
            ρ, u1, u2, T = fluctuations(F, vg, p.eps)
            @printf("  step %7d  t=%.4f  N=%6d  max|u|=%.3e  max|T|=%.3e  mean ρ_F err=%.2e  (%.1fs)\n",
                    n, t, length(x), maximum(hypot.(u1, u2)), maximum(abs, T),
                    mass(F) - m0, time() - tstart)
        end
    end
    moving(p) && verbose && @printf("  merged %d, inserted %d points\n", nm, ni)

    res = make_result(x, y, F, vg, p, h, dx0, t)
    if p.write_output
        res = merge(res, (file = output(res, p, ""; verbose = verbose), snapshots = snaps))
    end
    return res
end

"Assemble the result named tuple (fluctuations, vorticity, lattice resampling)."
function make_result(x, y, F, vg, p, h, dx0, t)
    ρ, u1, u2, T, ρF, uxF, uyF, TF = fluctuations(F, vg, p.eps)
    S  = build_stencils(x, y, h, p)
    ω  = vorticity(x, y, u1, u2, S)
    nl = p.Nx
    Gu = resample(x, y, permutedims(hcat(ρ, u1, u2, T, ω)), nl, p, h)
    xu = [(a - 0.5) * dx0 for b in 1:nl for a in 1:nl]
    yu = [(b - 0.5) * dx0 for b in 1:nl for a in 1:nl]
    return (x = x, y = y, rho = ρ, u1 = u1, u2 = u2, T = T, omega = ω,
            rhoF = ρF, uxF = uxF, uyF = uyF, TF = TF,
            xu = xu, yu = yu, rho_u = Gu[1, :], u1_u = Gu[2, :], u2_u = Gu[3, :],
            T_u = Gu[4, :], omega_u = Gu[5, :], F = F, vx = vg.vx, vy = vg.vy, t = t)
end

"Write the files of a result (suffix \"\" for the final state, \"_t<t>\" for snapshots) and plot."
function output(res, p, suffix; verbose = true)
    file = write_results(res, p; suffix = suffix, verbose = verbose)
    if p.plot && PLOT_HOOK[] !== nothing
        try
            Base.invokelatest(PLOT_HOOK[], file)
        catch err
            @warn "plotting failed" exception = err
        end
    end
    return file
end

"""
Function `file -> nothing` called by `run_case` after the output is written
(set by start.jl to plot the vorticity and save the PNG next to the data).
The solver itself does not depend on Plots.jl.
"""
const PLOT_HOOK = Ref{Any}(nothing)

"""
    file_tag(p) -> "bgk2d_eps<ε>_tau<τ>_<ale|eul>_Nx<Nx>"

Base name of the output files of a run.
"""
function file_tag(p::Params)
    fr = !moving(p) ? "eul" : theta(p) == 1 ? "ale" : @sprintf("ale%g", theta(p))
    w  = (p.weno == :full ? "" : "_wdir") * (p.stencil == :tiwari ? "_stiw" : "")
    return @sprintf("bgk2d_eps%g_tau%g_%s%s_Nx%d", p.eps, p.tau, fr, w, p.Nx)
end

"""
    write_results(res, p; verbose = true) -> path of the lattice file

Write to `p.outdir`
  `<tag>_uniform.dat`    values resampled to the Nx×Nx lattice (x fastest),
  `<tag>_particles.dat`  values at the points,
with `<tag> = bgk2d_eps<ε>_tau<τ>_<ale|eul>_Nx<Nx>` and columns
x  y  ρ  u1  u2  T  ω. The first line records the run parameters.
"""
function write_results(res, p::Params; suffix = "", verbose = true)
    mkpath(p.outdir)
    tag = file_tag(p)
    hdr = @sprintf("# x y rho u1 u2 T omega   eps=%g tau=%g t=%g Nx=%d Nv=%d order=%d frame=%s weno=%s alpha=%g stencil=%s N=%d",
                   p.eps, p.tau, res.t, p.Nx, p.Nv, p.order, (!moving(p) ? "Eulerian" : theta(p) == 1 ? "ALE" : @sprintf("ALE_w%g", theta(p))),
                   p.weno, p.alpha, p.stencil, length(res.x))
    function dump(file, cols)
        open(file, "w") do io
            println(io, hdr)
            for i in eachindex(cols[1])
                println(io, join((@sprintf("%.10e", c[i]) for c in cols), " "))
            end
        end
    end
    funi = abspath(joinpath(p.outdir, tag * suffix * "_uniform.dat"))
    fpar = abspath(joinpath(p.outdir, tag * suffix * "_particles.dat"))
    dump(funi, (res.xu, res.yu, res.rho_u, res.u1_u, res.u2_u, res.T_u, res.omega_u))
    dump(fpar, (res.x, res.y, res.rho, res.u1, res.u2, res.T, res.omega))
    verbose && println("  wrote ", funi, "\n        ", fpar)
    return funi
end

end # module
