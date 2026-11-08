if Base.find_package("ALEBGK2D") === nothing      # include()d without activating
    import Pkg; Pkg.activate(dirname(@__DIR__); io = devnull)
end
using ALEBGK2D, Test

# small, fast configurations
small(; kw...) = Params(; Nx = 24, Nv = 8, tfinal = 0.05, write_output = false, kw...)
rest(x, y, ε) = (0.0, 0.0, 0.0, 0.0)

@testset "ALEBGK2D" begin
    @testset "global equilibrium stays at rest" begin
        res = run_case(small(eps = 1.0, tau = 0.1, Nx = 12, Nv = 6, tfinal = 0.02,
                             init = rest); verbose = false)
        @test maximum(abs, res.rho) < 1e-10
        @test maximum(abs, res.u1)  < 1e-10
        @test maximum(abs, res.u2)  < 1e-10
        @test maximum(abs, res.T)   < 1e-10
    end

    # shear wave u1 = sin(2πy): for small ε it decays like exp(-4π²τt)
    limit = exp(-4π^2 * 0.05 * 0.05)
    amp(res) = sum(res.u1 .* sinpi.(2 .* res.y)) / sum(sinpi.(2 .* res.y) .^ 2)

    @testset "shear wave decay (Eulerian)" begin
        res = run_case(small(eps = 0.1, tau = 0.05, ale = false, init = shear_wave());
                       verbose = false)
        @test all(isfinite, res.F)
        @test abs(amp(res) - limit) < 0.02 * limit          # ≈ 0.6 % at Nx = 24
        @test maximum(abs, res.u2) < 0.02
        @test abs(sum(res.rhoF) / length(res.rhoF) - 1) < 1e-4
    end

    @testset "ALE agrees with Eulerian" begin
        e = run_case(small(eps = 0.1, tau = 0.05, ale = false, init = shear_wave());
                     verbose = false)
        a = run_case(small(eps = 0.1, tau = 0.05, ale = true,  init = shear_wave());
                     verbose = false)
        @test all(isfinite, a.F)
        @test abs(amp(a) - amp(e)) < 0.01 * amp(e)          # ≈ 0.2 %
        @test abs(sum(a.rhoF) / length(a.rhoF) - 1) < 1e-3
    end

    @testset "full smoothness indicator (weno = :full)" begin
        # prototype: amplitude 0.9003 at Nx = 24 (limit 0.906)
        res = run_case(small(eps = 0.1, tau = 0.05, ale = false, weno = :full,
                             alpha = 2.25, weno_eps = 1e-7, init = shear_wave());
                       verbose = false)
        @test all(isfinite, res.F)
        @test abs(amp(res) - limit) < 0.02 * limit
    end

    @testset "Tiwari stencils (stencil = :tiwari)" begin
        # exact for quadratics on a uniform lattice
        n = 16; dx0 = 1 / n; h = dx0 / 0.4
        xs = [(a - 0.5) * dx0 for b in 1:n for a in 1:n]
        ys = [(b - 0.5) * dx0 for b in 1:n for a in 1:n]
        p = Params(stencil = :tiwari, alpha = 2.25, hratio = 0.4)
        S = ALEBGK2D.build_stencils(xs, ys, h, p)
        i = n ÷ 2 + (n ÷ 2 - 1) * n                    # a point near the centre
        q(x, y) = 1 + 2x - 3y + 4x^2 + x * y - 2y^2   # f_x = 2 + 8x + y
        for s in 1:5
            fx = sum(S.W[ALEBGK2D.srow(s, 1), e] * (q(xs[S.idx[e]], ys[S.idx[e]]) - q(xs[i], ys[i]))
                     for e in S.start[i]:S.start[i+1]-1)
            @test abs(fx - (2 + 8xs[i] + ys[i])) < 1e-8
        end
        @test !any(S.cx_only) && !any(S.cy_only)
        res = run_case(small(eps = 0.1, tau = 0.05, ale = false, stencil = :tiwari, hratio = 0.4,
                             alpha = 2.25, init = shear_wave()); verbose = false)
        @test abs(amp(res) - limit) < 0.02 * limit     # prototype: 0.9032
    end

    @testset "partial grid velocity (grid_velocity = 0.5)" begin
        e = run_case(small(eps = 0.1, tau = 0.05, ale = false, init = shear_wave());
                     verbose = false)
        a = run_case(small(eps = 0.1, tau = 0.05, grid_velocity = 0.5, init = shear_wave());
                     verbose = false)
        @test all(isfinite, a.F)
        @test abs(amp(a) - amp(e)) < 0.01 * amp(e)
    end

    @testset "quadratic MLS is exact for quadratics (irregular, periodic)" begin
        n = 20; dx0 = 1 / n; h = dx0 / 0.35
        xs = Float64[]; ys = Float64[]; s = 0.37
        for b in 1:n, a in 1:n                  # deterministic jitter
            s = mod(s * 7.13 + 0.211, 1.0); push!(xs, mod((a - 0.5 + 0.6 * (s - 0.5)) * dx0, 1))
            s = mod(s * 7.13 + 0.211, 1.0); push!(ys, mod((b - 0.5 + 0.6 * (s - 0.5)) * dx0, 1))
        end
        q = (0.5, 0.5)
        f(x, y) = 1 + 2(x - q[1]) - 3(y - q[2]) + 4(x - q[1])^2 + (x - q[1]) * (y - q[2]) - 2(y - q[2])^2
        F = reshape([f(xs[i], ys[i]) for i in eachindex(xs)], 1, :)
        cl = ALEBGK2D.CellList(xs, ys, 1.0, h)
        g = ALEBGK2D.mls_value(cl, xs, ys, F, q[1], q[2], h, 6.0)
        @test abs(g[1] - f(q...)) < 1e-10
        # with excluded points (as used when merging a pair)
        g2 = ALEBGK2D.mls_value(cl, xs, ys, F, q[1], q[2], h, 6.0; exclude = (1, 2))
        @test abs(g2[1] - f(q...)) < 1e-10
        # linear MLS: exact for linear fields
        flin(x, y) = 1 + 2(x - q[1]) - 3(y - q[2])
        Fl = reshape([flin(xs[i], ys[i]) for i in eachindex(xs)], 1, :)
        gl = ALEBGK2D.mls_value(cl, xs, ys, Fl, q[1], q[2], h, 2.25; order = 1)
        @test abs(gl[1] - flin(q...)) < 1e-10
    end

    @testset "output file names and returned path" begin
        dir = mktempdir()
        res = run_case(Params(Nx = 12, Nv = 6, tfinal = 0.01, eps = 0.5, tau = 0.01,
                              ale = false, outdir = dir); verbose = false)
        @test isfile(res.file)
        @test basename(res.file) == "bgk2d_eps0.5_tau0.01_eul_Nx12_uniform.dat"
        @test isfile(replace(res.file, "_uniform" => "_particles"))
        @test occursin("frame=Eulerian", readline(res.file))
    end

    @testset "step shear layer initial data" begin
        @test step_shear_layer(0.25, 0.1, 0.1) == (0.0, 0.5 + 0.01, 0.01, 0.0)
        @test step_shear_layer(0.25, 0.5, 0.1)[2] ≈ -0.5 + 0.01
        res = run_case(small(eps = 0.1, tau = 0.001, ale = false, tfinal = 0.01,
                             init = step_shear_layer); verbose = false)
        @test all(isfinite, res.F)
    end

    @testset "first-order scheme runs; second order is more accurate" begin
        # first-order upwinding adds a numerical viscosity ~ Δx/ε:
        # amplitude ≈ 0.40 at Nx = 24 (limit 0.906); second order ≈ 0.90
        o1 = run_case(small(eps = 0.1, tau = 0.05, order = 1, ale = false,
                            init = shear_wave()); verbose = false)
        o2 = run_case(small(eps = 0.1, tau = 0.05, order = 2, ale = false,
                            init = shear_wave()); verbose = false)
        @test all(isfinite, o1.F)
        @test 0.3 * limit < amp(o1) < limit
        @test abs(amp(o2) - limit) < abs(amp(o1) - limit) / 5
    end
end
