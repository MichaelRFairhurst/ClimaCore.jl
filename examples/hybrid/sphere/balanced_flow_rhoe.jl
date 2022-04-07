using Test
using ClimaCorePlots, Plots
using ClimaCore.DataLayouts

include("baroclinic_wave_utilities.jl")

sponge = false

setups = [
    HybridDriverSetup(;
        additional_cache = make_additional_cache(sponge; κ₄ = FT(2e17)),
        additional_tendency! = make_additional_tendency(sponge),
        center_initial_condition = make_center_initial_condition(:ρe, true),
        face_initial_condition = make_face_initial_condition(),
        horizontal_mesh = cubed_sphere_mesh(; radius = R, h_elem = 4),
        npoly = 4,
        z_max = FT(30e3),
        z_elem = 10,
        t_end = FT(60 * 60),
        dt = FT(5),
        dt_save_to_sol = FT(50),
        dt_save_to_disk = FT(0), # 0 means don't save to disk
        ode_algorithm = SSPRK33,
    ),
]

function postprocessing(sols, output_dir)
    sol = sols[1]
    @info "L₂ norm of ρe at t = $(sol.t[1]): $(norm(sol.u[1].c.ρe))"
    @info "L₂ norm of ρe at t = $(sol.t[end]): $(norm(sol.u[end].c.ρe))"

    ᶜu_end = Geometry.UVVector.(sol.u[end].c.uₕ).components.data.:1
    Plots.png(Plots.plot(ᶜu_end, level = 3), joinpath(output_dir, "u_end.png"))

    ᶜw_end = Geometry.WVector.(sol.u[end].f.w).components.data.:1
    Plots.png(
        Plots.plot(ᶜw_end, level = 3 + half, clim = (-4, 4)),
        joinpath(output_dir, "w_end.png"),
    )

    ᶜu_start = Geometry.UVVector.(sol.u[1].c.uₕ).components.data.:1
    Plots.png(
        Plots.plot(ᶜu_end .- ᶜu_start, level = 3, clim = (-1, 1)),
        joinpath(output_dir, "Δu_end.png"),
    )

    @test sol.u[end].c.ρ ≈ sol.u[1].c.ρ rtol = 5e-2
    @test sol.u[end].c.ρe ≈ sol.u[1].c.ρe rtol = 5e-2
    @test sol.u[end].c.uₕ ≈ sol.u[1].c.uₕ rtol = 5e-2
end
