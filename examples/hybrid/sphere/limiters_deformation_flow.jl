using Test
using LinearAlgebra

import ClimaCore:
    ClimaCore,
    slab,
    Spaces,
    Domains,
    Meshes,
    Geometry,
    Topologies,
    Spaces,
    Fields,
    Operators,
    Limiters
import ClimaCore.Utilities: half

using OrdinaryDiffEq: ODEProblem, solve
using DiffEqBase
using ClimaTimeSteppers

import Logging
import TerminalLoggers
Logging.global_logger(TerminalLoggers.TerminalLogger())

# 3D deformation flow (DCMIP 2012 Test 1-1)
# Reference: http://www-personal.umich.edu/~cjablono/DCMIP-2012_TestCaseDocument_v1.7.pdf, Section 1.1

const R = 6.37122e6        # radius
const grav = 9.8           # gravitational constant
const R_d = 287.058        # R dry (gas constant / mol mass dry air)
const z_top = 1.2e4        # height position of the model top
const p_top = 25494.4      # pressure at the model top
const T_0 = 300            # isothermal atmospheric temperature
const H = R_d * T_0 / grav # scale height
const p_0 = 1.0e5          # reference pressure
const τ = 1036800.0        # period of motion
const ω_0 = 23000 * pi / τ # maxium of the vertical pressure velocity
const b = 0.2              # normalized pressure depth of divergent layer
const λ_c1 = 150.0         # initial longitude of first tracer
const λ_c2 = 210.0         # initial longitude of second tracer
const ϕ_c = 0.0            # initial latitude of tracers
const centers = [
    Geometry.LatLongZPoint(ϕ_c, λ_c1, 0.0),
    Geometry.LatLongZPoint(ϕ_c, λ_c2, 0.0),
]
const z_c = 5.0e3          # initial altitude of tracers
const R_t = R / 2          # horizontal half-width of tracers
const Z_t = 1000.0         # vertical half-width of tracers
const D₄ = 1.0e16          # hyperviscosity coefficient
const lim_flag = true      # limiters flag
const limiter_tol = 5e-14  # limiters least-square optmimum tolerance
T = 86400 * 12             # simulation times in seconds (12 days)
dt = 60 * 60               # time step in seconds (60 minutes)
zelems = 16
helems = 8

FT = Float64

# visualization artifacts
ENV["GKSwstype"] = "nul"
using ClimaCorePlots, Plots
Plots.GRBackend()

dirname = "limiters_deformation_flow"

if lim_flag == false
    dirname = "$(dirname)_no_lim"
end
if D₄ == 0
    dirname = "$(dirname)_D0"
end

path = joinpath(@__DIR__, "output", dirname)
mkpath(path)

function linkfig(figpath, alt = "")
    # buildkite-agent upload figpath
    # link figure in logs if we are running on CI
    if get(ENV, "BUILDKITE", "") == "true"
        artifact_url = "artifact://$figpath"
        print("\033]1338;url='$(artifact_url)';alt='$(alt)'\a\n")
    end
end

# set up function space
function sphere_3D(
    R = 6.37122e6,
    zlim = (0, 12.0e3);
    helem = 4,
    zelem = 12,
    npoly = 4,
)
    FT = Float64
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(zlim[1]),
        Geometry.ZPoint{FT}(zlim[2]);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = zelem)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)

    horzdomain = Domains.SphereDomain(R)
    horzmesh = Meshes.EquiangularCubedSphere(horzdomain, helem)
    horztopology = Topologies.Topology2D(horzmesh)
    quad = Spaces.Quadratures.GLL{npoly + 1}()
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)

    hv_center_space =
        Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    hv_face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(hv_center_space)
    return (horzspace, hv_center_space, hv_face_space)
end

# set up 3D domain
horzspace, hv_center_space, hv_face_space =
    sphere_3D(helem = helems, zelem = zelems)
global_geom = horzspace.global_geometry
topology = horzspace.topology

# Extract coordinates
center_coords = Fields.coordinate_field(hv_center_space)
face_coords = Fields.coordinate_field(hv_face_space)

# Initialize variables needed for limiters
horz_n_elems = Topologies.nlocalelems(horzspace.topology)
min_q1 = zeros(horz_n_elems, zelems)
max_q1 = zeros(horz_n_elems, zelems)
min_q2 = zeros(horz_n_elems, zelems)
max_q2 = zeros(horz_n_elems, zelems)
min_q3 = zeros(horz_n_elems, zelems)
max_q3 = zeros(horz_n_elems, zelems)
min_q4 = zeros(horz_n_elems, zelems)
max_q4 = zeros(horz_n_elems, zelems)
min_q5 = zeros(horz_n_elems, zelems)
max_q5 = zeros(horz_n_elems, zelems)

# Initialize pressure and density
p(z) = p_0 * exp(-z / H)
ρ_ref(z) = p(z) / R_d / T_0

y0 = map(center_coords) do coord
    z = coord.z
    zd = z - z_c
    λ = coord.long
    ϕ = coord.lat
    rd = Vector{Float64}(undef, 2)

    # great circle distances
    for i in 1:2
        rd[i] = Geometry.great_circle_distance(coord, centers[i], global_geom)
    end

    # scaled distance functions
    d = Vector{Float64}(undef, 2)
    for i in 1:2
        d[i] = min(1, (rd[i] / R_t)^2 + (zd / Z_t)^2)
    end

    q1 = 0.5 * (1 + cos(pi * d[1])) + 0.5 * (1 + cos(pi * d[2]))
    q2 = 0.9 - 0.8 * q1^2
    q3 = 0.0
    if d[1] < 0.5 || d[2] < 0.5
        q3 = 1.0
    else
        q3 = 0.1
    end

    if (z > z_c) && (ϕ > ϕ_c - rad2deg(1 / 8)) && (ϕ < ϕ_c + rad2deg(1 / 8))
        q3 = 0.1
    end
    q4 = 1 - 3 / 10 * (q1 + q2 + q3)
    q5 = 1

    ρq1 = ρ_ref(z) * q1
    ρq2 = ρ_ref(z) * q2
    ρq3 = ρ_ref(z) * q3
    ρq4 = ρ_ref(z) * q4
    ρq5 = ρ_ref(z) * q5

    return (ρ = ρ_ref(z), ρq1 = ρq1, ρq2 = ρq2, ρq3 = ρq3, ρq4 = ρq4, ρq5 = ρq5)
end

y0 = Fields.FieldVector(
    ρ = y0.ρ,
    ρq1 = y0.ρq1,
    ρq2 = y0.ρq2,
    ρq3 = y0.ρq3,
    ρq4 = y0.ρq4,
    ρq5 = y0.ρq5,
)

function rhs!(dydt, y, parameters, t, alpha, beta)

    # Set up operators
    # Spectral horizontal operators
    hwdiv = Operators.WeakDivergence()
    hgrad = Operators.Gradient()
    # Vertical staggered FD operators
    If2c = Operators.InterpolateF2C()
    Ic2f = Operators.InterpolateC2F(
        bottom = Operators.Extrapolate(),
        top = Operators.Extrapolate(),
    )
    vdivf2c = Operators.DivergenceF2C(
        top = Operators.SetValue(Geometry.Contravariant3Vector(FT(0.0))),
        bottom = Operators.SetValue(Geometry.Contravariant3Vector(FT(0.0))),
    )
    third_order_upwind_c2f = Operators.Upwind3rdOrderBiasedProductC2F(
        bottom = Operators.ThirdOrderOneSided(),
        top = Operators.ThirdOrderOneSided(),
    )

    # Define flow
    τ = parameters.τ
    center_coords = parameters.center_coords
    face_coords = parameters.face_coords

    ϕ = center_coords.lat
    λ = center_coords.long
    zc = center_coords.z
    zf = face_coords.z
    λp = λ .- 360 * t / τ
    k = 10 * R / τ

    ϕf = face_coords.lat
    λf = face_coords.long
    λpf = λf .- 360 * t / τ

    sp =
        @. 1 + exp((p_top - p_0) / b / p_top) - exp((p(zf) - p_0) / b / p_top) -
           exp((p_top - p(zf)) / b / p_top)
    ua = @. k * sind(λp)^2 * sind(2 * ϕ) * cos(pi * t / τ) +
       2 * pi * R / τ * cosd(ϕ)
    ud = @. ω_0 * R / b / p_top *
       cosd(λp) *
       cosd(ϕ)^2 *
       cos(2 * pi * t / τ) *
       (-exp((p(zc) - p_0) / b / p_top) + exp((p_top - p(zc)) / b / p_top))
    uu = @. ua + ud
    uv = @. k * sind(2 * λp) * cosd(ϕ) * cos(pi * t / τ)
    ω = @. ω_0 * sind(λpf) * cosd(ϕf) * cos(2 * pi * t / τ) * sp
    uw = @. -ω / ρ_ref(zf) / grav

    uₕ = Geometry.Covariant12Vector.(Geometry.UVVector.(uu, uv))
    w = Geometry.Covariant3Vector.(Geometry.WVector.(uw))

    # Compute vertical velocity by interpolating faces to centers
    cw = If2c.(w)
    cuvw = Geometry.Covariant123Vector.(uₕ) .+ Geometry.Covariant123Vector.(cw)

    ρ = y.ρ
    ρq1 = y.ρq1
    ρq2 = y.ρq2
    ρq3 = y.ρq3
    ρq4 = y.ρq4
    ρq5 = y.ρq5

    dρ = dydt.ρ
    dρq1 = dydt.ρq1
    dρq2 = dydt.ρq2
    dρq3 = dydt.ρq3
    dρq4 = dydt.ρq4
    dρq5 = dydt.ρq5

    # Define vertical fluxes
    vert_flux_wρ = vdivf2c.(w .* Ic2f.(ρ))
    vert_flux_wρq1 = vdivf2c.(Ic2f.(ρ) .* third_order_upwind_c2f.(w, ρq1 ./ ρ),)
    vert_flux_wρq2 = vdivf2c.(Ic2f.(ρ) .* third_order_upwind_c2f.(w, ρq2 ./ ρ),)
    vert_flux_wρq3 = vdivf2c.(Ic2f.(ρ) .* third_order_upwind_c2f.(w, ρq3 ./ ρ),)
    vert_flux_wρq4 = vdivf2c.(Ic2f.(ρ) .* third_order_upwind_c2f.(w, ρq4 ./ ρ),)
    vert_flux_wρq5 = vdivf2c.(Ic2f.(ρ) .* third_order_upwind_c2f.(w, ρq5 ./ ρ),)

    # Compute min_qi[] and max_qi[] that will be needed later in the stage limiters
    horz_neigh_elems_q1_min = Array{FT}(undef, 8, zelems)
    horz_neigh_elems_q1_max = Array{FT}(undef, 8, zelems)

    horz_neigh_elems_q2_min = Array{FT}(undef, 8, zelems)
    horz_neigh_elems_q2_max = Array{FT}(undef, 8, zelems)

    horz_neigh_elems_q3_min = Array{FT}(undef, 8, zelems)
    horz_neigh_elems_q3_max = Array{FT}(undef, 8, zelems)

    horz_neigh_elems_q4_min = Array{FT}(undef, 8, zelems)
    horz_neigh_elems_q4_max = Array{FT}(undef, 8, zelems)

    horz_neigh_elems_q5_min = Array{FT}(undef, 8, zelems)
    horz_neigh_elems_q5_max = Array{FT}(undef, 8, zelems)

    horz_q1_e = Array{Fields.Field}(undef, zelems)
    horz_q2_e = Array{Fields.Field}(undef, zelems)
    horz_q3_e = Array{Fields.Field}(undef, zelems)
    horz_q4_e = Array{Fields.Field}(undef, zelems)
    horz_q5_e = Array{Fields.Field}(undef, zelems)

    horz_q1_e_min = Array{FT}(undef, zelems)
    horz_q1_e_max = Array{FT}(undef, zelems)

    horz_q2_e_min = Array{FT}(undef, zelems)
    horz_q2_e_max = Array{FT}(undef, zelems)

    horz_q3_e_min = Array{FT}(undef, zelems)
    horz_q3_e_max = Array{FT}(undef, zelems)

    horz_q4_e_min = Array{FT}(undef, zelems)
    horz_q4_e_max = Array{FT}(undef, zelems)

    horz_q5_e_min = Array{FT}(undef, zelems)
    horz_q5_e_max = Array{FT}(undef, zelems)

    for v in 1:zelems
        for he in 1:horz_n_elems
            horz_q1_e[v] = Fields.slab(ρq1, v, he) ./ Fields.slab(ρ, v, he)
            horz_q2_e[v] = Fields.slab(ρq2, v, he) ./ Fields.slab(ρ, v, he)
            horz_q3_e[v] = Fields.slab(ρq3, v, he) ./ Fields.slab(ρ, v, he)
            horz_q4_e[v] = Fields.slab(ρq4, v, he) ./ Fields.slab(ρ, v, he)
            horz_q5_e[v] = Fields.slab(ρq5, v, he) ./ Fields.slab(ρ, v, he)

            horz_q1_e_min[v] = minimum(horz_q1_e[v])
            horz_q1_e_max[v] = maximum(horz_q1_e[v])

            horz_q2_e_min[v] = minimum(horz_q2_e[v])
            horz_q2_e_max[v] = maximum(horz_q2_e[v])

            horz_q3_e_min[v] = minimum(horz_q3_e[v])
            horz_q3_e_max[v] = maximum(horz_q3_e[v])

            horz_q4_e_min[v] = minimum(horz_q4_e[v])
            horz_q4_e_max[v] = maximum(horz_q4_e[v])

            horz_q5_e_min[v] = minimum(horz_q5_e[v])
            horz_q5_e_max[v] = maximum(horz_q5_e[v])

            horz_neigh_elems = Topologies.neighboring_elements(topology, he)
            for i in 1:length(horz_neigh_elems)
                if horz_neigh_elems[i] == 0
                    horz_neigh_elems_q1_min[i] = +Inf
                    horz_neigh_elems_q1_max[i] = -Inf

                    horz_neigh_elems_q2_min[i] = +Inf
                    horz_neigh_elems_q2_max[i] = -Inf

                    horz_neigh_elems_q3_min[i] = +Inf
                    horz_neigh_elems_q3_max[i] = -Inf

                    horz_neigh_elems_q4_min[i] = +Inf
                    horz_neigh_elems_q4_max[i] = -Inf

                    horz_neigh_elems_q5_min[i] = +Inf
                    horz_neigh_elems_q5_max[i] = -Inf
                else
                    horz_neigh_elems_q1_min[i] = Fields.minimum(
                        Fields.slab(ρq1, v, horz_neigh_elems[i]) ./
                        Fields.slab(ρ, v, horz_neigh_elems[i]),
                    )
                    horz_neigh_elems_q1_max[i] = Fields.maximum(
                        Fields.slab(ρq1, v, horz_neigh_elems[i]) ./
                        Fields.slab(ρ, v, horz_neigh_elems[i]),
                    )

                    horz_neigh_elems_q2_min[i] = Fields.minimum(
                        Fields.slab(ρq2, v, horz_neigh_elems[i]) ./
                        Fields.slab(ρ, v, horz_neigh_elems[i]),
                    )
                    horz_neigh_elems_q2_max[i] = Fields.maximum(
                        Fields.slab(ρq2, v, horz_neigh_elems[i]) ./
                        Fields.slab(ρ, v, horz_neigh_elems[i]),
                    )

                    horz_neigh_elems_q3_min[i] = Fields.minimum(
                        Fields.slab(ρq3, v, horz_neigh_elems[i]) ./
                        Fields.slab(ρ, v, horz_neigh_elems[i]),
                    )
                    horz_neigh_elems_q3_max[i] = Fields.maximum(
                        Fields.slab(ρq3, v, horz_neigh_elems[i]) ./
                        Fields.slab(ρ, v, horz_neigh_elems[i]),
                    )

                    horz_neigh_elems_q4_min[i] = Fields.minimum(
                        Fields.slab(ρq4, v, horz_neigh_elems[i]) ./
                        Fields.slab(ρ, v, horz_neigh_elems[i]),
                    )
                    horz_neigh_elems_q4_max[i] = Fields.maximum(
                        Fields.slab(ρq4, v, horz_neigh_elems[i]) ./
                        Fields.slab(ρ, v, horz_neigh_elems[i]),
                    )

                    horz_neigh_elems_q5_min[i] = Fields.minimum(
                        Fields.slab(ρq5, v, horz_neigh_elems[i]) ./
                        Fields.slab(ρ, v, horz_neigh_elems[i]),
                    )
                    horz_neigh_elems_q5_max[i] = Fields.maximum(
                        Fields.slab(ρq5, v, horz_neigh_elems[i]) ./
                        Fields.slab(ρ, v, horz_neigh_elems[i]),
                    )
                end
            end
            parameters.min_q1[he, v] =
                min(minimum(horz_neigh_elems_q1_min), horz_q1_e_min[v])
            parameters.max_q1[he, v] =
                max(maximum(horz_neigh_elems_q1_max), horz_q1_e_max[v])

            parameters.min_q2[he, v] =
                min(minimum(horz_neigh_elems_q2_min), horz_q2_e_min[v])
            parameters.max_q2[he, v] =
                max(maximum(horz_neigh_elems_q2_max), horz_q2_e_max[v])

            parameters.min_q3[he, v] =
                min(minimum(horz_neigh_elems_q3_min), horz_q3_e_min[v])
            parameters.max_q3[he, v] =
                max(maximum(horz_neigh_elems_q3_max), horz_q3_e_max[v])

            parameters.min_q4[he, v] =
                min(minimum(horz_neigh_elems_q4_min), horz_q4_e_min[v])
            parameters.max_q4[he, v] =
                max(maximum(horz_neigh_elems_q4_max), horz_q4_e_max[v])

            parameters.min_q5[he, v] =
                min(minimum(horz_neigh_elems_q5_min), horz_q5_e_min[v])
            parameters.max_q5[he, v] =
                max(maximum(horz_neigh_elems_q5_max), horz_q5_e_max[v])
        end
    end

    # Hyperdiffusion
    ystar = similar(y)
    # Compute hyperviscosity for the tracers equation by splitting it in two diffusion calls
    @. ystar.ρq1 = hwdiv(hgrad(ρq1 / ρ))
    Spaces.weighted_dss!(ystar.ρq1)
    @. ystar.ρq1 = -D₄ * hwdiv(ρ * hgrad(ystar.ρq1))

    @. ystar.ρq2 = hwdiv(hgrad(ρq2 / ρ))
    Spaces.weighted_dss!(ystar.ρq2)
    @. ystar.ρq2 = -D₄ * hwdiv(ρ * hgrad(ystar.ρq2))

    @. ystar.ρq3 = hwdiv(hgrad(ρq3 / ρ))
    Spaces.weighted_dss!(ystar.ρq3)
    @. ystar.ρq3 = -D₄ * hwdiv(ρ * hgrad(ystar.ρq3))

    @. ystar.ρq4 = hwdiv(hgrad(ρq4 / ρ))
    Spaces.weighted_dss!(ystar.ρq4)
    @. ystar.ρq4 = -D₄ * hwdiv(ρ * hgrad(ystar.ρq4))

    @. ystar.ρq5 = hwdiv(hgrad(ρq5 / ρ))
    Spaces.weighted_dss!(ystar.ρq5)
    @. ystar.ρq5 = -D₄ * hwdiv(ρ * hgrad(ystar.ρq5))

    # 1) Contintuity equation:
    @. dρ = beta * dρ - alpha * hwdiv(ρ * cuvw)
    @. dρ -= alpha * vdivf2c.(Ic2f.(ρ .* uₕ))
    @. dρ -= alpha * vert_flux_wρ

    # 2) Advection of tracers equations:
    @. dρq1 = beta * dρq1 - alpha * hwdiv(ρq1 * cuvw) + alpha * ystar.ρq1
    @. dρq1 -= alpha * vdivf2c.(Ic2f.(ρq1 .* uₕ))
    @. dρq1 -= alpha * vert_flux_wρq1

    @. dρq2 = beta * dρq2 - alpha * hwdiv(ρq2 * cuvw) + alpha * ystar.ρq2
    @. dρq2 -= alpha * vdivf2c.(Ic2f.(ρq2 .* uₕ))
    @. dρq2 -= alpha * vert_flux_wρq2

    @. dρq3 = beta * dρq3 - alpha * hwdiv(ρq3 * cuvw) + alpha * ystar.ρq3
    @. dρq3 -= alpha * vdivf2c.(Ic2f.(ρq3 .* uₕ))
    @. dρq3 -= alpha * vert_flux_wρq3

    @. dρq4 = beta * dρq4 - alpha * hwdiv(ρq4 * cuvw) + alpha * ystar.ρq4
    @. dρq4 -= alpha * vdivf2c.(Ic2f.(ρq4 .* uₕ))
    @. dρq4 -= alpha * vert_flux_wρq4

    @. dρq5 = beta * dρq5 - alpha * hwdiv(ρq5 * cuvw) + alpha * ystar.ρq5
    @. dρq5 -= alpha * vdivf2c.(Ic2f.(ρq5 .* uₕ))
    @. dρq5 -= alpha * vert_flux_wρq5

    # Apply the limiters:
    # First read in the min_q/max_q at the current time step
    min_q1 = parameters.min_q1
    max_q1 = parameters.max_q1

    min_q2 = parameters.min_q2
    max_q2 = parameters.max_q2

    min_q3 = parameters.min_q3
    max_q3 = parameters.max_q3

    min_q4 = parameters.min_q4
    max_q4 = parameters.max_q4

    min_q5 = parameters.min_q5
    max_q5 = parameters.max_q5

    if lim_flag
        # Call quasimonotone limiter, to find optimal ρq_i (where ρq_i gets updated in place)
        Limiters.quasimonotone_limiter!(
            ρq1,
            ρ,
            min_q1,
            max_q1,
            rtol = limiter_tol,
        )

        Limiters.quasimonotone_limiter!(
            ρq2,
            ρ,
            min_q2,
            max_q2,
            rtol = limiter_tol,
        )

        Limiters.quasimonotone_limiter!(
            ρq3,
            ρ,
            min_q3,
            max_q3,
            rtol = limiter_tol,
        )

        Limiters.quasimonotone_limiter!(
            ρq4,
            ρ,
            min_q4,
            max_q4,
            rtol = limiter_tol,
        )

        Limiters.quasimonotone_limiter!(
            ρq5,
            ρ,
            min_q5,
            max_q5,
            rtol = limiter_tol,
        )
    end
    Spaces.weighted_dss!(dydt.ρ)
    Spaces.weighted_dss!(dydt.ρq1)
    Spaces.weighted_dss!(dydt.ρq2)
    Spaces.weighted_dss!(dydt.ρq3)
    Spaces.weighted_dss!(dydt.ρq4)
    Spaces.weighted_dss!(dydt.ρq5)
end

# Set up vectors and parameters needed for the RHS function
ystar = copy(y0)

parameters = (
    horzspace = horzspace,
    min_q1 = min_q1,
    max_q1 = max_q1,
    min_q2 = min_q2,
    max_q2 = max_q2,
    min_q3 = min_q3,
    max_q3 = max_q3,
    min_q4 = min_q4,
    max_q4 = max_q4,
    min_q5 = min_q5,
    max_q5 = max_q5,
    τ = τ,
    center_coords = center_coords,
    face_coords = face_coords,
)

# Set up the RHS function
rhs!(ystar, y0, parameters, 0.0, dt, 1)

# Solve the ODE
prob = ODEProblem(IncrementingODEFunction(rhs!), copy(y0), (0.0, T), parameters)
sol = solve(
    prob,
    SSPRK33ShuOsher(),
    dt = dt,
    saveat = 0.99 * dt,
    progress = true,
    adaptive = false,
    progress_message = (dt, u, pm, t) -> t,
)

q1_error =
    norm(
        sol.u[end].ρq1 ./ ρ_ref.(center_coords.z) .-
        y0.ρq1 ./ ρ_ref.(center_coords.z),
    ) / norm(y0.ρq1 ./ ρ_ref.(center_coords.z))
@test q1_error ≈ 0.0 atol = 0.7

q2_error =
    norm(
        sol.u[end].ρq2 ./ ρ_ref.(center_coords.z) .-
        y0.ρq2 ./ ρ_ref.(center_coords.z),
    ) / norm(y0.ρq2 ./ ρ_ref.(center_coords.z))
@test q2_error ≈ 0.0 atol = 0.03

q3_error =
    norm(
        sol.u[end].ρq3 ./ ρ_ref.(center_coords.z) .-
        y0.ρq3 ./ ρ_ref.(center_coords.z),
    ) / norm(y0.ρq3 ./ ρ_ref.(center_coords.z))
@test q3_error ≈ 0.0 atol = 0.4

q4_error =
    norm(
        sol.u[end].ρq4 ./ ρ_ref.(center_coords.z) .-
        y0.ρq4 ./ ρ_ref.(center_coords.z),
    ) / norm(y0.ρq4 ./ ρ_ref.(center_coords.z))
@test q4_error ≈ 0.0 atol = 0.03

Plots.png(
    Plots.plot(
        sol.u[trunc(Int, end / 2)].ρq3 ./ ρ_ref.(center_coords.z),
        level = 5,
        clim = (-1, 1),
    ),
    joinpath(path, "q3_6day.png"),
)

Plots.png(
    Plots.plot(
        sol.u[end].ρq3 ./ ρ_ref.(center_coords.z),
        level = 5,
        clim = (-1, 1),
    ),
    joinpath(path, "q3_12day.png"),
)
