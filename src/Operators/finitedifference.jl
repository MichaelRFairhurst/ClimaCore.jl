# The geometry of the staggerd grid used by FiniteDifference in the vertical (sometimes called the C-grid)
# is (in one dimension) shown below

#         face   cell   face   cell   face
# left                                        right
#                 i-1            i
#                  ↓             ↓
#           |      ×      |      ×      |
#           ↑             ↑             ↑
#          i-1            i            i+1


#====

interpolate, multiply, gradient

  --->
ca  fa  fb  cb
    1 x F
  x       \
1           1    Boundary indices
  \       /
    2 - 2        --------------
  /       \
2           2    Interior indices
  \       /
    3 - 3
  /       \l
3           3

gradient with dirichlet

ca  fa
D - 1   modified stencil
  /
1
  \
    2   regular stencil
  /
2
  \
    3
  /
3


c2c laplacian, dirichlet

D - 1
  /   \
1       1
  \   /
    2   ---------------------
  /   \
2       2
  \   /
    3
  /   \
3

c2c laplacian, neumann
c   f   c
    N
      \
1       1
  \   /
    2  ----------------------
  /   \
2       2
  \   /
    n
  /   \
n       n
      /
   n+1
    =N


f2f laplacian, dirichlet
LaplacianF2F(left=Dirichlet(D))(x) = GradientC2F(left=Neumann(0))(GradientF2C(left=Dirichlet(D))(x))


dθ/dt = - ∇^2 θ
  --->
fa  ca  fb
D       _   effectively ignored: -dD/dt since is set by equations: use 2-point, 1-sided stencil?
  \
    1       boundary window
  /   \
2       2
  \   /
    2     ---------------
  /   \
3       3   interior
  \   /
    3

interior_indices1 = 2:...
interior_indices2 = 3:...


f2f laplacian, neumann
LaplacianF2F(left=Neumann(N))(x) = GradientC2F(left=Dirichlet(N))(GradientF2C(left=nothing)(x))

  --->
fa  ca  fb
1   N - 1   set by 1-sided stencil
  \   /
    1       boundary window
  /   \     ---------------
2       2
  \   /
    2
  /   \
3       3   interior
  \   /
    3

interior_indices1 = 1:n
interior_indices2 = 2:n-1
===#

struct PlusHalf{I <: Integer} <: Real
    i::I
end
PlusHalf{I}(h::PlusHalf{I}) where {I <: Integer} = h

const half = PlusHalf(0)

Base.:-(h::PlusHalf) = PlusHalf(-h.i - one(h.i))
Base.:+(i::Integer, h::PlusHalf) = PlusHalf(i + h.i)
Base.:+(h::PlusHalf, i::Integer) = PlusHalf(h.i + i)
Base.:+(h1::PlusHalf, h2::PlusHalf) = h1.i + h2.i + one(h1.i)
Base.:-(i::Integer, h::PlusHalf) = PlusHalf(i - h.i - one(h.i))
Base.:-(h::PlusHalf, i::Integer) = PlusHalf(h.i - i)
Base.:-(h1::PlusHalf, h2::PlusHalf) = h1.i - h2.i



Base.:<=(h1::PlusHalf, h2::PlusHalf) = h1.i <= h2.i
Base.:<(h1::PlusHalf, h2::PlusHalf) = h1.i < h2.i
Base.max(h1::PlusHalf, h2::PlusHalf) = PlusHalf(max(h1.i, h2.i))
Base.min(h1::PlusHalf, h2::PlusHalf) = PlusHalf(min(h1.i, h2.i))

left_idx(space::Spaces.CenterFiniteDifferenceSpace) = 1
right_idx(space::Spaces.CenterFiniteDifferenceSpace) = length(space.Δh_f2f)
left_idx(space::Spaces.FaceFiniteDifferenceSpace) = half
right_idx(space::Spaces.FaceFiniteDifferenceSpace) =
    PlusHalf(length(space.Δh_f2f))

left_face_boundary_idx(space::Spaces.FiniteDifferenceSpace) =
    left_idx(Spaces.FaceFiniteDifferenceSpace(space))
right_face_boundary_idx(space::Spaces.FiniteDifferenceSpace) =
    right_idx(Spaces.FaceFiniteDifferenceSpace(space))
left_center_boundary_idx(space::Spaces.FiniteDifferenceSpace) =
    left_idx(Spaces.CenterFiniteDifferenceSpace(space))
right_center_boundary_idx(space::Spaces.FiniteDifferenceSpace) =
    right_idx(Spaces.CenterFiniteDifferenceSpace(space))

left_face_boundary_idx(arg) = left_face_boundary_idx(axes(arg))
right_face_boundary_idx(arg) = right_face_boundary_idx(axes(arg))
left_center_boundary_idx(arg) = left_center_boundary_idx(axes(arg))
right_center_boundary_idx(arg) = right_center_boundary_idx(axes(arg))

Δh_f2f(space::Spaces.FiniteDifferenceSpace, idx::Integer) = space.Δh_f2f[idx]
Δh_c2c(space::Spaces.FiniteDifferenceSpace, idx::PlusHalf) =
    space.Δh_c2c[idx.i + 1]

Δh_left_bf2c(space::Spaces.FiniteDifferenceSpace) = space.Δh_c2c[1]
Δh_right_c2bf(space::Spaces.FiniteDifferenceSpace) = space.Δh_c2c[end]

abstract type BoundaryCondition end
"""
    SetValue(val)

Set the value at the boundary to be `val`. In the case of gradient operators,
this will set the input value from which the gradient is computed.
"""
struct SetValue{S} <: BoundaryCondition
    val::S
end

"""
    SetGradient(val)

Set the gradient at the boundary to be `val`. In the case of gradient operators
this will set the output value of the gradient.
"""
struct SetGradient{S} <: BoundaryCondition
    val::S
end

"""
    Extrapolate()

Set the value at the boundary to be the same as the closest interior point.
"""
struct Extrapolate <: BoundaryCondition end

abstract type Location end
abstract type Boundary <: Location end
abstract type BoundaryWindow <: Location end

struct Interior <: Location end
struct LeftBoundaryWindow{name} <: BoundaryWindow end
struct RightBoundaryWindow{name} <: BoundaryWindow end

abstract type FiniteDifferenceOperator end

return_eltype(::FiniteDifferenceOperator, arg) = eltype(arg)

boundary_width(op::FiniteDifferenceOperator, bc) =
    error("Boundary $(typeof(bc)) is not supported for operator $(typeof(op))")

get_boundary(
    op::FiniteDifferenceOperator,
    ::LeftBoundaryWindow{name},
) where {name} = getproperty(op.bcs, name)
get_boundary(
    op::FiniteDifferenceOperator,
    ::RightBoundaryWindow{name},
) where {name} = getproperty(op.bcs, name)

has_boundary(
    op::FiniteDifferenceOperator,
    ::LeftBoundaryWindow{name},
) where {name} = hasproperty(op.bcs, name)
has_boundary(
    op::FiniteDifferenceOperator,
    ::RightBoundaryWindow{name},
) where {name} = hasproperty(op.bcs, name)



abstract type AbstractStencilStyle <: Fields.AbstractFieldStyle end

# the .f field is an operator
struct StencilStyle <: AbstractStencilStyle end

# the .f field is a function, but one of the args (or their args) is a StencilStyle
struct CompositeStencilStyle <: AbstractStencilStyle end

abstract type InterpolationOperator <: FiniteDifferenceOperator end

"""
    InterpolateF2C()

Interpolate from face to center mesh. No boundary conditions are required (or supported).
"""
struct InterpolateF2C{BCS} <: InterpolationOperator
    bcs::BCS
end
InterpolateF2C(; kwargs...) = InterpolateF2C(NamedTuple(kwargs))

return_space(::InterpolateF2C, space::Spaces.FaceFiniteDifferenceSpace) =
    Spaces.CenterFiniteDifferenceSpace(space)

stencil_interior_width(::InterpolateF2C) = ((-half, half),)

function stencil_interior(::InterpolateF2C, loc, idx, arg)
    RecursiveApply.rdiv(
        (getidx(arg, loc, idx + half) ⊞ getidx(arg, loc, idx - half)),
        2,
    )
end

"""
    InterpolateC2F(;boundaries..)

Interpolate from center to face. Supported boundary conditions are:

- [`SetValue(val)`](@ref): set the value at the boundary face to be `val`.
- [`SetGradient`](@ref): set the value at the boundary such that the gradient is `val`.
- [`Extrapolate`](@ref): use the closest interior point as the boundary value
"""
struct InterpolateC2F{BCS} <: InterpolationOperator
    bcs::BCS
end
InterpolateC2F(; kwargs...) = InterpolateC2F(NamedTuple(kwargs))

return_space(::InterpolateC2F, space::Spaces.CenterFiniteDifferenceSpace) =
    Spaces.FaceFiniteDifferenceSpace(space)

stencil_interior_width(::InterpolateC2F) = ((-half, half),)

function stencil_interior(::InterpolateC2F, loc, idx, arg)
    space = axes(arg)
    #RecursiveApply.rdiv(((getidx(arg, loc, idx) ⊠ space.Δh_f2f[idx]) ⊞ (getidx(arg, loc, idx - 1) ⊠ space.Δh_f2f[idx-1])), 2 ⊠ space.Δh_c2c[idx])
    RecursiveApply.rdiv(
        getidx(arg, loc, idx + half) ⊞ getidx(arg, loc, idx - half),
        2,
    )
end

boundary_width(op::InterpolateC2F, ::SetValue) = 1
function stencil_left_boundary(::InterpolateC2F, bc::SetValue, loc, idx, arg)
    @assert idx == left_face_boundary_idx(arg)
    bc.val
end
function stencil_right_boundary(::InterpolateC2F, bc::SetValue, loc, idx, arg)
    space = axes(arg)
    @assert idx == right_face_boundary_idx(arg)
    bc.val
end

boundary_width(op::InterpolateC2F, ::SetGradient) = 1
function stencil_left_boundary(::InterpolateC2F, bc::SetGradient, loc, idx, arg)
    space = axes(arg)
    @assert idx == left_face_boundary_idx(arg)
    # Δh_c2c[1] is f1 to c1 distance
    getidx(arg, loc, idx + half) ⊟ (Δh_left_bf2c(space) ⊠ bc.val)
end
function stencil_right_boundary(
    ::InterpolateC2F,
    bc::SetGradient,
    loc,
    idx,
    arg,
)
    space = axes(arg)
    @assert idx == right_face_boundary_idx(arg)
    getidx(arg, loc, idx - half) ⊞ (Δh_right_c2bf(space) ⊠ bc.val)
end

boundary_width(op::InterpolateC2F, ::Extrapolate) = 1
function stencil_left_boundary(::InterpolateC2F, bc::Extrapolate, loc, idx, arg)
    space = axes(arg)
    @assert idx == left_face_boundary_idx(arg)
    getidx(arg, loc, idx + half)
end
function stencil_right_boundary(
    ::InterpolateC2F,
    bc::Extrapolate,
    loc,
    idx,
    arg,
)
    space = axes(arg)
    @assert idx == right_face_boundary_idx(arg)
    getidx(arg, loc, idx - half)
end

"""
    UpwindLeftC2F(;boundaries)

Interpolate from the left. Only the left boundary condition should be set:
- [`SetValue(val)`](@ref): set the value to be `val` on the boundary.
"""
struct UpwindLeftC2F{BCS} <: InterpolationOperator
    bcs::BCS
end
UpwindLeftC2F(; kwargs...) = UpwindLeftC2F(NamedTuple(kwargs))

return_space(::UpwindLeftC2F, space::Spaces.CenterFiniteDifferenceSpace) =
    Spaces.FaceFiniteDifferenceSpace(space)

stencil_interior_width(::UpwindLeftC2F) = ((-half, -half),)

stencil_interior(::UpwindLeftC2F, loc, idx, arg) = getidx(arg, loc, idx - half)

boundary_width(op::UpwindLeftC2F, ::SetValue) = 1

function stencil_left_boundary(::UpwindLeftC2F, bc::SetValue, loc, idx, arg)
    @assert idx == left_face_boundary_idx(arg)
    bc.val
end


abstract type BoundaryOperator <: FiniteDifferenceOperator end

"""
    SetBoundaryOperator(;boundaries...)

This operator only modifies the values at the boundary:
 - [`SetValue(val)`](@ref): set the value to be `val` on the boundary.
"""
struct SetBoundaryOperator{BCS} <: BoundaryOperator
    bcs::BCS
end
SetBoundaryOperator(; kwargs...) = SetBoundaryOperator(NamedTuple(kwargs))

return_space(::SetBoundaryOperator, space::Spaces.FaceFiniteDifferenceSpace) =
    space

stencil_interior_width(::SetBoundaryOperator) = ((0, 0),)

function stencil_interior(::SetBoundaryOperator, loc, idx, arg)
    getidx(arg, loc, idx)
end

boundary_width(op::SetBoundaryOperator, ::SetValue) = 1
function stencil_left_boundary(
    ::SetBoundaryOperator,
    bc::SetValue,
    loc,
    idx,
    arg,
)
    @assert idx == left_face_boundary_idx(arg)
    bc.val
end
function stencil_right_boundary(
    ::SetBoundaryOperator,
    bc::SetValue,
    loc,
    idx,
    arg,
)
    @assert idx == right_face_boundary_idx(arg)
    bc.val
end


abstract type GradientOperator <: FiniteDifferenceOperator end

"""
    GradientF2C(;boundaryname=boundarycondition...)

Centered-difference gradient operator from a `FaceFiniteDifferenceSpace` to a
`CenterFiniteDifferenceSpace`, applying the relevant boundary conditions. These
can be:
 - by default, the current value at the boundary face will be used.
 - [`SetValue(val)`](@ref): calculate the gradient assuming the value at the boundary is `val`.
 - [`Extrapolate()`](@ref): set the value at the center closest to the boundary
   to be the same as the neighbouring interior value.
"""
struct GradientF2C{BCS} <: GradientOperator
    bcs::BCS
end
GradientF2C(; kwargs...) = GradientF2C(NamedTuple(kwargs))

return_space(::GradientF2C, space::Spaces.FaceFiniteDifferenceSpace) =
    Spaces.CenterFiniteDifferenceSpace(space)

stencil_interior_width(::GradientF2C) = ((-half, half),)

function stencil_interior(::GradientF2C, loc, idx, arg)
    space = axes(arg)
    RecursiveApply.rdiv(
        (getidx(arg, loc, idx + half) ⊟ getidx(arg, loc, idx - half)),
        Δh_f2f(space, idx),
    )
end

boundary_width(op::GradientF2C, ::SetValue) = 1
function stencil_left_boundary(::GradientF2C, bc::SetValue, loc, idx, arg)
    space = axes(arg)
    @assert idx == left_center_boundary_idx(arg)
    RecursiveApply.rdiv(
        (getidx(arg, loc, idx + half) ⊟ bc.val),
        Δh_f2f(space, idx),
    )
end

function stencil_right_boundary(::GradientF2C, bc::SetValue, loc, idx, arg)
    space = axes(arg)
    # Δh_f2f = [f[2] - f[1], f[3] - f[2], ..., f[n] - f[n-1], f[n+1] - f[n]]
    @assert idx == right_center_boundary_idx(arg)
    RecursiveApply.rdiv(
        (bc.val ⊟ getidx(arg, loc, idx - half)),
        Δh_f2f(space, idx),
    )
end


boundary_width(op::GradientF2C, ::Extrapolate) = 1
function stencil_left_boundary(op::GradientF2C, ::Extrapolate, loc, idx, arg)
    space = axes(arg)
    @assert idx == left_center_boundary_idx(arg)
    stencil_interior(op, loc, idx + 1, arg)
end
function stencil_right_boundary(op::GradientF2C, ::Extrapolate, loc, idx, arg)
    space = axes(arg)
    @assert idx == right_center_boundary_idx(arg)
    stencil_interior(op, loc, idx - 1, arg)
end

"""
    GradientC2F(;boundaries)

Centered-difference gradient operator from a `CenterFiniteDifferenceSpace` to a
`FaceFiniteDifferenceSpace`, applying the relevant boundary conditions. These
can be:
 - [`SetValue(val)`](@ref): calculate the gradient assuming the value at the boundary is `val`.
 - [`SetGradient(val)`](@ref): set the value of the gradient at the boundary to be `val`.
"""
struct GradientC2F{BC} <: GradientOperator
    bcs::BC
end
GradientC2F(; kwargs...) = GradientC2F(NamedTuple(kwargs))

return_space(::GradientC2F, space::Spaces.CenterFiniteDifferenceSpace) =
    Spaces.FaceFiniteDifferenceSpace(space)

stencil_interior_width(::GradientC2F) = ((-half, half),)

function stencil_interior(::GradientC2F, loc, idx, arg)
    space = axes(arg)
    RecursiveApply.rdiv(
        (getidx(arg, loc, idx + half) ⊟ getidx(arg, loc, idx - half)),
        Δh_c2c(space, idx),
    )
end

boundary_width(op::GradientC2F, ::SetValue) = 1
boundary_width(op::GradientC2F, ::SetGradient) = 1

function stencil_left_boundary(::GradientC2F, bc::SetValue, loc, idx, arg)
    space = axes(arg)
    @assert idx == left_face_boundary_idx(arg)
    # Δh_c2c[1] is f1 to c1 distance
    RecursiveApply.rdiv(
        (getidx(arg, loc, idx + half) ⊟ bc.val),
        Δh_c2c(space, idx),
    )
end

function stencil_right_boundary(::GradientC2F, bc::SetValue, loc, idx, arg)
    space = axes(arg)
    # Δh_c2c = [c[1] - f[1], c[2] - c[1], ..., c[n] - c[n-1], f[n+1] - c[n]]
    @assert idx == right_face_boundary_idx(arg)
    # Δh_c2c[end] is c[n] to f[n+1] distance
    RecursiveApply.rdiv(
        (bc.val ⊟ getidx(arg, loc, idx - half)),
        Δh_c2c(space, idx),
    )
end

# left / right SetGradient boundary conditions
function stencil_left_boundary(::GradientC2F, bc::SetGradient, loc, idx, arg)
    @assert idx == left_face_boundary_idx(arg)
    # imposed flux boundary condition at left most face
    bc.val
end

function stencil_right_boundary(::GradientC2F, bc::SetGradient, loc, idx, arg)
    space = axes(arg)
    @assert idx == right_face_boundary_idx(arg)
    # imposed flux boundary condition at right most face
    bc.val
end


boundary_width(obj, loc) = 0
boundary_width(bc::Base.Broadcast.Broadcasted{StencilStyle}, loc) =
    has_boundary(bc.f, loc) ? boundary_width(bc.f, get_boundary(bc.f, loc)) : 0

function left_interor_window_idx(
    bc::Base.Broadcast.Broadcasted{StencilStyle},
    _,
    loc::LeftBoundaryWindow,
)
    space = axes(bc)
    arg_idxs = map(bc.args, stencil_interior_width(bc.f)) do arg, width
        left_interor_window_idx(arg, space, loc) - width[1]
    end
    return max(maximum(arg_idxs), left_idx(space) + boundary_width(bc, loc))
end
function right_interor_window_idx(
    bc::Base.Broadcast.Broadcasted{StencilStyle},
    _,
    loc::RightBoundaryWindow,
)
    space = axes(bc)
    arg_idxs = map(bc.args, stencil_interior_width(bc.f)) do arg, width
        right_interor_window_idx(arg, space, loc) - width[2]
    end
    return min(minimum(arg_idxs), right_idx(space) - boundary_width(bc, loc))
end




function left_interor_window_idx(
    bc::Base.Broadcast.Broadcasted{CompositeStencilStyle},
    _,
    loc::LeftBoundaryWindow,
)
    space = axes(bc)
    maximum(arg -> left_interor_window_idx(arg, space, loc), bc.args)
end
function right_interor_window_idx(
    bc::Base.Broadcast.Broadcasted{CompositeStencilStyle},
    _,
    loc::RightBoundaryWindow,
)
    space = axes(bc)
    minimum(arg -> right_interor_window_idx(arg, space, loc), bc.args)
end


function left_interor_window_idx(field::Field, _, loc::LeftBoundaryWindow)
    left_idx(axes(field))
end
function right_interor_window_idx(field::Field, _, loc::RightBoundaryWindow)
    right_idx(axes(field))
end

function left_interor_window_idx(
    bc::Base.Broadcast.Broadcasted{Style},
    _,
    loc::LeftBoundaryWindow,
) where {Style <: Fields.AbstractFieldStyle}
    left_idx(axes(bc))
end
function right_interor_window_idx(
    bc::Base.Broadcast.Broadcasted{Style},
    _,
    loc::RightBoundaryWindow,
) where {Style <: Fields.AbstractFieldStyle}
    right_idx(axes(bc))
end

function left_interor_window_idx(_, space, loc::LeftBoundaryWindow)
    left_idx(space)
end
function right_interor_window_idx(_, space, loc::RightBoundaryWindow)
    right_idx(space)
end

function getidx(
    bc::Base.Broadcast.Broadcasted{StencilStyle},
    loc::Interior,
    idx,
)
    stencil_interior(bc.f, loc, idx, bc.args...)
end

function getidx(
    bc::Base.Broadcast.Broadcasted{StencilStyle},
    loc::LeftBoundaryWindow,
    idx,
)
    space = axes(bc)
    op = bc.f
    if idx < left_idx(space) + boundary_width(bc, loc)
        stencil_left_boundary(op, get_boundary(op, loc), loc, idx, bc.args...)
    else
        # fallback to interior stencil
        stencil_interior(op, loc, idx, bc.args...)
    end
end

# faces: 1|2 3 4 ... 9 10|11
# center: 1|2 3 ....  9|10

function getidx(
    bc::Base.Broadcast.Broadcasted{StencilStyle},
    loc::RightBoundaryWindow,
    idx,
)
    space = axes(bc)
    op = bc.f
    n = length(axes(bc))
    if idx > right_idx(space) - boundary_width(bc, loc)
        stencil_right_boundary(op, get_boundary(op, loc), loc, idx, bc.args...)
    else
        # fallback to interior stencil
        stencil_interior(op, loc, idx, bc.args...)
    end
end

# broadcasting a StencilStyle gives a CompositeStencilStyle
# TODO: define precedence when multiple args are used
Base.Broadcast.BroadcastStyle(
    ::Type{<:Base.Broadcast.Broadcasted{StencilStyle}},
) = CompositeStencilStyle()

Base.Broadcast.BroadcastStyle(
    ::AbstractStencilStyle,
    ::Fields.AbstractFieldStyle,
) = CompositeStencilStyle()

Base.eltype(bc::Base.Broadcast.Broadcasted{StencilStyle}) =
    return_eltype(bc.f, bc.args...)

function getidx(
    bc::Fields.CenterFiniteDifferenceField,
    ::Location,
    idx::Integer,
)
    Fields.field_values(bc)[idx]
end
function getidx(bc::Fields.FaceFiniteDifferenceField, ::Location, idx::PlusHalf)
    Fields.field_values(bc)[idx.i + 1]
end
function setidx!(bc::Fields.CenterFiniteDifferenceField, idx::Integer, val)
    Fields.field_values(bc)[idx] = val
end
function setidx!(bc::Fields.FaceFiniteDifferenceField, idx::PlusHalf, val)
    Fields.field_values(bc)[idx.i + 1] = val
end



getidx(scalar, ::Location, idx) = scalar

function getidx(
    bc::Base.Broadcast.Broadcasted{FS},
    loc::Location,
    idx,
) where {FS <: Fields.AbstractFieldStyle}
    bc.f(map(arg -> getidx(arg, loc, idx), bc.args)...)
end

function Base.Broadcast.broadcasted(op::FiniteDifferenceOperator, args...)
    Base.Broadcast.broadcasted(StencilStyle(), op, args...)
end

function Base.Broadcast.broadcasted(
    ::StencilStyle,
    op::FiniteDifferenceOperator,
    args...,
)
    ax = return_space(op, map(axes, args)...)
    Base.Broadcast.Broadcasted{StencilStyle}(op, args, ax)
end

Base.Broadcast.instantiate(bc::Base.Broadcast.Broadcasted{StencilStyle}) = bc
Base.Broadcast._broadcast_getindex_eltype(
    bc::Base.Broadcast.Broadcasted{StencilStyle},
) = eltype(bc)

function Base.similar(
    bc::Base.Broadcast.Broadcasted{S},
    ::Type{Eltype},
) where {Eltype, S <: AbstractStencilStyle}
    sp = axes(bc)
    return Field(similar(Spaces.coordinates(sp), Eltype), sp)
end

function Base.copyto!(
    field_out::Field,
    bc::Base.Broadcast.Broadcasted{S},
) where {S <: AbstractStencilStyle}
    apply_stencil!(field_out, bc)
    return field_out
end

Spaces.interior_indices(field::Field) = Spaces.real_indices(axes(field))

Spaces.interior_indices(
    field::Base.Broadcast.Broadcasted{FS},
) where {FS <: Fields.AbstractFieldStyle} = Spaces.real_indices(axes(field))

function Spaces.interior_indices(bc::Base.Broadcast.Broadcasted{StencilStyle})
    width_op = stencil_interior_width(bc.f)   # tuple of 2-tuples
    indices_args = map(Spaces.interior_indices, bc.args) # tuple of indices
    @assert length(width_op) == length(indices_args)
    lo = minimum(map((o, a) -> first(a) - o[1], width_op, indices_args))
    hi = maximum(map((o, a) -> last(a) - o[2], width_op, indices_args))
    return lo:hi
end

function apply_stencil!(field_out, bc)
    space = axes(bc)
    n = length(space)
    lbw = LeftBoundaryWindow{Spaces.left_boundary_name(space)}()
    rbw = RightBoundaryWindow{Spaces.right_boundary_name(space)}()

    li = left_idx(space)
    lw = left_interor_window_idx(bc, space, lbw)
    ri = right_idx(space)
    rw = right_interor_window_idx(bc, space, rbw)
    @assert li <= lw <= rw <= ri
    for idx in li:(lw - 1)
        setidx!(field_out, idx, getidx(bc, lbw, idx))
    end
    for idx in lw:rw
        setidx!(field_out, idx, getidx(bc, Interior(), idx))
    end
    for idx in (rw + 1):ri
        setidx!(field_out, idx, getidx(bc, rbw, idx))
    end
    return field_out
end
