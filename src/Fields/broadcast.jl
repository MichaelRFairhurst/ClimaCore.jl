"""
    AbstractFieldStyle

The supertype of all broadcasting-like operations on Fields.
"""
abstract type AbstractFieldStyle <: Base.BroadcastStyle end

"""
    FieldStyle{DS <: DataStyle}

Standard broadcasting on Fields. Delegates the actual work to `DS`.
"""
struct FieldStyle{DS <: DataStyle} <: AbstractFieldStyle end

FieldStyle(::DS) where {DS <: DataStyle} = FieldStyle{DS}()

Base.Broadcast.BroadcastStyle(::Type{Field{V, S}}) where {V, S} =
    FieldStyle(DataStyle(V))

Base.Broadcast.BroadcastStyle(
    ::Base.Broadcast.AbstractArrayStyle{0},
    fs::AbstractFieldStyle,
) = fs

Base.Broadcast.BroadcastStyle(
    ::FieldStyle{DS1},
    ::FieldStyle{DS2},
) where {DS1, DS2} = FieldStyle(Base.Broadcast.BroadcastStyle(DS1(), DS2()))

Base.Broadcast.broadcastable(field::Field) = field

Base.eltype(bc::Base.Broadcast.Broadcasted{<:AbstractFieldStyle}) =
    Base.Broadcast.combine_eltypes(bc.f, bc.args)

# we implement our own to avoid the type-widening code, and throw a more useful error
@noinline error_inferred_eltype(bc) = error(
    "cannot infer concrete eltype of $(bc.f) on $(tuplemap(eltype, bc.args))",
)

function Base.copy(
    bc::Base.Broadcast.Broadcasted{Style},
) where {Style <: AbstractFieldStyle}
    ElType = eltype(bc)
    if !Base.isconcretetype(ElType)
        error_inferred_eltype(bc)
    end
    # We can trust it and defer to the simpler `copyto!`
    return copyto!(similar(bc, ElType), bc)
end

function slab(
    bc::Base.Broadcast.Broadcasted{Style},
    v,
    h,
) where {Style <: AbstractFieldStyle}
    _args = slab_args(bc.args, v, h)
    _axes = slab(axes(bc), v, h)
    Base.Broadcast.Broadcasted{Style}(bc.f, _args, _axes)
end

function column(
    bc::Base.Broadcast.Broadcasted{Style},
    i,
    j,
    h,
) where {Style <: AbstractFieldStyle}
    _args = column_args(bc.args, i, j, h)
    _axes = column(axes(bc), i, j, h)
    Base.Broadcast.Broadcasted{Style}(bc.f, _args, _axes)
end

# Return underlying DataLayout object, DataStyle of broadcasted
# for `Base.similar` of a Field
_todata_args(args::Tuple) = (todata(args[1]), _todata_args(Base.tail(args))...)
_todata_args(args::Tuple{Any}) = (todata(args[1]),)
_todata_args(::Tuple{}) = ()

todata(obj) = obj
todata(field::Field) = Fields.field_values(field)
function todata(bc::Base.Broadcast.Broadcasted{FieldStyle{DS}}) where {DS}
    _args = _todata_args(bc.args)
    Base.Broadcast.Broadcasted{DS}(bc.f, _args)
end

# same logic as Base.Broadcasted (which only defines it for Tuples)
Base.axes(bc::Base.Broadcast.Broadcasted{<:AbstractFieldStyle}) =
    _axes(bc, bc.axes)
_axes(bc, ::Nothing) = Base.Broadcast.combine_axes(bc.args...)
_axes(bc, axes) = axes

function Base.similar(
    bc::Base.Broadcast.Broadcasted{<:AbstractFieldStyle},
    ::Type{Eltype},
) where {Eltype}
    return Field(similar(todata(bc), Eltype), axes(bc))
end

function Base.copyto!(
    dest::Field,
    bc::Base.Broadcast.Broadcasted{<:AbstractFieldStyle},
)
    copyto!(field_values(dest), todata(bc))
    return dest
end

@noinline function error_mismatched_spaces(
    space1::Type{S},
    space2::Type{S},
) where {S <: AbstractSpace}
    error(
        "Broacasted spaces are the same ClimaCore.Spaces type but not the same instance",
    )
end

@noinline function error_mismatched_spaces(space1::Type, space2::Type)
    error("Broacasted spaces are not the same ClimaCore.Spaces type")
end

@inline function Base.Broadcast.broadcast_shape(
    space1::AbstractSpace,
    space2::AbstractSpace,
)
    if space1 !== space2
        error_mismatched_spaces(typeof(space1), typeof(space2))
    end
    return space1
end
@inline Base.Broadcast.broadcast_shape(space::AbstractSpace, ::Tuple{}) = space
@inline Base.Broadcast.broadcast_shape(::Tuple{}, space::AbstractSpace) = space

# Overload broadcast axes shape checking for more useful error message for Field Spaces
@inline function Base.Broadcast.check_broadcast_shape(
    space1::AbstractSpace,
    space2::AbstractSpace,
)
    if space1 !== space2
        error_mismatched_spaces(typeof(space1), typeof(space2))
    end
    return nothing
end
@inline function Base.Broadcast.check_broadcast_shape(
    space::AbstractSpace,
    ax2::Tuple,
)
    error_mismatched_spaces(typeof(space), typeof(ax2))
end
@inline function Base.Broadcast.check_broadcast_shape(
    ::AbstractSpace,
    ::Tuple{},
)
    return nothing
end

# Specialize handling of +, *, muladd, so that we can support broadcasting over NamedTuple element types
# Required for ODE solvers
Base.Broadcast.broadcasted(fs::AbstractFieldStyle, ::typeof(+), args...) =
    Base.Broadcast.broadcasted(fs, RecursiveApply.:⊞, args...)

Base.Broadcast.broadcasted(fs::AbstractFieldStyle, ::typeof(-), args...) =
    Base.Broadcast.broadcasted(fs, RecursiveApply.:⊟, args...)

Base.Broadcast.broadcasted(fs::AbstractFieldStyle, ::typeof(*), args...) =
    Base.Broadcast.broadcasted(fs, RecursiveApply.:⊠, args...)

Base.Broadcast.broadcasted(fs::AbstractFieldStyle, ::typeof(/), args...) =
    Base.Broadcast.broadcasted(fs, RecursiveApply.rdiv, args...)

Base.Broadcast.broadcasted(fs::AbstractFieldStyle, ::typeof(muladd), args...) =
    Base.Broadcast.broadcasted(fs, RecursiveApply.rmuladd, args...)

# Specialize handling of vector-based functions to automatically add LocalGeometry information
function Base.Broadcast.broadcasted(
    fs::AbstractFieldStyle,
    ::typeof(LinearAlgebra.norm),
    arg,
)
    space = Fields.axes(arg)
    # wrap in a Field so that the axes line up correctly (it just get's unwraped so effectively a no-op)
    Base.Broadcast.broadcasted(
        fs,
        Geometry._norm,
        arg,
        local_geometry_field(space),
    )
end
function Base.Broadcast.broadcasted(
    fs::AbstractFieldStyle,
    ::typeof(LinearAlgebra.norm_sqr),
    arg,
)
    space = Fields.axes(arg)
    # wrap in a Field so that the axes line up correctly (it just get's unwraped so effectively a no-op)
    Base.Broadcast.broadcasted(
        fs,
        Geometry._norm_sqr,
        arg,
        local_geometry_field(space),
    )
end

function Base.Broadcast.broadcasted(
    fs::AbstractFieldStyle,
    ::typeof(LinearAlgebra.cross),
    arg1,
    arg2,
)
    space = Fields.axes(arg1)
    # wrap in a Field so that the axes line up correctly (it just get's unwraped so effectively a no-op)
    Base.Broadcast.broadcasted(
        fs,
        Geometry._cross,
        arg1,
        arg2,
        local_geometry_field(space),
    )
end
function Base.Broadcast.broadcasted(
    fs::AbstractFieldStyle,
    ::typeof(Geometry.transform),
    arg1,
    arg2,
)
    space = Fields.axes(arg2)
    # wrap in a Field so that the axes line up correctly (it just get's unwraped so effectively a no-op)
    Base.Broadcast.broadcasted(
        fs,
        Geometry.transform,
        arg1,
        arg2,
        local_geometry_field(space),
    )
end


function Base.Broadcast.broadcasted(
    fs::AbstractFieldStyle,
    ::Type{V},
    arg,
) where {V <: Geometry.AxisVector}
    space = Fields.axes(arg)
    # wrap in a Field so that the axes line up correctly (it just get's unwraped so effectively a no-op)
    Base.Broadcast.broadcasted(fs, V, arg, local_geometry_field(space))
end

function Base.Broadcast.broadcasted(
    fs::AbstractFieldStyle,
    ::Type{V},
    arg,
) where {V <: Geometry.CartesianVector}
    space = Fields.axes(arg)
    # wrap in a Field so that the axes line up correctly (it just get's unwraped so effectively a no-op)
    Base.Broadcast.broadcasted(
        fs,
        V,
        arg,
        Ref(space.global_geometry),
        local_geometry_field(space),
    )
end

function Base.Broadcast.copyto!(
    field::Field,
    bc::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}},
)
    copyto!(Fields.field_values(field), bc)
    return field
end

function Base.Broadcast.copyto!(field::Field, nt::NamedTuple)
    copyto!(
        field,
        Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}}(
            identity,
            (nt,),
            axes(field),
        ),
    )
end

Base.fill!(field::Fields.Field, val) = field .= val
