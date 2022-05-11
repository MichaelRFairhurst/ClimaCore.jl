var documenterSearchIndex = {"docs":
[{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"CurrentModule = ClimateMachineCore","category":"page"},{"location":"api/#DataLayouts","page":"API","title":"DataLayouts","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"DataLayouts\nDataLayouts.IJFH","category":"page"},{"location":"api/#ClimateMachineCore.DataLayouts","page":"API","title":"ClimateMachineCore.DataLayouts","text":"ClimateMachineCore.DataLayouts\n\nNotation:\n\ni,j are horizontal node indices within an element\nk is the vertical node index within an element\nv is the vertical element index in a stack\nh is the element stack index\nf is the field index\n\nData layout is specified by the order in which they appear, e.g. IJKFVH indexes the underlying array as [i,j,k,f,v,h]\n\n\n\n\n\n","category":"module"},{"location":"api/#ClimateMachineCore.DataLayouts.IJFH","page":"API","title":"ClimateMachineCore.DataLayouts.IJFH","text":"IJFH{S,Nij}(ArrayType, nelements)\n\nConstruct an IJFH structure given the backing ArrayType, quadrature degrees of freedom Nij, and the number of mesh elements nelements.\n\n\n\n\n\n","category":"type"},{"location":"api/#Domains","page":"API","title":"Domains","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Domains.EquispacedRectangleDiscretization","category":"page"},{"location":"api/#ClimateMachineCore.Domains.EquispacedRectangleDiscretization","page":"API","title":"ClimateMachineCore.Domains.EquispacedRectangleDiscretization","text":"EquispacedRectangleDiscretization(domain::RectangleDomain, n1::Integer, n2::Integer)\n\nA regular discretization of domain with n1 elements in dimension 1, and n2 in dimension 2.\n\n\n\n\n\n","category":"type"},{"location":"api/#Topologies","page":"API","title":"Topologies","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Topologies","category":"page"},{"location":"api/#ClimateMachineCore.Topologies","page":"API","title":"ClimateMachineCore.Topologies","text":"ClimateMachineCore.Topologies\n\nObjects describing the horizontal connections between elements.\n\nAll elements are quadrilaterals, using the face and vertex numbering convention from p4est:\n\n          4\n      3-------4\n ^    |       |\n |  1 |       | 2\nx2    |       |\n      1-------2\n          3\n        x1-->\n\n\n\n\n\n","category":"module"},{"location":"api/#Meshes","page":"API","title":"Meshes","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Meshes\nMeshes.Quadratures.degrees_of_freedom\nMeshes.Quadratures.GL\nMeshes.Quadratures.Uniform\nMeshes.Quadratures.quadrature_points\nMeshes.Quadratures.GLL\nMeshes.Quadratures.polynomial_degree\nMeshes.Quadratures.QuadratureStyle","category":"page"},{"location":"api/#ClimateMachineCore.Meshes","page":"API","title":"ClimateMachineCore.Meshes","text":"Meshes\n\ndomain\ntopologyc\ncoordinates\nmetric terms (inverse partial derivatives)\nquadrature rules and weights\n\nReferences / notes\n\nceed\nQA\n\n\n\n\n\n","category":"module"},{"location":"api/#ClimateMachineCore.Meshes.Quadratures.degrees_of_freedom","page":"API","title":"ClimateMachineCore.Meshes.Quadratures.degrees_of_freedom","text":"degrees_of_freedom(QuadratureStyle) -> Int\n\nReturns the degreesoffreedom of the QuadratureStyle concrete type\n\n\n\n\n\n","category":"function"},{"location":"api/#ClimateMachineCore.Meshes.Quadratures.GL","page":"API","title":"ClimateMachineCore.Meshes.Quadratures.GL","text":"GL{Nq}()\n\nGauss-Legendre quadrature using Nq quadrature points.\n\n\n\n\n\n","category":"type"},{"location":"api/#ClimateMachineCore.Meshes.Quadratures.Uniform","page":"API","title":"ClimateMachineCore.Meshes.Quadratures.Uniform","text":"Uniform{Nq}()\n\nUniformly-spaced quadrature.\n\n\n\n\n\n","category":"type"},{"location":"api/#ClimateMachineCore.Meshes.Quadratures.quadrature_points","page":"API","title":"ClimateMachineCore.Meshes.Quadratures.quadrature_points","text":"points, weights = quadrature_points(::Type{FT}, quadrature_style)\n\nThe points and weights of the quadrature rule in floating point type FT.\n\n\n\n\n\n","category":"function"},{"location":"api/#ClimateMachineCore.Meshes.Quadratures.GLL","page":"API","title":"ClimateMachineCore.Meshes.Quadratures.GLL","text":"GLL{Nq}()\n\nGauss-Legendre-Lobatto quadrature using Nq quadrature points.\n\n\n\n\n\n","category":"type"},{"location":"api/#ClimateMachineCore.Meshes.Quadratures.polynomial_degree","page":"API","title":"ClimateMachineCore.Meshes.Quadratures.polynomial_degree","text":"polynomial_degree(QuadratureStyle) -> Int\n\nReturns the polynomial degree of the QuadratureStyle concrete type\n\n\n\n\n\n","category":"function"},{"location":"api/#ClimateMachineCore.Meshes.Quadratures.QuadratureStyle","page":"API","title":"ClimateMachineCore.Meshes.Quadratures.QuadratureStyle","text":"QuadratureStyle\n\nQuadrature style supertype. See sub-types:\n\nGLL\nGL\nUniform\n\n\n\n\n\n","category":"type"},{"location":"#ClimateMachineCore.jl","page":"Home","title":"ClimateMachineCore.jl","text":"","category":"section"}]
}
