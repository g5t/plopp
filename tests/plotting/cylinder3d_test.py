import plopp as pp
import scipp as sc
import numpy as np


def triangulate(*, at: sc.Variable, to: sc.Variable, edge: sc.Variable,
                elements: int = 6, rings: int = 1, unit: str | None = None, caps: bool = True,
                twist: bool = True):
    from scipp import sqrt, dot, arange, concat, flatten, fold, array
    from scipp.spatial import rotations_from_rotvecs
    import uuid
    if unit is None:
        unit = at.unit or 'm'

    l_vec = to.to(unit=unit) - at.to(unit=unit)
    ll = sqrt(dot(l_vec, l_vec))
    # *a* vector perpendicular to l (should we check that this _is_ perpendicular to l_vec?)
    p = edge.to(unit=unit) - at.to(unit=unit)

    # arange does _not_ include the stop value, by design
    a = arange(start=0., stop=360., step=360/elements, dim='ring', unit='degree')
    temp_dim = uuid.uuid4().hex
    full_a = a * l_vec / ll
    ring = fold(rotations_from_rotvecs(flatten(full_a, dims=full_a.dims, to=temp_dim)), dim=temp_dim, sizes=full_a.sizes) * p
    li = at.to(unit=unit) + arange(start=0, stop=rings + 1, dim='length') * l_vec / rings
    if twist:
        twists = arange(start=0., stop=rings+1, step=1, dim='length', unit='degree') * (-180 / elements)
        full_t = twists * l_vec / ll
        r_twist = fold(rotations_from_rotvecs(flatten(full_t, dims=full_t.dims, to=temp_dim)), dim=temp_dim, sizes=full_t.sizes)
        ring = r_twist * ring

    vertices = flatten(li + ring, dims=['length', 'ring'], to='vertices')  # the order in the addition is important
    if caps:
        # 0, elements*[0,elements), elements*elements + 1
        vertices = concat((at.to(unit=unit), vertices, to.to(unit=unit)), 'vertices')
    faces = []
    if caps:
        # bottom cap
        faces = [[0, (i + 1) % elements + 1, i + 1] for i in range(elements)]
    # between rings
    for j in range(rings):
        z = 1 + j * elements if caps else j * elements
        rf = [[[z + i, z + (i + 1) % elements, z + (i + 1) % elements + elements],
               [z + i, z + (i + 1) % elements + elements, z + i + elements]] for i in range(elements)]
        faces.extend([triangle for triangles in rf for triangle in triangles])
    if caps:
        # top cap
        last = len(vertices) - 1
        top = [[last, last - (i + 1) % elements - 1, last - i - 1] for i in range(elements)]
        faces.extend(top)
    return vertices, array(values=np.array(faces), dims=['faces', 'triangle'])


# def triangulate_all(data, segments: int = 6, caps: bool = True, twist: bool = False):
#     """Combine all triangulations as scipp Variables, where possible"""
#     from scipp import array, arange, flatten, transpose
#     bases, edges, fars = (data.coords[name] for name in ('base', 'edge', 'far'))
#     x = bases.dims[0]
#     vertices, faces = triangulate(at=bases, to=fars, edge=edges, elements=segments, caps=caps, twist=twist)
#     # vertices now has shape (cylinders, vertices), which we will flatten to only vertices
#     # first we need to expand faces:
#     count = vertices.sizes['vertices']
#     first = arange(start=0, stop=bases.sizes[x], dim=x) * count
#     faces = first + array(values=faces, dims=['faces', 'vertices'])
#     return (flatten(transpose(vertices, dims=[x, 'vertices']), to='vertices'),
#             flatten(transpose(faces, dims=[x,  'faces', 'vertices']), dims=[x, 'faces'], to='faces'),
#             first + count, count)


def cylinders(npoints=100, scale=10.0, seed=1) -> sc.DataGroup:
    """
    Generate cylindrical scatter data, based on a normal distribution
    Parameters
    ----------
    npoints:
        The number of points to generate
    scale:
        Standard deviation of the distribution
    seed
        The seed for the random number

    Returns
    -------
        A DataGroup with intensity 'data', cylinder indexing, and NXcylindrical_geometry vertices
    """
    rng = np.random.default_rng(seed)
    base = sc.vectors(dims=['cylinder'], unit='m', values=scale * rng.standard_normal(size=[npoints, 3]))
    length = sc.vectors(dims=['cylinder'], unit='m', values=scale * rng.standard_normal(size=[npoints, 3]))
    radius = sc.array(dims=['cylinder'], unit='m', values=rng.standard_normal(size=[npoints]))
    v = sc.cross(sc.vector(value=[0, 1, 0]), length / sc.sqrt(sc.dot(length, length)))
    vertices = sc.concat((base, base + v * radius, base + length), dim='vertices').transpose().flatten(to='vertices')
    cyl = sc.arange(start=0, stop=3 * npoints, dim='flat').fold(dim='flat', sizes={'cylinder': npoints, 'index': 3})
    intensity = sc.array(dims=['cylinder'], unit='counts', values=rng.standard_normal(size=[npoints]))
    return sc.DataGroup(data=intensity, cylinders=cyl, vertices=vertices)


def cylinder(radius, length) -> sc.DataGroup:
    """
    Generate a single cylinder
    Parameters
    ----------
    radius:
        The radius of the generated cylinder
    length:
        The length of the generated cylinder

    Returns
    -------
        A DataGroup with intensity 'data', cylinder indexing, and NXcylindrical_geometry vertices of a single cylinder
    """
    base = sc.vectors(dims=['cylinder'], unit='m', values=[[0, 0, 0]])
    radius = sc.array(dims=['cylinder'], unit='m', values=[radius])
    length = sc.array(dims=['cylinder'], unit='m', values=[length])
    vertices = sc.concat((base,
                          base + radius * sc.vector(value=[0, 1, 0]),
                          base + length * sc.vector(value=[0, 0, 1])),
                         dim='vertices').transpose().flatten(to='vertices')
    cyl = sc.arange(start=0, stop=3, dim='flat').fold(dim='flat', sizes={'cylinder': 1, 'index': 3})
    intensity = sc.array(dims=['cylinder'], unit='counts', values=[1])
    return sc.DataGroup(data=intensity, cylinders=cyl, vertices=vertices)


def cylinders_to_mesh(data: sc.DataGroup) -> tuple[sc.DataArray, sc.Variable]:
    """Convert from NXCylindrical geometry to full-rank vertex information"""
    # nx_vertices contains any number of vertices, indexed by a (N_cylinders, 3) array, nx_cylinders
    # where max(nx_cylinders) < nx_vertices.shape[0]
    nx_vertices = data['vertices']    # (N_{unique}_vertices,) -- vector3 elements
    nx_cylinders = data['cylinders']  # (N_cylinders, 3)  -- integer elements

    assert data['data'].ndim == 1          # data _should_ have shape (N_cylinders,)
    data_dim = data['data'].dims[0]        # extract the coordinate name in case it's something else
    index_dim = data['cylinders'].dims[1]

    def extract(index):
        return nx_vertices[nx_cylinders[index_dim, index].values].flatten(to=data_dim)

    # the characteristic vertices for each cylinder:
    #  at == the center of one end-cap faces
    #  to == the center of the second end-cap face
    #  edge == 'any' point on the wall of the cylinder -- expected to be in the first end-cap plane
    at, edge, to = [extract(i) for i in range(3)]

    # convert from characteristic vertices to an N-polygon representation
    # and return the regular face indexes that apply to any cylinder's vertices
    vertices, faces = triangulate(at=at, to=to, edge=edge)

    # vertices _should_ be (N_cylinders, N_vertices(has_caps, segments))
    #       where N_vertices is a constant defined by the number of segments used to
    #       approximate a circle and whether the cylinder end caps are to be triangulated
    # faces is (N_faces(has_caps, segments), 3)
    #       where N_faces is a constant defined by the number of segments, etc.
    non_cylinder = [x for x in vertices.dims if x is not data_dim]
    assert len(non_cylinder) == 1
    n_vertices = vertices.sizes[non_cylinder[0]]

    # this feels _very_ weird, but makes sense in some way since each element of 'data'
    # applies to one cylinder, which has some number of vertices.
    mesh = sc.DataArray(data=vertices, coords={'intensity': data['data']})

    return mesh, faces


def test_single_cylinder3d():
    dg = cylinder(0.1, 2)
    mesh, faces = cylinders_to_mesh(dg)
    pp.mesh3d(mesh, faces)


def test_multiple_cylinder3d():
    dg = cylinders()
    mesh, faces = cylinders_to_mesh(dg)
    pp.mesh3d(mesh, faces)
