---
title: |
  'votess: A multi-target, GPU-capable, parallel Voronoi tessellator'
tags:
  - C++
  - GPU
  - SYCL
  - parallel algorithms
  - Voronoi Tesellation
authors:
  - name: Samridh Dev Singh
    orcid: 0009-0008-8620-3648
    affiliation: 1
  - name: Chris Byrohl
    orcid: 0000-0002-0885-8090
    affiliation: 2
  - name: Dylan Nelson
    orcid: 0000-0001-8421-5890
    affiliation: 2
affiliations:
  - index: 1
    name: Grinnell College, 1115 8th Avenue, 50112 Grinnell, United States of America
  - index: 2
    name: Heidelberg University, Institute for Theoretical Astronomy, Albert-Ueberle-Str. 2, 69120 Heidelberg, Germany
date: 15 September 2024
bibliography: paper.bib
---

# Statement of need
 
A Voronoi tessellation is a spatial decomposition that partitions space into a
set of convex hulls based on proximity to a seed points with interesting to the
applications in biology, data science, geography, and physics. One compute
intensive application is its use in astrophysics, such as the analysis of
matter distribution [@weygaert], optimal transport theory for early-universe
reconstruction [@opticaltransport], and in observational data analysis and
numerical simulations of cosmic structure formation [@Springel2010].

The increasing size of datasets produced today have underscored the need for
more efficient algorithms to both generate and analyse these datasets, and the
rise of heterogenous computing facilities would enable such new algorithms to
be run. There do exist several sequential and parallel implementations of the
Voronoi diagram problem [@Marot]  [@WU2023102995]  [@cgal2018]  [@geogram2018],
however, they are mostly restricted to CPU or specific GPU architectures, thus
limiting their potential as a portable multi-architecture algorithm. 

# Summary

`votess` is a library, implementing Ray's meshless algorithm [@ray2018], for
computing parallel three dimensional Voronoi tessellations using the C++/SYCL
framework for CPU, GPUs and other future architectures.

One advantage of this algorithm is the ability for each cell to be computed
independently [@ray2018], making it suitable for parallel execution. It also
produces the geometry of the Voronoi cells via their neighbor connectivity
information, rather than a full combinatorial mesh data structure, thus making
it more ammenable to data parallel architectures than alternatives such as
sequential insertion or the Bowyer-Watson algorithm [@boyer1]  [@watson1].
 
The core method of `votess` consists of two main steps. First, the input set of
points is sorted into a grid, and a k-nearest neighbors search is performed.
Once the k nearest neighbors are identified for each point, the Voronoi cell is
computed by iteratively clipping a bounding box using the perpendicular
bisectors between the point and the identified neighbors. A *security radius*
condition [@securityradius] ensures that the resulting Voronoi cell is valid,
and if the cell cannot be validated, an CPU fallback mechanism is used.

## Performance

![](./loglog.png)

In Figure 1, we benchmark votess against two well-established libraries: CGAL,
which builds parallel CPU-based Delaunay meshes [@cgal2018], and Voro++, which
performs single-core, cell-based 3D tessellations [@rycroft2009voro]. All tests
use a float32 white-noise dataset, which provide a worst case scenario for
uniform datasets [@ray2018]. The use of clustured datasets lies outside of both
`votess` and the reference implementation [@ray2018], but remains in mind for
future iterations of `votess`.  Other Multithreaded Voronoi tesellelation codes
exist, including `ParVoro++` [@WU2023102995], and `GEOGRAM` [@geogram2018].
However, they do not natively support GPU architectures, but are acknowledged
as established alternatives.

From the graph above, `votess` outperforms `Voro++`, however, when compared
`CGAL`, both the CPU and GPU version falls short by around a factor of 6.
Ongoing optimizations for both backends are underway to close this gap.
 
# Features

`votess` is designed to be versatile. It supports various outputs, including
the natural neighbor information for each Voronoi cell. This is a 2D jagged
array of neighbor indices of the sorted input dataset.
 
Users can invoke `votess` in three ways: through the C++ library, a
command-line interface `clvotess`, and a Python wrapper interface `pyvotess`.
The C++ library offers a simple interface with a `tessellate` function that
computes the mesh. The Python wrapper, mirrors the functionality of the C++
version, with native numpy array support, providing ease of use for
Python-based workflows.
 
The behavior of `votess` can be fine-tuned with run time parameters in order to
(optionally) optimize runtime performance. 
 
# Acknowledgements
 
CB and DN acknowledge funding from the Deutsche Forschungsgemeinschaft (DFG)
through an Emmy Noether Research Group (grant number NE 2441/1-1).

# References

