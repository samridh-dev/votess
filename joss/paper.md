---
title: 'votess: TODO'
tags:
  - C++
authors:
  - name: Samridh Dev Singh
    orcid: TODO
    affiliation: 1
  - name: Chris Byrohl
    orcid: 0000-0002-0885-8090
    affiliation: 2
  - name: Dylan Nelson
    orcid: 0000-0001-8421-5890
    affiliation: 2
affiliations:
 - name: TODO
 - index: 1
 - name: Heidelberg University, Institute for Theoretical Astronomy, Albert-Ueberle-Str. 2, 69120 Heideberg, Germany 
 - index: 2 
 - date: 15 Sep 2024 
 - bibliography: paper.bib
---


# Statement of need

With the rise of parallel architectures, it has now become possible to solve problems that would have been computationally too expensive in the past. One example would be a Voronoi tesellation on large datasets. Many projects utilize such an algorithm, be it from Cosmology, Earth Science, Material Science, Biochemistry, etc.

There isn't however, a tesellation program that can run on the variety of platforms without severe modification to the source code. Thus forcing each problem requiring a voronoi tesellation to have a bespoke solution.

# Summary

`votess` is a library for performing 3D Voronoi tessellations on heterogeneous platforms via SYCL framework. `votess` was designed to be portable, but performant, with an easy to use interface.

The underlying algorithm is based on the work of Ray et al. (2018), which describes how to compute a Voronoi diagram without the need for a combinatorial mesh data structure, as required by classical approaches like the Bowyer-Watson algorithm. The core algorithm employed by `votess` consists of two main steps. First, given an input set of points, a k-nearest neighbors search is performed after sorting the points into a grid. With the nearest neighbors identified for each point, the Voronoi cell is computed by iteratively clipping a bounding box using the perpendicular bisectors of the point and its neighbors. To avoid iterating through all neighbors, a security radius condition is applied. If a Voronoi cell cannot be validated, a CPU fallback mechanism ensures robustness.

This simple, efficient algorithm allows for independent thread execution, making it well-suited for GPU parallelism.

## Performance
    
With a working implementation of `votess`, it can be seen that it outperforms several single-threaded applications:

![](./bar.png)

It can be seen that for both devices, `votess` has a speedup of around a factor of 10, not counting for optimizations that had been performed during the benchmark run. 

# Features

`votess` provides a simple C++ interface to compute tesellations, that being a single function `tesellate`. There also exists an interface to select ipnuts
One can select the target device to run the tesellation on, and currently CPU and GPU devices are supported. 

A class`vtargs` is provided, with usage following closely to the `std::unordered_map` STL class. It enables users to tune parameters, which could help with the runtime performance of the application, if that is ever necessary.

The tesellate function returns a templated class `dnn`, as a 2 dimensional jagged array of neighbhors that contribute to the voronoi cell of each particle in the sorted dataset. How the dataset is sorted can be tuned by class `vtargs`.

There also exists a python wrapper to `tesellate`, named `pyvotess`, with the same usage as the C++ implementation.

# Acknowledgements

CB and DN acknowledge funding from the Deutsche Forschungsgemeinschaft (DFG) through an Emmy Noether Research Group (grant number NE 2441/1-1).

# References

Nicolas Ray, Dmitry Sokolov, Sylvain Lefebvre, Bruno Lévy. Meshless Voronoi on the GPU. ACM Transactions on Graphics, 2018, 37 (6), pp.1-12. 10.1145/3272127.3275092. hal-01927559
