# kodama

### Changes made over the original [crate](https://github.com/diffeo/kodama): 

- Update Rust edition from 2018 to 2021
- Merge [`ag/updates`](https://github.com/diffeo/kodama/tree/ag/updates) branch into master
- Heavier use of `#[inline]`
- Use `codegen = 1`, `lto = true` and `opt-level = 3` in `cargo.toml` for release version
- Use `panic = abort` (plays nicely with inlining and making more code fit in instructions cache)
- A few other small changes

This crate provides a fast implementation of agglomerative
[hierarchical clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering).

[![Linux build status](https://travis-ci.org/diffeo/kodama.svg?branch=master)](https://travis-ci.org/diffeo/kodama)
[![](https://img.shields.io/crates/v/kodama.svg)](https://crates.io/crates/kodama)

This library is released under the MIT license.

The ideas and implementation in this crate are heavily based on the work of
Daniel Müllner, and in particular, his 2011 paper,
[Modern hierarchical, agglomerative clustering algorithms](https://arxiv.org/pdf/1109.2378.pdf).
Parts of the implementation have also been inspired by his C++
library, [`fastcluster`](http://danifold.net/fastcluster.html).
Müllner's work, in turn, is based on the hierarchical clustering facilities
provided by MATLAB and
[SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html).

The runtime performance of this library is on par with Müllner's `fastcluster`
implementation.

For a more detailed example of how to use hierarchical clustering, see the
[example in the API documentation](https://docs.rs/kodama/0.1.0/kodama/#example).

### Documentation

https://docs.rs/kodama

### Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
kodama = { git = "https://github.com/SubconsciousCompute/kodama" }
```

and this to your crate root:

```rust
use kodama;
```

### C API and Go bindings

This repository includes
[`kodama-capi`](https://github.com/diffeo/kodama/tree/master/kodama-capi),
which provides a C interface to hierarchical clustering.

This repository also includes
[Go FFI bindings via cgo](https://github.com/diffeo/kodama/tree/master/go-kodama)
to the aforementioned C API. Documentation for the Go library can be found at
[godoc.org/github.com/diffeo/kodama/go-kodama](http://godoc.org/github.com/diffeo/kodama/go-kodama).
