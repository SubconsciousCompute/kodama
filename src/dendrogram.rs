use std::mem;
use std::ops;

use crate::float::Float;

/// A step-wise dendrogram that represents a hierarchical clustering as a
/// binary tree.
///
/// A dendrogram consists of a series of `N - 1` steps, where `N` is the number
/// of observations that were clustered. Each step corresponds to a merge
/// between two other clusters (where a cluster might consist of one or more
/// observations). Each step includes the labels for the pair of clusters that
/// were merged, the number of total observations in the new merged cluster
/// and the dissimilarity between the two merged clusters.
///
/// The labels of clusters are assigned as follows:
///
/// 1. A cluster that corresponds to a single observation is assigned a label
///    that corresponds to the given observation's index in the pairwise
///    dissimilarity matrix.
/// 2. A cluster with more than one observation has the label `N + i`, where
///    `N` is the total number of observations and `i` corresponds to the the
///    `i`th step in which the cluster was created. So for example, the very
///    first step in a dendrogram creates a cluster with the label `N` and the
///    last step in a dendrogram creates a cluster with the label
///    `(N + N - 1) - 1` (since there are always `N - 1` steps in a
///    dendrogram).
///
/// This labeling scheme corresponds to the same labeling scheme used by `SciPy`.
///
/// The type parameter `T` refers to the type of dissimilarity used in the
/// steps. In practice, `T` is a floating point type.
#[derive(Debug, Eq, Hash, PartialEq)]
pub struct Dendrogram<T> {
    steps: Vec<Step<T>>,
    observations: usize,
}

/// A single merge step in a dendrogram.
///
/// A step always corresponds to a merge between two clusters, where each
/// cluster has at least one observation. Each step itself corresponds to a new
/// cluster containing the observations of the merged clusters.
///
/// By convention, the smaller label is assigned to `cluster1`.
///
/// The type parameter `T` refers to the type of dissimilarity used. In
/// practice, `T` is a floating point type.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Step<T> {
    /// The label corresponding to the first cluster.
    ///
    /// The algorithm for labeling clusters is documented on
    /// [`Dendrogram`](struct.Dendrogram.html).
    pub cluster1: usize,
    /// The label corresponding to the second cluster.
    ///
    /// The algorithm for labeling clusters is documented on
    /// [`Dendrogram`](struct.Dendrogram.html).
    pub cluster2: usize,
    /// The dissimilarity between `cluster1` and `cluster2`.
    ///
    /// If both `cluster1` and `cluster2` correspond to singleton clusters,
    /// then this dissimilarity is equivalent to the pairwise dissimilarity
    /// between the clusters' corresponding observations. Otherwise, the
    /// dissimilarity is computed according to the clustering
    /// [`Method`](enum.Method.html) used.
    pub dissimilarity: T,
    /// The total number of observations in this merged cluster. This is
    /// always equivalent to the total number of observations in `cluster1`
    /// plus the total number of observations in `cluster2`.
    pub size: usize,
}

impl<T> Dendrogram<T> {
    /// Return a new dendrogram with space for the given number of
    /// observations.
    #[inline]
    pub fn new(observations: usize) -> Dendrogram<T> {
        Dendrogram { steps: Vec::with_capacity(observations), observations }
    }

    /// Clear this dendrogram and ensure there is space for the given number
    /// of observations.
    ///
    /// This method is useful for reusing a dendrogram's allocation.
    ///
    /// Note that this method does not need to be called before passing it to
    /// one of the clustering functions. The clustering functions will reset
    /// the dendrogram for you.
    #[inline]
    pub fn reset(&mut self, observations: usize) {
        self.steps.clear();
        self.observations = observations;
    }

    /// Push a new step on to this dendrogram.
    ///
    /// # Panics
    ///
    /// This method panics if the dendrogram has `N - 1` steps, where `N` is
    /// the number of observations supported by this dendrogram.
    #[inline]
    pub fn push(&mut self, step: Step<T>) {
        assert!(self.len() < self.observations().saturating_sub(1));
        self.steps.push(step);
    }

    /// Returns the steps in the dendrogram.
    #[inline]
    pub fn steps(&self) -> &[Step<T>] {
        &self.steps
    }

    /// Return a mutable slice of the steps in this dendrogram.
    #[inline]
    pub fn steps_mut(&mut self) -> &mut [Step<T>] {
        &mut self.steps
    }

    /// Return the number of steps in this dendrogram.
    #[inline]
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Return true if and only if this dendrogram has no steps.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Return the number of observations that this dendrogram supports.
    #[inline]
    pub fn observations(&self) -> usize {
        self.observations
    }

    /// Returns the total number of observations in the cluster identified by
    /// the following label.
    ///
    /// The label may be any value in the half-open interval
    /// `[0, N + N - 1)`, where `N` is the total number of observations.
    #[inline]
    pub fn cluster_size(&self, label: usize) -> usize {
        if label < self.observations() {
            1
        } else {
            self[label - self.observations()].size
        }
    }
}

impl<T: Float> Dendrogram<T> {
    /// Compare two dendrograms for approximate equality.
    ///
    /// Approximate equality in this case refers to the dissimilarities in each
    /// step. In particular, two dissimilarities are considered equal if and
    /// only if the absolute value of their difference is less than or equal to
    /// the given `epsilon` value.
    #[inline]
    pub fn eq_with_epsilon(&self, other: &Dendrogram<T>, epsilon: T) -> bool {
        if self.len() != other.len() {
            return false;
        }
        for (s1, s2) in self.steps().iter().zip(other.steps()) {
            if !s1.eq_with_epsilon(s2, epsilon) {
                return false;
            }
        }
        true
    }
}

impl<T> ops::Index<usize> for Dendrogram<T> {
    type Output = Step<T>;
    #[inline]
    fn index(&self, i: usize) -> &Step<T> {
        &self.steps[i]
    }
}

impl<T> ops::IndexMut<usize> for Dendrogram<T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut Step<T> {
        &mut self.steps[i]
    }
}

impl<T> Step<T> {
    /// Create a new a step that can be added to a dendrogram.
    ///
    /// Note that the clustering labels given are normalized such that the
    /// smallest label is always assigned to `cluster1`.
    #[inline]
    pub fn new(
        mut cluster1: usize,
        mut cluster2: usize,
        dissimilarity: T,
        size: usize,
    ) -> Step<T> {
        if cluster2 < cluster1 {
            mem::swap(&mut cluster1, &mut cluster2);
        }
        Step { cluster1, cluster2, dissimilarity, size }
    }

    /// Set the cluster labels on this step.
    ///
    /// Note that the clustering labels given are normalized such that the
    /// smallest label is always assigned to `cluster1`.
    #[inline]
    pub fn set_clusters(&mut self, mut cluster1: usize, mut cluster2: usize) {
        if cluster2 < cluster1 {
            mem::swap(&mut cluster1, &mut cluster2);
        }
        self.cluster1 = cluster1;
        self.cluster2 = cluster2;
    }
}

impl<T: Float> Step<T> {
    /// Compare two steps for approximate equality.
    ///
    /// Approximate equality in this case refers to the dissimilarity in each
    /// step. In particular, two dissimilarity are considered equal if and only
    /// if the absolute value of their difference is less than or equal to the
    /// given `epsilon` value.
    #[inline]
    pub fn eq_with_epsilon(&self, other: &Step<T>, epsilon: T) -> bool {
        if self == other {
            return true;
        }
        let key1 = (self.cluster1, self.cluster2, self.size);
        let key2 = (other.cluster1, other.cluster2, other.size);
        if key1 != key2 {
            return false;
        }
        if (self.dissimilarity - other.dissimilarity).abs() > epsilon {
            return false;
        }
        true
    }
}
