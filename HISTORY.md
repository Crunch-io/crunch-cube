# History of Changes

#### 1.8.1
- Major Cube refactor and a couple bugfixes

#### 1.8.0
- Major refactor of `Dimension` class

#### 1.7.3
- Implement pruning for index tables

#### 1.7.2
- Implement correct index table functionality
- Deprecate old index functionality

#### 1.7.1 Fix index error
- Fix peculiar case of CA x CAT (single elem) index error
- Support with unit tests
- Create base for fixing exporter issues

#### 1.7.0 Normalization, PEP8 and bugfix
- Refactored to remove a couple modules
- Fixed pesky numpy warnings
- Replaced vulnerable lazyproperty implementation

#### 1.6.11 Deprecate `shape`
- Deprecate the `CubeSlice` `shape` property
- Use `get_shape(prune=False)` instead
- Will be removed in future versions

#### 1.6.10 Fix README on pypi

#### 1.6.9 Bugfix
- When Categorical Array variable is selected in multitable export, and Scale Means is selected, the cube fails, because it tries to access the non-existing slice (the CA is only _interpreted_ as multiple slices in tabbooks). This fix makes sure that the export cube doesn't fail in such case.

#### 1.6.8 Scale Means Marginal
- Add capability to calculate the scale means marginal. This is used when analysing a 2D cube, and obtaining a sort of a "scale mean _total_" for each of the variables constituting a cube.

#### 1.6.7 Population fraction
- Various bugfixes and optimizations.
- Add property `population_fraction`. This is needed for the exporter to be able to calculate the correct population counts, based on weighted/unweighted and filtered/unfiltered states of the cube.
- Apply newly added `population_fraction` to the calculation of `population_counts`.
- Modify API for `scale_means`. It now accepts additional parameters `hs_dims` (defaults to `None`) and `prune` (defaults to `False`). Also, the format of the return value is slightly different in nature. It is a list of lists of numpy arrrays. It functions like this:

    - The outermost list corresponds to cube slices. If cube.ndim < 3, then it's a single-element list
    - Inner lists have either 1 or 2 elements (if they're a 1D cube slice, or a 2D cube slice, respectively).
    - If there are scale means defined on the corresponding dimension of the cube slice, then the inner list element is a numpy array with scale means. If it doesn't have scale means defined (numeric values), then the element is `None`.

- Add property `ca_dim_ind` to `CubeSlice`.
- Add property `is_double_mr` to `CubeSlice` (which is needed since it differs from the interpretation of the cube. E.g. MR x CA x MR will render slices which are *not* double MRs).
- Add `shape`, `ndim`, and `scale_means` to `CubeSlice`, for accessibility.
- `index` now also operates on slices (no api change).

#### 1.6.6 Added support for CubeSlice, which always represents a
- 2D cube (even if they're the slices of a 3D cube).
- Various fixes for support of wide-export

#### 1.6.5 Fixes for Pruning and Headers and subtotals.
- Population size support.
- Fx various calculations in 3d cubes.

#### 1.6.4 Fixes for 3d Pruning.

#### 1.6.2 support "Before" and "After" in variable transformations since they exist in zz9 data.

#### 1.6.1 `standardized_residuals` are now included.

#### 1.6.0 Z-Score and bug fixes.

#### 1.5.3 Fix bugs with invalid input data for H&S

#### 1.5.2 Fix bugs with `anchor: 0` for H&S

#### 1.5.1 Implement index for MR x MR

#### 1.5.0 Start implementing index table functionality

#### 1.4.5 Fix MR x MR proportions calculation

#### 1.4.4 Implement obtaining labels with category ids (useful for H&S in exporter)

#### 1.4.3 Fix bug (exporting 2D crtab with H&S on row only)

#### 1.4.2 Fix bugs discovered by first `cr.exporter` deploy to alpha

#### 1.4.1 Update based on deck tests from `cr.server`

#### 1.4 Update based on tabbook tests from `cr.lib`

#### 1.3 Implement Headers & Subtotals

#### 1.2 Support exporter

#### 1.1 Fix stray ipdb.

#### 1.0 Initial release
