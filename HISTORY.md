# History of Changes

### 2.1.23
- Valid weighted and unweighted counts per cell measures

### 2.1.22
- Fix t_stats sign for mean differences

### 2.1.21
- Fix overlaps pairwise significance

### 2.1.20
- Fix sig letters for mean differences

### 2.1.19
- Fix pairwise bug (with empty indices)

### 2.1.18
- Fix alt pairwise bug (with no alt alpha)

### 2.1.17
- Fix overlaps pairwise signs
- Add overlaps pairwise indices

### 2.1.16
- Fix overlaps pairwise significance for MR

### 2.1.15
- Pairwise t test for mean differences

### 2.1.14
- Improve pairwise t test performance

### 2.1.13
- Handle hiding transforms with subvar alias and id
- Additional share of sum measures
- Overlaps for MRxMR matrix

### 2.1.12
- Bug fixes for subtotal differences

### 2.1.11
- Bug fix for numeric array with weighted counts

### 2.1.10
- Add pairwise t test considering overlaps
- Add hare of sum measure

### 2.1.9
- Improvements to subtotal differences

### 2.1.8
- Add cube std deviation measure

### 2.1.7
- Add cube sum measure

### 2.1.6
- Enable explicit ordering by subvar IDs (strings)

### 2.1.5
- Bug fix for shape calculation on numeric arrays.

### 2.1.4
- Change `population_moe` -> `population_counts_moe` for `_Strand`

### 2.1.3
- Transpose dimension for numeric arrays

### 2.1.2
- Handle numeric array explicit order

### 2.1.1
- Custom column bases for Numeric Array matrix types

### 2.1.0
- Measure Consolidation

### 2.0.3
- Fix mean measure for CubeSet

### 2.0.2
- Expose `cube.valid_counts` and `cube.valid_counts_summary`

### 2.0.1
- Fix row standard error for MR x MR

### 2.0.0
- De-vectorize matrix.py and add sort-by-value
- Remove old api interface

### 1.12.11
- Numeric array measures available

### 1.12.10
- Selected category labels partition interface

### 1.12.9
- Margin of error for row %
- Margin of error for population
- Std deviation and std error for row %

### 1.12.8
- Fix pairwise t-test for scale means
- Fix UserWarning for smoothing measures
- Move cr.cube.enum -> cr.cube.enums

### 1.12.7
- Margin of error for 1D cubes
- Allow pairwise significance for CA_SUBVAR

### 1.12.6
- T-stats scale means for multiple response
- Margin of error for column percentages

### 1.12.4
- Measure expression evaluation method
- Multiple response allowed for pairwise comparison

### 1.12.3
- Bug fix for t_stats scale means

### 1.12.2
- Smoothing on scale means

### 1.12.1
- Smoothing on column percentages and column index

### 1.11.37
- PR 216: Document matrix.py classes and properties

### 1.11.36
- Hypotesis testing for subtotals (heading and insertions)

### 1.11.35
- Bug fix for hypothesis testing with overlaps

### 1.11.34
- Bug fix for augmented MRxMR matrices

#### 1.11.33
- Manage augmentation for MRxMR matrices

#### 1.11.32
- Handle hidden option for insertions

#### 1.11.31
- Use bases instead of margin for MR `standard_error` calculation

#### 1.11.30
- Fix `standard_error` calculation for MR types

#### 1.11.29
- Fix `standard_error` denominator for `Strand`

#### 1.11.28
- Fix collapsed `scale-mean-pairwise-indices`

#### 1.11.27
- Standard deviation and standard error for `Strand`

#### 1.11.26
- Fix `pairwise_indices()` array collapse when all values empty

#### 1.11.25
- Expose two-level pairwise-t-test

#### 1.11.24
- Bug fix for scale_median calculation

#### 1.11.23
- Expose population fraction in cube partitions

#### 1.11.22
- Additional summary measures for scale (`std_dev`, `std_error`, `median`)

#### 1.11.21
- Fix slicing for CA + single col filter

#### 1.11.20
- Fix cube title payload discrepancy

#### 1.11.19
- Fix problem where pre-ordering anchor-idx was used for locating inserted subtotal vectors
- Enable handling of filter-only multitable-template placeholders.
- New measures: table and columns standard deviation and standard error

#### 1.11.18
- Fix wrong proportions and base values when explicit order is expressed

#### 1.11.17
- Fix incorrect means values after hiding

#### 1.11.16
- New base for t_stats overlaps

#### 1.11.15
- Fix t_stats values for overlaps

#### 1.11.14
- Bug fix for margin property in MeanVector

#### 1.11.13
- Correct tstatistcs for multiple response
- New tstats measure in MRxMR Matrix
- New pairwise significance test for CATxMRxITSELF (5D) cubes
- New cube partition methods for residuals values

#### 1.11.12
- Bug fix on residuals for subtotals

#### 1.11.11
- Two-tailed t-tests for scale means
- Pairwise significance measure for scale means

#### 1.11.10
- zscore and pval measures for headers and subtotals

#### 1.11.9
- 100% test coverage
- New `is_empty` property in each of cube partition

#### 1.11.8
- Increase test coverage
- Fix 2D cubes that have means with insertions

#### 1.11.7
- Fix a bug when MR x MR table is pruned on both dimensions

#### 1.11.6
- Calculate population size fraction using complete cases

#### 1.11.5
- Fix pval calculation issues with defective matrices (rank < 2)
- Fix occasional overflowing issues, that arise from `np.empty` usage (use `np.zeros` instead)

#### 1.11.3
- Add `cr.cube.cube.CubeSet` and automatic dimension inflation for (0D, 1D, 1D, ...)
  cube sets.

#### 1.11.2
- Mostly renaming and support for numeric means in tabbooks

#### 1.11.1
- Fix `fill` for insertions

#### 1.11.0
- Significant refactor of the frozen cube code (even thought most of the logic is the same)

#### 1.10.6
- Fix index error by fixing the indexing array type to int (it used to default to float when the indexed array is empty)

#### 1.10.5
- Implement (frozen) `Cube` - responsible for (frozen) `_Slice` creation

#### 1.10.4
- Column index with insertions (as dashes)

#### 1.10.4
- Fix means on `_Slice` having subtotals.

#### 1.10.3
- Refactor hidden and pruned slices

#### 1.10.2
- Fix getting element ids from transforms shim
- Check for both int and str versions in incoming dictionaries
- This needs to be properly fixed in the shim code, but this code "just" provides extra safety

#### 1.10.1
- Add `fill` property to `_Element`, and provide fill information through `FrozenSlice` API.
- Increase test coverage (for various MR and Means cases)

#### 1.10.0
- Initial stab at `FrozenSlice`

#### 1.9.19
- Fix `None` anchor

#### 1.9.18
- Pairwise summary as T-Stats

#### 1.9.17
- Unweighted N as basis for t-stats

#### 1.9.16
- Proper t-stats for cubes with H&S

#### 1.9.15
- Implement pairwise indices for Wishart, directly in cube

#### 1.9.14
- Fix how Headings and Subtotals are treated in pairwise indices
- Row dimension is treated when calculating indices, while the column dimension
  is treated by inserting NaN placeholders

#### 1.9.13
- Parametrize pairwise comparisons based on column
- Add placeholders for insertions

#### 1.9.12
- Implement pairwise comparisons based on T-Stats

#### 1.9.11
- Eliminate `memoize` from `Dimension`, and thus reduce probability of threading bugs

#### 1.9.10
- Fix scale means for cubes with single-element categories

#### 1.9.9
- Enable other category-like types when comparing pairwise (datetime, text and binned numeric)

#### 1.9.8
- Enable pruning for min-base-size masks

#### 1.9.7
- Implement Min Base Size suppression masks

#### 1.9.8
- Enable pruning for min-base-size masks

#### 1.9.7
- Implement Min Base Size suppression masks

#### 1.9.6
- Make `margin` explicit in CubeSlice
- Fix calculation of `scale_means_margin` as a result

#### 1.9.5
- Fix calculating population counts for CAxCAT slices, that need to be treated as 0th cubes in tabbooks

#### 1.9.4
- Enable `CA_CAT` as a dimension type when calculating Pairwise Comparisons

#### 1.9.3
- Support H&S when calculating Pairwise Comparisons

#### 1.9.2
- Fix `scale_means` for Categorical Array (as a 0th slice in Tabbooks) where categorical doesn't have any numerical values

#### 1.9.1
- Fix `scale_means` for Categorical Array (as a 0th slice in Tabbooks)

#### 1.9.0
- Implement pairwise comparisons

#### 1.8.6
- Fix pruning for single element MRs

#### 1.8.5
- Fix `index_table` for MR x MR where either dimension has a only single element

#### 1.8.4
- Fix `index_table` for MR (single element) x CAT

#### 1.8.3
- fix second "broadcast error" bug (different cause)
- refactor to extract `_Measures` object and related
- other general factoring improvements in `cr.cube.crunch_cube`

#### 1.8.2
- fix "broadcast error" bug
- improve test coverage
- relocate test fixtures and add cached fixture lazy-loading

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
