# Data attribution and provenance

## Scope

The MIT licence in `LICENSE` covers the code in this repository only. The C-MAPSS
files in `data_raw/` are NASA data and are not covered by that licence; they are
attributed and described below.

## Dataset and citation

The data is the NASA C-MAPSS turbofan engine degradation simulation data set
(sub-sets FD001 to FD004). NASA's recommended citation:

> A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set",
> NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

Repository page:
<https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>

## Paper

The simulation is described in:

> A. Saxena, K. Goebel, D. Simon and N. Eklund, "Damage Propagation Modeling for
> Aircraft Engine Run-to-Failure Simulation", Proceedings of the 1st International
> Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.
> DOI: 10.1109/PHM.2008.4711414

## Terms

The C-MAPSS data in `data_raw/` is provided by the NASA Ames Prognostics Center of
Excellence. NASA's repository page states no licence; it asks users to acknowledge
the repository and the data donors, and it states that users employ the data at their
own risk. The files are redistributed here unmodified (SHA256 in `data_raw/SHA256SUMS`)
for reproducibility. Check the source page for current terms.

## Provenance

Retrieved on 2026-10-17 from two independent sources, which were compared file by file.

| Item | Value |
|------|-------|
| Official source (NASA) | `https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip` |
| Content-Length | 12429152 |
| ETag | `"a83e8f128c59fc5614a4ca2e42a2e30c"` |
| Last-Modified | Sun, 18 Sep 2022 03:42:08 GMT |
| Outer zip SHA256 | `c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2` |
| Inner `CMAPSSData.zip` SHA256 | `74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f` |
| Mirror used in the original project run | `https://raw.githubusercontent.com/edwardzjl/CMAPSSData/master/<name>` |

Result: all 12 files are byte-identical between the NASA download and the GitHub
mirror, and both match the expected SHA256 values. The committed files are the NASA
copies, taken from the inner `CMAPSSData.zip`. The per-file SHA256 values are in
`data_raw/SHA256SUMS` (`sha256sum -c SHA256SUMS` from inside `data_raw/` verifies
them). The `readme.txt` and `Damage Propagation Modeling.pdf` that ship in the inner
zip are not redistributed here.

## File format

The files use LF line endings and whitespace-separated columns, with a trailing space
before every newline. `.gitattributes` marks `data_raw/* -text` so that Git never
converts line endings on Windows checkouts (a CRLF conversion would change the
bytes and break the checksums). The files are never edited.

## Errata

NASA's `readme.txt` swaps the FD004 engine counts: it says 248 training and 249 test
engines. The data has 249 training and 248 test engines, which the tests assert. Do not
change the tests to match the readme.

## Counts

| Subset | Train engines | Train rows | Test engines | Test rows |
|--------|--------------:|-----------:|-------------:|----------:|
| FD001 | 100 | 20,631 | 100 | 13,096 |
| FD002 | 260 | 53,759 | 259 | 33,991 |
| FD003 | 100 | 24,720 | 100 | 16,596 |
| FD004 | 249 | 61,249 | 248 | 41,214 |
