# Changelog

All notable changes to this project are documented here.

Each entry is marked by a date-based version with one `## YYYY.M.DD (PR#NN)`
section per PR, newest first. Versions follow a date-based scheme, `YYYY.M.DD`,
bumped per PR to the date it lands. When two PRs land on the same day, the
second and later append a micro segment (`YYYY.M.DD.MICRO`, e.g. `2026.6.29.1`)
so each version stays unique.

If no PR is attatched, remove the `PR#NN` notation. Major changes should get at
least one other set of eyes before the PR lands.

## 2026.11.17

- Add `plot_results.py` that plots `results.csv` on two graphs showing runtime and speedup comparisons
- `nbody.c` now creates `results.csv`, then runs `plot_results.py` right after. All you have to do is \
 run `nbody.exe` and it does everything for you

## 2026.11.16

- Add `sequential.c/h`, `rowwise.c/h`, and `tiled.c/h` files to seperate implementations to their own files
- `nbody.c` now calls from the previously listed files to run and calculate time elapsed for implementations
- Changed `Makefile` to include all the new files and read/write to the correct folders
- Add `source\` directory containing .c files
  - Add `headers\` directory containing .h files
- Add `object\` directory containing .o files

## 2026.11.15

- Add `docs/` directory for building reports and sharing related notes.
  - Add, in `docs/` a markdown document to collect notes for building the preliminary report.
- Add `CHANGELOG.md` to track notable changes and versions to help write reports.

## 2026.11.10

- Initial commit:
  - Add sequential implementation `nbody.c` and associated `Makefile`.
  - Add general initial `README.md`
