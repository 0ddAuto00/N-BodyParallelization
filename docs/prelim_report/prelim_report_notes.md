# Document to hold notes for our preliminary report (Due Nov 19)

*This will be updated in-step with changelog. Updates here will be reflected there, and vice versa.*

## Checklist - All points are listed below

- [ ] An overview of the progress made on your project.
  - Complete when: Header 1 has a complete bullet list of all tasks completed
- [ ] Reference the milestones outlined in the project
  - Complete when: Header 1 bullets reference applicable milestones
- [ ] Specific challenges you have addressed in the project thus far
  - Complete when: Header 5 has full bullet list
- [ ] Remaining items that you need to address in your project
  - Complete when: Headers 2, 3, 4 have full bullet list

- [ ] Preliminary results (Visual plots)
  - Complete when: Preliminary result figures in `docs/prelim_report/`

- [ ] Updated Milestone list
  - Complete when: Headers 1-4 have full bullet lists

## 1. Tasks Completed - Include changelog version entry

- Config style setup and modularization
- Config Validation
- timing scripts

## 2. Tasks In progress (On-schedule)

- Rowwise implementation, ran into a race condition
- Implementation of tiled

## 3. Tasks In progress (Behind schedule)

- Debugging of rowwise

## 4. Changes to project timeline or scope

- Add grid search of parameters (`.dt`, `.softening`) in `nbody.cfg` file to make sure energy is reasonably conserved for all runs. These values might depend on the number of bodies? (@Fady)
- Add averaging of times rather than a single time

## 5. Specific challenges addressed in the project thus far

- Parallel rowwise implementation cannot compute just an upper matrix because values need to be attibuted to both bodies. This version is not as comparable to sequential because it computes all possible combinations, not just an upper matrix.
- Played with the idea of it being a stencil application but determined this would not be comparable directly to the tiled implementation and is outside proposal scope

## Guidance for visuals

- Speedup tests at number of bodies (`n`) = 256, 1024, 4096, up to 10^4

**If you are improving the performance of an algorithm or application, a result
may include:** a plot of response time vs. number of threads, a speedup plot, number of cache misses
before and after optimization, the number of nodes accessed in a data structure, etc.
