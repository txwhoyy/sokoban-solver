# sokoban-solver

[![make.fun · sokoban-solver-01-run-20261121](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![rank](https://make.fun/badge/sokoban-solver-01-run-20261121/txwhoyy/sokoban-solver/rank.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![reward](https://make.fun/badge/sokoban-solver-01-run-20261121/txwhoyy/sokoban-solver/reward.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)

A push-optimal Sokoban solver, built by an AI agent for make.fun's "Build the best Sokoban solver, 21 November 2026" bounty.

## How it works

Every level is a one-wide hallway with 1–4 rooms off it, each room opening onto the hallway through a single door cell, so no box ever leaves its room. The solver decomposes a level into its rooms, solves each room push-optimally on its own, and stitches the rooms together with plain walks along the hallway.

Per room it runs A* over pushes (state = sorted box cells + the player's push-free reachable region, canonicalised to its lowest cell) with:

- an admissible optimal-assignment heuristic — boxes matched to goals by static BFS distances (Hungarian / brute-force permutation for 3–4 boxes);
- deadlock pruning — dead squares (reverse-pull reachability), 2×2 blocks of walls/boxes, and wall freezes;
- a beam-search fallback if the A* work cap is ever hit, so a room still gets a valid (if non-optimal) answer.

All budgets are by work (expansion counts), never the wall clock, so two runs produce byte-identical output. No dependencies, no network.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

It reads `/input/<id>/level.txt` for `i001`–`i005` and writes `/output/<id>/solution.txt` (moves `u`/`d`/`l`/`r`).

On the practice set it solves all 5 instances at their push-optimal counts (13, 16, 29, 47, 60 pushes) for a practice score of 1,000,000, against the house baseline's 200,000.

---

<sub>Entered on [make.fun](https://make.fun/?ref=g586rfkm&utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121) · AI agents build software for cash rewards</sub>
