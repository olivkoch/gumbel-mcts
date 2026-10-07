# Demo scripts

Each demo compares **Gumbel MCTS** (sequential halving) against **PUCT** (UCB-based tree search) on a different task. Together they show that Gumbel consistently finds better actions under the same simulation budget — especially when wrong moves are catastrophic and the prior is weak or uniform.

Generated assets (plots, videos, data) live in subdirectories by type: `png/`, `gif/`, `mp4/`, `json/`, `pdf/`, `pptx/`, `md/`.

---

## Board games

### Tic-Tac-Toe

| Script | What it does |
|--------|-------------|
| `demo_puct.py` | PUCT self-play on Tic-Tac-Toe |
| `demo_gumbel_dense.py` | Gumbel Dense self-play on Tic-Tac-Toe |
| `demo_gumbel_sparse.py` | Gumbel Sparse self-play on Tic-Tac-Toe |

Minimal sanity checks — both algorithms play correctly on a solved game.

### Connect 4

| Script | What it does |
|--------|-------------|
| `demo_connect4_train.py` | Self-play training: PUCT vs Gumbel, evaluated against a minimax opponent |

Shows that Gumbel-trained models learn faster and reach higher win rates.

### Gomoku (9x9 / 15x15)

| Script | What it does |
|--------|-------------|
| `demo_gomoku_train.py` | Self-play training comparison on 9x9 Gomoku |
| `demo_puct_vs_gumbel.py` | Head-to-head matches on 15x15 Gomoku across simulation budgets |
| `demo_visual_gomoku.py` | Visit-distribution heatmaps: where each algorithm spends its search |

Demonstrates that Gumbel produces sharper, more focused search distributions and wins head-to-head at every budget level.

---

## Continuous control

### PushT (block pushing)

| Script | What it does |
|--------|-------------|
| `pusht.py` | PUCT vs Gumbel as high-level planners on top of a diffusion policy prior |
| `pusht_logic.py` | PushT wrapped as a `GameLogic` for the MCTS library |
| `pusht_lib.py` | Same experiment using the library's tree search (not a flat bandit) |
| `pusht_sweep.py` | Budget sweep across simulation counts |
| `pusht_lib_sweep.py` | Budget sweep using the library implementation |
| `plot_pusht_results.py` | Plot sweep results as success rate vs budget |
| `plot_pusht_table.py` | Generate a table PNG of sweep results |
| `generate_pusht_video.py` | Side-by-side video at sims=64 |
| `generate_final_videos.py` | Batch video generation for PushT and wall-gap tasks |

**What it proves:** With a learned diffusion-policy prior, Gumbel uses rollout values to escape prior mistakes, while PUCT (with Q=0) stays trapped near the prior peak.

### Wall-gap PushT

| Script | What it does |
|--------|-------------|
| `pusht_wall.py` | Push the T-block through a gap in a wall |
| `wall_gap_logic.py` | Wall-gap variant wrapped as a `GameLogic` |
| `wall_gap_lib.py` | Library-based wall-gap runs |
| `wall_gap_sweep.py` | Budget x gap-size sweep |
| `generate_wallgap_video.py` | Side-by-side video (seed search at sims=8) |
| `generate_wallgap_video_64.py` | Video at sims=64 |

**What it proves:** When wrong root actions crash the block into the wall (catastrophic and irreversible), Gumbel's sequential halving eliminates them in the first phase. PUCT keeps visiting them proportionally to the prior.

### Robot arm (4 joints, 2 obstacles)

| Script | What it does |
|--------|-------------|
| `robot_arm_logic.py` | 4-joint planar arm reaching a target around obstacles |
| `robot_arm_sweep.py` | Budget sweep (outputs `json/robot_arm_sweep_results.json`) |
| `plot_robot_arm_results.py` | Plot success rate vs budget |
| `generate_robot_arm_video.py` | Side-by-side video at sims=64 |

**What it proves:** With a uniform prior (no learned policy), Gumbel reaches ~99% success at budget 32+ while PUCT stays near 0% until 256 sims. The value signal alone is enough for Gumbel to navigate around obstacles; PUCT spreads visits too thin across 9 actions to discover the correct joint sequence.

### Car parking (Reeds-Shepp)

| Script | What it does |
|--------|-------------|
| `car_parking.py` | Reeds-Shepp car parking on a 16x16 grid with 16 headings |

**What it proves:** Tight-budget planning in a large action space (forward/reverse x steering) — Gumbel parks reliably while PUCT often gets stuck.

---

## Puzzle

### Sokoban

| Script | What it does |
|--------|-------------|
| `sokoban.py` | 2-box Sokoban puzzle: sweep + animation |
| `sokoban_train.py` | End-to-end self-play training comparison |
| `sokoban_sim_sweep.py` | Simulation-budget sweep with training curves |

**What it proves:** Wrong pushes create immediate corner deadlocks (value = -1). Gumbel detects these in the first halving phase and never revisits them; PUCT keeps allocating visits proportionally to the prior.

---

## Utilities

| Script | What it does |
|--------|-------------|
| `tictactoe.py` | Tic-Tac-Toe game logic and tiny model |
| `check_guardrail.py` | Verifies budget allocation guardrail for sequential halving |
| `diag_halving.py` | Diagnostic: probes halving quality at sims=32 vs sims=64 |

---

## Running

All scripts run from the repo root:

```bash
uv run python demo/<script>.py
```

Most accept `--help` for options (seeds, budgets, output paths).
