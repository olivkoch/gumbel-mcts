"""Robot arm budget sweep: PUCT vs Gumbel. Outputs JSON results."""

import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from robot_arm_logic import RobotArmLogic, RobotArmModel, end_effector, N_JOINTS, SUCCESS_DIST
from generate_robot_arm_video import PythonPUCT, PythonGumbelDense

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def pick_action(algo, logic, model, board, budget):
    mx = max(budget * 8 + 100, 600)
    if algo == "puct":
        tree = PythonPUCT(n_games=1, max_nodes=mx, logic=logic, device="cpu")
        tree.initialize_roots([0], board[None], np.array([1]))
        tree.run_simulation_batch(model, [0], num_simulations=budget)
        visits, _ = tree.get_all_root_data(n_active=1)
        return int(np.argmax(visits[0]))
    else:
        tree = PythonGumbelDense(n_games=1, max_nodes=mx, logic=logic, device="cpu")
        tree.initialize_roots([0], board[None], np.array([1]))
        return int(tree.run_simulation_batch(model, [0], num_simulations=budget)[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budgets", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128, 256])
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--output", type=str, default=os.path.join(OUT_DIR, "json", "robot_arm_sweep_results.json"))
    args = p.parse_args()

    seeds = [int(s * 137 + 42) for s in range(args.seeds)]
    t0 = time.perf_counter()
    results = {}

    for budget in args.budgets:
        puct_ok = 0; gumbel_ok = 0
        puct_dists = []; gumbel_dists = []

        print(f"\n{'='*60}")
        print(f" Budget = {budget}")
        print(f"{'='*60}")

        for seed in seeds:
            for algo, aseed in [("puct", seed), ("gumbel", seed + 1)]:
                logic = RobotArmLogic(seed=seed); logic.reset()
                model = RobotArmModel(logic)
                board = logic.get_initial_board()
                np.random.seed(aseed); torch.manual_seed(aseed)
                done = False
                for step in range(args.n_steps):
                    action = pick_action(algo, logic, model, board, budget)
                    _, _, done, board = logic.fast_step(board.copy(), action, 1)
                    if done:
                        dist = np.linalg.norm(end_effector(board[:N_JOINTS]) - board[N_JOINTS:])
                        if dist < SUCCESS_DIST:
                            break
                        else:
                            done = False; break

                dist = np.linalg.norm(end_effector(board[:N_JOINTS]) - board[N_JOINTS:])
                success = dist < SUCCESS_DIST
                if algo == "puct":
                    puct_ok += success; puct_dists.append(float(dist))
                else:
                    gumbel_ok += success; gumbel_dists.append(float(dist))

            print(f"  seed={seed:5d}  puct={puct_dists[-1]:.0f}  gumbel={gumbel_dists[-1]:.0f}")

        n = len(seeds)
        print(f"\n  PUCT:   success={puct_ok/n:.1%}  mean_dist={np.mean(puct_dists):.0f}")
        print(f"  Gumbel: success={gumbel_ok/n:.1%}  mean_dist={np.mean(gumbel_dists):.0f}")

        results[str(budget)] = {
            "puct_success": puct_ok / n,
            "gumbel_success": gumbel_ok / n,
            "puct_dists": puct_dists,
            "gumbel_dists": gumbel_dists,
            "n_seeds": n,
        }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")
    print(f"Total time: {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
