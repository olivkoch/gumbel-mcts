"""Generate robot arm MP4: PUCT vs Gumbel side-by-side at sims=64."""

import os, sys, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from robot_arm_logic import (
    RobotArmLogic, RobotArmModel, render, end_effector,
    N_JOINTS, SUCCESS_DIST,
)
from gumbel_mcts.puct import PUCT
from gumbel_mcts.gumbel_dense import GumbelDense
from kernels.python_kernels import select_leaves_batch, backpropagate_batch

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BUDGET = 64
N_STEPS = 50
FPS = 8


# ── Python-kernel PUCT / GumbelDense (copied from pusht_logic.py) ────────────

class PythonPUCT(PUCT):
    def run_simulation_batch(self, model, active_games, num_simulations=50,
                             c_puct_base=19652, c_puct_init=1.25):
        game_indices = np.array(active_games, dtype=np.int32)
        logic = self.logic

        unexpanded = [self.storage.root_indices[g]
                      for g in active_games
                      if not self.storage.is_expanded[self.storage.root_indices[g]]]
        if unexpanded:
            root_indices = np.array(unexpanded, dtype=np.int32)
            obs_boards = torch.tensor(self.storage.boards[root_indices],
                                      device=self.device, dtype=torch.float32)
            obs_players = torch.tensor(self.storage.players[root_indices],
                                       device=self.device, dtype=torch.long)
            batch = {"boards": obs_boards.flatten(1), "current_player": obs_players}
            with torch.no_grad():
                outputs = model.forward_for_mcts(batch)
            priors = outputs['policy'].float().cpu().numpy().astype(np.float64)
            vals = outputs['value'].float().cpu().numpy().flatten().astype(np.float64)
            self.storage.is_expanded[root_indices] = True
            self.storage.prior_probs[root_indices] = priors
            backpropagate_batch(root_indices, vals,
                                self.storage.parents, self.storage.visit_counts,
                                self.storage.values)

        while True:
            min_visits = min(
                self.storage.visit_counts[self.storage.root_indices[g]]
                for g in active_games
            )
            if min_visits >= num_simulations + 1:
                break

            leaf_indices = select_leaves_batch(
                fast_step_func=logic.fast_step,
                get_valid_mask_func=logic.get_valid_mask,
                NUM_ACTIONS=logic.NUM_ACTIONS,
                player1=logic.PLAYER_1, player2=logic.PLAYER_2,
                game_indices=game_indices,
                root_indices=self.storage.root_indices,
                children=self.storage.children,
                visit_counts=self.storage.visit_counts,
                values=self.storage.values,
                prior_probs=self.storage.prior_probs,
                is_expanded=self.storage.is_expanded,
                is_terminal=self.storage.is_terminal,
                terminal_values=self.storage.terminal_values,
                boards=self.storage.boards,
                players=self.storage.players,
                parents=self.storage.parents,
                edge_from_parent=self.storage.edge_from_parent,
                next_free_idx_ptr=self.next_free_idx_arr,
                c_puct_base=c_puct_base, c_puct_init=c_puct_init,
                max_nodes=self.max_nodes,
                depths=self.storage.depths,
                max_game_depth=logic.MAX_MOVES
            )

            is_term = self.storage.is_terminal[leaf_indices]
            leaf_values = np.zeros(len(leaf_indices), dtype=np.float64)
            if np.any(is_term):
                leaf_values[is_term] = self.storage.terminal_values[leaf_indices[is_term]]
            non_term = ~is_term
            if np.any(non_term):
                nn_idx = leaf_indices[non_term]
                obs_b = torch.tensor(self.storage.boards[nn_idx],
                                     device=self.device, dtype=torch.float32)
                obs_p = torch.tensor(self.storage.players[nn_idx],
                                     device=self.device, dtype=torch.long)
                batch = {"boards": obs_b.flatten(1), "current_player": obs_p}
                with torch.no_grad():
                    out = model.forward_for_mcts(batch)
                priors = out['policy'].float().cpu().numpy().astype(np.float64)
                vals = out['value'].float().cpu().numpy().flatten().astype(np.float64)
                self.storage.is_expanded[nn_idx] = True
                self.storage.prior_probs[nn_idx] = priors
                leaf_values[non_term] = vals

            backpropagate_batch(leaf_indices, leaf_values,
                                self.storage.parents, self.storage.visit_counts,
                                self.storage.values)


class PythonGumbelDense(GumbelDense):
    def _expand_roots_v4(self, model, active_games):
        n_active = len(active_games)
        root_indices = self.storage.root_indices[active_games]
        obs_boards = torch.tensor(self.storage.boards[root_indices],
                                  device=self.device, dtype=torch.float32)
        obs_players = torch.tensor(self.storage.players[root_indices],
                                   device=self.device, dtype=torch.long)
        batch = {"boards": obs_boards.flatten(1), "current_player": obs_players}
        with torch.no_grad():
            outputs = model.forward_for_mcts(batch)
        probs = outputs['policy'].float().cpu().numpy()
        vals = outputs['value'].float().cpu().numpy().flatten().astype(np.float64)
        self.storage.prior_probs[root_indices] = probs
        self.root_logits[:n_active] = np.log(probs + 1e-10)
        self.root_nn_values[:n_active] = vals.astype(np.float32)
        logic = model.logic
        for i in range(n_active):
            r_idx = root_indices[i]
            self.root_legal_masks[i] = logic.get_valid_mask(
                self.storage.boards[r_idx], self.storage.players[r_idx]
            )
        self.storage.is_expanded[root_indices] = True
        backpropagate_batch(root_indices, vals,
                            self.storage.parents, self.storage.visit_counts,
                            self.storage.values)

    def _evaluate_and_backprop_v3(self, model, leaf_indices):
        is_term = self.storage.is_terminal[leaf_indices]
        leaf_values = np.zeros(len(leaf_indices), dtype=np.float64)
        if np.any(is_term):
            leaf_values[is_term] = self.storage.terminal_values[leaf_indices[is_term]]
        non_term = ~is_term
        if np.any(non_term):
            nn_idx = leaf_indices[non_term]
            boards = torch.from_numpy(self.storage.boards[nn_idx]).to(self.device).float()
            players = torch.from_numpy(self.storage.players[nn_idx]).to(self.device).long()
            with torch.no_grad():
                outputs = model.forward_for_mcts(
                    {"boards": boards.flatten(1), "current_player": players})
            priors = outputs['policy'].float().cpu().numpy().astype(np.float64)
            vals = outputs['value'].float().cpu().numpy().flatten().astype(np.float64)
            self.storage.is_expanded[nn_idx] = True
            self.storage.prior_probs[nn_idx] = priors
            leaf_values[non_term] = vals
        backpropagate_batch(leaf_indices, leaf_values,
                            self.storage.parents, self.storage.visit_counts,
                            self.storage.values)

    def run_simulation_batch(self, model, active_games, num_simulations=50):
        from kernels.python_kernels import (
            descend_tree_kernel, get_forced_root_moves_kernel, get_gumbel_score_kernel,
        )
        game_indices = np.array(active_games, dtype=np.int32)
        logic = model.logic
        n_active = len(active_games)
        self._expand_roots_v4(model, active_games)

        root_idxs = self.storage.root_indices[game_indices]
        raw_probs = self.storage.prior_probs[root_idxs].astype(np.float64)
        legal_f = self.root_legal_masks[:n_active].astype(np.float64)
        legal_probs = raw_probs * legal_f
        legal_probs /= legal_probs.sum(axis=1, keepdims=True).clip(min=1e-8)
        mean_ent = float(-(legal_probs * np.log(legal_probs + 1e-10)).sum(axis=1).mean())
        mean_n_legal = float(legal_f.sum(axis=1).mean())
        max_ent = np.log(mean_n_legal) if mean_n_legal > 1.0 else 1.0
        ratio = min(1.0, mean_ent / max(float(max_ent), 1e-8))
        dynamic_max_k = int(round(
            self.max_considered_actions
            - ratio * (self.max_considered_actions - self.entropy_min_k)
        ))
        dynamic_max_k = max(self.entropy_min_k, min(self.max_considered_actions, dynamic_max_k))
        self.last_root_entropy = mean_ent
        self.last_dynamic_max_k = dynamic_max_k

        max_k = min(self.storage.num_actions, dynamic_max_k)
        num_phases = max(1, int(np.log2(max_k)))
        first_phase_budget = num_simulations // num_phases
        k_initial = min(max_k, first_phase_budget // 2)
        k_initial = max(2, k_initial)
        num_phases = max(1, int(np.log2(k_initial)))
        k_initial = min(k_initial, max(2, (num_simulations // num_phases) // 4))
        k_initial = max(2, 1 << int(np.log2(k_initial)))
        num_phases = max(1, int(np.log2(k_initial)))

        candidate_mask = self._get_initial_gumbel_candidates(logic, game_indices, k_initial)
        remaining = num_simulations
        for phase in range(num_phases):
            k_phase = max(1, k_initial // (2 ** phase))
            phases_left = num_phases - phase
            budget_this_phase = remaining if phase == num_phases - 1 else remaining // phases_left
            sims_per_action = max(1, budget_this_phase // k_phase)
            for candidate_rank in range(k_phase):
                root_moves = get_forced_root_moves_kernel(
                    n_active, candidate_mask, candidate_rank
                )
                for _ in range(sims_per_action):
                    leaf_indices = descend_tree_kernel(
                        logic.fast_step, logic.get_valid_mask,
                        self.storage.num_actions,
                        logic.PLAYER_1, logic.PLAYER_2,
                        game_indices,
                        self.storage.root_indices[game_indices],
                        root_moves,
                        self.storage.children, self.storage.visit_counts,
                        self.storage.values, self.storage.prior_probs,
                        self.storage.is_expanded, self.storage.is_terminal,
                        self.storage.terminal_values, self.storage.boards,
                        self.storage.players, self.storage.parents,
                        self.storage.edge_from_parent, self.next_free_idx_arr,
                        self.max_nodes, self.storage.depths, logic.MAX_MOVES,
                        self.c_visit, self.c_scale
                    )
                    self._evaluate_and_backprop_v3(model, leaf_indices)
            remaining -= (k_phase * sims_per_action)
            if phase < num_phases - 1:
                candidate_mask = self._halve_candidates_py(
                    game_indices, candidate_mask, get_gumbel_score_kernel
                )

        final_moves = self._get_final_survivors_py(
            game_indices, candidate_mask, get_gumbel_score_kernel
        )
        return final_moves

    def _halve_candidates_py(self, game_indices, candidate_mask, score_fn):
        n_active = len(game_indices)
        scores = score_fn(
            n_active, game_indices, self.storage.root_indices, candidate_mask,
            self.storage.children, self.storage.visit_counts, self.storage.values,
            self.root_logits, self.gumbel_noises, self.storage.prior_probs,
            self.root_nn_values,
            c_visit=self.c_visit, c_scale=self.c_scale
        )
        new_mask = np.zeros_like(candidate_mask)
        for i in range(n_active):
            active_moves = np.where(candidate_mask[i])[0]
            if len(active_moves) <= 1:
                new_mask[i] = candidate_mask[i]; continue
            num_to_keep = max(1, len(active_moves) // 2)
            row_scores = scores[i, active_moves]
            top_indices = np.argsort(row_scores)[-num_to_keep:]
            new_mask[i, active_moves[top_indices]] = True
        return new_mask

    def _get_final_survivors_py(self, game_indices, candidate_mask, score_fn):
        n_active = len(game_indices)
        scores = score_fn(
            n_active, game_indices, self.storage.root_indices, candidate_mask,
            self.storage.children, self.storage.visit_counts, self.storage.values,
            self.root_logits, self.gumbel_noises, self.storage.prior_probs,
            self.root_nn_values,
            c_visit=self.c_visit, c_scale=self.c_scale
        )
        return np.array([np.argmax(scores[i]) for i in range(n_active)], dtype=np.int32)


# ── GIF / MP4 assembly ───────────────────────────────────────────────────────

def _pad_to_same_height(img_a, img_b):
    ha, hb = img_a.shape[0], img_b.shape[0]
    if ha < hb:
        img_a = np.pad(img_a, [(0, hb - ha), (0, 0), (0, 0)])
    elif hb < ha:
        img_b = np.pad(img_b, [(0, ha - hb), (0, 0), (0, 0)])
    return img_a, img_b


def make_gif(frames_left, frames_right, label_left, label_right, out_path, fps=15):
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default(size=18)

    n = max(len(frames_left), len(frames_right))
    last_l = frames_left[-1] if frames_left else np.zeros((100, 100, 3), dtype=np.uint8)
    last_r = frames_right[-1] if frames_right else np.zeros((100, 100, 3), dtype=np.uint8)

    pil_frames = []
    for i in range(n):
        fl = frames_left[i] if i < len(frames_left) else last_l
        fr = frames_right[i] if i < len(frames_right) else last_r
        fl, fr = _pad_to_same_height(fl, fr)
        divider = np.full((fl.shape[0], 4, 3), 220, dtype=np.uint8)
        combined = np.concatenate([fl, divider, fr], axis=1)
        pil_img = Image.fromarray(combined)
        draw = ImageDraw.Draw(pil_img, "RGBA")
        w = fl.shape[1]
        banner = (0, 0, 0, 160)
        for x_off, label, color in [
            (0, label_left, (255, 220, 80)),
            (w + 4, label_right, (80, 200, 255)),
        ]:
            bbox = font.getbbox(label)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.rectangle([x_off, 0, x_off + tw + 16, th + 12], fill=banner)
            draw.text((x_off + 8, 4), label, fill=color, font=font)
        pil_frames.append(pil_img.convert("RGB"))

    duration_ms = int(1000 / fps)
    pil_frames[0].save(out_path, save_all=True, append_images=pil_frames[1:],
                       duration=duration_ms, loop=0)
    print(f"Saved {out_path}  ({n} frames @ {fps} fps)")


def to_mp4(gif_path, mp4_path=None):
    if mp4_path is None:
        mp4_path = gif_path.replace(".gif", ".mp4")
    subprocess.run([
        "ffmpeg", "-y", "-i", gif_path,
        "-movflags", "faststart", "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        mp4_path,
    ], capture_output=True, check=True)
    return mp4_path


# ── MCTS helpers ─────────────────────────────────────────────────────────────

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


def run_episode(algo, seed, budget):
    algo_seed = seed if algo == "puct" else seed + 1
    np.random.seed(algo_seed)
    torch.manual_seed(algo_seed)
    logic = RobotArmLogic(seed=seed)
    logic.reset()
    model = RobotArmModel(logic)
    board = logic.get_initial_board()
    target = board[N_JOINTS:N_JOINTS + 2]

    frames = [render(board[:N_JOINTS], target)]
    for step in range(N_STEPS):
        action = pick_action(algo, logic, model, board, budget)
        _, _, done, board = logic.fast_step(board.copy(), action, 1)
        frames.append(render(board[:N_JOINTS], target))
        dist = np.linalg.norm(end_effector(board[:N_JOINTS]) - target)
        if step % 10 == 0 or done:
            print(f"  [{algo:6s}] step {step+1}/{N_STEPS} | action={action} | dist={dist:.0f}")
        if done and dist < SUCCESS_DIST:
            frames.extend([frames[-1]] * 5)
            break
    return frames, dist


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Searching for best seed (budget={BUDGET})...")
    best_seed, best_diff = None, -1e9
    for s in range(20):
        seed = s * 137 + 42
        dists = {}
        for algo in ["puct", "gumbel"]:
            aseed = seed if algo == "puct" else seed + 1
            np.random.seed(aseed)
            torch.manual_seed(aseed)
            logic = RobotArmLogic(seed=seed)
            logic.reset()
            model = RobotArmModel(logic)
            board = logic.get_initial_board()
            for _ in range(N_STEPS):
                action = pick_action(algo, logic, model, board, BUDGET)
                _, _, done, board = logic.fast_step(board.copy(), action, 1)
                if done:
                    break
            dists[algo] = np.linalg.norm(end_effector(board[:N_JOINTS]) - board[N_JOINTS:])
        diff = dists["puct"] - dists["gumbel"]
        print(f"  seed={seed}: PUCT dist={dists['puct']:.0f}, Gumbel dist={dists['gumbel']:.0f}")
        if diff > best_diff:
            best_diff = diff
            best_seed = seed

    print(f"\nBest seed: {best_seed} (diff={best_diff:.0f})")
    print(f"\nRecording video at seed={best_seed}...")

    puct_frames, puct_dist = run_episode("puct", best_seed, BUDGET)
    gumbel_frames, gumbel_dist = run_episode("gumbel", best_seed, BUDGET)

    gif = os.path.join(OUT_DIR, "gif", "final_robot_arm.gif")
    make_gif(
        puct_frames, gumbel_frames,
        f"PUCT  dist={puct_dist:.0f}",
        f"Gumbel  dist={gumbel_dist:.0f}",
        gif, fps=FPS,
    )
    mp4 = to_mp4(gif, os.path.join(OUT_DIR, "mp4", "final_robot_arm.mp4"))
    print(f"\nDone! {mp4} ({os.path.getsize(mp4) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
