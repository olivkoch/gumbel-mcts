"""
demo/pusht.py — Comparing PUCT vs Gumbel as high-level planners on PushT.

Both planners sit on top of a trained diffusion policy (lerobot/diffusion_pusht_keypoints)
that provides a meaningful prior over the 8 macro-actions. Neither planner has access to a
learned value function (value=0), making this a fair test of planning without a critic.

PUCT (Q=0): UCB = C·P(a)·√N / (1 + n(a))
  With no value signal, visits concentrate around the prior peak and never improve beyond it.

Gumbel sequential halving:
  Initial ranking uses log P(a) + Gumbel noise (Gumbel-top-k trick).
  Phase halving uses actual rollout IoU, so the budget concentrates on high-reward actions
  even when the prior points the wrong way.

Usage:
    uv run python demo/pusht.py
    uv run python demo/pusht.py --seed 7 --n-macros 10 --budget 32
    uv run python demo/pusht.py --no-prior   # uniform prior (ablation)
"""

import argparse
import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import gymnasium as gym
import gym_pusht  # noqa: F401
from PIL import Image

# ── Macro-action constants ─────────────────────────────────────────────────────

NUM_ACTIONS   = 8
APPROACH_DIST = 80
PUSH_DEPTH    = 60
N_APPROACH    = 12
N_PUSH        = 18
N_SETTLE      = 12

WORKSPACE_LO, WORKSPACE_HI = 0.0, 512.0

# ── T-block geometry ───────────────────────────────────────────────────────────
# scale=30, length=4 ⟹
#   shape1 (horizontal bar): v0=(-60,30) v1=(60,30) v2=(60,0) v3=(-60,0)
#   shape2 (stem):           v4=(-15,30) v5=(-15,120) v6=(15,120) v7=(15,30)
# Outer faces we push on:
#   action 0 — bar top   : midpoint(kp[0], kp[1])
#   action 1 — bar right : midpoint(kp[1], kp[2])
#   action 2 — stem end  : midpoint(kp[5], kp[6])
#   action 3 — bar left  : midpoint(kp[3], kp[0])
# Corner pushes:
#   action 4 — kp[0]  (bar top-left)
#   action 5 — kp[1]  (bar top-right)
#   action 6 — kp[5]  (stem far-left)
#   action 7 — kp[6]  (stem far-right)


def _keypoints(block) -> np.ndarray:
    """World-space keypoints of the T-block, shape (8, 2)."""
    pts = []
    for shape in block.shapes:
        for v in shape.get_vertices():
            w = v.rotated(shape.body.angle) + shape.body.position
            pts.append(np.array(w, dtype=np.float64))
    return np.vstack(pts)


def _macro_targets(raw_env):
    """
    Returns (approach_pts, push_pts) each of shape (NUM_ACTIONS, 2).
    approach_pts[i] — EE goes here first (outside the surface)
    push_pts[i]     — EE then aims here (through the surface toward COG)
    """
    kp = _keypoints(raw_env.block)
    face_pts = np.array([
        (kp[0] + kp[1]) / 2,  # 0: bar top edge
        (kp[1] + kp[2]) / 2,  # 1: bar right edge
        (kp[5] + kp[6]) / 2,  # 2: stem far edge
        (kp[3] + kp[0]) / 2,  # 3: bar left edge
        kp[0],                  # 4: bar top-left corner
        kp[1],                  # 5: bar top-right corner
        kp[5],                  # 6: stem far-left corner
        kp[6],                  # 7: stem far-right corner
    ])
    cog = kp.mean(axis=0)
    d = face_pts - cog
    outward = d / np.linalg.norm(d, axis=1, keepdims=True).clip(min=1e-6)

    approach = np.clip(face_pts + APPROACH_DIST * outward, WORKSPACE_LO, WORKSPACE_HI)
    push_end  = np.clip(face_pts - PUSH_DEPTH   * outward, WORKSPACE_LO, WORKSPACE_HI)
    return approach, push_end


# ── Diffusion prior ────────────────────────────────────────────────────────────

def _load_policy(model_id: str = "lerobot/diffusion_pusht_keypoints"):
    """Load lerobot DiffusionPolicy from HuggingFace. Returns None if unavailable."""
    try:
        import json, tempfile
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

        print(f"Loading {model_id} ...")
        cfg_path = hf_hub_download(model_id, "config.json")
        weights_path = hf_hub_download(model_id, "model.safetensors")

        with open(cfg_path) as f:
            cfg = json.load(f)

        # The HF checkpoint uses the old lerobot config schema (pre-0.4).
        # Translate to the current draccus-based format if needed.
        if "type" not in cfg:
            sd = load_file(weights_path)
            norm_stats = {k: v for k, v in sd.items() if "normalize" in k}

            cfg["type"] = "diffusion"
            cfg["input_features"] = {
                k: {"type": "ENV" if s[0] > 2 else "STATE", "shape": s}
                for k, s in cfg.pop("input_shapes", {}).items()
            }
            cfg["output_features"] = {
                k: {"type": "ACTION", "shape": s}
                for k, s in cfg.pop("output_shapes", {}).items()
            }
            cfg["normalization_mapping"] = {
                ft["type"]: mode.upper()
                for modes_key in ("input_normalization_modes", "output_normalization_modes")
                for k, mode in cfg.pop(modes_key, {}).items()
                for ft in [cfg.get("input_features", {}).get(k) or
                           cfg.get("output_features", {}).get(k, {})]
            }

            tmpdir = tempfile.mkdtemp()
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(cfg, f)
            os.symlink(weights_path, os.path.join(tmpdir, "model.safetensors"))

            policy = DiffusionPolicy.from_pretrained(tmpdir)
            policy._norm_stats = norm_stats
        else:
            policy = DiffusionPolicy.from_pretrained(model_id)
            policy._norm_stats = None

        policy.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        policy = policy.to(device)
        print(f"  ready on {device}")
        return policy
    except Exception as e:
        print(f"[warn] Could not load diffusion policy ({e}). Using uniform prior.")
        return None


def _make_obs(raw_env) -> dict:
    """Snapshot the current observation from the raw gym env."""
    return {
        "environment_state": _keypoints(raw_env.block).flatten().astype(np.float32),
        "agent_pos": np.array(raw_env.agent.position, dtype=np.float32),
    }


def _compute_prior(policy, raw_env, obs_history: list,
                   temperature: float = 0.1) -> np.ndarray:
    """
    Run the diffusion policy on the last 2 obs to predict a continuous (x,y) target,
    then weight macro-actions by proximity of their push_end to that target.
    Returns a (NUM_ACTIONS,) probability vector.
    """
    uniform = np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS, dtype=np.float32)
    if policy is None or not obs_history:
        return uniform

    import torch

    # 2-frame sliding window; duplicate first frame if history is short
    history = obs_history[-2:] if len(obs_history) >= 2 else [obs_history[0]] * 2
    env_states = np.stack([h["environment_state"] for h in history])  # (2, 16)
    agent_poss  = np.stack([h["agent_pos"]          for h in history])  # (2, 2)

    dev = next(policy.parameters()).device
    env_t = torch.tensor(env_states, dtype=torch.float32).unsqueeze(0).to(dev)
    state_t = torch.tensor(agent_poss, dtype=torch.float32).unsqueeze(0).to(dev)

    # Old checkpoints need manual min-max normalization (buffers stripped on load).
    ns = getattr(policy, "_norm_stats", None)
    if ns:
        def _norm(x, key):
            mn = ns[f"normalize_inputs.buffer_{key.replace('.', '_')}.min"].to(dev)
            mx = ns[f"normalize_inputs.buffer_{key.replace('.', '_')}.max"].to(dev)
            return (x - mn) / (mx - mn + 1e-8) * 2 - 1
        env_t = _norm(env_t, "observation.environment_state")
        state_t = _norm(state_t, "observation.state")

    batch = {
        "observation.environment_state": env_t,
        "observation.state":             state_t,
    }

    try:
        policy.reset()
        with torch.no_grad():
            actions = policy.predict_action_chunk(batch)  # (1, horizon, 2)

        # Unnormalize actions for old checkpoints.
        if ns:
            a_min = ns["unnormalize_outputs.buffer_action.min"].to(dev)
            a_max = ns["unnormalize_outputs.buffer_action.max"].to(dev)
            actions = (actions + 1) / 2 * (a_max - a_min) + a_min

        target = actions[0, 0].cpu().numpy()  # (2,) first predicted target in [0, 512]
    except Exception as e:
        print(f"[warn] Policy inference failed: {e}")
        return uniform

    # Score each macro-action's push_end by distance to the predicted target
    _, push_end = _macro_targets(raw_env)
    dists = np.linalg.norm(push_end - target, axis=1)  # (8,)
    logits = -dists / (temperature * 512.0)
    logits -= logits.max()
    probs = np.exp(logits)
    return (probs / probs.sum()).astype(np.float32)


# ── PushTMacroEnv ─────────────────────────────────────────────────────────────

class PushTMacroEnv:
    """PushT environment wrapped at the macro-action level."""

    def __init__(self, seed: int = 42, record: bool = False):
        self.seed = seed
        self.record = record
        # gym-pusht registers with max_episode_steps=300. MCTS simulations
        # count toward that counter, so we strip the TimeLimit wrapper.
        _env = gym.make(
            "gym_pusht/PushT-v0",
            obs_type="state",
            render_mode="rgb_array" if record else None,
        )
        self.env = _env.env if isinstance(_env, gym.wrappers.TimeLimit) else _env
        self._rng = np.random.default_rng(seed)
        self.frames: list[np.ndarray] = []

    def reset(self):
        obs, info = self.env.reset(seed=self.seed)
        self.frames.clear()
        if self.record:
            self.frames.append(self.env.render())
        return obs

    def close(self):
        self.env.close()

    # ── State clone / restore ──────────────────────────────────────────────────

    def save_state(self) -> np.ndarray:
        return np.array(self.env.unwrapped.get_obs(), dtype=np.float64)

    def restore_state(self, state: np.ndarray):
        raw = self.env.unwrapped
        raw.agent.position = list(state[:2])
        raw.agent.velocity = (0.0, 0.0)
        # Angle must be set BEFORE position: with a non-zero COG offset (0, 45),
        # cpBodySetAngle updates the body transform in a way that shifts the position
        # reported by cpBodyGetPosition. Setting angle first keeps the subsequent
        # position write as the final word.
        raw.block.angle = float(state[4])
        raw.block.position = list(state[2:4])
        raw.block.velocity = (0.0, 0.0)
        raw.block.angular_velocity = 0.0

    # ── Execution primitives ───────────────────────────────────────────────────

    def _run_to(self, target: np.ndarray, n_steps: int, record: bool) -> tuple[float, bool]:
        coverage, done = 0.0, False
        for _ in range(n_steps):
            _, _, terminated, truncated, info = self.env.step(target.astype(np.float32))
            coverage = info.get("coverage", 0.0)
            done = terminated or truncated
            if record:
                self.frames.append(self.env.render())
            if done:
                break
        return coverage, done

    def _execute_macro(self, action_id: int, record: bool) -> tuple[float, bool]:
        raw = self.env.unwrapped
        approach, push_end = _macro_targets(raw)
        cov, done = 0.0, False
        for target, n in [
            (approach[action_id], N_APPROACH),
            (push_end[action_id], N_PUSH),
            (push_end[action_id], N_SETTLE),
        ]:
            cov, done = self._run_to(target, n, record=record)
            if done:
                break
        return cov, done

    def step_macro(self, action_id: int) -> tuple[float, bool]:
        return self._execute_macro(action_id, record=self.record)

    def simulate_macro(self, action_id: int) -> float:
        cov, _ = self._execute_macro(action_id, record=False)
        return cov

    def simulate_rollout(self, first_action: int, depth: int) -> float:
        """Simulate first_action then (depth-1) random follow-up actions. Return final IoU."""
        cov = 0.0
        actions = [first_action] + [self.random_action() for _ in range(depth - 1)]
        for action in actions:
            cov, _ = self._execute_macro(action, record=False)
        return cov

    # ── Action selection ───────────────────────────────────────────────────────

    def random_action(self) -> int:
        return int(self._rng.integers(NUM_ACTIONS))

    def greedy_action(self) -> int:
        """Evaluate all 8 depth-1 outcomes, pick best IoU. (8 simulations)"""
        root = self.save_state()
        best_a, best_q = 0, -1.0
        for a in range(NUM_ACTIONS):
            self.restore_state(root)
            q = self.simulate_macro(a)
            if q > best_q:
                best_q, best_a = q, a
        self.restore_state(root)
        return best_a

    def puct_action(self, prior: np.ndarray, n_sims: int = 32, C: float = 1.5,
                    rollout_depth: int = 1) -> int:
        """
        PUCT with Q updates.  UCB = Q(a) + C · P(a) · √N / (1 + n(a)).

        Each rollout outcome updates Q(a) as a running mean, so the planner
        learns which actions lead to high IoU and shifts visits accordingly.
        Final action = argmax visit count.
        """
        root = self.save_state()
        N = np.zeros(NUM_ACTIONS, dtype=float)
        Q = np.zeros(NUM_ACTIONS, dtype=float)

        for _ in range(n_sims):
            N_total = max(1.0, N.sum())
            ucb = Q + C * prior * np.sqrt(N_total) / (1.0 + N)
            action = int(np.argmax(ucb))
            self.restore_state(root)
            iou = self.simulate_rollout(action, rollout_depth)
            N[action] += 1.0
            Q[action] += (iou - Q[action]) / N[action]

        self.restore_state(root)
        return int(np.argmax(N))

    def gumbel_action(self, prior: np.ndarray | None = None,
                      n_sims: int = 32, max_k: int = 8,
                      rollout_depth: int = 1) -> int:
        """
        Gumbel sequential halving with log-prior perturbation.

        Initial candidate ranking uses the Gumbel-top-k trick:
            score(a) = log P(a) + Gumbel(0,1) noise
        so candidates are sampled without replacement proportional to P(a).

        Phase halving ranks survivors by mean_Q(a) + Gumbel score, letting
        actual rollout IoU override a misguided prior over time.

        When rollout_depth > 1, each simulation executes the candidate action
        followed by (rollout_depth - 1) random actions, returning the final IoU.
        This gives the planner a longer horizon to discover multi-step sequences
        (e.g. rotations) that only pay off several actions later.
        """
        if prior is None:
            prior = np.ones(NUM_ACTIONS, dtype=np.float32) / NUM_ACTIONS

        root = self.save_state()
        visits   = np.zeros(NUM_ACTIONS, dtype=int)
        total_q  = np.zeros(NUM_ACTIONS)

        # Gumbel-top-k: perturb log-prior so initial ranking ∝ P(a)
        log_prior = np.log(prior.clip(min=1e-8))
        u = np.random.uniform(1e-8, 1 - 1e-8, NUM_ACTIONS)
        gumbel = log_prior - np.log(-np.log(u))

        k = min(max_k, NUM_ACTIONS)
        phases = max(1, int(np.log2(k)))
        budget_pp = max(1, n_sims // phases)

        candidates = list(np.argsort(gumbel)[::-1][:k])
        remaining = n_sims

        for phase in range(phases):
            k_phase = max(1, len(candidates))
            budget = remaining if phase == phases - 1 else budget_pp
            spc = max(1, budget // k_phase)

            for action in candidates:
                for _ in range(spc):
                    self.restore_state(root)
                    iou = self.simulate_rollout(action, rollout_depth)
                    visits[action] += 1
                    total_q[action] += iou
                    remaining -= 1

            if phase < phases - 1 and len(candidates) > 1:
                mean_q = np.array([
                    total_q[a] / visits[a] if visits[a] > 0 else 0.0
                    for a in candidates
                ])
                g = np.array([gumbel[a] for a in candidates])
                scores = mean_q + g
                half = max(1, len(candidates) // 2)
                top = np.argsort(scores)[::-1][:half]
                candidates = [candidates[i] for i in top]

        self.restore_state(root)
        mean_q = np.where(visits > 0, total_q / visits, -1.0)
        return int(np.argmax(mean_q))


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(strategy: str, seed: int, n_macros: int, budget: int,
                policy, record: bool,
                rollout_depth: int = 1) -> tuple[list[float], list[np.ndarray]]:
    env = PushTMacroEnv(seed=seed, record=record)
    raw = env.env.unwrapped
    env.reset()

    obs_history = [_make_obs(raw)]
    coverages = []

    for step in range(n_macros):
        # Prior computed once from current state; all simulations reuse it.
        prior = _compute_prior(policy, raw, obs_history)

        t0 = time.perf_counter()
        if strategy == "puct":
            action = env.puct_action(prior, n_sims=budget,
                                     rollout_depth=rollout_depth)
        elif strategy == "gumbel":
            action = env.gumbel_action(prior, n_sims=budget,
                                       rollout_depth=rollout_depth)
        elif strategy == "random":
            action = env.random_action()
        elif strategy == "greedy":
            action = env.greedy_action()
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")
        dt = time.perf_counter() - t0

        cov, done = env.step_macro(action)
        coverages.append(cov)
        obs_history.append(_make_obs(raw))

        prior_str = " ".join(f"{p:.2f}" for p in prior)
        print(f"[{strategy:6s}] step {step+1}/{n_macros} | action={action} | "
              f"IoU={cov:.3f} | plan={dt:.2f}s")
        print(f"          prior=[{prior_str}]")
        if done:
            print("  → solved!")
            break

    frames = list(env.frames)
    env.close()
    return coverages, frames


# ── GIF assembly ───────────────────────────────────────────────────────────────

def _pad_to_same_height(img_a: np.ndarray, img_b: np.ndarray) -> tuple:
    ha, hb = img_a.shape[0], img_b.shape[0]
    if ha < hb:
        img_a = np.pad(img_a, [(0, hb - ha), (0, 0), (0, 0)])
    elif hb < ha:
        img_b = np.pad(img_b, [(0, ha - hb), (0, 0), (0, 0)])
    return img_a, img_b


def make_gif(frames_left, frames_right, label_left, label_right, out_path, fps=15):
    from PIL import ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default(size=18)

    n = max(len(frames_left), len(frames_right))
    last_l = frames_left[-1]  if frames_left  else np.zeros((100, 100, 3), dtype=np.uint8)
    last_r = frames_right[-1] if frames_right else np.zeros((100, 100, 3), dtype=np.uint8)

    pil_frames = []
    for i in range(n):
        fl = frames_left[i]  if i < len(frames_left)  else last_l
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


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--n-macros",   type=int,   default=10,
                   help="Max macro-actions per episode")
    p.add_argument("--budget",     type=int,   default=32,
                   help="Simulation budget per macro-step")
    p.add_argument("--rollout-depth", type=int, default=1,
                   help="Lookahead depth per simulation (receding horizon)")
    p.add_argument("--strategy-a", default="puct",
                   choices=["random", "greedy", "puct", "gumbel"])
    p.add_argument("--strategy-b", default="gumbel",
                   choices=["random", "greedy", "puct", "gumbel"])
    p.add_argument("--no-prior",   action="store_true",
                   help="Skip loading the diffusion policy (use uniform prior)")
    p.add_argument("--no-gif",     action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"\n{'='*60}")
    print(f" PushT  —  PUCT (Q=0) vs Gumbel  |  prior: diffusion policy")
    print(f" seed={args.seed}  n_macros={args.n_macros}  budget={args.budget}  rollout_depth={args.rollout_depth}")
    print(f" comparing: {args.strategy_a}  vs  {args.strategy_b}")
    print(f"{'='*60}\n")

    policy = None if args.no_prior else _load_policy()
    record = not args.no_gif

    print(f"\n── Strategy A: {args.strategy_a} ──")
    covs_a, frames_a = run_episode(
        args.strategy_a, args.seed, args.n_macros, args.budget, policy, record,
        rollout_depth=args.rollout_depth,
    )

    print(f"\n── Strategy B: {args.strategy_b} ──")
    covs_b, frames_b = run_episode(
        args.strategy_b, args.seed, args.n_macros, args.budget, policy, record,
        rollout_depth=args.rollout_depth,
    )

    print(f"\n{'='*60}")
    print(f"  {args.strategy_a:6s}  final IoU: {covs_a[-1]:.3f}  |  mean: {np.mean(covs_a):.3f}")
    print(f"  {args.strategy_b:6s}  final IoU: {covs_b[-1]:.3f}  |  mean: {np.mean(covs_b):.3f}")
    print(f"{'='*60}\n")

    if record:
        gif_path = os.path.join(out_dir, "pusht_comparison.gif")
        label_a = f"{args.strategy_a}  IoU={covs_a[-1]:.2f}"
        label_b = f"{args.strategy_b}  IoU={covs_b[-1]:.2f}"
        make_gif(frames_a, frames_b, label_a, label_b, gif_path, fps=15)


if __name__ == "__main__":
    main()
