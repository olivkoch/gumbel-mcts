"""Verify budget allocation before and after guardrail for various sim budgets."""
import numpy as np

def halving_params_original(sims, max_k=16, num_actions=81):
    max_k = min(num_actions, max_k)
    num_phases = max(1, int(np.log2(max_k)))
    first_phase_budget = sims // num_phases
    k_initial = min(max_k, first_phase_budget // 2)
    k_initial = max(2, k_initial)
    num_phases = max(1, int(np.log2(k_initial)))
    # Simulate the phase loop
    remaining = sims
    phases = []
    for phase in range(num_phases):
        k_phase = max(1, k_initial // (2 ** phase))
        phases_left = num_phases - phase
        if phase == num_phases - 1:
            budget = remaining
        else:
            budget = remaining // phases_left
        spa = max(1, budget // k_phase)
        phases.append((k_phase, spa))
        remaining -= k_phase * spa
    return k_initial, num_phases, phases

MIN_SPA0 = 4

def halving_params_guarded(sims, max_k=16, num_actions=81):
    max_k = min(num_actions, max_k)
    num_phases = max(1, int(np.log2(max_k)))
    first_phase_budget = sims // num_phases
    k_initial = min(max_k, first_phase_budget // 2)
    k_initial = max(2, k_initial)
    num_phases = max(1, int(np.log2(k_initial)))

    # Guardrail: ensure phase 0 has >= MIN_SPA0 sims/candidate
    k_initial = min(k_initial, max(2, (sims // num_phases) // MIN_SPA0))
    # Round down to nearest power of 2 for clean halving
    k_initial = max(2, 1 << int(np.log2(k_initial)))
    # Recompute phases for the guarded k
    num_phases = max(1, int(np.log2(k_initial)))

    remaining = sims
    phases = []
    for phase in range(num_phases):
        k_phase = max(1, k_initial // (2 ** phase))
        phases_left = num_phases - phase
        if phase == num_phases - 1:
            budget = remaining
        else:
            budget = remaining // phases_left
        spa = max(1, budget // k_phase)
        phases.append((k_phase, spa))
        remaining -= k_phase * spa
    return k_initial, num_phases, phases

budgets = [4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 512, 1024]

print(f"{'sims':>6}  {'orig k0':>7} {'orig spa0':>9}  {'guard k0':>8} {'guard spa0':>10}  phase breakdown (guarded)")
print("-" * 90)
for sims in budgets:
    k0_o, np_o, ph_o = halving_params_original(sims)
    k0_g, np_g, ph_g = halving_params_guarded(sims)
    spa0_o = ph_o[0][1] if ph_o else 0
    spa0_g = ph_g[0][1] if ph_g else 0
    changed = " ←" if k0_g != k0_o else ""
    breakdown = "  ".join(f"k={k},spa={s}" for k,s in ph_g)
    print(f"{sims:>6}  {k0_o:>7} {spa0_o:>9}  {k0_g:>8} {spa0_g:>10}  [{breakdown}]{changed}")
