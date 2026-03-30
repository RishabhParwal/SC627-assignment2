"""Trajectory optimization for a 2D double-integrator drone on a 6x6 map.

This script solves two open-loop trajectory optimization problems with CasADi:
1) Minimum energy trajectory
2) Minimum time trajectory

Obstacle avoidance is handled with linear inequalities by using a convex safe corridor
around the rectangle obstacle (separating-hyperplane style constraints).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import csv


# =========================
# Easily configurable values
# =========================
MAP_X_MIN = -5.0
MAP_X_MAX = 5.0
MAP_Y_MIN = -5.0
MAP_Y_MAX = 5.0

# Rectangle obstacle: width 0.2 (x from -0.1 to 0.1), height 3.0 (y from -1.5 to 1.5)
OBSTACLE_X_MIN = -0.1
OBSTACLE_X_MAX = 0.1
OBSTACLE_Y_MIN = -1.5
OBSTACLE_Y_MAX = 1.5

START_POS = np.array([-2.0, 0.0])
GOAL_POS = np.array([2.0, 0.0])
START_VEL = np.array([0.0, 0.0])
GOAL_VEL = np.array([0.0, 0.0])

N_TIMESTEPS = 50
DEFAULT_TIME_HORIZON = 6.0
DRONE_MASS = 1.0
MIN_TIME_FIXED_DT = 0.12

# Restored constraints (as used earlier)
MAX_ACCEL_X = 5.0
MAX_ACCEL_Y = 5.0
MAX_VEL_X = 6.0
MAX_VEL_Y = 6.0

SAFETY_MARGIN = 0.2
# More interior samples reduce missed corner-grazing between knot points.
COLLISION_SAMPLE_FRACTIONS = tuple(np.linspace(0.1, 0.9, 9))
SHOW_TRAJECTORY_MARKERS = False
OBSTACLE_CLEARANCE_EPS = 0.05

# Time optimization bounds
MIN_TOTAL_TIME = 1.0
MAX_TOTAL_TIME = 12.0


@dataclass
class TrajectoryResult:
    name: str
    success: bool
    solve_time_s: float
    states: np.ndarray
    controls: np.ndarray
    total_time: float
    energy_cost: float
    objective_value: float


def continuous_dynamics(x: ca.MX, u: ca.MX) -> ca.MX:
    """Continuous-time double integrator dynamics.

    x = [px, py, vx, vy], u = [ax, ay]
    """
    return ca.vertcat(x[2], x[3], u[0], u[1])


def rk4_step(xk: ca.MX, uk: ca.MX, dt: ca.MX) -> ca.MX:
    """One RK4 integration step for x_dot = f(x, u)."""
    k1 = continuous_dynamics(xk, uk)
    k2 = continuous_dynamics(xk + 0.5 * dt * k1, uk)
    k3 = continuous_dynamics(xk + 0.5 * dt * k2, uk)
    k4 = continuous_dynamics(xk + dt * k3, uk)
    return xk + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def apply_common_constraints(
    opti: ca.Opti,
    x: ca.MX,
    u: ca.MX,
    dt: ca.MX,
    n_steps: int = N_TIMESTEPS,
) -> None:
    """Apply dynamics, bounds, and direct obstacle-avoidance constraints."""
    # Initial and terminal conditions
    opti.subject_to(x[0, 0] == START_POS[0])
    opti.subject_to(x[1, 0] == START_POS[1])
    opti.subject_to(x[2, 0] == START_VEL[0])
    opti.subject_to(x[3, 0] == START_VEL[1])

    opti.subject_to(x[0, n_steps] == GOAL_POS[0])
    opti.subject_to(x[1, n_steps] == GOAL_POS[1])
    opti.subject_to(x[2, n_steps] == GOAL_VEL[0])
    opti.subject_to(x[3, n_steps] == GOAL_VEL[1])

    # Dynamics + acceleration bounds
    for k in range(n_steps):
        x_next = rk4_step(x[:, k], u[:, k], dt)
        opti.subject_to(x[:, k + 1] == x_next)

        opti.subject_to(opti.bounded(-MAX_ACCEL_X, u[0, k], MAX_ACCEL_X))
        opti.subject_to(opti.bounded(-MAX_ACCEL_Y, u[1, k], MAX_ACCEL_Y))

    # Position and velocity bounds
    for k in range(n_steps + 1):
        opti.subject_to(opti.bounded(MAP_X_MIN, x[0, k], MAP_X_MAX))
        opti.subject_to(opti.bounded(MAP_Y_MIN, x[1, k], MAP_Y_MAX))
        opti.subject_to(opti.bounded(-MAX_VEL_X, x[2, k], MAX_VEL_X))
        opti.subject_to(opti.bounded(-MAX_VEL_Y, x[3, k], MAX_VEL_Y))

    # Direct obstacle avoidance (without forced left/top/right route)
    x_left = OBSTACLE_X_MIN - SAFETY_MARGIN
    x_right = OBSTACLE_X_MAX + SAFETY_MARGIN
    y_low = OBSTACLE_Y_MIN - SAFETY_MARGIN
    y_high = OBSTACLE_Y_MAX + SAFETY_MARGIN

    def outside_metric(px: ca.MX, py: ca.MX) -> ca.MX:
        # Outside inflated rectangle iff at least one face inequality is satisfied.
        # max(px-x_right, x_left-px, py-y_high, y_low-py) >= 0.
        t1 = px - x_right
        t2 = x_left - px
        t3 = py - y_high
        t4 = y_low - py
        return ca.fmax(ca.fmax(t1, t2), ca.fmax(t3, t4))

    for k in range(n_steps + 1):
        opti.subject_to(outside_metric(x[0, k], x[1, k]) >= OBSTACLE_CLEARANCE_EPS)

    # Extra collision checks at sampled points between nodes to avoid line-segment
    # interpolation cutting through the obstacle even when node constraints pass.
    for k in range(n_steps):
        for s in COLLISION_SAMPLE_FRACTIONS:
            p_sample = (1.0 - s) * x[:, k] + s * x[:, k + 1]
            opti.subject_to(outside_metric(p_sample[0], p_sample[1]) >= OBSTACLE_CLEARANCE_EPS)


def build_initial_guess(n_steps: int = N_TIMESTEPS) -> tuple[np.ndarray, np.ndarray]:
    """Build a feasible-ish warm start for states and controls."""
    x_guess = np.zeros((4, n_steps + 1), dtype=float)
    u_guess = np.zeros((2, n_steps), dtype=float)

    for k in range(n_steps + 1):
        alpha = k / n_steps
        x_guess[0, k] = (1.0 - alpha) * START_POS[0] + alpha * GOAL_POS[0]
        x_guess[1, k] = (1.0 - alpha) * START_POS[1] + alpha * GOAL_POS[1]

    # Nudge midpoint above obstacle to help initialize a feasible guess.
    mid = n_steps // 2
    x_guess[1, mid] = max(x_guess[1, mid], OBSTACLE_Y_MAX + SAFETY_MARGIN + 0.1)

    return x_guess, u_guess


def configure_solver(opti: ca.Opti) -> None:
    """Configure IPOPT settings for quiet, stable solves."""
    p_opts = {"expand": True}
    s_opts = {
        "max_iter": 2000,
        "print_level": 0,
        "sb": "yes",
        "tol": 1e-6,
    }
    opti.solver("ipopt", p_opts, s_opts)


def solve_min_energy() -> TrajectoryResult:
    """Minimize integrated control effort with fixed total time.

    Control-effort energy per node: ax^2 + ay^2.
    """
    opti = ca.Opti()

    x = opti.variable(4, N_TIMESTEPS + 1)
    u = opti.variable(2, N_TIMESTEPS)

    dt = DEFAULT_TIME_HORIZON / N_TIMESTEPS
    apply_common_constraints(opti, x, u, dt)

    objective = 0
    for k in range(N_TIMESTEPS):
        objective += ca.dot(u[:, k], u[:, k]) * dt

    opti.minimize(objective)

    x_guess, u_guess = build_initial_guess(N_TIMESTEPS)
    opti.set_initial(x, x_guess)
    opti.set_initial(u, u_guess)

    configure_solver(opti)

    t0 = time.time()
    try:
        sol = opti.solve()
        success = True
    except RuntimeError:
        sol = opti.debug
        success = False
    solve_time = time.time() - t0

    x_sol = np.array(sol.value(x), dtype=float)
    u_sol = np.array(sol.value(u), dtype=float)
    total_time = DEFAULT_TIME_HORIZON
    dt_sol = total_time / N_TIMESTEPS
    energy = float(np.sum(np.sum(u_sol * u_sol, axis=0)) * dt_sol)
    obj_val = float(sol.value(objective))

    return TrajectoryResult(
        name="minimum_control_energy",
        success=success,
        solve_time_s=solve_time,
        states=x_sol,
        controls=u_sol,
        total_time=total_time,
        energy_cost=energy,
        objective_value=obj_val,
    )


def solve_min_time() -> TrajectoryResult:
    """Minimize total time with fixed dt by searching over variable N."""
    dt_fixed = MIN_TIME_FIXED_DT
    n_min = max(2, int(np.ceil(MIN_TOTAL_TIME / dt_fixed)))
    n_max = max(n_min, int(np.floor(MAX_TOTAL_TIME / dt_fixed)))

    total_solve_time = 0.0
    best_result: TrajectoryResult | None = None

    for n_steps in range(n_min, n_max + 1):
        opti = ca.Opti()
        x = opti.variable(4, n_steps + 1)
        u = opti.variable(2, n_steps)

        apply_common_constraints(opti, x, u, dt_fixed, n_steps=n_steps)

        # Secondary objective for numerical stability and tie-breaking at fixed N.
        objective = 0
        for k in range(n_steps):
            objective += ca.dot(u[:, k], u[:, k]) * dt_fixed
        opti.minimize(objective)

        x_guess, u_guess = build_initial_guess(n_steps)
        opti.set_initial(x, x_guess)
        opti.set_initial(u, u_guess)
        configure_solver(opti)

        t0 = time.time()
        try:
            sol = opti.solve()
            success = True
        except RuntimeError:
            sol = opti.debug
            success = False
        solve_time = time.time() - t0
        total_solve_time += solve_time

        if not success:
            continue

        x_sol = np.array(sol.value(x), dtype=float)
        u_sol = np.array(sol.value(u), dtype=float)
        t_sol = n_steps * dt_fixed
        energy = float(np.sum(np.sum(u_sol * u_sol, axis=0)) * dt_fixed)
        obj_val = float(sol.value(objective))

        best_result = TrajectoryResult(
            name="minimum_time",
            success=True,
            solve_time_s=total_solve_time,
            states=x_sol,
            controls=u_sol,
            total_time=t_sol,
            energy_cost=energy,
            objective_value=obj_val,
        )
        break

    if best_result is not None:
        return best_result

    # If no feasible N found, return a failure placeholder.
    return TrajectoryResult(
        name="minimum_time",
        success=False,
        solve_time_s=total_solve_time,
        states=np.zeros((4, n_max + 1), dtype=float),
        controls=np.zeros((2, n_max), dtype=float),
        total_time=n_max * dt_fixed,
        energy_cost=float("inf"),
        objective_value=float("inf"),
    )


def solve_min_kinetic_energy() -> TrajectoryResult:
    """Minimize integrated kinetic energy with fixed total time.

    Kinetic energy per node: (1/2) * m * (vx^2 + vy^2).
    """
    opti = ca.Opti()

    x = opti.variable(4, N_TIMESTEPS + 1)
    u = opti.variable(2, N_TIMESTEPS)

    dt = DEFAULT_TIME_HORIZON / N_TIMESTEPS
    apply_common_constraints(opti, x, u, dt)

    objective = 0
    for k in range(N_TIMESTEPS):
        vx_k = x[2, k]
        vy_k = x[3, k]
        objective += 0.5 * DRONE_MASS * (vx_k * vx_k + vy_k * vy_k) * dt

    opti.minimize(objective)

    x_guess, u_guess = build_initial_guess()
    opti.set_initial(x, x_guess)
    opti.set_initial(u, u_guess)

    configure_solver(opti)

    t0 = time.time()
    try:
        sol = opti.solve()
        success = True
    except RuntimeError:
        sol = opti.debug
        success = False
    solve_time = time.time() - t0

    x_sol = np.array(sol.value(x), dtype=float)
    u_sol = np.array(sol.value(u), dtype=float)
    total_time = DEFAULT_TIME_HORIZON
    dt_sol = total_time / N_TIMESTEPS
    vx = x_sol[2, :-1]
    vy = x_sol[3, :-1]
    energy = float(np.sum(0.5 * DRONE_MASS * (vx * vx + vy * vy)) * dt_sol)
    obj_val = float(sol.value(objective))

    return TrajectoryResult(
        name="minimum_kinetic_energy",
        success=success,
        solve_time_s=solve_time,
        states=x_sol,
        controls=u_sol,
        total_time=total_time,
        energy_cost=energy,
        objective_value=obj_val,
    )


def verify_obstacle_clearance(states: np.ndarray) -> tuple[bool, float]:
    """Check minimum signed clearance from the rectangle (positive = outside buffer)."""
    px = states[0, :]
    py = states[1, :]

    x_min = OBSTACLE_X_MIN - SAFETY_MARGIN
    x_max = OBSTACLE_X_MAX + SAFETY_MARGIN
    y_min = OBSTACLE_Y_MIN - SAFETY_MARGIN
    y_max = OBSTACLE_Y_MAX + SAFETY_MARGIN

    clearances = []
    for xk, yk in zip(px, py):
        dx_out = max(x_min - xk, 0.0, xk - x_max)
        dy_out = max(y_min - yk, 0.0, yk - y_max)
        if dx_out > 0.0 or dy_out > 0.0:
            c = np.hypot(dx_out, dy_out)
        else:
            c = -min(xk - x_min, x_max - xk, yk - y_min, y_max - yk)
        clearances.append(c)

    min_clearance = float(np.min(clearances))
    return min_clearance >= -1e-6, min_clearance


def save_trajectory_to_csv(result: TrajectoryResult, filename: str) -> None:
    """Save the trajectory to a CSV file assuming fixed dt."""
    if not result.success:
        print(f"Cannot save failed trajectory {result.name} to CSV.")
        return

    n_nodes = result.states.shape[1]
    dt_sol = result.total_time / (n_nodes - 1)
    
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
        
        for k in range(n_nodes):
            t = k * dt_sol
            x = result.states[0, k]
            y = result.states[1, k]
            z = 1.03  # Fixed altitude for 2D map
            vx = result.states[2, k]
            vy = result.states[3, k]
            vz = 0.0
            writer.writerow([t, x, y, z, vx, vy, vz])
            
    print(f"Saved {result.name} trajectory to {filename}")


def print_summary(result: TrajectoryResult) -> None:
    """Print compact run summary."""
    clear_ok, min_clearance = verify_obstacle_clearance(result.states)
    print(f"[{result.name}]")
    print(f"  success        : {result.success}")
    print(f"  solve_time_s   : {result.solve_time_s:.4f}")
    print(f"  total_time_s   : {result.total_time:.4f}")
    print(f"  energy_cost    : {result.energy_cost:.6f}")
    print(f"  objective      : {result.objective_value:.6f}")
    print(f"  clearance_ok   : {clear_ok}")
    print(f"  min_clearance  : {min_clearance:.6f}")


def plot_results(results: list[TrajectoryResult]) -> None:
    """Plot trajectories over map and obstacle."""
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot(
        [MAP_X_MIN, MAP_X_MAX, MAP_X_MAX, MAP_X_MIN, MAP_X_MIN],
        [MAP_Y_MIN, MAP_Y_MIN, MAP_Y_MAX, MAP_Y_MAX, MAP_Y_MIN],
        "k-",
        linewidth=1.5,
        label="map",
    )

    ox = [OBSTACLE_X_MIN, OBSTACLE_X_MAX, OBSTACLE_X_MAX, OBSTACLE_X_MIN, OBSTACLE_X_MIN]
    oy = [OBSTACLE_Y_MIN, OBSTACLE_Y_MIN, OBSTACLE_Y_MAX, OBSTACLE_Y_MAX, OBSTACLE_Y_MIN]
    ax.fill(ox, oy, color="tab:red", alpha=0.35, label="obstacle")

    bx_min = OBSTACLE_X_MIN - SAFETY_MARGIN
    bx_max = OBSTACLE_X_MAX + SAFETY_MARGIN
    by_min = OBSTACLE_Y_MIN - SAFETY_MARGIN
    by_max = OBSTACLE_Y_MAX + SAFETY_MARGIN
    bxp = [bx_min, bx_max, bx_max, bx_min, bx_min]
    byp = [by_min, by_min, by_max, by_max, by_min]
    ax.plot(bxp, byp, "r--", linewidth=1.0, label="obstacle buffer")

    styles = {
        "minimum_control_energy": {"color": "tab:blue", "label": "min control energy"},
        "minimum_time": {"color": "tab:green", "label": "min time"},
        "minimum_kinetic_energy": {"color": "tab:orange", "label": "min kinetic energy"},
    }

    for result in results:
        cfg = styles.get(result.name, {"color": "tab:gray", "label": result.name})
        ax.plot(result.states[0, :], result.states[1, :], color=cfg["color"], linewidth=2.0, label=cfg["label"])

        if SHOW_TRAJECTORY_MARKERS:
            # Show node sampling points and inter-node collision-check points.
            ax.scatter(
                result.states[0, :],
                result.states[1, :],
                color=cfg["color"],
                s=14,
                alpha=0.8,
                zorder=3,
                label=f"{cfg['label']} nodes",
            )

            sample_x = []
            sample_y = []
            n_local = result.states.shape[1] - 1
            for k in range(n_local):
                p0 = result.states[:, k]
                p1 = result.states[:, k + 1]
                for s in COLLISION_SAMPLE_FRACTIONS:
                    ps = (1.0 - s) * p0 + s * p1
                    sample_x.append(ps[0])
                    sample_y.append(ps[1])

            ax.scatter(
                sample_x,
                sample_y,
                color=cfg["color"],
                s=8,
                alpha=0.55,
                marker="x",
                zorder=2,
                label=f"{cfg['label']} samples",
            )

    ax.scatter([START_POS[0]], [START_POS[1]], color="black", marker="o", s=60, label="start")
    ax.scatter([GOAL_POS[0]], [GOAL_POS[1]], color="black", marker="*", s=120, label="goal")

    ax.set_title("Double Integrator Trajectory Optimization")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(MAP_X_MIN - 0.2, MAP_X_MAX + 0.2)
    ax.set_ylim(MAP_Y_MIN - 0.2, MAP_Y_MAX + 0.2)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig('trajectory.png', dpi=300)
    print("Plot saved to trajectory.png")
    # plt.show()


def main() -> None:
    print("=== Double Integrator Trajectory Optimization ===")
    print(f"Map: [{MAP_X_MIN}, {MAP_X_MAX}] x [{MAP_Y_MIN}, {MAP_Y_MAX}]")
    print(
        "Obstacle: "
        f"x in [{OBSTACLE_X_MIN}, {OBSTACLE_X_MAX}], "
        f"y in [{OBSTACLE_Y_MIN}, {OBSTACLE_Y_MAX}]"
    )
    print()

    res_energy = solve_min_energy()
    res_time = solve_min_time()
    res_kinetic = solve_min_kinetic_energy()

    print_summary(res_energy)
    print()
    print_summary(res_time)
    print()
    print_summary(res_kinetic)

    print()
    print("Comparison:")
    print(f"  min-energy time  : {res_energy.total_time:.4f} s")
    print(f"  min-time time    : {res_time.total_time:.4f} s")
    print(f"  min-kinetic time : {res_kinetic.total_time:.4f} s")
    print(f"  min-energy energy: {res_energy.energy_cost:.6f}")
    print(f"  min-time energy  : {res_time.energy_cost:.6f}")
    print(f"  min-kinetic energy: {res_kinetic.energy_cost:.6f}")

    # Save the minimum time trajectory to be used by the feeder node
    save_trajectory_to_csv(res_time, "optimal_trajectory.csv")

    plot_results([res_energy, res_time, res_kinetic])


if __name__ == "__main__":
    main()
