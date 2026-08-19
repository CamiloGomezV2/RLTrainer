"""Gymnasium + tabular agent episode runner."""

from __future__ import annotations

import base64
import atexit
import io
import secrets
from contextlib import contextmanager
from typing import Any, Iterator

import gymnasium as gym
import numpy as np
from PIL import Image
from agents.TableAgents import MC, Q_learning, SARSA
from gymnasium.spaces import Box, Discrete, Tuple

DEFAULT_MAX_TIMESTEPS = 100
DEFAULT_GAMMA = 0.99
DEFAULT_TRAINING_EPISODES = 100

TABULAR_AGENTS = {
    "MC": MC,
    "SARSA": SARSA,
    "Q_learning": Q_learning,
}

# In-memory runtime keyed by Flask session id
_RUNTIMES: dict[str, dict[str, Any]] = {}


def _close_env(env) -> None:
    if env is None:
        return
    try:
        env.close()
    except Exception:
        pass


def ensure_session_id(session) -> str:
    if "sid" not in session:
        session["sid"] = secrets.token_hex(16)
    return session["sid"]


def _copy_learned_state(agent) -> dict[str, Any] | None:
    """Snapshot Q, policy, and visit counts so visualization cannot learn."""
    if agent is None:
        return None
    snapshot: dict[str, Any] = {}
    if hasattr(agent, "Q"):
        snapshot["Q"] = np.array(agent.Q, copy=True)
    if hasattr(agent, "policy"):
        snapshot["policy"] = np.array(agent.policy, copy=True)
    if hasattr(agent, "N"):
        snapshot["N"] = np.array(agent.N, copy=True)
    return snapshot


def _restore_learned_state(agent, snapshot: dict[str, Any] | None) -> None:
    if agent is None or snapshot is None:
        return
    if "Q" in snapshot and hasattr(agent, "Q"):
        q_table = agent.Q
        if isinstance(q_table, np.ndarray) and q_table.shape == snapshot["Q"].shape:
            q_table[...] = snapshot["Q"]
        else:
            agent.Q = np.array(snapshot["Q"], copy=True)
    if "policy" in snapshot and hasattr(agent, "policy"):
        policy = agent.policy
        if isinstance(policy, np.ndarray) and policy.shape == snapshot["policy"].shape:
            policy[...] = snapshot["policy"]
        else:
            agent.policy = np.array(snapshot["policy"], copy=True)
    if "N" in snapshot and hasattr(agent, "N"):
        visits = agent.N
        if isinstance(visits, np.ndarray) and visits.shape == snapshot["N"].shape:
            visits[...] = snapshot["N"]
        else:
            agent.N = np.array(snapshot["N"], copy=True)


@contextmanager
def _without_learning(agent) -> Iterator[None]:
    snapshot = _copy_learned_state(agent)
    try:
        yield
    finally:
        _restore_learned_state(agent, snapshot)


def _reset_agent_knowledge(agent) -> None:
    """Zero learned values (Q-table / policy) if the agent supports it."""
    if agent is None:
        return
    reset = getattr(agent, "reset", None)
    if callable(reset):
        reset()
        return
    if hasattr(agent, "Q"):
        agent.Q = np.zeros_like(np.asarray(agent.Q), dtype=float)


def invalidate_runtime(session) -> None:
    sid = session.get("sid")
    if not sid:
        return
    runtime = _RUNTIMES.pop(sid, None)
    if runtime is None:
        return
    _reset_agent_knowledge(runtime.get("agent"))
    _close_env(runtime.get("env"))


def reset_experiment_runtime(session) -> None:
    """
    Discard the current environment/agent so the next use starts from scratch.

    Applying configuration must leave the Q-table at 0 even if the environment,
    agent, and hyperparameters did not change.
    """
    session["runtime_generation"] = int(session.get("runtime_generation", 0)) + 1
    session.modified = True
    invalidate_runtime(session)


def close_all_runtimes() -> None:
    for sid in list(_RUNTIMES):
        runtime = _RUNTIMES.pop(sid, None)
        if runtime is not None:
            _close_env(runtime.get("env"))


atexit.register(close_all_runtimes)


def _frame_to_data_url(frame) -> str | None:
    if frame is None:
        return None
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    image = Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _serialize_observation(observation):
    if isinstance(observation, (np.integer, int)):
        return int(observation)
    if isinstance(observation, (np.floating, float)):
        return float(observation)
    if isinstance(observation, tuple):
        return [_serialize_observation(item) for item in observation]
    if isinstance(observation, np.ndarray):
        return observation.tolist()
    return str(observation)


def observation_space_size(space) -> int:
    if isinstance(space, Discrete):
        return int(space.n)
    if isinstance(space, Tuple):
        size = 1
        for subspace in space.spaces:
            if not isinstance(subspace, Discrete):
                raise ValueError("Only Discrete Tuple observation spaces are supported.")
            size *= int(subspace.n)
        return size
    raise ValueError(
        "Tabular agents require a Discrete (or Discrete Tuple) observation space."
    )


def encode_observation(observation, space) -> int:
    if isinstance(space, Discrete):
        return int(observation)
    if isinstance(space, Tuple):
        index = 0
        multiplier = 1
        for value, subspace in zip(reversed(observation), reversed(space.spaces)):
            index += int(value) * multiplier
            multiplier *= int(subspace.n)
        return index
    raise ValueError(
        "Tabular agents require a Discrete (or Discrete Tuple) observation space."
    )


def _state_q_values(runtime: dict[str, Any], observation: Any = None) -> dict | None:
    """Return Q(s, a) for every action using the same table as /api/agent/q-table."""
    env = runtime.get("env")
    try:
        table = get_q_table(runtime)
    except ValueError:
        return None
    if env is None:
        return None

    q_table = np.asarray(table["q_table"], dtype=float)
    if q_table.ndim != 2:
        return None

    if observation is None:
        agent = runtime.get("agent")
        if not getattr(agent, "states", None):
            return None
        state = int(agent.states[-1])
    elif isinstance(observation, (int, np.integer)):
        state = int(observation)
    else:
        try:
            state = encode_observation(observation, env.observation_space)
        except (ValueError, TypeError, AttributeError):
            return None

    if state < 0 or state >= q_table.shape[0]:
        return None

    values = q_table[state]
    max_q = float(np.max(values))
    greedy_actions = [
        int(action) for action, value in enumerate(values) if float(value) == max_q
    ]
    return {
        "state_index": state,
        "values": [float(value) for value in values],
        "greedy_actions": greedy_actions,
    }


def create_agent(agent_name: str, env: gym.Env, config: dict):
    if agent_name not in TABULAR_AGENTS:
        raise ValueError(
            f"Agent '{agent_name}' is not a tabular agent from TableAgents.py."
        )
    if isinstance(env.action_space, Box):
        raise ValueError("Tabular agents require a Discrete action space.")

    parameters = {
        "nS": observation_space_size(env.observation_space),
        "nA": int(env.action_space.n),
        "gamma": float(config.get("discount_factor", DEFAULT_GAMMA)),
        "epsilon": float(config["exploration_probability"]),
        "alpha": float(config["learning_rate"]),
        "first_visit": True,
    }
    return TABULAR_AGENTS[agent_name](parameters)


def get_or_create_runtime(session, config: dict, need_agent: bool = False) -> dict[str, Any]:
    sid = ensure_session_id(session)
    runtime = _RUNTIMES.get(sid)
    fingerprint = (
        config["environment"],
        config["agent"],
        float(config["learning_rate"]),
        float(config["exploration_probability"]),
        float(config.get("discount_factor", DEFAULT_GAMMA)),
        int(session.get("runtime_generation", 0)),
    )

    if runtime is not None and runtime.get("fingerprint") == fingerprint:
        if need_agent and runtime.get("agent") is None:
            runtime["agent"] = create_agent(config["agent"], runtime["env"], config)
        return runtime

    if runtime is not None:
        _reset_agent_knowledge(runtime.get("agent"))
        _close_env(runtime.get("env"))

    env = gym.make(config["environment"], render_mode="rgb_array")
    agent = None
    if need_agent or config["agent"] in TABULAR_AGENTS:
        try:
            agent = create_agent(config["agent"], env, config)
        except ValueError:
            if need_agent:
                _close_env(env)
                raise
            agent = None

    runtime = {
        "env": env,
        "agent": agent,
        "fingerprint": fingerprint,
        "config": dict(config),
        "timestep": 0,
        "accumulated_reward": 0.0,
    }
    _RUNTIMES[sid] = runtime
    return runtime


def reset_environment(runtime: dict[str, Any], seed: int | None = 0) -> dict:
    env = runtime["env"]
    agent = runtime.get("agent")
    observation, info = env.reset(seed=seed)
    if agent is not None:
        agent.restart()
        state = encode_observation(observation, env.observation_space)
        agent.states.append(state)
    runtime["timestep"] = 0
    runtime["accumulated_reward"] = 0.0
    frame = env.render()
    return {
        "observation": _serialize_observation(observation),
        "info": {key: _serialize_observation(value) for key, value in info.items()},
        "image": _frame_to_data_url(frame),
        "timestep": 0,
        "max_timesteps": DEFAULT_MAX_TIMESTEPS,
        "accumulated_reward": 0.0,
        "q_values": _state_q_values(runtime, observation),
    }


def _resolve_action(runtime: dict[str, Any], action_choice: Any):
    env = runtime["env"]
    agent = runtime.get("agent")

    if action_choice is None or action_choice == "policy":
        if agent is None:
            raise ValueError("Agent policy requires a tabular agent.")
        if not agent.states:
            raise ValueError("Environment has not been initialized.")
        return agent.make_decision()

    if isinstance(env.action_space, Discrete):
        action = int(action_choice)
        if action < 0 or action >= env.action_space.n:
            raise ValueError(f"Action must be in 0..{env.action_space.n - 1}.")
        return action

    if isinstance(env.action_space, Box):
        value = float(action_choice)
        low = float(env.action_space.low.flat[0])
        high = float(env.action_space.high.flat[0])
        if value < low or value > high:
            raise ValueError(f"Action must be in [{low}, {high}].")
        return np.array([value], dtype=np.float32)

    raise ValueError("Unsupported action space.")


def run_single_action(runtime: dict[str, Any], action_choice: Any = "policy") -> dict:
    """Take one environment step without updating the Q-table."""
    env = runtime["env"]
    agent = runtime.get("agent")

    if agent is None:
        raise ValueError("Run an action currently requires a tabular agent.")

    with _without_learning(agent):
        # Start a fresh episode if needed or if the timestep budget was exhausted.
        if not agent.states or runtime.get("timestep", 0) >= DEFAULT_MAX_TIMESTEPS:
            reset_environment(runtime, seed=None)

        action = _resolve_action(runtime, action_choice)
        agent.actions.append(
            action if isinstance(action, (int, np.integer)) else int(action)
        )

        observation, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        next_state = encode_observation(observation, env.observation_space)

        agent.rewards.append(reward)
        agent.dones.append(done)

        runtime["timestep"] = int(runtime.get("timestep", 0)) + 1
        runtime["accumulated_reward"] = float(
            runtime.get("accumulated_reward", 0.0)
        ) + float(reward)

        frame = env.render()
        image = _frame_to_data_url(frame)
        serialized_observation = _serialize_observation(observation)
        serialized_action = (
            int(action)
            if isinstance(action, (int, np.integer))
            else _serialize_observation(action)
        )
        next_image = None
        next_observation = None
        if done:
            observation, info = env.reset()
            agent.restart()
            state = encode_observation(observation, env.observation_space)
            agent.states.append(state)
            runtime["accumulated_reward"] = 0.0
            next_image = _frame_to_data_url(env.render())
            next_observation = _serialize_observation(observation)
        else:
            agent.states.append(next_state)

    payload = {
        "timestep": runtime["timestep"],
        "max_timesteps": DEFAULT_MAX_TIMESTEPS,
        "accumulated_reward": runtime["accumulated_reward"],
        "reward": float(reward),
        "done": done,
        "action": serialized_action,
        "image": image,
        "observation": serialized_observation,
        "q_values": _state_q_values(runtime, serialized_observation),
        "reset_after_done": done,
    }
    if done:
        payload["next_image"] = next_image
        payload["next_observation"] = next_observation
        payload["next_q_values"] = _state_q_values(runtime, next_observation)
    return payload


def run_until_max_timesteps(
    runtime: dict[str, Any],
    max_timesteps: int = DEFAULT_MAX_TIMESTEPS,
    action_choice: Any = "policy",
) -> dict:
    """
    Run up to max_timesteps environment steps without learning.

    Uses the selected action (or the frozen agent policy) at every step.
    When an episode ends (terminated or truncated), restart the env and agent
    episode buffers. The Q-table is not modified.
    """
    env = runtime["env"]
    agent = runtime.get("agent")
    if agent is None:
        raise ValueError("No tabular agent is available for this configuration.")

    with _without_learning(agent):
        initial = reset_environment(runtime, seed=None)
        frames = [
            {
                "timestep": 0,
                "accumulated_reward": 0.0,
                "reward": 0.0,
                "done": False,
                "image": initial["image"],
                "observation": initial["observation"],
                "q_values": None,
            }
        ]

        accumulated_reward = 0.0

        for timestep in range(1, max_timesteps + 1):
            action = _resolve_action(runtime, action_choice)
            agent.actions.append(
                action if isinstance(action, (int, np.integer)) else int(action)
            )

            observation, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            next_state = encode_observation(observation, env.observation_space)

            agent.rewards.append(reward)
            agent.dones.append(done)

            accumulated_reward += float(reward)
            serialized_observation = _serialize_observation(observation)
            frame = env.render()
            frames.append(
                {
                    "timestep": timestep,
                    "accumulated_reward": accumulated_reward,
                    "reward": float(reward),
                    "done": done,
                    "action": int(action)
                    if isinstance(action, (int, np.integer))
                    else serialized_observation,
                    "image": _frame_to_data_url(frame),
                    "observation": serialized_observation,
                    "q_values": None,
                }
            )

            if done:
                observation, info = env.reset()
                agent.restart()
                state = encode_observation(observation, env.observation_space)
                agent.states.append(state)
                accumulated_reward = 0.0
            else:
                agent.states.append(next_state)

    for frame_payload in frames:
        frame_payload["q_values"] = _state_q_values(
            runtime, frame_payload["observation"]
        )

    runtime["timestep"] = max_timesteps
    runtime["accumulated_reward"] = frames[-1]["accumulated_reward"]

    return {
        "frames": frames,
        "timestep": max_timesteps,
        "max_timesteps": max_timesteps,
        "accumulated_reward": frames[-1]["accumulated_reward"],
    }


def get_q_table(runtime: dict[str, Any]) -> dict:
    """Serialize the tabular agent's Q-table for analysis views (read-only)."""
    agent = runtime.get("agent")
    if agent is None:
        raise ValueError(
            "Q-table analysis requires a tabular agent (MC, SARSA, or Q-learning)."
        )
    if not hasattr(agent, "Q"):
        raise ValueError("The current agent does not expose a Q-table.")

    q_values = np.asarray(agent.Q, dtype=float)
    if q_values.ndim != 2:
        raise ValueError("Unexpected Q-table shape.")

    return {
        "n_states": int(q_values.shape[0]),
        "n_actions": int(q_values.shape[1]),
        "q_table": q_values.tolist(),
    }


def apply_agent_hyperparameters(runtime: dict[str, Any], config: dict) -> None:
    """Update in-memory tabular agent hyperparameters from the active config."""
    agent = runtime.get("agent")
    if agent is None:
        return
    agent.epsilon = float(config["exploration_probability"])
    agent.gamma = float(config.get("discount_factor", DEFAULT_GAMMA))
    if hasattr(agent, "alpha"):
        agent.alpha = float(config["learning_rate"])
    agent.parameters["epsilon"] = agent.epsilon
    agent.parameters["gamma"] = agent.gamma
    if "alpha" in agent.parameters:
        agent.parameters["alpha"] = float(config["learning_rate"])


def run_training_episode(
    runtime: dict[str, Any],
    max_timesteps: int = DEFAULT_MAX_TIMESTEPS,
) -> dict:
    """
    Run one training episode with the tabular agent.

    Uses the agent's policy, steps the environment, and updates Q-values.
    Visualization runners do not learn; only this training path does.
    """
    env = runtime["env"]
    agent = runtime.get("agent")
    if agent is None:
        raise ValueError("Training currently requires a tabular agent.")

    observation, info = env.reset()
    agent.restart()
    state = encode_observation(observation, env.observation_space)
    agent.states.append(state)

    episode_reward = 0.0
    steps = 0
    done = False

    for _ in range(max_timesteps):
        action = agent.make_decision()
        agent.actions.append(
            action if isinstance(action, (int, np.integer)) else int(action)
        )

        observation, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        next_state = encode_observation(observation, env.observation_space)

        agent.update(next_state, reward, done)
        agent.rewards.append(reward)
        agent.dones.append(done)

        episode_reward += float(reward)
        steps += 1

        if done:
            break
        agent.states.append(next_state)

    runtime["timestep"] = steps
    runtime["accumulated_reward"] = episode_reward

    return {
        "episode_reward": episode_reward,
        "steps": steps,
        "done": done,
        "max_timesteps": max_timesteps,
    }
