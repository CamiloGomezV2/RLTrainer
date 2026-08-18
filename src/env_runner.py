"""Gymnasium + tabular agent episode runner."""

from __future__ import annotations

import base64
import atexit
import io
import secrets
from typing import Any

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


def invalidate_runtime(session) -> None:
    sid = session.get("sid")
    if sid and sid in _RUNTIMES:
        runtime = _RUNTIMES.pop(sid)
        _close_env(runtime.get("env"))


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
    )

    if runtime is not None and runtime.get("fingerprint") == fingerprint:
        if need_agent and runtime.get("agent") is None:
            runtime["agent"] = create_agent(config["agent"], runtime["env"], config)
        return runtime

    if runtime is not None:
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
    """Take one environment step using the selected action (or the agent policy)."""
    env = runtime["env"]
    agent = runtime.get("agent")

    if agent is None:
        raise ValueError("Run an action currently requires a tabular agent.")

    # Start a fresh episode if needed or if the timestep budget was exhausted.
    if not agent.states or runtime.get("timestep", 0) >= DEFAULT_MAX_TIMESTEPS:
        reset_environment(runtime, seed=None)

    action = _resolve_action(runtime, action_choice)
    agent.actions.append(action if isinstance(action, (int, np.integer)) else int(action))

    observation, reward, terminated, truncated, info = env.step(action)
    done = bool(terminated or truncated)
    next_state = encode_observation(observation, env.observation_space)

    agent.update(next_state, reward, done)
    agent.rewards.append(reward)
    agent.dones.append(done)

    runtime["timestep"] = int(runtime.get("timestep", 0)) + 1
    runtime["accumulated_reward"] = float(runtime.get("accumulated_reward", 0.0)) + float(reward)

    frame = env.render()
    payload = {
        "timestep": runtime["timestep"],
        "max_timesteps": DEFAULT_MAX_TIMESTEPS,
        "accumulated_reward": runtime["accumulated_reward"],
        "reward": float(reward),
        "done": done,
        "action": int(action) if isinstance(action, (int, np.integer)) else _serialize_observation(action),
        "image": _frame_to_data_url(frame),
        "observation": _serialize_observation(observation),
    }

    if done:
        observation, info = env.reset()
        agent.restart()
        state = encode_observation(observation, env.observation_space)
        agent.states.append(state)
        runtime["accumulated_reward"] = 0.0
        payload["reset_after_done"] = True
        payload["next_image"] = _frame_to_data_url(env.render())
        payload["next_observation"] = _serialize_observation(observation)
    else:
        agent.states.append(next_state)
        payload["reset_after_done"] = False

    return payload


def run_until_max_timesteps(runtime: dict[str, Any], max_timesteps: int = DEFAULT_MAX_TIMESTEPS) -> dict:
    """
    Run up to max_timesteps environment steps with the selected tabular agent.
    When an episode ends (terminated or truncated), restart the env and agent episode buffers.
    """
    env = runtime["env"]
    agent = runtime.get("agent")
    if agent is None:
        raise ValueError("No tabular agent is available for this configuration.")

    initial = reset_environment(runtime, seed=None)
    frames = [
        {
            "timestep": 0,
            "accumulated_reward": 0.0,
            "reward": 0.0,
            "done": False,
            "image": initial["image"],
            "observation": initial["observation"],
        }
    ]

    accumulated_reward = 0.0

    for timestep in range(1, max_timesteps + 1):
        action = agent.make_decision()
        agent.actions.append(action)

        observation, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        next_state = encode_observation(observation, env.observation_space)

        agent.update(next_state, reward, done)
        agent.rewards.append(reward)
        agent.dones.append(done)

        accumulated_reward += float(reward)
        frame = env.render()
        frames.append(
            {
                "timestep": timestep,
                "accumulated_reward": accumulated_reward,
                "reward": float(reward),
                "done": done,
                "action": int(action),
                "image": _frame_to_data_url(frame),
                "observation": _serialize_observation(observation),
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

    Uses the same decision / step / update flow as the existing visualization
    runners, without collecting render frames.
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
