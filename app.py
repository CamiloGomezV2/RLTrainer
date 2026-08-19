import os
import sys
from pathlib import Path

# Allow Gymnasium rgb_array rendering without an interactive display.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flask import Flask, jsonify, render_template, request, session

from env_runner import (
    DEFAULT_GAMMA,
    DEFAULT_MAX_TIMESTEPS,
    DEFAULT_TRAINING_EPISODES,
    apply_agent_hyperparameters,
    get_or_create_runtime,
    get_q_table,
    reset_environment,
    reset_experiment_runtime,
    run_single_action,
    run_training_episode,
    run_until_max_timesteps,
)

app = Flask(__name__)
app.secret_key = "rl-trainer-dev-secret"


@app.after_request
def _disable_api_caching(response):
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
    return response

ENV_LABELS = {
    "Blackjack-v1": "Blackjack",
    "Taxi-v3": "Taxi",
    "FrozenLake-v1": "Frozen Lake",
    "CliffWalking-v0": "Cliff Walking",
    "Acrobot-v1": "Acrobot",
    "CartPole-v1": "CartPole",
    "MountainCar-v0": "Mountain Car",
    "MountainCarContinuous-v0": "Continuous Mountain Car",
    "Pendulum-v1": "Pendulum",
}

AGENT_LABELS = {
    "MC": "MC",
    "SARSA": "SARSA",
    "Q_learning": "Q-learning",
    "drl-sb3": "DRL agents from stable baselines 3",
}

# Observation / action space specs from Gymnasium docs
ENV_SPACES = {
    "Blackjack-v1": {
        "states": "704 (32 × 11 × 2)",
        "actions": "2",
        "action_space": {"type": "discrete", "n": 2},
    },
    "Taxi-v3": {
        "states": "500",
        "actions": "6",
        "action_space": {"type": "discrete", "n": 6},
    },
    "FrozenLake-v1": {
        "states": "16",
        "actions": "4",
        "action_space": {"type": "discrete", "n": 4},
    },
    "CliffWalking-v0": {
        "states": "48",
        "actions": "4",
        "action_space": {"type": "discrete", "n": 4},
    },
    "Acrobot-v1": {
        "states": (
            "Box(6,), "
            "low=[-1, -1, -1, -1, -12.57, -28.27], "
            "high=[1, 1, 1, 1, 12.57, 28.27]"
        ),
        "actions": "3",
        "action_space": {"type": "discrete", "n": 3},
    },
    "CartPole-v1": {
        "states": (
            "Box(4,), "
            "low=[-4.8, -∞, -0.4189, -∞], "
            "high=[4.8, ∞, 0.4189, ∞]"
        ),
        "actions": "2",
        "action_space": {"type": "discrete", "n": 2},
    },
    "MountainCar-v0": {
        "states": "Box(2,), low=[-1.2, -0.07], high=[0.6, 0.07]",
        "actions": "3",
        "action_space": {"type": "discrete", "n": 3},
    },
    "MountainCarContinuous-v0": {
        "states": "Box(2,), low=[-1.2, -0.07], high=[0.6, 0.07]",
        "actions": "Box(1,), low=-1.0, high=1.0",
        "action_space": {"type": "box", "low": -1.0, "high": 1.0, "shape": (1,)},
    },
    "Pendulum-v1": {
        "states": "Box(3,), low=[-1, -1, -8], high=[1, 1, 8]",
        "actions": "Box(1,), low=-2.0, high=2.0",
        "action_space": {"type": "box", "low": -2.0, "high": 2.0, "shape": (1,)},
    },
}

DEFAULT_CONFIG = {
    "environment": "FrozenLake-v1",
    "agent": "Q_learning",
    "learning_rate": 0.1,
    "exploration_probability": 0.1,
    "discount_factor": DEFAULT_GAMMA,
    "training_episodes": DEFAULT_TRAINING_EPISODES,
}


def get_config():
    config = dict(DEFAULT_CONFIG)
    config.update(session.get("config", {}))
    # Migrate older session values.
    if config.get("agent") == "tabular":
        config["agent"] = DEFAULT_CONFIG["agent"]
    if config.get("environment") == "CliffWalking-v1":
        config["environment"] = "CliffWalking-v0"
    return config


def config_for_template(config):
    env_id = config["environment"]
    agent_id = config["agent"]
    space = ENV_SPACES.get(env_id, {})
    return {
        **config,
        "environment_label": ENV_LABELS.get(env_id, env_id),
        "agent_label": AGENT_LABELS.get(agent_id, agent_id),
        "action_space": space.get("action_space"),
        "states": space.get("states", ""),
        "actions": space.get("actions", ""),
        "max_timesteps": DEFAULT_MAX_TIMESTEPS,
    }


@app.route("/")
def index():
    return render_template(
        "index.html",
        config=config_for_template(get_config()),
    )


@app.route("/load")
def load():
    return render_template(
        "load.html",
        env_spaces=ENV_SPACES,
        config=config_for_template(get_config()),
        default_config=DEFAULT_CONFIG,
    )


@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    if request.method == "GET":
        return jsonify(config_for_template(get_config()))

    data = request.get_json(silent=True) or {}
    environment = data.get("environment", DEFAULT_CONFIG["environment"])
    agent = data.get("agent", DEFAULT_CONFIG["agent"])

    if environment not in ENV_SPACES:
        return jsonify({"error": "Unknown environment"}), 400
    if agent not in AGENT_LABELS:
        return jsonify({"error": "Unknown agent"}), 400

    try:
        learning_rate = float(data.get("learning_rate", DEFAULT_CONFIG["learning_rate"]))
        exploration_probability = float(
            data.get(
                "exploration_probability",
                DEFAULT_CONFIG["exploration_probability"],
            )
        )
        discount_factor = float(
            data.get("discount_factor", DEFAULT_CONFIG["discount_factor"])
        )
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid hyperparameter values"}), 400

    current = get_config()
    session["config"] = {
        "environment": environment,
        "agent": agent,
        "learning_rate": learning_rate,
        "exploration_probability": exploration_probability,
        "discount_factor": discount_factor,
        "training_episodes": current.get(
            "training_episodes", DEFAULT_CONFIG["training_episodes"]
        ),
    }
    reset_experiment_runtime(session)
    return jsonify(config_for_template(session["config"]))


@app.route("/api/config/reset", methods=["POST"])
def api_config_reset():
    session["config"] = dict(DEFAULT_CONFIG)
    reset_experiment_runtime(session)
    return jsonify(config_for_template(session["config"]))


@app.route("/api/environment/initial")
def api_environment_initial():
    config = get_config()
    env_id = config["environment"]
    if env_id not in ENV_SPACES:
        return jsonify({"error": "Unknown environment"}), 400
    try:
        runtime = get_or_create_runtime(session, config)
        payload = reset_environment(runtime, seed=0)
    except Exception as exc:  # noqa: BLE001 - surface Gymnasium / agent errors to the UI
        return jsonify({"error": str(exc)}), 500
    payload["environment"] = env_id
    payload["environment_label"] = ENV_LABELS.get(env_id, env_id)
    return jsonify(payload)


@app.route("/api/environment/run-episode", methods=["POST"])
def api_environment_run_episode():
    config = get_config()
    if config["agent"] not in {"MC", "SARSA", "Q_learning"}:
        return jsonify(
            {
                "error": (
                    "Run episode currently supports tabular agents from "
                    "TableAgents.py (MC, SARSA, Q-learning)."
                )
            }
        ), 400
    data = request.get_json(silent=True) or {}
    action_choice = data.get("action", "policy")
    if action_choice == "manual":
        action_choice = data.get("manual_action")

    try:
        runtime = get_or_create_runtime(session, config, need_agent=True)
        payload = run_until_max_timesteps(
            runtime, DEFAULT_MAX_TIMESTEPS, action_choice
        )
    except Exception as exc:  # noqa: BLE001 - surface Gymnasium / agent errors to the UI
        return jsonify({"error": str(exc)}), 500
    return jsonify(payload)


@app.route("/api/environment/run-action", methods=["POST"])
def api_environment_run_action():
    config = get_config()
    if config["agent"] not in {"MC", "SARSA", "Q_learning"}:
        return jsonify(
            {
                "error": (
                    "Run an action currently supports tabular agents from "
                    "TableAgents.py (MC, SARSA, Q-learning)."
                )
            }
        ), 400

    data = request.get_json(silent=True) or {}
    action_choice = data.get("action", "policy")
    if action_choice == "manual":
        action_choice = data.get("manual_action")

    try:
        runtime = get_or_create_runtime(session, config, need_agent=True)
        payload = run_single_action(runtime, action_choice)
    except Exception as exc:  # noqa: BLE001 - surface Gymnasium / agent errors to the UI
        return jsonify({"error": str(exc)}), 500
    return jsonify(payload)


@app.route("/api/agent/q-table")
def api_agent_q_table():
    config = get_config()
    if config["agent"] not in {"MC", "SARSA", "Q_learning"}:
        return jsonify(
            {
                "error": (
                    "Q-table analysis currently supports tabular agents "
                    "(MC, SARSA, Q-learning)."
                )
            }
        ), 400
    try:
        runtime = get_or_create_runtime(session, config, need_agent=True)
        payload = get_q_table(runtime)
    except Exception as exc:  # noqa: BLE001 - surface agent errors to the UI
        return jsonify({"error": str(exc)}), 500
    payload["environment"] = config["environment"]
    payload["environment_label"] = ENV_LABELS.get(
        config["environment"], config["environment"]
    )
    payload["agent"] = config["agent"]
    payload["agent_label"] = AGENT_LABELS.get(config["agent"], config["agent"])
    return jsonify(payload)


@app.route("/api/training/start", methods=["POST"])
def api_training_start():
    config = get_config()
    if config["agent"] not in {"MC", "SARSA", "Q_learning"}:
        return jsonify(
            {
                "error": (
                    "Training currently supports tabular agents "
                    "(MC, SARSA, Q-learning)."
                )
            }
        ), 400

    data = request.get_json(silent=True) or {}
    try:
        episodes = int(data.get("episodes", config.get("training_episodes", DEFAULT_TRAINING_EPISODES)))
        learning_rate = float(data.get("learning_rate", config["learning_rate"]))
        exploration_probability = float(
            data.get("exploration_probability", config["exploration_probability"])
        )
        discount_factor = float(data.get("discount_factor", config["discount_factor"]))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid training configuration values."}), 400

    if episodes < 1:
        return jsonify({"error": "Episodes must be at least 1."}), 400
    if not 0 <= learning_rate <= 1:
        return jsonify({"error": "Learning rate must be between 0 and 1."}), 400
    if not 0 <= exploration_probability <= 1:
        return jsonify({"error": "Exploration must be between 0 and 1."}), 400
    if not 0 <= discount_factor <= 1:
        return jsonify({"error": "Discount factor must be between 0 and 1."}), 400

    session["config"] = {
        **config,
        "learning_rate": learning_rate,
        "exploration_probability": exploration_probability,
        "discount_factor": discount_factor,
        "training_episodes": episodes,
    }
    reset_experiment_runtime(session)

    try:
        runtime = get_or_create_runtime(session, session["config"], need_agent=True)
        apply_agent_hyperparameters(runtime, session["config"])
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500

    return jsonify(
        {
            "status": "ready",
            "episodes": episodes,
            "learning_rate": learning_rate,
            "exploration_probability": exploration_probability,
            "discount_factor": discount_factor,
            "environment": session["config"]["environment"],
            "environment_label": ENV_LABELS.get(
                session["config"]["environment"], session["config"]["environment"]
            ),
            "agent": session["config"]["agent"],
            "agent_label": AGENT_LABELS.get(
                session["config"]["agent"], session["config"]["agent"]
            ),
        }
    )


@app.route("/api/training/episode", methods=["POST"])
def api_training_episode():
    config = get_config()
    if config["agent"] not in {"MC", "SARSA", "Q_learning"}:
        return jsonify(
            {
                "error": (
                    "Training currently supports tabular agents "
                    "(MC, SARSA, Q-learning)."
                )
            }
        ), 400

    data = request.get_json(silent=True) or {}
    try:
        episode = int(data.get("episode", 1))
        total_episodes = int(
            data.get("total_episodes", config.get("training_episodes", DEFAULT_TRAINING_EPISODES))
        )
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid episode progress values."}), 400

    try:
        runtime = get_or_create_runtime(session, config, need_agent=True)
        apply_agent_hyperparameters(runtime, config)
        payload = run_training_episode(runtime, DEFAULT_MAX_TIMESTEPS)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500

    payload.update(
        {
            "episode": episode,
            "total_episodes": total_episodes,
            "status": "completed" if episode >= total_episodes else "training",
        }
    )
    return jsonify(payload)


@app.route("/visualize-environment")
def visualize_environment():
    return render_template(
        "visualize_environment.html",
        config=config_for_template(get_config()),
    )


@app.route("/visualize-q-table")
def visualize_q_table():
    return render_template(
        "agent_analysis.html",
        config=config_for_template(get_config()),
    )


@app.route("/training")
def training():
    return render_template(
        "training.html",
        config=config_for_template(get_config()),
        default_episodes=DEFAULT_TRAINING_EPISODES,
    )


if __name__ == "__main__":
    # use_reloader=False avoids multiprocessing semaphore leaks from Pygame/Gymnasium.
    app.run(debug=True, use_reloader=False)
