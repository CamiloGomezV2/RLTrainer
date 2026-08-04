# RL Trainer

A Flask web app for visualizing reinforcement learning agents in Gymnasium environments. Choose an environment and tabular agent (MC, SARSA, or Q-learning), set hyperparameters, then step through or run episodes while watching the environment render.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser.
