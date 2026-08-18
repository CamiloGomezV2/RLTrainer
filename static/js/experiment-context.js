(function () {
  const root = document.getElementById("experiment-context");
  if (!root) {
    window.ExperimentContext = {
      setStatus: function () {},
      setEnvironment: function () {},
      setAgent: function () {},
    };
    return;
  }

  const statusEl = document.getElementById("experiment-status");
  const statusLabel = statusEl ? statusEl.querySelector(".status-badge__label") : null;
  const environmentEl = document.getElementById("experiment-environment");
  const agentEl = document.getElementById("experiment-agent");

  function setStatus(state, label) {
    if (!statusEl || !statusLabel) {
      return;
    }
    statusEl.dataset.state = state;
    statusLabel.textContent = label;
  }

  function setEnvironment(environmentId, environmentLabel) {
    if (!environmentEl || !environmentId) {
      return;
    }
    environmentEl.textContent = environmentId;
    if (environmentLabel) {
      environmentEl.title = environmentLabel;
    }
    root.dataset.environment = environmentId;
  }

  function setAgent(agentId, agentLabel) {
    if (!agentEl) {
      return;
    }
    if (agentLabel) {
      agentEl.textContent = agentLabel;
    } else if (agentId) {
      agentEl.textContent = agentId;
    }
    if (agentId) {
      agentEl.title = agentId;
      root.dataset.agent = agentId;
    }
  }

  window.ExperimentContext = {
    setStatus,
    setEnvironment,
    setAgent,
  };
})();
