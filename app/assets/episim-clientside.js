window.dash_clientside = Object.assign({}, window.dash_clientside, {
  episim: {}
});
window.episimPerf = window.episimPerf || {};

(function () {
  const colors = {
    Susceptible: "#f4efe8",
    Exposed: "#ff9f43",
    Infectious: "#ff3b4d",
    Recovered: "#5ed6ff"
  };

  function coalesce(dragValue, releaseValue) {
    return dragValue == null ? releaseValue : dragValue;
  }

  function downsample(values, maxPoints) {
    if (!maxPoints || values.length <= maxPoints) {
      return values;
    }
    const step = Math.ceil(values.length / maxPoints);
    const result = [];
    for (let index = 0; index < values.length; index += step) {
      result.push(values[index]);
    }
    if (result[result.length - 1] !== values[values.length - 1]) {
      result.push(values[values.length - 1]);
    }
    return result;
  }

  function downsampleSeries(time, compartments, maxPoints) {
    if (!maxPoints || time.length <= maxPoints) {
      return { time, compartments };
    }
    const step = Math.ceil(time.length / maxPoints);
    const indices = [];
    for (let index = 0; index < time.length; index += step) {
      indices.push(index);
    }
    if (indices[indices.length - 1] !== time.length - 1) {
      indices.push(time.length - 1);
    }
    const sampledCompartments = {};
    Object.entries(compartments).forEach(([name, values]) => {
      sampledCompartments[name] = indices.map((index) => values[index]);
    });
    return {
      time: indices.map((index) => time[index]),
      compartments: sampledCompartments
    };
  }

  function applyIntervention(beta, time, interventionDay, interventionStrength) {
    if (interventionDay == null || time < interventionDay) {
      return beta;
    }
    return beta * (1 - interventionStrength);
  }

  function rk4SIR(population, initialInfected, beta, gamma, days, interventionDay, interventionStrength) {
    const dt = 0.25;
    const steps = Math.ceil(days / dt);
    const time = new Array(steps + 1);
    const susceptible = new Array(steps + 1);
    const infectious = new Array(steps + 1);
    const recovered = new Array(steps + 1);

    let s = population - initialInfected;
    let i = initialInfected;
    let r = 0;

    for (let step = 0; step <= steps; step += 1) {
      time[step] = step * dt;
      susceptible[step] = s;
      infectious[step] = i;
      recovered[step] = r;

      if (step === steps) {
        break;
      }

      const deriv = (t, state) => {
        const [currentS, currentI, currentR] = state;
        const effectiveBeta = applyIntervention(beta, t, interventionDay, interventionStrength);
        const force = effectiveBeta * currentS * currentI / population;
        return [-force, force - gamma * currentI, gamma * currentI];
      };

      const state = [s, i, r];
      const k1 = deriv(time[step], state);
      const k2 = deriv(time[step] + dt / 2, state.map((value, index) => value + dt * k1[index] / 2));
      const k3 = deriv(time[step] + dt / 2, state.map((value, index) => value + dt * k2[index] / 2));
      const k4 = deriv(time[step] + dt, state.map((value, index) => value + dt * k3[index]));

      s = Math.max(0, s + dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6);
      i = Math.max(0, i + dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6);
      r = Math.max(0, r + dt * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6);
    }

    return {
      modelName: "SIR",
      population,
      time,
      compartments: {
        Susceptible: susceptible,
        Infectious: infectious,
        Recovered: recovered
      }
    };
  }

  function rk4SEIR(population, initialInfected, initialExposed, beta, sigma, gamma, days, interventionDay, interventionStrength) {
    const dt = 0.25;
    const steps = Math.ceil(days / dt);
    const time = new Array(steps + 1);
    const susceptible = new Array(steps + 1);
    const exposed = new Array(steps + 1);
    const infectious = new Array(steps + 1);
    const recovered = new Array(steps + 1);

    let s = population - initialInfected - initialExposed;
    let e = initialExposed;
    let i = initialInfected;
    let r = 0;

    for (let step = 0; step <= steps; step += 1) {
      time[step] = step * dt;
      susceptible[step] = s;
      exposed[step] = e;
      infectious[step] = i;
      recovered[step] = r;

      if (step === steps) {
        break;
      }

      const deriv = (t, state) => {
        const [currentS, currentE, currentI, currentR] = state;
        const effectiveBeta = applyIntervention(beta, t, interventionDay, interventionStrength);
        const force = effectiveBeta * currentS * currentI / population;
        return [
          -force,
          force - sigma * currentE,
          sigma * currentE - gamma * currentI,
          gamma * currentI
        ];
      };

      const state = [s, e, i, r];
      const k1 = deriv(time[step], state);
      const k2 = deriv(time[step] + dt / 2, state.map((value, index) => value + dt * k1[index] / 2));
      const k3 = deriv(time[step] + dt / 2, state.map((value, index) => value + dt * k2[index] / 2));
      const k4 = deriv(time[step] + dt, state.map((value, index) => value + dt * k3[index]));

      s = Math.max(0, s + dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6);
      e = Math.max(0, e + dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6);
      i = Math.max(0, i + dt * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6);
      r = Math.max(0, r + dt * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) / 6);
    }

    return {
      modelName: "SEIR",
      population,
      time,
      compartments: {
        Susceptible: susceptible,
        Exposed: exposed,
        Infectious: infectious,
        Recovered: recovered
      }
    };
  }

  function summarize(result) {
    const infectious = result.compartments.Infectious;
    const susceptible = result.compartments.Susceptible;
    let peakIndex = 0;
    for (let index = 1; index < infectious.length; index += 1) {
      if (infectious[index] > infectious[peakIndex]) {
        peakIndex = index;
      }
    }

    let extinctionIndex = null;
    for (let index = peakIndex; index < infectious.length; index += 1) {
      if (infectious[index] <= 1) {
        extinctionIndex = index;
        break;
      }
    }

    const finalOutbreakSize = result.population - susceptible[susceptible.length - 1];
    return {
      peakInfectious: infectious[peakIndex],
      peakDay: result.time[peakIndex],
      finalOutbreakSize,
      finalOutbreakShare: finalOutbreakSize / result.population,
      timeToExtinction: extinctionIndex == null ? null : result.time[extinctionIndex]
    };
  }

  function formatMetrics(summary) {
    return [
      Math.round(summary.peakInfectious).toLocaleString(),
      summary.peakDay.toFixed(1),
      Math.round(summary.finalOutbreakSize).toLocaleString(),
      `${(summary.finalOutbreakShare * 100).toFixed(1)}%`,
      summary.timeToExtinction == null ? "Not reached" : `${summary.timeToExtinction.toFixed(1)} days`
    ];
  }

  function buildFigure(result) {
    const summary = summarize(result);
    const sampled = downsampleSeries(result.time, result.compartments, 181);
    const data = Object.entries(sampled.compartments).map(([name, values]) => ({
      type: "scattergl",
      x: sampled.time,
      y: values,
      mode: "lines",
      name,
      line: { width: 3, color: colors[name] || "#f4efe8" }
    }));

    return {
      data,
      layout: {
        template: "plotly_dark",
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { family: "IBM Plex Sans, sans-serif", color: "#f5f0ea" },
        margin: { l: 20, r: 20, t: 92, b: 24 },
        height: 500,
        title: {
          text: `${result.modelName} compartment dynamics`,
          x: 0.02,
          y: 0.97,
          pad: { b: 22 }
        },
        legend: {
          orientation: "h",
          yanchor: "bottom",
          y: 1.01,
          xanchor: "left",
          x: 0
        },
        xaxis: { title: "Day", gridcolor: "rgba(255,255,255,0.08)", zeroline: false },
        yaxis: { title: "People", gridcolor: "rgba(255,255,255,0.08)", zeroline: false },
        shapes: [{
          type: "line",
          xref: "x",
          yref: "paper",
          x0: summary.peakDay,
          x1: summary.peakDay,
          y0: 0,
          y1: 1,
          line: { dash: "dash", color: "#f4efe8", width: 1 }
        }],
        annotations: [{
          x: summary.peakDay,
          y: 1.03,
          xref: "x",
          yref: "paper",
          text: `Peak day ${summary.peakDay.toFixed(1)}`,
          showarrow: false,
          font: { size: 12, color: "#f4efe8" }
        }]
      }
    };
  }

  function buildSirLivePayload(population, initialInfected, beta, gamma, days, interventionDay, interventionStrength) {
    const result = rk4SIR(population, initialInfected, beta, gamma, days, interventionDay, interventionStrength);
    const summary = summarize(result);
    return {
      figure: buildFigure(result),
      metrics: formatMetrics(summary)
    };
  }

  function buildSeirLivePayload(population, initialInfected, initialExposed, beta, sigma, gamma, days, interventionDay, interventionStrength) {
    const result = rk4SEIR(population, initialInfected, initialExposed, beta, sigma, gamma, days, interventionDay, interventionStrength);
    const summary = summarize(result);
    return {
      figure: buildFigure(result),
      metrics: formatMetrics(summary)
    };
  }

  window.episimPerf.buildSirLivePayload = buildSirLivePayload;
  window.episimPerf.buildSeirLivePayload = buildSeirLivePayload;

  window.dash_clientside.episim.updateSirLive = function (
    populationDrag, populationValue,
    infectedDrag, infectedValue,
    betaDrag, betaValue,
    gammaDrag, gammaValue,
    daysDrag, daysValue,
    dayDrag, dayValue,
    strengthDrag, strengthValue
  ) {
    const payload = buildSirLivePayload(
      coalesce(populationDrag, populationValue),
      coalesce(infectedDrag, infectedValue),
      coalesce(betaDrag, betaValue),
      coalesce(gammaDrag, gammaValue),
      coalesce(daysDrag, daysValue),
      coalesce(dayDrag, dayValue),
      coalesce(strengthDrag, strengthValue)
    );
    return [payload.figure].concat(payload.metrics);
  };

  window.dash_clientside.episim.updateSeirLive = function (
    populationDrag, populationValue,
    infectedDrag, infectedValue,
    exposedDrag, exposedValue,
    betaDrag, betaValue,
    sigmaDrag, sigmaValue,
    gammaDrag, gammaValue,
    daysDrag, daysValue,
    dayDrag, dayValue,
    strengthDrag, strengthValue
  ) {
    const payload = buildSeirLivePayload(
      coalesce(populationDrag, populationValue),
      coalesce(infectedDrag, infectedValue),
      coalesce(exposedDrag, exposedValue),
      coalesce(betaDrag, betaValue),
      coalesce(sigmaDrag, sigmaValue),
      coalesce(gammaDrag, gammaValue),
      coalesce(daysDrag, daysValue),
      coalesce(dayDrag, dayValue),
      coalesce(strengthDrag, strengthValue)
    );
    return [payload.figure].concat(payload.metrics);
  };
})();
