/**
 * charts.js — Chart.js wrappers for the statistics page.
 */

let _hourlyChart = null;

function drawHourlyChart(stats) {
  const canvas = document.getElementById("hourly-chart");
  if (!canvas) return;

  const labels    = stats.map(r => {
    const d = new Date(r.hour + "Z");
    return d.toLocaleString("vi-VN", {month:"2-digit", day:"2-digit",
                                       hour:"2-digit", minute:"2-digit"});
  });
  const occupied  = stats.map(r => r.occupied_pct);
  const empty     = stats.map(r => r.empty_pct);
  const unknown   = stats.map(r => r.unknown_pct);

  const chartData = {
    labels,
    datasets: [
      {
        label: "Có xe (%)",
        data: occupied,
        borderColor: "#ef4444",
        backgroundColor: "rgba(239,68,68,0.15)",
        tension: 0.3, fill: true, pointRadius: 2,
      },
      {
        label: "Trống (%)",
        data: empty,
        borderColor: "#22c55e",
        backgroundColor: "rgba(34,197,94,0.12)",
        tension: 0.3, fill: true, pointRadius: 2,
      },
      {
        label: "Không xác định (%)",
        data: unknown,
        borderColor: "#f97316",
        backgroundColor: "rgba(249,115,22,0.10)",
        tension: 0.3, fill: true, pointRadius: 2,
      },
    ],
  };

  if (_hourlyChart) {
    _hourlyChart.data = chartData;
    _hourlyChart.update();
    return;
  }

  _hourlyChart = new Chart(canvas.getContext("2d"), {
    type: "line",
    data: chartData,
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: "#8892a4" } },
        tooltip: { mode: "index", intersect: false },
      },
      scales: {
        x: {
          ticks: { color: "#8892a4", maxTicksLimit: 12 },
          grid:  { color: "rgba(255,255,255,0.05)" },
        },
        y: {
          min: 0, max: 100,
          ticks: { color: "#8892a4", callback: v => v + "%" },
          grid:  { color: "rgba(255,255,255,0.05)" },
        },
      },
    },
  });
}


// ── Mini occupancy chart on dashboard ───────────────────────────

let _miniChart = null;

async function loadOccupancyChart(cameraId) {
  const canvas = document.getElementById("occupancy-chart");
  if (!canvas) return;

  const stats = await fetch(
    `/api/v1/stats/cameras/${cameraId}/hourly?since_hours=24`
  ).then(r => r.json()).catch(() => []);

  const labels   = stats.map(r => new Date(r.hour + "Z")
                                    .toLocaleTimeString("vi-VN",
                                      {hour:"2-digit", minute:"2-digit"}));
  const occupied = stats.map(r => r.occupied_pct);

  if (_miniChart) {
    _miniChart.data.labels = labels;
    _miniChart.data.datasets[0].data = occupied;
    _miniChart.update();
    return;
  }

  _miniChart = new Chart(canvas.getContext("2d"), {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "Có xe (%)",
        data: occupied,
        borderColor: "#3b82f6",
        backgroundColor: "rgba(59,130,246,0.15)",
        tension: 0.3, fill: true, pointRadius: 0,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#8892a4", maxTicksLimit: 8 },
             grid:  { color: "rgba(255,255,255,0.05)" } },
        y: { min: 0, max: 100,
             ticks: { color: "#8892a4", callback: v => v + "%" },
             grid:  { color: "rgba(255,255,255,0.05)" } },
      },
    },
  });
}
