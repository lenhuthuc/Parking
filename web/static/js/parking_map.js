/**
 * parking_map.js
 * Draws a canvas-based 2-D parking map from slot polygons + state data.
 */

const ParkingMap = (() => {
  const COLORS = {
    TRỐNG:              { fill: "rgba(34,197,94,0.40)",  stroke: "#22c55e" },
    "CÓ XE":            { fill: "rgba(239,68,68,0.45)",  stroke: "#ef4444" },
    "KHÔNG XÁC ĐỊNH":   { fill: "rgba(249,115,22,0.40)", stroke: "#f97316" },
    _unknown:           { fill: "rgba(100,100,120,0.3)", stroke: "#6b7280" },
  };

  let _canvas = null;
  let _ctx    = null;
  let _layout = null;        // { parking_slots: [...], no_parking_zones: [...] }
  let _states = [];          // array of state strings, same order as _layout.parking_slots

  function init(canvasId) {
    _canvas = document.getElementById(canvasId);
    if (!_canvas) return;
    _ctx = _canvas.getContext("2d");
  }

  async function loadLayout(cameraId) {
    try {
      const slots = await fetch(`/api/v1/cameras/${cameraId}/slots/`)
                          .then(r => r.json());
      _layout = slots;
      // Seed initial states from the REST response so the map draws immediately
      _states = slots.map(s => s.state || "KHÔNG XÁC ĐỊNH");
    } catch(e) {
      console.warn("ParkingMap: could not load layout", e);
    }
  }

  function updateStates(states) {
    _states = states;
    draw();
  }

  function draw() {
    if (!_ctx || !_layout || _layout.length === 0) {
      _drawPlaceholder();
      return;
    }

    const W = _canvas.width;
    const H = _canvas.height;
    _ctx.clearRect(0, 0, W, H);

    // Background
    _ctx.fillStyle = "#1a1d27";
    _ctx.fillRect(0, 0, W, H);

    // Compute bounding box of all slot polygons to auto-scale
    const allPoints = _layout.flatMap(s => s.polygon || []);
    if (allPoints.length === 0) { _drawPlaceholder(); return; }

    const xs = allPoints.map(p => p[0]);
    const ys = allPoints.map(p => p[1]);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMin = Math.min(...ys), yMax = Math.max(...ys);

    const pad = 30;
    const scaleX = (W - pad * 2) / (xMax - xMin || 1);
    const scaleY = (H - pad * 2) / (yMax - yMin || 1);
    const scale  = Math.min(scaleX, scaleY);

    const tx = px => pad + (px - xMin) * scale;
    const ty = py => pad + (py - yMin) * scale;

    // Draw slots
    _layout.forEach((slot, i) => {
      const state  = _states[i] || "KHÔNG XÁC ĐỊNH";
      const colors = COLORS[state] || COLORS._unknown;
      const pts    = slot.polygon;

      _ctx.beginPath();
      _ctx.moveTo(tx(pts[0][0]), ty(pts[0][1]));
      for (let j = 1; j < pts.length; j++)
        _ctx.lineTo(tx(pts[j][0]), ty(pts[j][1]));
      _ctx.closePath();

      _ctx.fillStyle = colors.fill;
      _ctx.fill();
      _ctx.strokeStyle = colors.stroke;
      _ctx.lineWidth = 2;
      _ctx.stroke();

      // Label
      const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
      const cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
      _ctx.fillStyle = "#e2e8f0";
      _ctx.font = `bold ${Math.max(9, Math.round(12 * scale))}px sans-serif`;
      _ctx.textAlign = "center";
      _ctx.textBaseline = "middle";
      _ctx.fillText(slot.slot_id, tx(cx), ty(cy));
    });
  }

  function _drawPlaceholder() {
    if (!_ctx) return;
    const W = _canvas.width, H = _canvas.height;
    _ctx.fillStyle = "#1a1d27";
    _ctx.fillRect(0, 0, W, H);
    _ctx.fillStyle = "#4b5563";
    _ctx.font = "16px sans-serif";
    _ctx.textAlign = "center";
    _ctx.textBaseline = "middle";
    _ctx.fillText("Chọn camera để xem sơ đồ", W / 2, H / 2);
  }

  return { init, loadLayout, updateStates, draw };
})();


// ─── Global hook called from dashboard.html ──────────────────────

async function updateParkingMap(states, cameraId) {
  if (!window._mapInitDone) {
    ParkingMap.init("parking-map");
    window._mapInitDone = true;
  }
  if (window._mapCameraId !== cameraId) {
    await ParkingMap.loadLayout(cameraId);
    window._mapCameraId = cameraId;
  }
  ParkingMap.updateStates(states);
}
