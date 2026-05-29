/**
 * dashboard.js — shared utilities loaded on every page.
 */

// ── Toast notifications ─────────────────────────────────────────
function showToast(message, type = "info") {
  const existing = document.getElementById("toast-container");
  const container = existing || (() => {
    const el = document.createElement("div");
    el.id = "toast-container";
    Object.assign(el.style, {
      position: "fixed", bottom: "1.5rem", right: "1.5rem",
      display: "flex", flexDirection: "column", gap: "0.5rem",
      zIndex: "9999",
    });
    document.body.appendChild(el);
    return el;
  })();

  const toast = document.createElement("div");
  const colors = { info: "#3b82f6", success: "#22c55e",
                   warning: "#f97316", error: "#ef4444" };
  Object.assign(toast.style, {
    background: "#252836",
    borderLeft: `4px solid ${colors[type] || colors.info}`,
    color: "#e2e8f0",
    padding: "10px 16px",
    borderRadius: "8px",
    boxShadow: "0 4px 20px rgba(0,0,0,0.4)",
    fontSize: "0.85rem",
    maxWidth: "320px",
    opacity: "0",
    transition: "opacity 0.3s",
  });
  toast.textContent = message;
  container.appendChild(toast);
  requestAnimationFrame(() => { toast.style.opacity = "1"; });

  setTimeout(() => {
    toast.style.opacity = "0";
    setTimeout(() => toast.remove(), 350);
  }, 4000);
}


// ── Date/time helpers ───────────────────────────────────────────

function formatDateVN(isoString) {
  return new Date(isoString + (isoString.endsWith("Z") ? "" : "Z"))
    .toLocaleString("vi-VN", {
      year: "numeric", month: "2-digit", day: "2-digit",
      hour: "2-digit", minute: "2-digit", second: "2-digit",
    });
}


// ── Periodic page refresh for static pages ──────────────────────

function autoRefresh(intervalMs, callback) {
  callback();
  return setInterval(callback, intervalMs);
}


// ── Keyboard shortcut: Alt+D → dashboard etc. ───────────────────

document.addEventListener("keydown", e => {
  if (!e.altKey) return;
  const map = { d: "/", v: "/violations", s: "/statistics", c: "/settings" };
  if (map[e.key]) { e.preventDefault(); location.href = map[e.key]; }
});
