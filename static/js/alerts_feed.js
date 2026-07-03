/*
 * Shared live PPE alert feed poller.
 * Used by webcam / cctv / video monitoring pages.
 * Renders into #alertContainer using the themed .alert-card component.
 */
(function () {
  const container = document.getElementById("alertContainer");
  if (!container) return;

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function fmtType(t) {
    return (t || "PPE violation").replace(/_/g, " ");
  }
  function shortId(id) {
    id = String(id || "Unknown");
    return id.length > 8 ? "…" + id.slice(-6) : id;
  }
  function fmtTime(ts) {
    if (!ts) return "";
    var m = String(ts).match(/T(\d{2}:\d{2}:\d{2})/);
    return m ? m[1] : String(ts).slice(0, 19);
  }

  function render(alerts) {
    if (!alerts || !alerts.length) {
      container.innerHTML =
        '<div class="empty-state" style="padding:30px 10px"><p>No recent violations.</p></div>';
      return;
    }
    container.innerHTML = alerts
      .map(function (a) {
        const thumb = a.snapshot
          ? '<img src="' + esc(a.snapshot) + '" class="alert-thumb">'
          : '<div class="alert-thumb ph">!</div>';
        const missing = (a.missing || []);
        const chips = missing.map(function (m) { return '<span class="miss-chip">✕ ' + esc(fmtType(m)) + "</span>"; }).join("");
        return (
          '<div class="alert-card is-ongoing">' +
          thumb +
          '<div class="alert-body">' +
          '<div class="miss-row">' + chips + "</div>" +
          '<div class="alert-meta">' + esc(shortId(a.track_id)) + "  ·  " + esc(fmtTime(a.created_at)) + "</div>" +
          "</div>" +
          "</div>"
        );
      })
      .join("");
  }

  var lastSig = null;

  function signature(alerts) {
    return (alerts || [])
      .map(function (a) { return [a.track_id, (a.missing || []).join(","), a.created_at, a.snapshot].join("|"); })
      .join("~");
  }

  async function load() {
    try {
      const res = await fetch("/api/latest-alerts");
      const alerts = await res.json();
      // Only re-render when the alert set actually changed — otherwise every
      // poll rebuilds the DOM and forces all thumbnails to reload, which looks
      // like flickering / wrong images.
      const sig = signature(alerts);
      if (sig === lastSig) return;
      lastSig = sig;
      render(alerts);
    } catch (err) {
      console.error(err);
    }
  }

  load();
  setInterval(load, 3000);
})();
