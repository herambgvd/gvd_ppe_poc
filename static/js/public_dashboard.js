/*
# ======================================
# PUBLIC_DASHBOARD.JS — wall display client
# ======================================
Self-contained realtime client for /public (does NOT use dashboard.js — the wall
page has its own minimal DOM). Rides the same Socket.IO events the operator
dashboard uses: metrics_update (KPIs + trend), active_violation (alerts feed +
spoken alert via tts.js), active_events_snapshot (initial feed fill).
*/
(function () {
  const socket = io();
  let trendChart = null;

  /* ---------- clock ---------- */
  function tickClock() {
    const el = document.getElementById('clock');
    if (el) el.textContent = new Date().toLocaleString();
  }
  setInterval(tickClock, 1000); tickClock();

  /* ---------- KPIs + trend ---------- */
  function setText(id, v) { const el = document.getElementById(id); if (el) el.textContent = v; }

  function updateAnalytics(a) {
    if (!a) return;
    if (a.violations_today !== undefined) setText('kpiToday', a.violations_today);
    setText('kpiActive', a.active_events);
    setText('kpiCompliance', a.compliance_pct + '%');
    renderTrend(a.timeline || []);
  }

  function renderTrend(timeline) {
    const canvas = document.getElementById('trendChart');
    if (!canvas || typeof Chart === 'undefined') return;
    const labels = timeline.map(x => (x.bucket || '').slice(11, 16));   // HH:MM
    const counts = timeline.map(x => x.count);
    if (!trendChart) {
      trendChart = new Chart(canvas, {
        type: 'line',
        data: { labels: labels, datasets: [{ label: 'Violations', data: counts, tension: 0.35, borderColor: '#ff6b6b', backgroundColor: 'rgba(255,107,107,.15)', fill: true, pointRadius: 0 }] },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { ticks: { color: '#777', maxTicksLimit: 8 }, grid: { color: '#1e1e1e' } },
            y: { ticks: { color: '#777', precision: 0 }, grid: { color: '#1e1e1e' }, beginAtZero: true }
          }
        }
      });
    } else {
      trendChart.data.labels = labels;
      trendChart.data.datasets[0].data = counts;
      trendChart.update('none');
    }
  }

  /* ---------- alerts feed ---------- */
  const MAX_ALERTS = 30;
  function alertCard(e) {
    const div = document.createElement('div');
    div.className = 'alert-item';
    const vtype = String(e.violation_type || 'violation').replace(/_/g, ' ');
    const cam = e.camera_id || '-';
    const ts = (e.timestamp_start || e.created_at || '').slice(11, 19);
    div.innerHTML = `<b>${vtype}</b> · ${e.state || ''}<div class="meta">Camera ${cam} · ${ts}</div>`;
    return div;
  }

  function prependAlert(e) {
    const list = document.getElementById('alerts');
    if (!list) return;
    if (list.querySelector('.muted')) list.innerHTML = '';
    list.prepend(alertCard(e));
    while (list.children.length > MAX_ALERTS) list.removeChild(list.lastChild);
  }

  function fillAlerts(events) {
    const list = document.getElementById('alerts');
    if (!list) return;
    list.innerHTML = '';
    const live = (events || []).filter(e => ['NEW', 'ACTIVE'].includes(e.state));
    if (!live.length) { list.innerHTML = '<p class="muted">No active violations 🎉</p>'; return; }
    live.slice(0, MAX_ALERTS).forEach(e => list.appendChild(alertCard(e)));
  }

  /* ---------- socket wiring ---------- */
  socket.on('connect', () => socket.emit('request_metrics'));
  socket.on('metrics_update', (p) => { if (p && p.analytics) updateAnalytics(p.analytics); });
  socket.on('active_events_snapshot', (events) => fillAlerts(events || []));
  socket.on('active_violation', (event) => {
    prependAlert(event);
    if (window.PPE_TTS) PPE_TTS.onViolation(event);   // spoken alert (NEW only, throttled)
  });

  /* initial paint from server-rendered analytics */
  window.addEventListener('load', () => { if (window.initialAnalytics) updateAnalytics(window.initialAnalytics); });
})();
