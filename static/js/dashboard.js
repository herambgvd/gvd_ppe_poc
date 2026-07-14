/*
# ======================================
# DASHBOARD.JS
# ======================================

# ======================================
# PURPOSE
# ======================================
Explain:
- This file handles Socket.IO connection, dashboard metric updates, camera start/stop actions, and Chart.js rendering.
- It solves the real-time UI requirement without polling every page refresh.
- Enterprise frontends keep live update code modular so future React/Vue migration can reuse the same backend events.
*/
const socket = io();
let timelineChart = null;
let typeChart = null;

socket.on('connect', () => {
  const el = document.getElementById('socketStatus');
  if (el) { el.textContent = 'Connected'; el.classList.remove('danger'); el.classList.add('success'); }
  socket.emit('request_metrics');
});

socket.on('disconnect', () => {
  const el = document.getElementById('socketStatus');
  if (el) { el.textContent = 'Disconnected'; el.classList.remove('success'); el.classList.add('danger'); }
});

socket.on('metrics_update', (payload) => {
  if (payload.analytics) updateAnalytics(payload.analytics);
});

socket.on('active_events_snapshot', (events) => {
  renderActiveEvents(events || []);
});

socket.on('active_violation', (event) => {
  prependActiveEvent(event);
  // Spoken alert (tts.js): speaks only state === "NEW" incidents, throttled + mutable.
  if (window.PPE_TTS) PPE_TTS.onViolation(event);
});

socket.on('camera_status', (payload) => {
  console.log('camera_status', payload);
});

/* ---------- Toast feedback ---------- */
function showToast(msg, kind) {
  let t = document.getElementById('appToast');
  if (!t) { t = document.createElement('div'); t.id = 'appToast'; document.body.appendChild(t); }
  t.className = 'toast show ' + (kind || '');
  t.textContent = msg;
  clearTimeout(t._timer);
  t._timer = setTimeout(() => { t.className = 'toast ' + (kind || ''); }, 2800);
}

/* ---------- Single toggle button (Start <-> Stop) ---------- */
function _applyToggleState(btn, running) {
  btn.dataset.state = running ? 'running' : 'stopped';
  btn.classList.toggle('is-on', running);
  btn.innerHTML = '<span class="tg-dot"></span> ' + (running ? 'Stop Monitoring' : 'Start Monitoring');
  const card = btn.closest('.feed-card, .camera-card');
  if (card) {
    const pill = card.querySelector('[data-status-pill]');
    if (pill) {
      pill.className = 'pill ' + (running ? 'ok' : 'neutral');
      pill.setAttribute('data-status-pill', '');
      pill.textContent = running ? 'Live' : 'Stopped';
    }
    const feed = card.querySelector('img.feed-video, img.live-feed');
    if (feed && feed.dataset.src) feed.src = feed.dataset.src.split('?')[0] + '?t=' + Date.now();
  }
}

async function toggleCamera(btn) {
  const id = btn.dataset.cam;
  const running = btn.dataset.state === 'running';
  const action = running ? 'stop' : 'start';
  const prev = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = '<span class="spin"></span> ' + (running ? 'Stopping…' : 'Starting…');
  try {
    const r = await fetch(`/api/cameras/${id}/${action}`, { method: 'POST' });
    const data = await r.json();
    if (!data.ok) throw new Error('not ok');
    _applyToggleState(btn, !running);
    showToast(!running ? 'Camera started' : 'Camera stopped', !running ? 'ok' : '');
  } catch (e) {
    btn.innerHTML = prev;
    showToast('Action failed — check the camera / stream URL.', 'bad');
  } finally {
    btn.disabled = false;
  }
}

/* Back-compat wrappers (older inline handlers) */
function startCamera(cameraId) {
  fetch(`/api/cameras/${cameraId}/start`, { method: 'POST' }).then(r => r.json())
    .then(d => showToast(d.ok ? 'Camera started' : 'Failed to start camera', d.ok ? 'ok' : 'bad'));
}
function stopCamera(cameraId) {
  fetch(`/api/cameras/${cameraId}/stop`, { method: 'POST' }).then(r => r.json())
    .then(d => showToast(d.ok ? 'Camera stopped' : 'Failed to stop camera', d.ok ? 'ok' : 'bad'));
}

function updateAnalytics(analytics) {
  setText('kpiViolations', analytics.total_violations);
  setText('kpiActive', analytics.active_events);
  setText('kpiResolved', analytics.resolved_events);
  setText('kpiCompliance', `${analytics.compliance_pct}%`);
  renderCharts(analytics);
  if (analytics.recent_events) renderActiveEvents(analytics.recent_events.filter(e => ['NEW','ACTIVE'].includes(e.state)));
}

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function renderActiveEvents(events) {
  const list = document.getElementById('activeEventsList');
  if (!list) return;
  list.innerHTML = '';
  if (!events.length) {
    list.innerHTML = '<p class="muted">No active events.</p>';
    return;
  }
  events.slice(0, 20).forEach(e => list.appendChild(eventCard(e)));
}

function prependActiveEvent(event) {
  const list = document.getElementById('activeEventsList');
  if (!list) return;
  if (list.querySelector('.muted')) list.innerHTML = '';
  list.prepend(eventCard(event));
}

function eventCard(e) {
  const div = document.createElement('div');
  div.className = 'event-item';
  const evidence = e.screenshot_path ? `<a href="${e.screenshot_path}" target="_blank">Evidence</a>` : '';
  div.innerHTML = `<strong>${e.violation_type || 'violation'} · ${e.state || ''}</strong><span>Camera: ${e.camera_id || '-'} · Track: ${e.track_id || '-'}</span><br>${evidence}`;
  return div;
}

function renderCharts(analytics) {
  if (!document.getElementById('timelineChart')) return;
  const timelineLabels = (analytics.timeline || []).map(x => x.bucket);
  const timelineCounts = (analytics.timeline || []).map(x => x.count);
  const typeLabels = Object.keys(analytics.violations_by_type || {});
  const typeCounts = Object.values(analytics.violations_by_type || {});

  if (!timelineChart) {
    timelineChart = new Chart(document.getElementById('timelineChart'), {
      type: 'line',
      data: { labels: timelineLabels, datasets: [{ label: 'Violations', data: timelineCounts, tension: 0.35 }] },
      options: { responsive: true, plugins: { legend: { display: false } } }
    });
  } else {
    timelineChart.data.labels = timelineLabels; timelineChart.data.datasets[0].data = timelineCounts; timelineChart.update();
  }

  if (!typeChart && document.getElementById('typeChart')) {
    typeChart = new Chart(document.getElementById('typeChart'), {
      type: 'bar',
      data: { labels: typeLabels, datasets: [{ label: 'Count', data: typeCounts }] },
      options: { responsive: true, plugins: { legend: { display: false } } }
    });
  } else if (typeChart) {
    typeChart.data.labels = typeLabels; typeChart.data.datasets[0].data = typeCounts; typeChart.update();
  }
}

window.addEventListener('load', () => {
  if (window.initialAnalytics) renderCharts(window.initialAnalytics);
});
