/*
# ======================================
# TTS.JS — spoken violation alerts
# ======================================
Browser SpeechSynthesis on new violations (no server dependency). Included by
base.html (all operator pages) and public_dashboard.html (wall display).

- PPE_TTS.onViolation(event): speak "No helmet detected on <camera>" for a NEW
  incident. The EventManager already dedupes (persistence + 30s cooldown), and a
  4s throttle here stops a burst of workers talking over each other.
- Mute toggle (🔊/🔇) injected top-right, persisted in localStorage. Chrome's
  autoplay policy blocks speech until a user gesture — the first click on the
  toggle doubles as the unlock.
- Plays static/sounds/alert.wav as an attention chime before the phrase.
*/
(function () {
  const PHRASES = {
    missing_helmet: 'No helmet detected',
    missing_vest: 'No safety vest detected',
    missing_gloves: 'No gloves detected',
    missing_goggles: 'No goggles detected',
    missing_boots: 'No safety boots detected',
    missing_mask: 'No mask detected',
  };
  const THROTTLE_MS = 4000;
  const LS_KEY = 'ppe_tts_muted';

  let lastSpokeAt = 0;
  let cameraNames = {};
  let chime = null;

  function muted() { return localStorage.getItem(LS_KEY) === '1'; }
  function setMuted(m) {
    localStorage.setItem(LS_KEY, m ? '1' : '0');
    const btn = document.getElementById('ttsToggle');
    if (btn) { btn.textContent = m ? '🔇' : '🔊'; btn.title = m ? 'Voice alerts muted — click to unmute' : 'Voice alerts on — click to mute'; }
  }

  // Camera display names for the spoken phrase (the event only carries camera_id).
  fetch('/api/cameras').then(r => r.json()).then(list => {
    (list || []).forEach(c => { cameraNames[c.camera_id] = c.name || c.camera_id; });
  }).catch(() => {});

  function phraseFor(event) {
    const base = PHRASES[event.violation_type] || String(event.violation_type || 'P P E violation').replace(/_/g, ' ');
    const cam = cameraNames[event.camera_id];
    return cam ? `${base} on camera ${cam}` : base;
  }

  function speak(text) {
    try {
      if (!('speechSynthesis' in window)) return;
      const u = new SpeechSynthesisUtterance(text);
      u.rate = 1.0;
      u.volume = 1.0;
      window.speechSynthesis.speak(u);
    } catch (e) { /* speech unavailable — chime already played */ }
  }

  function playChime() {
    try {
      if (!chime) { chime = new Audio('/static/sounds/alert.wav'); chime.preload = 'auto'; }
      chime.currentTime = 0;
      const p = chime.play();
      if (p && p.catch) p.catch(() => {});   // autoplay-blocked until first gesture
    } catch (e) { /* no audio */ }
  }

  function onViolation(event) {
    if (!event || muted()) return;
    // Speak only when a NEW incident is confirmed (EventManager persistence passed);
    // ACTIVE updates of the same incident stay silent.
    if ((event.state || '') !== 'NEW') return;
    const now = Date.now();
    if (now - lastSpokeAt < THROTTLE_MS) return;
    lastSpokeAt = now;
    playChime();
    speak(phraseFor(event));
  }

  // Inject the mute toggle (top-right, unobtrusive) on DOM ready.
  function injectToggle() {
    if (document.getElementById('ttsToggle')) return;
    const btn = document.createElement('button');
    btn.id = 'ttsToggle';
    btn.style.cssText = 'position:fixed;top:12px;right:12px;z-index:9999;background:rgba(20,20,20,.75);color:#fff;border:1px solid rgba(255,255,255,.25);border-radius:8px;padding:6px 10px;font-size:16px;cursor:pointer;';
    btn.onclick = function () {
      setMuted(!muted());
      // First gesture unlocks speech + audio under Chrome's autoplay policy.
      try { const u = new SpeechSynthesisUtterance(' '); u.volume = 0; window.speechSynthesis.speak(u); } catch (e) {}
      if (!muted()) speak('Voice alerts enabled');
    };
    document.body.appendChild(btn);
    setMuted(muted());
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', injectToggle);
  else injectToggle();

  window.PPE_TTS = { onViolation: onViolation, speak: speak };
})();
