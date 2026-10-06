// Polls state.json and renders the jukebox cabinet. Fully offline, no dependencies.
// RGB values mirror COLOR_MAP in src/vcm/dispatch.py.
const COLOR_MAP = {
  red: [1, 0, 0], green: [0, 1, 0], blue: [0, 0, 1], yellow: [1, 1, 0],
  white: [1, 1, 1], orange: [1, 0.5, 0], purple: [0.5, 0, 1],
  pink: [1, 0.4, 0.7], warm: [1, 0.6, 0.3], cool: [0.7, 0.85, 1],
};
const POLL_MS = 400;
// UI-invented display durations (ms) for each transient screen; timer is computed.
const SHOW_MS = { calling: 5000, message: 3500, time: 5000, weather: 5000, alarm: 4500,
                  temperature: 4500, reminder: 5500, timer_done: 6000 };
const TEMP_MIN = 10, TEMP_MAX = 30, ARC = 264;

const $ = (id) => document.getElementById(id);
const jukebox = $("jukebox"), screen = $("screen"), glow = $("glow");
let last = "";
let loaded = false;    // false until the first poll: whatever is in the file then is old news
let phoneId = null;    // last transient phone-style event seen
let phoneTimer = null, countTimer = null;
let musicState = { state: "idle" };   // last known music state, for the idle-vs-music screen fallback

function rgb(color, k) {
  const c = COLOR_MAP[color] || COLOR_MAP.white;
  return `rgb(${c.map((v) => Math.round(255 * v * k)).join(",")})`;
}

function renderLights(s) {
  const on = !!s.lights_on;
  const bri = Math.max(0, Math.min(100, Number(s.brightness) || 0));
  const color = s.color || "white";
  jukebox.dataset.power = on ? "on" : "off";
  $("power").textContent = on ? "ON" : "OFF";
  $("bri").textContent = on ? bri + "%" : "--";
  $("briBar").style.width = on ? bri + "%" : "0";
  $("col").textContent = on ? color : "--";
  $("colDot").style.background = on ? rgb(color, 1) : "";
  if (on) {
    glow.style.setProperty("--c", rgb(color, 1));
    glow.style.opacity = 0.25 + 0.55 * (bri / 100);
  } else {
    glow.style.opacity = 0;
  }
}

// Persistent state (stays visible at rest, like lights) -- drives the vinyl
// spin/tonearm and, when nothing else is showing, the screen's idle view.
function renderMusic(m) {
  musicState = m || { state: "idle" };
  const state = musicState.state || "idle";
  jukebox.dataset.music = state;
  $("musicTrack").textContent = musicState.track || "--";
  $("musicPlaylist").textContent = musicState.playlist || "--";
  const vol = musicState.volume != null ? Math.max(0, Math.min(100, Number(musicState.volume))) : null;
  $("musicVol").textContent = vol != null ? vol + "%" : "--";
  $("musicVolBar").style.width = vol != null ? vol + "%" : "0";
  // If the screen is currently sitting at rest (idle or music), keep it in
  // sync with the latest music state; a transient event (call, time, ...)
  // takes priority and isn't interrupted.
  if (screen.dataset.state === "idle" || screen.dataset.state === "music") {
    setScreenRest();
  }
}

function setScreenRest() {
  screen.dataset.state = musicState.state && musicState.state !== "idle" ? "music" : "idle";
}

function setScreen(state) {
  screen.dataset.state = state;
}

function renderDevice(d) {
  d = d || {};
  $("devAlarm").textContent = d.alarm || "--";
  $("devTemp").textContent = d.temperature != null ? d.temperature + "°" : "--";
  const n = (d.reminders || []).length;
  $("devRem").textContent = n ? n + (n === 1 ? " reminder" : " reminders") : "None";
}

function tempArc(t) {
  const f = Math.max(0, Math.min(1, (t - TEMP_MIN) / (TEMP_MAX - TEMP_MIN)));
  const arc = $("tempArc");
  arc.style.strokeDashoffset = ARC * (1 - f);
  arc.style.stroke = `hsl(${210 - 190 * f}, 85%, 58%)`; // cool blue -> warm orange
}

function startTimer(total, label) {
  clearInterval(countTimer);
  const end = Date.now() + total * 1000;
  $("timerLabel").textContent = label;
  const tick = () => {
    const left = Math.max(0, (end - Date.now()) / 1000);
    $("timerNum").textContent = Math.ceil(left);
    $("timerArc").style.strokeDashoffset = ARC * (1 - left / total);
    if (left <= 0) clearInterval(countTimer);
  };
  tick();
  countTimer = setInterval(tick, 100);
}

// Fill the screen for one event and return how long to show it.
function showEvent(p) {
  const d = p.data || {};
  switch (p.status) {
    case "message":
      $("msgText").textContent = p.text || "Message sent";
      break;
    case "time": {
      const [h, m] = (d.clock || "0:0").split(":").map(Number);
      $("hHand").setAttribute("transform", `rotate(${(h % 12) * 30 + m / 2} 50 50)`);
      $("mHand").setAttribute("transform", `rotate(${m * 6} 50 50)`);
      $("timeText").textContent = (p.text || "").replace(/^It's\s*/i, "");
      break;
    }
    case "weather": {
      const m = /(-?\d+)\s*degrees/i.exec(p.text || "");
      $("weatherTemp").textContent = m ? m[1] + "°" : "";
      $("weatherText").textContent = /sunny/i.test(p.text || "") ? "Sunny" : p.text;
      break;
    }
    case "alarm":
      $("alarmTime").textContent = d.alarm_time || "";
      break;
    case "temperature":
      $("tempNum").textContent = d.temperature + "°";
      tempArc(Number(d.temperature));
      break;
    case "reminder": {
      const list = d.reminders || [];
      $("remTitle").textContent = /added/i.test(p.text || "") ? p.text : list.length ? "Your reminders" : "No reminders";
      const ul = $("remList");
      ul.textContent = "";
      list.forEach((r, i) => {
        const li = document.createElement("li");
        li.textContent = r;
        if (/added/i.test(p.text || "") && i === list.length - 1) li.className = "new";
        ul.appendChild(li);
      });
      break;
    }
    case "timer": {
      const total = d.seconds || (d.minutes || 0) * 60;
      startTimer(total, d.seconds ? (total === 1 ? "second" : "seconds") : (d.minutes === 1 ? "minute" : "minutes"));
      return total * 1000 + 800; // stay until it should have finished
    }
    case "timer_done":
      $("doneLabel").textContent = (d.label || "") + " timer";
      break;
  }
  return SHOW_MS[p.status] || 4000;
}

// The backend only writes "this happened"; idle is computed here. A new event
// (changed id) shows for a few seconds, then reverts to the music/idle rest
// state. Timing is local, so clock differences between the Pi and this
// device don't matter. Whatever was already in the file at page load is
// treated as old.
function renderPhone(p) {
  const id = p ? p.seq + "|" + p.updated_at : "none";
  if (!loaded) { phoneId = id; renderDevice(p && p.device); return; }
  renderDevice(p && p.device);
  if (!p || id === phoneId) return;
  phoneId = id;
  if (p.status !== "timer") clearInterval(countTimer);
  const ms = showEvent(p);
  setScreen(p.status);
  clearTimeout(phoneTimer);
  phoneTimer = setTimeout(setScreenRest, ms);
}

function setLink(ok) {
  const el = $("link");
  el.className = "link " + (ok ? "ok" : "bad");
  el.lastChild.textContent = ok ? "Live" : "Waiting for Pi";
}

async function poll() {
  try {
    const r = await fetch("state.json?t=" + Date.now(), { cache: "no-store" });
    if (!r.ok) { setLink(false); return; }
    const text = await r.text();
    setLink(true);
    if (text === last) return;
    const s = JSON.parse(text);
    last = text;
    renderLights(s);
    renderMusic(s.music);
    renderPhone(s.phone);
    loaded = true;
  } catch (e) { setLink(false); /* missing or mid-write: keep last state */ }
}

renderLights({ lights_on: false, brightness: 100, color: null });
renderMusic(null);
poll();
setInterval(poll, POLL_MS);
