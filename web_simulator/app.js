// Polls state.json and renders lights + phone state. Fully offline, no dependencies.
// RGB values mirror COLOR_MAP in src/vcm/dispatch.py.
const COLOR_MAP = {
  red: [1, 0, 0], green: [0, 1, 0], blue: [0, 0, 1], yellow: [1, 1, 0],
  white: [1, 1, 1], orange: [1, 0.5, 0], purple: [0.5, 0, 1],
  pink: [1, 0.4, 0.7], warm: [1, 0.6, 0.3], cool: [0.7, 0.85, 1],
};
const POLL_MS = 400;
// UI-invented display durations (ms) for each phone screen; timer is computed.
const SHOW_MS = { calling: 5000, message: 3500, time: 5000, weather: 5000, alarm: 4500,
                  temperature: 4500, reminder: 5500, timer_done: 6000 };
const PILL = { idle: "IDLE", calling: "CALLING", message: "MESSAGE", time: "TIME", weather: "WEATHER",
               alarm: "ALARM", timer: "TIMER", timer_done: "DONE", temperature: "THERMOSTAT", reminder: "REMINDERS" };
const TEMP_MIN = 10, TEMP_MAX = 30, ARC = 264;

const MUSIC_PILL = { idle: "IDLE", playing: "PLAYING", paused: "PAUSED" };

const $ = (id) => document.getElementById(id);
const lightCard = $("lightCard"), phoneCard = $("phoneCard"), musicCard = $("musicCard");
const glow = $("glow"), fill = $("bulbFill");
let last = "";
let loaded = false;    // false until the first poll: whatever is in the file then is old news
let phoneId = null;    // last phone event seen
let phoneTimer = null, countTimer = null;

function rgb(color, k) {
  const c = COLOR_MAP[color] || COLOR_MAP.white;
  return `rgb(${c.map((v) => Math.round(255 * v * k)).join(",")})`;
}

function renderLights(s) {
  const on = !!s.lights_on;
  const bri = Math.max(0, Math.min(100, Number(s.brightness) || 0));
  const color = s.color || "white";
  lightCard.classList.toggle("on", on);
  $("power").textContent = on ? "ON" : "OFF";
  $("bri").textContent = on ? bri + "%" : "--";
  $("briBar").style.width = on ? bri + "%" : "0";
  $("col").textContent = on ? color : "--";
  $("colDot").style.background = on ? rgb(color, 1) : "";
  if (on) {
    fill.style.fill = rgb(color, 0.35 + 0.65 * (bri / 100));
    glow.style.setProperty("--c", rgb(color, 1));
    glow.style.opacity = (bri / 100) * 0.9;
  } else {
    fill.style.fill = "";
    glow.style.opacity = 0;
  }
}

// Persistent state like lights (stays visible at rest), not a transient
// phone-style event -- "what's playing" should still show after a poll
// rather than flash and revert.
function renderMusic(m) {
  m = m || {};
  const state = m.state || "idle";
  musicCard.dataset.state = state;
  $("musicPill").textContent = MUSIC_PILL[state] || state.toUpperCase();
  $("musicTrack").textContent = m.track || "--";
  $("musicPlaylist").textContent = m.playlist || "--";
  const vol = m.volume != null ? Math.max(0, Math.min(100, Number(m.volume))) : null;
  $("musicVol").textContent = vol != null ? vol + "%" : "--";
  $("musicVolBar").style.width = vol != null ? vol + "%" : "0";
}

function setPhone(state) {
  phoneCard.dataset.state = state;
  $("phonePill").textContent = PILL[state] || state.toUpperCase();
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
// (changed id) shows for a few seconds, then reverts. Timing is local, so clock
// differences between the Pi and this device don't matter. Whatever was already
// in the file at page load is treated as old.
function renderPhone(p) {
  const id = p ? p.seq + "|" + p.updated_at : "none";
  if (!loaded) { phoneId = id; renderDevice(p && p.device); return; }
  renderDevice(p && p.device);
  if (!p || id === phoneId) return;
  phoneId = id;
  if (p.status !== "timer") clearInterval(countTimer);
  const ms = showEvent(p);
  setPhone(p.status);
  clearTimeout(phoneTimer);
  phoneTimer = setTimeout(() => setPhone("idle"), ms);
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
