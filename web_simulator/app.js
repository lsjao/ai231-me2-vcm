// Polls state.json and renders lights + phone state. Fully offline, no dependencies.
// RGB values mirror COLOR_MAP in src/vcm/dispatch.py.
const COLOR_MAP = {
  red: [1, 0, 0], green: [0, 1, 0], blue: [0, 0, 1], yellow: [1, 1, 0],
  white: [1, 1, 1], orange: [1, 0.5, 0], purple: [0.5, 0, 1],
  pink: [1, 0.4, 0.7], warm: [1, 0.6, 0.3], cool: [0.7, 0.85, 1],
};
const POLL_MS = 400;
const PHONE_MS = { calling: 5000, message: 3500 }; // UI-invented display durations

const $ = (id) => document.getElementById(id);
const lightCard = $("lightCard"), phoneCard = $("phoneCard");
const glow = $("glow"), fill = $("bulbFill");
let last = "";
let phoneSeq = null;   // last phone event seen (null until first poll)
let phoneTimer = null;

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

function setPhone(state) {
  phoneCard.dataset.state = state;
  $("phonePill").textContent = state === "calling" ? "CALLING" : state === "message" ? "MESSAGE" : "IDLE";
}

// The backend only writes "this happened"; idle is computed here. A new event
// (changed seq) shows for a few seconds, then reverts. Timing is local, so
// clock differences between the Pi and this device don't matter. Whatever was
// already in the file at page load is treated as old.
function renderPhone(p) {
  if (!p || !p.status) return;
  const id = p.seq + "|" + p.updated_at;
  if (phoneSeq === null) { phoneSeq = id; return; }
  if (id === phoneSeq) return;
  phoneSeq = id;
  setPhone(p.status);
  $("lastEvent").textContent = (p.status === "calling" ? "Call" : "Message") + " at " + (p.updated_at.split("T")[1] || "");
  clearTimeout(phoneTimer);
  phoneTimer = setTimeout(() => setPhone("idle"), PHONE_MS[p.status] || 3500);
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
    renderPhone(s.phone);
  } catch (e) { setLink(false); /* missing or mid-write: keep last state */ }
}

function tickClock() {
  const d = new Date();
  $("clock").textContent = d.getHours() + ":" + String(d.getMinutes()).padStart(2, "0");
}

renderLights({ lights_on: false, brightness: 100, color: null });
tickClock();
setInterval(tickClock, 10000);
poll();
setInterval(poll, POLL_MS);
