// Polls state.json and renders light state. Fully offline, no dependencies.
// RGB values mirror COLOR_MAP in src/vcm/dispatch.py.
const COLOR_MAP = {
  red: [1, 0, 0], green: [0, 1, 0], blue: [0, 0, 1], yellow: [1, 1, 0],
  white: [1, 1, 1], orange: [1, 0.5, 0], purple: [0.5, 0, 1],
  pink: [1, 0.4, 0.7], warm: [1, 0.6, 0.3], cool: [0.7, 0.85, 1],
};
const POLL_MS = 400;

const $ = (id) => document.getElementById(id);
const stage = $("stage"), glow = $("glow"), fill = $("bulbFill");
let last = "";

function rgb(color, k) {
  const c = COLOR_MAP[color] || COLOR_MAP.white;
  return `rgb(${c.map((v) => Math.round(255 * v * k)).join(",")})`;
}

function render(s) {
  const on = !!s.lights_on;
  const bri = Math.max(0, Math.min(100, Number(s.brightness) || 0));
  const color = s.color || "white";
  stage.classList.toggle("on", on);
  document.body.classList.toggle("on", on);
  $("power").textContent = on ? "ON" : "OFF";
  $("bri").textContent = on ? bri + "%" : "--";
  $("col").textContent = on && s.color ? s.color : on ? "white" : "--";
  if (on) {
    const k = 0.35 + 0.65 * (bri / 100);
    fill.style.fill = rgb(color, k);
    glow.style.setProperty("--c", rgb(color, 1));
    glow.style.opacity = (bri / 100) * 0.9;
  } else {
    fill.style.fill = "";
    glow.style.opacity = 0;
  }
}

async function poll() {
  try {
    const r = await fetch("state.json?t=" + Date.now(), { cache: "no-store" });
    if (!r.ok) return;
    const text = await r.text();
    if (text === last) return;
    const s = JSON.parse(text);
    last = text;
    render(s);
  } catch (e) { /* missing or mid-write: keep last state */ }
}

render({ lights_on: false, brightness: 100, color: null });
poll();
setInterval(poll, POLL_MS);
