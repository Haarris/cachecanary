// Usage: python3 scripts/serve_site.py site 8765 headers & node scripts/check_site.mjs http://127.0.0.1:8765/ <out-dir>
// Drive headless Chrome over the DevTools protocol: load a URL, collect console/CSP/network
// problems, check that CSS and JS actually applied, click the Copy button, take screenshots.
import { spawn } from "node:child_process";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const [url, outDir] = process.argv.slice(2);
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const profile = mkdtempSync(join(tmpdir(), "cdp-"));
const port = 9333;
const chrome = spawn(CHROME, ["--headless=new", "--disable-gpu", `--remote-debugging-port=${port}`,
  `--user-data-dir=${profile}`, "--no-first-run", "about:blank"], { stdio: "ignore" });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let target;
for (let i = 0; i < 50 && !target; i++) {
  try { target = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find((t) => t.type === "page"); } catch {}
  if (!target) await sleep(200);
}
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener("open", r));
let id = 0; const pending = new Map(); const problems = [];
ws.addEventListener("message", (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); return; }
  if (msg.method === "Log.entryAdded") problems.push(`log/${msg.params.entry.source}/${msg.params.entry.level}: ${msg.params.entry.text}`);
  if (msg.method === "Runtime.exceptionThrown") problems.push(`exception: ${msg.params.exceptionDetails.text} ${msg.params.exceptionDetails.exception?.description ?? ""}`);
  if (msg.method === "Runtime.consoleAPICalled" && ["error", "warning"].includes(msg.params.type)) problems.push(`console.${msg.params.type}: ${msg.params.args.map((a) => a.value ?? a.description).join(" ")}`);
  if (msg.method === "Network.loadingFailed") problems.push(`network failed: ${msg.params.errorText} ${msg.params.blockedReason ?? ""}`);
  if (msg.method === "Network.responseReceived" && msg.params.response.status >= 400) problems.push(`HTTP ${msg.params.response.status}: ${msg.params.response.url}`);
});
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const evaluate = async (expr) => (await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true })).result?.result?.value;

for (const d of ["Log", "Runtime", "Network", "Page"]) await send(`${d}.enable`);
await send("Browser.grantPermissions", { permissions: ["clipboardReadWrite", "clipboardSanitizedWrite"], origin: new URL(url).origin }).catch(() => {});

const report = {};
for (const scheme of ["light", "dark"]) {
  for (const [label, width, height, mobile] of [["desktop", 1280, 900, false], ["phone", 390, 844, true]]) {
    await send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: mobile ? 2 : 1, mobile });
    await send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: scheme }] });
    await send("Page.navigate", { url });
    await sleep(1200);
    const info = await evaluate(`(() => {
      const cs = getComputedStyle(document.body);
      const term = document.querySelector('.term');
      return { innerWidth, scrollWidth: document.documentElement.scrollWidth,
               bodyBg: cs.backgroundColor, bodyFont: cs.fontFamily.slice(0, 30),
               termBorder: term ? getComputedStyle(term).borderTopWidth : null,
               installWraps: (() => { const s = document.getElementById('install-cmd'); return s ? s.getClientRects().length > 1 : null; })(),
               h1: document.querySelector('h1')?.textContent, links: [...document.querySelectorAll('a')].map(a => a.href),
               thirdParty: [...new Set(performance.getEntriesByType('resource').map(e => new URL(e.name)).filter(u => u.host !== location.host).map(u => u.host + u.pathname.split('/').slice(0, 2).join('/')))] };
    })()`);
    // Click Copy and read back what it did.
    const copy = await evaluate(`(async () => {
      const b = document.getElementById('copy'); b.click();
      await new Promise(r => setTimeout(r, 300));
      let clip = null; try { clip = await navigator.clipboard.readText(); } catch (e) { clip = 'read-failed: ' + e.name; }
      return { label: b.textContent, clipboard: clip, selection: String(getSelection()) };
    })()`);
    const shot = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true,
      clip: { x: 0, y: 0, width, height: await evaluate("document.documentElement.scrollHeight"), scale: 1 } });
    writeFileSync(join(outDir, `${label}-${scheme}.png`), Buffer.from(shot.result.data, "base64"));
    report[`${label}-${scheme}`] = { ...info, links: undefined, copy };
    report.links = info.links;
  }
}
console.log(JSON.stringify({ problems: [...new Set(problems)], report }, null, 2));
ws.close(); chrome.kill(); await sleep(300); rmSync(profile, { recursive: true, force: true });
process.exit(0);
