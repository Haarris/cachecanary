// Cache hit rate checker for the numbers Amazon Bedrock publishes in CloudWatch. Runs only in the browser;
// nothing is sent anywhere. The maths and prices mirror the CLI (src/cachecanary/pricing.py), and
// tests/test_check_page.py checks both against it.
(function (root) {
  "use strict";

  // Global-endpoint list prices in USD per million input tokens, and the cache read price as a share of it.
  // Cache writes cost 1.25x input (5-minute cache). min = tokens needed before a cache point is used.
  // Same rows as GLOBAL_PRICES and CLAUDE_LIMITS in the CLI.
  var MODELS = {
    "claude-fable-5-1": { input: 10, read: 0.025, min: 512 },
    "claude-mythos-5-1": { input: 10, read: 0.025, min: 512 },
    "claude-fable-5": { input: 10, read: 0.1, min: 512 },
    "claude-mythos-5": { input: 10, read: 0.1, min: 512 },
    "claude-opus-5-5": { input: 4, read: 0.05, min: 512 },
    "claude-opus-5": { input: 5, read: 0.1, min: 512 },
    "claude-opus-4-8": { input: 5, read: 0.1, min: 1024 },
    "claude-opus-4-7": { input: 5, read: 0.1, min: 4096 },
    "claude-opus-4-6": { input: 5, read: 0.1, min: 4096 },
    "claude-opus-4-5": { input: 5, read: 0.1, min: 4096 },
    "claude-sonnet-5-5": { input: 2, read: 0.05, min: 512 },
    "claude-sonnet-5": { input: 2, read: 0.1, min: 1024 },
    "claude-sonnet-4-6": { input: 3, read: 0.1, min: 1024 },
    "claude-sonnet-4-5": { input: 3, read: 0.1, min: 1024 },
    "claude-haiku-5-5": { input: 0.1, read: 0.1, min: 512 },
    "claude-haiku-4-5": { input: 1, read: 0.1, min: 4096 },
    "claude-sonnet-4": { input: 3, read: 0.1, min: 1024, legacy: true },
    "claude-opus-4-1": { input: 15, read: 0.1, min: 1024, legacy: true },
    "claude-opus-4": { input: 15, read: 0.1, min: 1024, legacy: true },
    "claude-3-5-haiku": { input: 0.8, read: 0.1, min: 2048, legacy: true }
  };
  var REGIONAL_PREMIUM = 1.1; // regional and geo cross-Region IDs cost 10% more, except legacy models
  var WRITE_5M = 1.25;
  var DEFAULT_READ = 0.1; // read share used with a custom price for a model the table doesn't know

  function label(key) {
    // "claude-sonnet-4-6" -> "Claude Sonnet 4.6", "claude-3-5-haiku" -> "Claude 3.5 Haiku"
    var parts = key.split("-").slice(1), words = [], nums = [];
    parts.forEach(function (p) { (/^\d+$/.test(p) ? nums : words).push(p); });
    var name = words.map(function (w) { return w.charAt(0).toUpperCase() + w.slice(1); }).join(" ");
    var version = nums.join(".");
    return "Claude " + (/^claude-\d/.test(key) ? version + " " + name : name + " " + version);
  }

  // Rates for a model, endpoint ("global" or "regional") and optional custom input price (USD per million).
  function rates(key, endpoint, customInput) {
    var m = MODELS[key];
    var readShare = m ? m.read : DEFAULT_READ;
    var input;
    if (customInput !== null && customInput !== undefined) {
      input = customInput; // your own rate: no regional premium added
    } else {
      if (!m) return null;
      input = m.input * (endpoint === "global" || m.legacy ? 1 : REGIONAL_PREMIUM);
    }
    return { input: input, read: input * readShare, write: input * WRITE_5M };
  }

  // Same as pricing.input_cost with all writes priced as 5-minute writes (CloudWatch doesn't split them).
  function inputCost(u, r, w, rt) {
    return (u * rt.input + r * rt.read + w * rt.write) / 1e6;
  }

  // Same as pricing.lost_to_misses: extra input spend compared with the same calls at the target hit rate.
  function lostToMisses(u, r, w, rt, target) {
    var total = u + r + w;
    var shortfall = target * total - r;
    var notRead = u + w;
    if (total === 0 || shortfall <= 0 || notRead === 0) return 0;
    var paidForNotRead = (u * rt.input + w * rt.write) / notRead;
    return shortfall * (paidForNotRead - rt.read) / 1e6;
  }

  function hitRate(u, r, w) {
    var total = u + r + w;
    return total ? r / total : 0;
  }

  // Same wording as the CLI's dollar figures.
  function money(usd) {
    if (usd === 0) return "$0.00";
    if (usd < 0.01) return "under $0.01";
    return "$" + usd.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  // Token counts as people paste them: "1,234,567", "1 234 567", "2.5M", "830k". Returns null if not a count.
  function parseCount(text) {
    var s = String(text === undefined || text === null ? "" : text).trim().replace(/[,\s_]/g, "");
    if (s === "") return null;
    var m = /^(\d+(?:\.\d+)?|\.\d+)([kmb])?$/i.exec(s);
    if (!m) return null;
    var n = parseFloat(m[1]) * ({ k: 1e3, m: 1e6, b: 1e9 }[(m[2] || "").toLowerCase()] || 1);
    return isFinite(n) ? Math.round(n) : null;
  }

  function parsePercent(text) {
    var s = String(text === undefined || text === null ? "" : text).trim().replace(/%$/, "").trim();
    if (!/^\d+(?:\.\d+)?$/.test(s)) return null;
    var n = parseFloat(s);
    return n > 0 && n <= 100 ? n / 100 : null;
  }

  function parsePrice(text) {
    var s = String(text === undefined || text === null ? "" : text).trim().replace(/^\$/, "");
    if (s === "") return undefined; // not given: use list prices
    if (!/^\d+(?:\.\d+)?$/.test(s)) return null;
    var n = parseFloat(s);
    return n > 0 && n < 1000 ? n : null;
  }

  // Everything the page shows, from raw inputs. Throws nothing; returns {error} for bad input.
  function evaluate(input) {
    var u = parseCount(input.uncached), r = parseCount(input.read), w = parseCount(input.write);
    var bad = [];
    if (u === null) bad.push("InputTokenCount");
    if (r === null) bad.push("CacheReadInputTokenCount");
    if (w === null) bad.push("CacheWriteInputTokenCount");
    if (bad.length) {
      return { error: "Enter a token count for " + bad.join(", ") + " (whole numbers; commas, k and M are fine)." };
    }
    var target = parsePercent(input.target === undefined || input.target === "" ? "90" : input.target);
    if (target === null) return { error: "The target hit rate should be a percentage between 1 and 100." };
    var custom = parsePrice(input.price);
    if (custom === null) return { error: "Your price should be a number of US dollars per million input tokens, like 3." };
    var model = MODELS[input.model] ? input.model : null;
    if (!model) return { error: "Pick a model." };
    var total = u + r + w;
    if (total === 0) return { error: "All three numbers are 0, so there's nothing to check yet." };
    var rt = rates(model, input.endpoint, custom);
    var hit = hitRate(u, r, w);
    var result = {
      model: model, hit: hit, target: target, total: total, uncached: u, read: r, write: w,
      cost: inputCost(u, r, w, rt), lost: lostToMisses(u, r, w, rt, target),
      price: custom === undefined ? "list" : "custom", min: MODELS[model].min, label: label(model)
    };
    if (r === 0 && w === 0) result.verdict = "none";
    else if (r === 0) result.verdict = "never-read";
    else if (hit >= target) result.verdict = "healthy";
    else if (hit >= 0.5) result.verdict = "leaking";
    else result.verdict = "poor";
    result.writesExceedReads = w > r && r > 0;
    return result;
  }

  var api = { MODELS: MODELS, REGIONAL_PREMIUM: REGIONAL_PREMIUM, WRITE_5M: WRITE_5M, label: label, rates: rates,
              inputCost: inputCost, lostToMisses: lostToMisses, hitRate: hitRate, money: money,
              parseCount: parseCount, parsePercent: parsePercent, parsePrice: parsePrice, evaluate: evaluate };
  if (typeof module !== "undefined" && module.exports) { module.exports = api; return; }

  // ---- page wiring (browser only) ----
  var $ = function (id) { return document.getElementById(id); };
  var form = $("checker");
  if (!form) return;

  var select = $("model");
  Object.keys(MODELS).forEach(function (key) {
    var opt = document.createElement("option");
    opt.value = key;
    opt.textContent = label(key) + (MODELS[key].legacy ? " (legacy)" : "");
    if (key === "claude-sonnet-4-6") opt.selected = true;
    select.appendChild(opt);
  });

  function pct(x) {
    var p = x * 100;
    return (p > 0 && p < 1 ? p.toFixed(1) : Math.round(p)) + "%";
  }
  function n(x) { return x.toLocaleString("en-US"); }
  function setText(id, text) { $(id).textContent = text; }

  var VERDICTS = {
    healthy: "Your cache is working well.",
    leaking: "Your cache works, but it misses often.",
    poor: "Your cache is barely working.",
    "never-read": "Your cache is written, but never read back.",
    none: "Nothing was cached at all."
  };
  var CAUSES = {
    healthy: ["Keep it that way: check saved requests in CI with <code>cachecanary lint</code>, so a library upgrade can't quietly break it."],
    leaking: [
      "Agents that run many tools at once: Bedrock only looks back about 20 blocks for the cached part.",
      "Calls more than 5 minutes apart, so the cache expires in between.",
      "Something in the cached part changes on some requests, like a date, a user name or the order of your tools."
    ],
    poor: [
      "Something in the cached part changes on most requests, like a date, a request ID or the order of your tools.",
      "Agents that run many tools at once: Bedrock only looks back about 20 blocks for the cached part.",
      "Calls more than 5 minutes apart, so the cache expires in between."
    ],
    "never-read": [
      "The cached part changes on every request, like today's date or a request ID in the system prompt.",
      "Requests come more than 5 minutes apart, so each one finds the cache expired.",
      "Your requests go to different Regions (a <code>us.</code> or <code>global.</code> ID can route them), so they don't share a cache."
    ],
    none: [
      "No cache points in your requests. Many libraries leave caching off until you turn it on.",
      "The part before the cache point is shorter than the model's minimum, so Bedrock skips it without an error.",
      "Your library doesn't recognise your model ID or inference profile, so it stops asking for caching."
    ]
  };

  function render() {
    var res = evaluate({
      model: select.value, endpoint: $("endpoint").value, uncached: $("uncached").value,
      read: $("read").value, write: $("write").value, target: $("target").value, price: $("price").value
    });
    var out = $("result"), err = $("error");
    var empty = !$("uncached").value.trim() && !$("read").value.trim() && !$("write").value.trim();
    if (empty) { out.hidden = true; err.hidden = true; return; }
    if (res.error) { out.hidden = true; err.hidden = false; err.textContent = res.error; return; }
    err.hidden = true;
    out.hidden = false;
    out.className = "result " + res.verdict;
    setText("r-verdict", VERDICTS[res.verdict]);
    setText("r-hit", pct(res.hit));
    setText("r-hit-note", "of your " + n(res.total) + " input tokens came from the cache");
    setText("r-cost", money(res.cost));
    setText("r-cost-note", "input cost " + (res.price === "custom" ? "at your price" : "at Bedrock list price"));
    setText("r-lost", money(res.lost));
    setText("r-lost-note", res.lost > 0 ? "lost to cache misses, against a " + pct(res.target) + " hit rate"
      : "lost: you're at or above the " + pct(res.target) + " target");
    var causes = CAUSES[res.verdict].slice();
    if (res.verdict === "none") {
      causes[1] = "The part before the cache point is shorter than " + n(res.min) + " tokens, the minimum for " +
        res.label + ", so Bedrock skips it without an error.";
    }
    if (res.writesExceedReads) {
      causes.unshift("You write more to the cache than you read back. Each write costs 1.25 times the normal input price, so writes that are never read cost more than no caching.");
    }
    var list = $("r-causes");
    list.innerHTML = "";
    causes.forEach(function (c) { var li = document.createElement("li"); li.innerHTML = c; list.appendChild(li); });
    $("r-causes-title").textContent = res.verdict === "healthy" ? "Next" : "What usually causes this";
  }

  form.addEventListener("submit", function (e) { e.preventDefault(); render(); });
  form.addEventListener("input", render);
  form.addEventListener("change", render);
  render();
})(this);
