// Laya Bridge: relays observe/act requests from the local Laya app to CLAIMED tabs only.
// Nothing runs on a tab until you claim it (popup button); claimed tabs show a banner with a Stop button.
const DEFAULT_PORT = 8766;
let ws = null, backoff = 500, claimed = new Set();

const store = {
  get: (k) => new Promise((r) => chrome.storage.local.get(k, (v) => r(v[k]))),
  set: (o) => new Promise((r) => chrome.storage.local.set(o, r)),
};

async function loadClaims() { claimed = new Set((await store.get('claimed')) || []); }
async function saveClaims() { await store.set({ claimed: [...claimed] }); badge(); pushTabs(); }
function badge() { chrome.action.setBadgeText({ text: claimed.size ? String(claimed.size) : '' }); chrome.action.setBadgeBackgroundColor({ color: '#ff4b4b' }); }

async function tabsInfo() {
  const out = [];
  for (const id of claimed) {
    try { const t = await chrome.tabs.get(id); out.push({ id, url: t.url, title: t.title, claimed: true }); }
    catch (e) { claimed.delete(id); }
  }
  return out;
}
async function pushTabs() { if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: 'tabs', tabs: await tabsInfo() })); }

async function showBanner(tabId, on) {
  try {
    await chrome.scripting.executeScript({
      target: { tabId }, args: [on],
      func: (on) => {
        const id = '__laya_banner';
        const old = document.getElementById(id); if (old) old.remove();
        if (!on) return;
        const d = document.createElement('div'); d.id = id;
        d.style.cssText = 'position:fixed;top:0;left:50%;transform:translateX(-50%);z-index:2147483647;background:#ff4b4b;color:#fff;font:13px system-ui;padding:5px 12px;border-radius:0 0 8px 8px;box-shadow:0 2px 8px rgba(0,0,0,.3)';
        d.textContent = 'Laya is controlling this tab ';
        const b = document.createElement('button'); b.textContent = 'Stop'; b.style.cssText = 'margin-left:8px;cursor:pointer';
        b.onclick = () => chrome.runtime.sendMessage({ type: 'release_all' });
        d.appendChild(b); document.documentElement.appendChild(d);
      },
    });
  } catch (e) { /* restricted pages (chrome://) cannot show it */ }
}

async function claim(tabId) { claimed.add(tabId); await saveClaims(); await showBanner(tabId, true); }
async function unclaim(tabId) { claimed.delete(tabId); await saveClaims(); await showBanner(tabId, false); }
async function releaseAll() { const ids = [...claimed]; claimed.clear(); await saveClaims(); for (const id of ids) await showBanner(id, false); }
chrome.tabs.onRemoved.addListener((id) => { if (claimed.delete(id)) saveClaims(); });

async function ensureScript(tabId) {
  await chrome.scripting.executeScript({ target: { tabId }, files: ['observe.js'] });
}

async function handle(msg) {
  const { method, tabId, params } = msg;
  if (method === 'release_all') { await releaseAll(); return { ok: true }; }
  if (method === 'tabs') return { tabs: await tabsInfo() };
  if (!claimed.has(tabId)) return { error: 'tab_not_claimed' };   // the one rule everything else depends on
  if (method === 'observe') {
    await ensureScript(tabId);
    const [{ result }] = await chrome.scripting.executeScript({ target: { tabId }, func: () => window.__laya.observe() });
    return { result };
  }
  if (method === 'act') {
    await ensureScript(tabId);
    const [{ result }] = await chrome.scripting.executeScript({ target: { tabId }, args: [params], func: (a) => window.__laya.act(a) });
    return { result };
  }
  if (method === 'navigate') {
    await chrome.tabs.update(tabId, { url: params.url });
    await new Promise((res) => { const f = (id, info) => { if (id === tabId && info.status === 'complete') { chrome.tabs.onUpdated.removeListener(f); res(); } }; chrome.tabs.onUpdated.addListener(f); setTimeout(res, 15000); });
    return { result: { ok: true } };
  }
  if (method === 'screenshot') {
    const tab = await chrome.tabs.get(tabId);
    const url = await chrome.tabs.captureVisibleTab(tab.windowId, { format: 'png' });
    return { result: { dataUrl: url } };
  }
  return { error: 'unknown_method' };
}

async function connect() {
  const token = await store.get('token');
  if (!token) { setTimeout(connect, 2000); return; }
  const port = (await store.get('port')) || DEFAULT_PORT;
  ws = new WebSocket(`ws://127.0.0.1:${port}/bridge`);
  ws.onopen = async () => { backoff = 500; ws.send(JSON.stringify({ type: 'hello', token, version: '0.1.0' })); await pushTabs(); };
  ws.onmessage = async (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.method === undefined) return;
    let reply;
    try { reply = await handle(msg); } catch (e) { reply = { error: String(e && e.message || e) }; }
    ws.send(JSON.stringify({ id: msg.id, ...reply }));
  };
  ws.onclose = () => { ws = null; setTimeout(connect, backoff = Math.min(backoff * 2, 5000)); };
  ws.onerror = () => { try { ws.close(); } catch (e) {} };
}

chrome.runtime.onMessage.addListener((m, _s, respond) => {
  (async () => {
    if (m.type === 'release_all') { await releaseAll(); respond({ ok: true }); }
    else if (m.type === 'claim') { await claim(m.tabId); respond({ ok: true }); }
    else if (m.type === 'unclaim') { await unclaim(m.tabId); respond({ ok: true }); }
    else if (m.type === 'status') { respond({ connected: !!(ws && ws.readyState === 1), claimed: [...claimed], hasToken: !!(await store.get('token')) }); }
    else if (m.type === 'set_token') { await store.set({ token: m.token, ...(m.port ? { port: Number(m.port) } : {}) }); if (ws) ws.close(); else connect(); respond({ ok: true }); }
  })();
  return true;
});

globalThis.laya = { claim, unclaim, releaseAll, status: () => ({ connected: !!(ws && ws.readyState === 1), claimed: [...claimed] }), setToken: (t, port) => store.set({ token: t, ...(port ? { port } : {}) }).then(() => { if (ws) ws.close(); else connect(); }) };
loadClaims().then(() => { badge(); connect(); });
