const $ = (id) => document.getElementById(id);
const send = (m) => new Promise((r) => chrome.runtime.sendMessage(m, r));
let tab = null;
async function refresh() {
  [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const st = await send({ type: 'status' });
  const mine = st.claimed.includes(tab.id);
  $('s').textContent = (st.hasToken ? (st.connected ? 'Connected to Laya. ' : 'Waiting for the Laya app... ') : 'Paste the token below first. ') + `${st.claimed.length} tab(s) controlled.`;
  $('toggle').textContent = mine ? 'Stop controlling this tab' : 'Let Laya control this tab';
}
$('toggle').onclick = async () => { const st = await send({ type: 'status' }); await send({ type: st.claimed.includes(tab.id) ? 'unclaim' : 'claim', tabId: tab.id }); refresh(); };
$('stop').onclick = async () => { await send({ type: 'release_all' }); refresh(); };
$('save').onclick = async () => { await send({ type: 'set_token', token: $('tok').value.trim() }); $('tok').value = ''; setTimeout(refresh, 800); };
refresh();
