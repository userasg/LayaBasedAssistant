// Page observation and action script. ONE source, used by the Playwright test driver and by the Chrome extension's
// content script, so both see and act on pages identically.
//
// observe() -> {url, title, text, alerts, dialogs, elements:[{id, role, label, value, type, checked, disabled, required, href, visible, inViewport}]}
// act({kind, id, value, key, dy, clear}) -> {ok, error?, value?, checked?}
// Element ids ("e1", "e2", ...) are stored on the element (data-laya-id) so they stay stable across observations.
(() => {
  if (window.__laya) return;
  let counter = 0;
  const INTERACTIVE = 'a[href],button,input:not([type=hidden]),select,textarea,summary,[role=button],[role=link],[role=checkbox],[role=radio],[role=tab],[role=menuitem],[role=option],[role=switch],[role=combobox],[role=textbox],[role=searchbox],[contenteditable=""],[contenteditable=true],[onclick]';

  const isVisible = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) return false;
    const s = getComputedStyle(el);
    return s.visibility !== 'hidden' && s.display !== 'none' && parseFloat(s.opacity || '1') > 0.05;
  };
  const inViewport = (el) => {
    const r = el.getBoundingClientRect();
    return r.bottom > 0 && r.right > 0 && r.top < innerHeight && r.left < innerWidth;
  };
  const clean = (t) => (t || '').replace(/\s+/g, ' ').trim();
  const labelOf = (el) => {
    const aria = el.getAttribute('aria-label');
    if (aria) return clean(aria);
    if (el.id) { const l = document.querySelector(`label[for="${CSS.escape(el.id)}"]`); if (l) return clean(l.innerText); }
    const wrap = el.closest('label');
    if (wrap) return clean(wrap.innerText);
    const tag = el.tagName.toLowerCase();
    if (tag === 'input') {
      const t = (el.type || 'text').toLowerCase();
      if (['button', 'submit', 'reset'].includes(t)) return clean(el.value);
      return clean(el.placeholder || el.name || el.title || '');
    }
    if (tag === 'select') return clean(el.getAttribute('name') || el.title || '');
    return clean(el.innerText || el.title || el.alt || el.placeholder || '');
  };
  const roleOf = (el) => {
    const r = el.getAttribute('role');
    if (r) return r === 'searchbox' ? 'textbox' : r;
    const tag = el.tagName.toLowerCase();
    if (tag === 'a') return 'link';
    if (tag === 'button' || tag === 'summary') return 'button';
    if (tag === 'select') return 'select';
    if (tag === 'textarea') return 'textbox';
    if (tag === 'input') {
      const t = (el.type || 'text').toLowerCase();
      if (['button', 'submit', 'reset', 'image'].includes(t)) return 'button';
      if (t === 'checkbox') return 'checkbox';
      if (t === 'radio') return 'radio';
      return 'textbox';
    }
    if (el.isContentEditable) return 'textbox';
    return 'button';
  };
  const idOf = (el) => {
    if (!el.dataset.layaId) el.dataset.layaId = 'e' + (++counter);
    return el.dataset.layaId;
  };

  function observe() {
    const elements = [];
    document.querySelectorAll(INTERACTIVE).forEach((el) => {
      if (!isVisible(el)) return;
      const tag = el.tagName.toLowerCase();
      const type = tag === 'input' ? (el.type || 'text').toLowerCase() : '';
      const isPw = type === 'password';
      elements.push({
        id: idOf(el),
        role: roleOf(el),
        label: labelOf(el).slice(0, 90),
        value: isPw ? (el.value ? '***' : '') : (tag === 'select' ? (el.selectedOptions[0] ? clean(el.selectedOptions[0].text) : '') : clean(el.value || (el.isContentEditable ? el.textContent : '')).slice(0, 120)),
        type,
        checked: !!el.checked,
        disabled: !!el.disabled || el.getAttribute('aria-disabled') === 'true',
        required: !!el.required,
        href: tag === 'a' ? el.getAttribute('href') || '' : '',
        inViewport: inViewport(el),
      });
    });
    const alerts = [...document.querySelectorAll('[role=alert],.error,.alert,[aria-live=assertive]')].filter(isVisible).map((e) => clean(e.innerText)).filter(Boolean).slice(0, 3);
    const dialogs = [...document.querySelectorAll('dialog[open],[role=dialog],[role=alertdialog],[aria-modal=true]')].filter(isVisible).length;
    return {
      url: location.href,
      title: document.title,
      text: clean(document.body ? document.body.innerText : '').slice(0, 700),
      alerts, dialogs, elements,
    };
  }

  const setNative = (el, value) => {
    const proto = el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : el instanceof HTMLSelectElement ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, value);
  };
  const fire = (el, ...types) => types.forEach((t) => el.dispatchEvent(new Event(t, { bubbles: true })));

  function act(a) {
    if (a.kind === 'scroll') { window.scrollBy(0, a.dy || 500); return { ok: true }; }
    const el = a.id ? document.querySelector(`[data-laya-id="${a.id}"]`) : null;
    if (!el) return { ok: false, error: 'element_not_found' };
    if (!isVisible(el)) return { ok: false, error: 'element_not_visible' };
    el.scrollIntoView({ block: 'center', inline: 'center' });
    const tag = el.tagName.toLowerCase();
    const type = tag === 'input' ? (el.type || 'text').toLowerCase() : '';
    if (a.kind === 'click') { el.focus && el.focus(); el.click(); return { ok: true }; }
    if (a.kind === 'type') {
      if (type === 'password') return { ok: false, error: 'password_field' };  // defence in depth: the agent never types secrets
      el.focus();
      if (el.isContentEditable) { el.textContent = a.clear === false ? el.textContent + a.value : a.value; fire(el, 'input'); return { ok: true, value: el.textContent }; }
      setNative(el, a.clear === false ? el.value + a.value : a.value);
      fire(el, 'input', 'change');
      return { ok: true, value: el.value };
    }
    if (a.kind === 'select') {
      const want = String(a.value).toLowerCase();
      const opt = [...el.options].find((o) => o.text.toLowerCase() === want || o.value.toLowerCase() === want) || [...el.options].find((o) => o.text.toLowerCase().includes(want));
      if (!opt) return { ok: false, error: 'option_not_found' };
      setNative(el, opt.value); fire(el, 'input', 'change');
      return { ok: true, value: opt.text };
    }
    if (a.kind === 'check') {
      const want = a.value === undefined ? true : !['false', '0', 'off', 'no', false].includes(a.value);
      if (el.checked !== want) el.click();
      return { ok: true, checked: el.checked };
    }
    if (a.kind === 'key') {
      el.focus();
      const key = a.key || 'Enter';
      for (const t of ['keydown', 'keypress', 'keyup']) el.dispatchEvent(new KeyboardEvent(t, { key, code: key, bubbles: true }));
      if (key === 'Enter' && el.form && tag === 'input') { el.form.requestSubmit ? el.form.requestSubmit() : el.form.submit(); }
      return { ok: true };
    }
    return { ok: false, error: 'unknown_action' };
  }
  window.__laya = { observe, act };
})();
