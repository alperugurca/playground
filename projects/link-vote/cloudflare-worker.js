// Ascent — links rise to the surface with every vote.
// Inspired only by the underwater photo: deep blue, surface light, bubbles.
// Cloudflare Worker + D1. Edit only CONFIG.

/* CONFIG */
const BRAND = "Ascent";
const HOURLY = { add: 5, vote: 10 };

/* STORE */
let started;
const start = (db) => started ??= (async () => {
  await db.prepare(
    "CREATE TABLE IF NOT EXISTS links(id INTEGER PRIMARY KEY, label TEXT, href TEXT UNIQUE, up INTEGER DEFAULT 0, down INTEGER DEFAULT 0)").run();
  // Older tables only had net "points": add up/down columns and split the net into them.
  for (const col of ["up", "down"]) try { await db.prepare(`ALTER TABLE links ADD COLUMN ${col} INTEGER DEFAULT 0`).run(); } catch {}
  try { await db.prepare("UPDATE links SET up = MAX(points, 0), down = MAX(-points, 0) WHERE points != 0 AND up = 0 AND down = 0").run(); } catch {}
})();
const ranked = async (db) =>
  (await db.prepare("SELECT *, (up - down) AS points FROM links ORDER BY points DESC, id").all()).results;

/* ESCAPE (user text -> safe HTML) */
const safe = (s) => String(s).replace(/[&<>"']/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m]));

/* VIEW */
const view = (rows) => {
  return new Response(`<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>${BRAND}</title>
<style>
  *{box-sizing:border-box;margin:0}
  body{font-family:system-ui;color:#463E5E;min-height:100vh;padding:84px 20px 90px;background:#EDE9F6;zoom:1.25}
  header,main{max-width:680px;margin:auto}
  header{text-align:center;margin-bottom:42px}
  h1{font-size:56px;font-weight:250;letter-spacing:.12em;text-indent:.12em}
  .counts{display:flex;flex-direction:row;gap:10px;align-items:center}
  .counts .total{font-size:19px;font-weight:800;min-width:34px;text-align:center;color:#463E5E}
  #new{display:flex;gap:10px;margin-bottom:20px}
  #new input{flex:1;min-width:0;border:1px solid #DCD2F0;background:#E7DFF8;border-radius:0;clip-path:polygon(8px 0,calc(100% - 8px) 0,100% 8px,100% calc(100% - 8px),calc(100% - 8px) 100%,8px 100%,0 calc(100% - 8px),0 8px);color:#463E5E;padding:12px 14px;font-size:14px}
  #new input:focus{outline:none;border-color:#C3B2E0} #new input::placeholder{color:#9A90B3}
  #new button{border:0;border-radius:0;padding:12px 20px;background:#B79FE0;color:#463E5E;font-weight:650;cursor:pointer;clip-path:polygon(8px 0,calc(100% - 8px) 0,100% 8px,100% calc(100% - 8px),calc(100% - 8px) 100%,8px 100%,0 calc(100% - 8px),0 8px)}
  ol{list-style:none;padding:0}
  li{position:relative;display:flex;align-items:center;gap:16px;background:#C9B6E8;clip-path:polygon(10px 0,calc(100% - 10px) 0,100% 10px,100% calc(100% - 10px),calc(100% - 10px) 100%,10px 100%,0 calc(100% - 10px),0 10px);padding:11px 17px;margin-bottom:10px}
  li::before{content:"";position:absolute;inset:1px;background:#E7DFF8;clip-path:polygon(10px 0,calc(100% - 10px) 0,100% 10px,100% calc(100% - 10px),calc(100% - 10px) 100%,10px 100%,0 calc(100% - 10px),0 10px)}
  li>*{position:relative}
  .rank{color:#9A90B3;font-size:15px;font-weight:800;min-width:24px;text-align:right}
  .txt{flex:1;min-width:0}
  .txt a{color:#463E5E;font-size:17px;font-weight:650;text-decoration:none}
  .txt a:hover{color:#7C6AAE}
  .txt .host{display:block;color:#9A90B3;font-size:12.5px;font-weight:400;margin-top:3px;text-decoration:none}
  .txt .host:hover{color:#7C6AAE;text-decoration:underline}
  /* Vote buttons: bare icons, no boxes, on the right. */
  form.rise{display:flex;flex-direction:column;align-items:center;gap:1px}
  form.rise button{cursor:pointer;background:none;border:0;box-shadow:none;width:60px;height:50px;padding:0;transition:transform .12s}
  form.rise button:hover{transform:scale(1.18)}
  form.rise button:active{transform:scale(.92)}
  form.rise svg{width:37px;height:37px;display:block;margin:auto}
  form.rise .up{color:#3FA46A;position:relative}
  form.rise .badge{position:absolute;top:1px;left:0;right:0;display:flex;justify-content:center;font-size:9px;font-weight:700;color:#9A90B3;opacity:.3}
  form.rise .down{color:#E0655A;position:relative}
  form.rise .badgeDown{position:absolute;bottom:1px;left:0;right:0;display:flex;justify-content:center;font-size:9px;font-weight:700;color:#9A90B3;opacity:.3}
  .none{color:#9A90B3;text-align:center;padding:26px 0}
</style>
<header><h1>${BRAND}</h1>
</header>
<main>
  <form id="new" method="POST">
    <input name="label" placeholder="Title" maxlength="120" required>
    <input name="href" placeholder="https://…" maxlength="500" required>
    <button>Add</button>
  </form>
  <ol id="sea">
${rows.map((r, i) => `    <li>
      <div class="rank">${i + 1}</div>
      <div class="txt"><a href="${safe(r.href)}" target="_blank" rel="noopener">${safe(r.label)}</a><a class="host" href="${safe(r.href)}" target="_blank" rel="noopener">${safe(new URL(r.href).hostname)}</a></div>
      <form class="rise" method="POST"><input type="hidden" name="id" value="${r.id}">
        <div class="counts" data-up="${r.up}" data-down="${r.down}"><button class="up" name="go" value="1" aria-label="Upvote"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M5 15l7-7 7 7"/></svg><span class="badge">${r.up}</span></button><b class="total">${r.up - r.down}</b><button class="down" name="go" value="-1" aria-label="Downvote"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M19 9l-7 7-7-7"/></svg><span class="badgeDown">${r.down}</span></button></div></form>
    </li>`).join("\n") || '    <p class="none">The water is empty. Drop the first link.</p>'}
  </ol>
</main>
<script>
  const cap = (key, max) => {
    const t = Date.now(), hits = JSON.parse(localStorage.getItem(key) || "[]").filter((x) => t - x < 36e5);
    if (hits.length >= max) return false;
    return localStorage.setItem(key, JSON.stringify([...hits, t])), true;
  };
  const net = (li) => { const c = li.querySelector(".counts"); return +c.dataset.up - +c.dataset.down; };
  const floatUp = () => {
    const sea = document.getElementById("sea"), items = [...sea.querySelectorAll("li")];
    const sorted = [...items].sort((a, b) => net(b) - net(a));
    if (sorted.every((el, i) => el === items[i])) return; // sira ayni: DOM'a dokunma (yoksa gecis bastan baslar)
    sorted.forEach((li, i) => { sea.append(li); li.querySelector(".rank").textContent = i + 1; });
  };
  document.addEventListener("submit", async (e) => {
    const form = e.target;
    if (form.id === "new") {
      if (!cap("adds", ${HOURLY.add})) { e.preventDefault(); alert("Limit: ${HOURLY.add} entries per hour."); }
      return;
    }
    e.preventDefault();
    if (!cap("votes", ${HOURLY.vote})) return alert("Limit: ${HOURLY.vote} votes per hour.");
    const li = form.closest("li"), box = li.querySelector(".counts"), step = +(e.submitter?.value || 1);
    if (step > 0) box.dataset.up++; else box.dataset.down++;
    box.querySelector(".total").textContent = +box.dataset.up - +box.dataset.down;
    li.querySelector(".badge").textContent = box.dataset.up;
    li.querySelector(".badgeDown").textContent = box.dataset.down;
    floatUp();
    const body = new FormData(form); body.set("go", step);
    const fresh = await fetch(location.href, { method: "POST", headers: { "x-ajax": "1" }, body }).then((r) => r.json()).catch(() => null);
    if (!fresh) return location.reload();
    document.querySelectorAll("li").forEach((item) => {
      const hit = fresh.find((x) => x.id === +item.querySelector("input").value);
      if (!hit) return;
      const b = item.querySelector(".counts");
      b.dataset.up = hit.up; b.dataset.down = hit.down;
      b.querySelector(".total").textContent = hit.up - hit.down;
      item.querySelector(".badge").textContent = hit.up;
      item.querySelector(".badgeDown").textContent = hit.down;
    });
    floatUp();
  });
</script>`, { headers: { "Content-Type": "text/html; charset=utf-8" } });
};

/* SERVER */
export default {
  async fetch(request, env) {
    await start(env.DB);
    if (request.method !== "POST") return view(await ranked(env.DB));

    const data = await request.formData();
    if (data.get("id")) {
      const step = Math.sign(parseInt(data.get("go")) || 0);
      if (step) await env.DB.prepare(`UPDATE links SET ${step > 0 ? "up" : "down"} = ${step > 0 ? "up" : "down"} + 1 WHERE id = ?`).bind(data.get("id")).run();
      if (request.headers.get("x-ajax"))
        return Response.json((await env.DB.prepare("SELECT id, up, down FROM links").all()).results);
    } else {
      const label = String(data.get("label") || "").trim().slice(0, 120);
      let href = String(data.get("href") || "").trim().slice(0, 500);
      if (!/^https?:\/\//i.test(href)) href = "https://" + href;
      if (label) try { new URL(href); await env.DB.prepare("INSERT OR IGNORE INTO links(label, href) VALUES (?, ?)").bind(label, href).run(); } catch {}
    }
    return Response.redirect(request.url);
  },
};
