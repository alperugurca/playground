// AI Course Vote — Cloudflare Worker (API + statik dosya sunumu)
// Oylar sinirsizdir (ayni kisi tekrar oy verebilir), ancak asagidaki
// hiz sinirlari botlarin otomatik olarak binlerce oy gondermesini engeller.
const VOTE_LIMIT = 30; // ayni IP'den pencere icinde maks. oy
const VOTE_WINDOW_SEC = 60; // 60 saniyede 30 oy
const COURSE_LIMIT = 10; // ayni IP'den pencere icinde maks. kurs ekleme
const COURSE_WINDOW_SEC = 3600; // 1 saatte 10 kurs

const JSON_HEADERS = { "content-type": "application/json; charset=utf-8" };

function json(data, status = 200, extraHeaders = {}) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { ...JSON_HEADERS, ...extraHeaders },
  });
}

function fail(message, status = 400, extraHeaders = {}) {
  return json({ error: message }, status, extraHeaders);
}

function getClientIp(request) {
  const cfIp = request.headers.get("CF-Connecting-IP");
  if (cfIp) return cfIp.trim();
  const forwarded = request.headers.get("X-Forwarded-For");
  if (forwarded) return forwarded.split(",")[0].trim();
  return "unknown";
}

async function hashIp(ip) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(ip));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

// Pencere icinde limit asilmissa kalan bekleme suresini (sn) dondurur, yoksa 0.
async function rateLimitRetryAfter(db, table, ipHash, limit, windowSec) {
  const now = Math.floor(Date.now() / 1000);
  const row = await db
    .prepare(
      `SELECT COUNT(*) AS cnt, MIN(created_at) AS oldest FROM ${table}
       WHERE ip_hash = ? AND created_at > ?`
    )
    .bind(ipHash, now - windowSec)
    .first();
  if (!row || row.cnt < limit) return 0;
  const oldest = row.oldest ?? now;
  return Math.max(1, oldest + windowSec - now);
}

function rateLimited(retryAfter) {
  return fail(
    "Çok fazla istek gönderdiniz. Lütfen biraz bekleyip tekrar deneyin.",
    429,
    { "Retry-After": String(retryAfter) }
  );
}

function validateTitle(raw) {
  if (typeof raw !== "string") return null;
  const title = raw.trim();
  if (title.length < 2 || title.length > 120) return null;
  return title;
}

function validateUrl(raw) {
  if (typeof raw !== "string") return null;
  const value = raw.trim();
  if (!value || value.length > 2048) return null;
  try {
    const url = new URL(value);
    if (url.protocol !== "http:" && url.protocol !== "https:") return null;
    return url.href;
  } catch {
    return null;
  }
}

const COURSE_COLUMNS =
  "id, title, url, upvotes, downvotes, (upvotes - downvotes) AS score, created_at";

async function listCourses(env) {
  const { results } = await env.DB.prepare(
    `SELECT ${COURSE_COLUMNS} FROM courses
     ORDER BY score DESC, upvotes DESC, created_at DESC, id DESC`
  ).all();
  return json({ courses: results });
}

async function createCourse(request, env) {
  let body;
  try {
    body = await request.json();
  } catch {
    return fail("Geçersiz istek gövdesi.");
  }

  const title = validateTitle(body?.title);
  if (!title) return fail("Başlık 2–120 karakter arasında olmalıdır.");
  const url = validateUrl(body?.url);
  if (!url) return fail("Geçerli bir http(s) URL girin.");

  const ipHash = await hashIp(getClientIp(request));
  const retry = await rateLimitRetryAfter(env.DB, "submit_events", ipHash, COURSE_LIMIT, COURSE_WINDOW_SEC);
  if (retry > 0) return rateLimited(retry);

  const now = Math.floor(Date.now() / 1000);
  const inserted = await env.DB.prepare("INSERT INTO courses (title, url) VALUES (?, ?) RETURNING id")
    .bind(title, url)
    .first();
  await env.DB.prepare("INSERT INTO submit_events (ip_hash, created_at) VALUES (?, ?)")
    .bind(ipHash, now)
    .run();

  const course = await env.DB.prepare(`SELECT ${COURSE_COLUMNS} FROM courses WHERE id = ?`)
    .bind(inserted.id)
    .first();
  return json({ course }, 201);
}

async function vote(request, env, courseId) {
  if (!Number.isInteger(courseId) || courseId <= 0) return fail("Geçersiz kurs.", 404);

  let body;
  try {
    body = await request.json();
  } catch {
    return fail("Geçersiz istek gövdesi.");
  }
  const direction = body?.direction;
  if (direction !== "up" && direction !== "down") {
    return fail('Oy yönü "up" veya "down" olmalıdır.');
  }

  const exists = await env.DB.prepare("SELECT id FROM courses WHERE id = ?").bind(courseId).first();
  if (!exists) return fail("Kurs bulunamadı.", 404);

  const ipHash = await hashIp(getClientIp(request));
  const retry = await rateLimitRetryAfter(env.DB, "vote_events", ipHash, VOTE_LIMIT, VOTE_WINDOW_SEC);
  if (retry > 0) return rateLimited(retry);

  const now = Math.floor(Date.now() / 1000);
  const delta = direction === "up" ? 1 : -1;
  const column = direction === "up" ? "upvotes" : "downvotes";

  await env.DB.batch([
    env.DB.prepare("INSERT INTO vote_events (course_id, direction, ip_hash, created_at) VALUES (?, ?, ?, ?)")
      .bind(courseId, delta, ipHash, now),
    env.DB.prepare(`UPDATE courses SET ${column} = ${column} + 1 WHERE id = ?`).bind(courseId),
  ]);

  const course = await env.DB.prepare(`SELECT ${COURSE_COLUMNS} FROM courses WHERE id = ?`)
    .bind(courseId)
    .first();
  return json({ course });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const { pathname } = url;

    try {
      if (pathname === "/api/health") return json({ ok: true });

      if (pathname === "/api/courses") {
        if (request.method === "GET") return await listCourses(env);
        if (request.method === "POST") return await createCourse(request, env);
        return fail("Yöntem desteklenmiyor.", 405);
      }

      const voteMatch = pathname.match(/^\/api\/courses\/(\d+)\/vote$/);
      if (voteMatch) {
        if (request.method !== "POST") return fail("Yöntem desteklenmiyor.", 405);
        return await vote(request, env, Number(voteMatch[1]));
      }

      if (pathname.startsWith("/api/")) return fail("Bulunamadı.", 404);

      // API disi her sey statik dosyalardan sunulur.
      return await env.ASSETS.fetch(request);
    } catch (err) {
      console.error(err);
      return fail("Sunucu hatası.", 500);
    }
  },
};
