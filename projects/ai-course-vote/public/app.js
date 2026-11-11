const listEl = document.getElementById("course-list");
const emptyEl = document.getElementById("empty-state");
const statusEl = document.getElementById("status");
const form = document.getElementById("add-form");
const formMessage = document.getElementById("form-message");
const addButton = document.getElementById("add-button");
const titleInput = document.getElementById("input-title");
const urlInput = document.getElementById("input-url");

let courses = [];
let lastVoted = null;
let lastVotedTimer = null;

function setMessage(el, text, kind = "") {
  el.textContent = text;
  el.className = `message${kind ? ` ${kind}` : ""}`;
}

function sortCourses() {
  courses.sort((a, b) => b.score - a.score || b.upvotes - a.upvotes || b.id - a.id);
}

const CHEVRON_UP =
  '<svg viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M3.5 10.5 8 6l4.5 4.5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const CHEVRON_DOWN =
  '<svg viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M3.5 5.5 8 10l4.5-4.5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>';

function render() {
  sortCourses();
  listEl.replaceChildren();

  const totalVotes = courses.reduce((sum, c) => sum + c.upvotes + c.downvotes, 0);
  document.getElementById("course-count").textContent =
    courses.length > 0 ? `${courses.length} kurs` : "Henüz kurs yok";
  document.getElementById("stats").textContent =
    courses.length > 0 ? `${courses.length} kurs · ${totalVotes} oy kullanıldı` : "";

  courses.forEach((course, index) => {
    let host = "";
    try {
      host = new URL(course.url).hostname.replace(/^www\./, "");
    } catch {
      host = "";
    }

    const item = document.createElement("article");
    item.className = "course";

    const voteBox = document.createElement("div");
    voteBox.className = "vote";

    const net = document.createElement("div");
    net.className = "net";
    const netScore = document.createElement("span");
    netScore.className = "net-score";
    netScore.textContent = String(course.score);
    const netLabel = document.createElement("span");
    netLabel.className = "net-label";
    netLabel.textContent = "net";
    net.append(netScore, netLabel);

    const upBtn = document.createElement("button");
    upBtn.type = "button";
    upBtn.className = "up";
    upBtn.innerHTML = CHEVRON_UP;
    const upCount = document.createElement("span");
    upCount.textContent = String(course.upvotes);
    upBtn.append(upCount);
    upBtn.title = "Olumlu oy ver";
    upBtn.setAttribute("aria-label", `${course.title} için olumlu oy ver`);
    if (lastVoted?.id === course.id && lastVoted?.direction === "up") upBtn.classList.add("voted");
    upBtn.addEventListener("click", () => vote(course.id, "up"));

    const downBtn = document.createElement("button");
    downBtn.type = "button";
    downBtn.className = "down";
    downBtn.innerHTML = CHEVRON_DOWN;
    const downCount = document.createElement("span");
    downCount.textContent = String(course.downvotes);
    downBtn.append(downCount);
    downBtn.title = "Olumsuz oy ver";
    downBtn.setAttribute("aria-label", `${course.title} için olumsuz oy ver`);
    if (lastVoted?.id === course.id && lastVoted?.direction === "down") downBtn.classList.add("voted");
    downBtn.addEventListener("click", () => vote(course.id, "down"));

    voteBox.append(net, upBtn, downBtn);

    const main = document.createElement("div");
    main.className = "course-main";

    const rank = document.createElement("span");
    rank.className = "rank";
    rank.textContent = String(index + 1).padStart(2, "0");

    const text = document.createElement("div");
    text.className = "course-text";

    const link = document.createElement("a");
    link.className = "course-title";
    link.href = course.url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = course.title;
    const ext = document.createElement("span");
    ext.className = "ext";
    ext.textContent = "↗";
    ext.setAttribute("aria-hidden", "true");
    link.append(ext);

    const meta = document.createElement("span");
    meta.className = "course-meta";
    const total = course.upvotes + course.downvotes;
    meta.textContent = `${host ? `${host} · ` : ""}${total} oy`;

    text.append(link, meta);
    main.append(rank, text);
    item.append(main, voteBox);
    listEl.append(item);
  });

  emptyEl.hidden = courses.length > 0;
}

async function api(path, options) {
  const res = await fetch(path, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(data.error || "Bir hata oluştu.");
    err.status = res.status;
    throw err;
  }
  return data;
}

async function loadCourses() {
  try {
    const data = await api("/api/courses");
    courses = data.courses;
    render();
  } catch (err) {
    setMessage(statusEl, err.message, "error");
  }
}

// Sayfa yenilenmeden oy verilir; sunucudan donen guncel kurs listeye islenir
// ve liste net puana gore yeniden siralanir.
async function vote(courseId, direction) {
  setMessage(statusEl, "");
  try {
    const data = await api(`/api/courses/${courseId}/vote`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ direction }),
    });
    const idx = courses.findIndex((c) => c.id === courseId);
    if (idx !== -1) courses[idx] = data.course;
    lastVoted = { id: courseId, direction };
    render();
    clearTimeout(lastVotedTimer);
    lastVotedTimer = setTimeout(() => {
      lastVoted = null;
      render();
    }, 1400);
  } catch (err) {
    setMessage(statusEl, err.message, "error");
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setMessage(formMessage, "");

  const title = titleInput.value.trim();
  const url = urlInput.value.trim();
  if (title.length < 2 || title.length > 120) {
    setMessage(formMessage, "Başlık 2–120 karakter arasında olmalıdır.", "error");
    return;
  }
  try {
    const parsed = new URL(url);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") throw new Error();
  } catch {
    setMessage(formMessage, "Geçerli bir http(s) URL girin.", "error");
    return;
  }

  addButton.disabled = true;
  try {
    const data = await api("/api/courses", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ title, url }),
    });
    courses.push(data.course);
    render();
    form.reset();
    setMessage(formMessage, "Kurs eklendi.", "success");
  } catch (err) {
    setMessage(formMessage, err.message, "error");
  } finally {
    addButton.disabled = false;
  }
});

loadCourses();
