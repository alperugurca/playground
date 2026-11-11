CREATE TABLE IF NOT EXISTS courses (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  title      TEXT    NOT NULL,
  url        TEXT    NOT NULL,
  upvotes    INTEGER NOT NULL DEFAULT 0,
  downvotes  INTEGER NOT NULL DEFAULT 0,
  created_at TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- Her oy bir satir olarak da loglanir: hiz siniri (rate limit) bu tablo
-- uzerinden IP bazinda hesaplanir. Oy sayaclari courses tablosundadir.
CREATE TABLE IF NOT EXISTS vote_events (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  course_id  INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
  direction  INTEGER NOT NULL CHECK (direction IN (1, -1)),
  ip_hash    TEXT    NOT NULL,
  created_at INTEGER NOT NULL -- unix epoch (saniye)
);
CREATE INDEX IF NOT EXISTS idx_vote_events_ip_time ON vote_events (ip_hash, created_at);

-- Kurs ekleme islemleri icin hiz siniri logu.
CREATE TABLE IF NOT EXISTS submit_events (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  ip_hash    TEXT    NOT NULL,
  created_at INTEGER NOT NULL -- unix epoch (saniye)
);
CREATE INDEX IF NOT EXISTS idx_submit_events_ip_time ON submit_events (ip_hash, created_at);
