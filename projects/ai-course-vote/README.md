# AI Course Vote

Tek sayfada AI kurslarını listeleyen, kurs ekleme ve upvote/downvote özellikli temel sürüm.
Oylar Cloudflare D1'de saklanır, liste en yüksek net puana (▲ − ▼) göre sıralanır.

## Özellikler

- Tek sayfa: tüm kurslar, net puana göre sıralı
- Kurs ekleme: yalnızca başlık + URL
- Upvote / downvote — sayfa yenilenmeden (fetch API)
- **Tekrarlı oya izin verilir** — oy sınırı yoktur
- Mobil uyumlu, sade tasarım

## Hız sınırı (rate limit)

Oylar sınırsız olsa da botların otomatik olarak binlerce oy göndermesini engellemek için
IP bazlı (IP'ler SHA-256 ile hash'lenerek saklanır) temel sınırlar vardır — `src/index.js`:

| İşlem | Sınır |
|---|---|
| Oy verme | 30 oy / 60 saniye / IP |
| Kurs ekleme | 10 kurs / 1 saat / IP |

Sınır aşılırsa API `429` + `Retry-After` döner. Normal insan kullanımında bu sınırlara ulaşılmaz;
sabitler `src/index.js` başından değiştirilebilir.

## Kurulum ve çalıştırma

```bash
npm install

# D1 veritabanını oluşturun ve çıkan database_id değerini wrangler.toml'a yazın
npm run db:create

# Şemayı uygulayın (yerel)
npm run db:migrate:local

# Geliştirme sunucusu
npm run dev
```

## Yayınlama (deploy)

```bash
npm run db:migrate:remote
npm run deploy
```

## API

| Metot | Yol | Açıklama |
|---|---|---|
| GET | `/api/courses` | Kurslar (net puan ↓ sıralı) |
| POST | `/api/courses` | `{ "title": "...", "url": "https://..." }` |
| POST | `/api/courses/:id/vote` | `{ "direction": "up" \| "down" }` |
| GET | `/api/health` | Sağlık kontrolü |

## Yapı

```
src/index.js          Worker: API + statik dosya sunumu + rate limit
public/               index.html, styles.css, app.js
migrations/           D1 şeması (courses, vote_events, submit_events)
wrangler.toml         Worker + D1 + assets yapılandırması
```
