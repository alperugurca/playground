# Link Vote — bağlantı oylama sitesi

Kullanıcılar başlık + link ekler, oy verir; liste oy sayısına göre
çoktan aza sıralanır. Herkes aynı listeyi görür.

## Cloudflare Workers — `cloudflare-worker.js`

Kod ziyaretçiye en yakın Cloudflare noktasından çalışır, o yüzden açılış/tıklama hızlıdır.
Ücretsiz sınır: günde 100 bin istek — bu site için fazlasıyla yeter.

### Kurulum (panelden, ~5 dakika)

1. https://dash.cloudflare.com → ücretsiz üye ol
2. Sol menü **Workers & Pages → Create an app → Start with Hello World!** → isim ver (örn. `link-vote`) → **Deploy**
3. **Edit code** → içindeki her şeyi sil → `cloudflare-worker.js` içeriğini yapıştır → **Deploy**
4. Sol menü **Storage & Databases → D1 → Create database** → isim ver (örn. `vote-db`) → **Create**
5. Worker'ına dön → **Settings → Bindings → Add → D1 database**
   → Variable name: **`DB`** (büyük harf, kod böyle bekliyor) → veritabanını seç → **Deploy**
6. Worker linkini aç (`link-vote.<kullanici>.workers.dev`) — tablo ilk açılışta otomatik oluşur, site hazır

Oyları/öğeleri sıfırlamak için: D1 panelinde veritabanını silip yeniden oluştur (adım 4-5).

## Neden sunucu/platform var?

"Herkes aynı oyu görsün" istenirse oyların ortak bir yerde toplanması şarttır.
Tek HTML dosyası oyları sadece kendi tarayıcısında saklar. Bu yüzden en basit
ortak çözüm, hazır platformda çalışan tek dosyalık bu küçük koddur.

## Sınır

Giriş yok → aynı kişi tekrar oy verebilir. Kesin engel üyelik sistemi ister;
cookie kontrolü sadece basit caydırıcı olur, istenirse eklenir.
