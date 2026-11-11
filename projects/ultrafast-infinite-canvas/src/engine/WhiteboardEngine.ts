export type ToolType = 'select' | 'rect' | 'ellipse' | 'line';

export interface TelemetryStats {
  total: number;
  visible: number;
  renderMs: number;
  zoom: number;
}

export type TelemetryCallback =
  | ((total: number, renderMs: number) => void)
  | ((stats: TelemetryStats) => void);

export const PALETTE = [
  '#60a5fa', // Mavi
  '#34d399', // Yeşil
  '#f472b6', // Pembe
  '#fbbf24', // Sarı
  '#a78bfa', // Mor
  '#f87171', // Kırmızı
] as const;

export const MAX_SHAPES = 100_000;

export class WhiteboardEngine {
  // 1. DÜZ VERİ TAMPONLARI (Struct-of-Arrays - Sıfır GC Maliyeti)
  private coords = new Float32Array(MAX_SHAPES * 4); // [x1, y1, x2, y2, ...]
  private types = new Uint8Array(MAX_SHAPES);       // 1: Rect, 2: Ellipse, 3: Line
  private colors = new Uint8Array(MAX_SHAPES);      // PALETTE indeksleri (0-5)
  public shapeCount = 0;

  // Renk bazlı toplu çizim (Color Batching) için önceden ayrılmış tamponlar
  private colorBuckets = PALETTE.map(() => new Int32Array(MAX_SHAPES));
  private bucketCounts = new Int32Array(PALETTE.length);

  // 2. KAMERA VE UYGULAMA DURUMU
  public currentTool: ToolType = 'rect';
  public currentColorIndex = 0;
  public camX = 0;
  public camY = 0;
  public camZoom = 1;

  private isPanning = false;
  private isSpaceDown = false;
  private isDrawing = false;
  private isDraggingShape = false;
  private startX = 0;
  private startY = 0;
  private currX = 0;
  private currY = 0;
  public selectedIndex = -1;
  private dragOffsetX1 = 0;
  private dragOffsetY1 = 0;
  private dragOffsetX2 = 0;
  private dragOffsetY2 = 0;

  // 3. CANVAS BAĞLAMLARI
  private sCtx: CanvasRenderingContext2D;
  private aCtx: CanvasRenderingContext2D;
  private dpr = 1;

  // Izgara ve statik desenler için OffscreenCanvas önbelleği
  private gridCache: OffscreenCanvas | HTMLCanvasElement;
  private gridPattern: CanvasPattern | null = null;

  // Olay dinleyici referansları (Temizleme / Unmount için)
  private boundResize: () => void;
  private boundPointerDown: (e: PointerEvent) => void;
  private boundPointerMove: (e: PointerEvent) => void;
  private boundPointerUp: (e: PointerEvent) => void;
  private boundWheel: (e: WheelEvent) => void;
  private boundKeyDown: (e: KeyboardEvent) => void;
  private boundKeyUp: (e: KeyboardEvent) => void;

  constructor(
    private staticCanvas: HTMLCanvasElement,
    private activeCanvas: HTMLCanvasElement,
    private onTelemetryUpdate?: TelemetryCallback
  ) {
    this.sCtx = this.staticCanvas.getContext('2d', { alpha: false })!;
    this.aCtx = this.activeCanvas.getContext('2d', { desynchronized: true })!;

    // Offscreen Canvas oluşturma
    const size = 32;
    if (typeof OffscreenCanvas !== 'undefined') {
      this.gridCache = new OffscreenCanvas(size, size);
    } else {
      this.gridCache = document.createElement('canvas');
      this.gridCache.width = size;
      this.gridCache.height = size;
    }

    this.initGridCache();

    // Dinleyicileri bağlama (Cleanup için referanslar saklanır)
    this.boundResize = () => this.resize();
    this.boundPointerDown = (e) => this.onPointerDown(e);
    this.boundPointerMove = (e) => this.onPointerMove(e);
    this.boundPointerUp = (e) => this.onPointerUp(e);
    this.boundWheel = (e) => this.onWheel(e);
    this.boundKeyDown = (e) => this.onKeyDown(e);
    this.boundKeyUp = (e) => this.onKeyUp(e);

    this.bindEvents();
    this.resize();
    this.addDefaultShapes();
  }

  private initGridCache() {
    const size = 32;
    this.gridCache.width = size;
    this.gridCache.height = size;
    const ctx = this.gridCache.getContext('2d') as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null;
    if (ctx) {
      ctx.fillStyle = '#22222a';
      ctx.fillRect(0, 0, 2, 2); // Noktalı ızgara deseni GPU dokusuna bir kez yazılır
      try {
        this.gridPattern = this.sCtx.createPattern(this.gridCache as any, 'repeat');
      } catch {
        this.gridPattern = null;
      }
    }
  }

  private bindEvents() {
    window.addEventListener('resize', this.boundResize);
    window.addEventListener('keydown', this.boundKeyDown);
    window.addEventListener('keyup', this.boundKeyUp);

    this.activeCanvas.addEventListener('pointerdown', this.boundPointerDown);
    this.activeCanvas.addEventListener('pointermove', this.boundPointerMove);
    this.activeCanvas.addEventListener('pointerup', this.boundPointerUp);
    this.activeCanvas.addEventListener('wheel', this.boundWheel, { passive: false });
  }

  public destroy() {
    window.removeEventListener('resize', this.boundResize);
    window.removeEventListener('keydown', this.boundKeyDown);
    window.removeEventListener('keyup', this.boundKeyUp);

    this.activeCanvas.removeEventListener('pointerdown', this.boundPointerDown);
    this.activeCanvas.removeEventListener('pointermove', this.boundPointerMove);
    this.activeCanvas.removeEventListener('pointerup', this.boundPointerUp);
    this.activeCanvas.removeEventListener('wheel', this.boundWheel);
  }

  public resize() {
    this.dpr = window.devicePixelRatio || 1;
    const w = window.innerWidth;
    const h = window.innerHeight;

    for (const c of [this.staticCanvas, this.activeCanvas]) {
      c.width = w * this.dpr;
      c.height = h * this.dpr;
    }
    this.renderStaticLayer();
    this.renderActiveLayer();
  }

  // Ekran koordinatını Dünya (World) koordinatına çevir
  public toWorldX(screenX: number): number {
    return (screenX - this.camX) / this.camZoom;
  }

  public toWorldY(screenY: number): number {
    return (screenY - this.camY) / this.camZoom;
  }

  public setTool(tool: ToolType) {
    this.currentTool = tool;
    this.selectedIndex = -1;
    this.activeCanvas.style.cursor = tool === 'select' ? 'default' : 'crosshair';
    this.renderActiveLayer();
  }

  public setColor(colorIndex: number) {
    this.currentColorIndex = Math.max(0, Math.min(colorIndex, PALETTE.length - 1));
  }

  // Şekil Ekleme (Sıfır bellek tahsisi)
  public addShape(
    type: number,
    x1: number,
    y1: number,
    x2: number,
    y2: number,
    colorIdx: number = this.currentColorIndex
  ) {
    if (this.shapeCount >= MAX_SHAPES) return;
    const i4 = this.shapeCount * 4;

    if (type !== 3) {
      this.coords[i4]     = Math.min(x1, x2);
      this.coords[i4 + 1] = Math.min(y1, y2);
      this.coords[i4 + 2] = Math.max(x1, x2);
      this.coords[i4 + 3] = Math.max(y1, y2);
    } else {
      this.coords[i4]     = x1;
      this.coords[i4 + 1] = y1;
      this.coords[i4 + 2] = x2;
      this.coords[i4 + 3] = y2;
    }

    this.types[this.shapeCount] = type;
    this.colors[this.shapeCount] = colorIdx % PALETTE.length;
    this.shapeCount++;
  }

  // Seçili şekli sil (O(1) Swap-and-Pop ile sıfır GC)
  public deleteSelected(): boolean {
    if (this.selectedIndex === -1 || this.shapeCount === 0) return false;
    const last = this.shapeCount - 1;

    if (this.selectedIndex !== last) {
      const src4 = last * 4;
      const dst4 = this.selectedIndex * 4;
      this.coords[dst4]     = this.coords[src4];
      this.coords[dst4 + 1] = this.coords[src4 + 1];
      this.coords[dst4 + 2] = this.coords[src4 + 2];
      this.coords[dst4 + 3] = this.coords[src4 + 3];
      this.types[this.selectedIndex] = this.types[last];
      this.colors[this.selectedIndex] = this.colors[last];
    }

    this.shapeCount--;
    this.selectedIndex = -1;
    this.renderStaticLayer();
    this.renderActiveLayer();
    return true;
  }

  // Ters sırayla AABB Hit-Testing
  public hitTest(wx: number, wy: number): number {
    const pad = 6 / this.camZoom;
    for (let i = this.shapeCount - 1; i >= 0; i--) {
      const i4 = i * 4;
      const x1 = Math.min(this.coords[i4], this.coords[i4 + 2]) - pad;
      const y1 = Math.min(this.coords[i4 + 1], this.coords[i4 + 3]) - pad;
      const x2 = Math.max(this.coords[i4], this.coords[i4 + 2]) + pad;
      const y2 = Math.max(this.coords[i4 + 1], this.coords[i4 + 3]) + pad;
      if (wx >= x1 && wx <= x2 && wy >= y1 && wy <= y2) return i;
    }
    return -1;
  }

  /**
   * 4. ANA STATİK RENDER MOTORU (Viewport Culling + Color Batching)
   */
  public renderStaticLayer() {
    const t0 = performance.now();
    const w = window.innerWidth;
    const h = window.innerHeight;

    this.sCtx.save();
    this.sCtx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);

    // Arka planı temizle
    this.sCtx.fillStyle = '#121212';
    this.sCtx.fillRect(0, 0, w, h);

    // Noktalı ızgara deseni (GPU pattern cache)
    if (this.gridPattern) {
      try {
        if ('setTransform' in this.gridPattern && typeof DOMMatrix !== 'undefined') {
          const m = new DOMMatrix().translateSelf(this.camX, this.camY).scaleSelf(this.camZoom, this.camZoom);
          this.gridPattern.setTransform(m);
        }
        this.sCtx.fillStyle = this.gridPattern;
        this.sCtx.fillRect(0, 0, w, h);
      } catch {
        // Pattern transform desteklenmeyen durumlarda sessizce atla
      }
    }

    // Kamera dönüşümü
    this.sCtx.translate(this.camX, this.camY);
    this.sCtx.scale(this.camZoom, this.camZoom);

    // Görünür Dünya Sınırları (Viewport Culling için)
    const viewMinX = -this.camX / this.camZoom;
    const viewMinY = -this.camY / this.camZoom;
    const viewMaxX = (w - this.camX) / this.camZoom;
    const viewMaxY = (h - this.camY) / this.camZoom;

    // 1. AŞAMA: Kırpma (Culling) ve Renk Gruplama
    this.bucketCounts.fill(0);
    let visibleCount = 0;

    for (let i = 0; i < this.shapeCount; i++) {
      if (i === this.selectedIndex && this.isDraggingShape) continue; // Sürüklenen şekil üst katmanda çizilir
      const i4 = i * 4;
      const x1 = this.coords[i4], y1 = this.coords[i4 + 1];
      const x2 = this.coords[i4 + 2], y2 = this.coords[i4 + 3];

      // AABB Görünüm testi
      if (
        Math.max(x1, x2) < viewMinX || Math.min(x1, x2) > viewMaxX ||
        Math.max(y1, y2) < viewMinY || Math.min(y1, y2) > viewMaxY
      ) {
        continue;
      }

      const cIdx = this.colors[i];
      this.colorBuckets[cIdx][this.bucketCounts[cIdx]++] = i;
      visibleCount++;
    }

    // 2. AŞAMA: Renk Bazlı Toplu Çizim
    this.sCtx.lineWidth = 2 / this.camZoom;
    for (let c = 0; c < PALETTE.length; c++) {
      const count = this.bucketCounts[c];
      if (count === 0) continue;

      this.sCtx.strokeStyle = PALETTE[c];
      this.sCtx.beginPath();
      const bucket = this.colorBuckets[c];

      for (let b = 0; b < count; b++) {
        const idx = bucket[b];
        const i4 = idx * 4;
        const x1 = this.coords[i4], y1 = this.coords[i4 + 1];
        const x2 = this.coords[i4 + 2], y2 = this.coords[i4 + 3];
        const t = this.types[idx];

        if (t === 1) { // Dikdörtgen
          this.sCtx.rect(x1, y1, x2 - x1, y2 - y1);
        } else if (t === 2) { // Elips
          const rx = (x2 - x1) * 0.5, ry = (y2 - y1) * 0.5;
          const cx = x1 + rx, cy = y1 + ry;
          this.sCtx.moveTo(cx + rx, cy);
          this.sCtx.ellipse(cx, cy, Math.abs(rx), Math.abs(ry), 0, 0, Math.PI * 2);
        } else if (t === 3) { // Çizgi
          this.sCtx.moveTo(x1, y1);
          this.sCtx.lineTo(x2, y2);
        }
      }
      this.sCtx.stroke(); // Binlerce şekli tek GPU komutunda çizer!
    }

    this.sCtx.restore();

    // Telemetri Güncellemesi
    const dt = performance.now() - t0;
    this.emitTelemetry(dt, visibleCount);
  }

  /**
   * 5. AKTİF KATMAN RENDER MOTORU (Sadece o an çizilen/taşınan şekli çizer)
   * Maliyet: ~0.05 ms
   */
  public renderActiveLayer() {
    const w = window.innerWidth;
    const h = window.innerHeight;

    this.aCtx.save();
    this.aCtx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this.aCtx.clearRect(0, 0, w, h);

    if (!this.isDrawing && this.selectedIndex === -1) {
      this.aCtx.restore();
      return;
    }

    this.aCtx.translate(this.camX, this.camY);
    this.aCtx.scale(this.camZoom, this.camZoom);
    this.aCtx.lineWidth = 2 / this.camZoom;

    // Yeni şekil çiziliyorsa
    if (this.isDrawing) {
      this.aCtx.strokeStyle = PALETTE[this.currentColorIndex];
      this.aCtx.beginPath();
      this.drawSingleShape(this.aCtx, this.getToolTypeId(this.currentTool), this.startX, this.startY, this.currX, this.currY);
      this.aCtx.stroke();
    }

    // Seçili bir şekil varsa (veya sürükleniyorsa)
    if (this.selectedIndex !== -1) {
      const i4 = this.selectedIndex * 4;
      const x1 = this.coords[i4], y1 = this.coords[i4 + 1];
      const x2 = this.coords[i4 + 2], y2 = this.coords[i4 + 3];

      if (this.isDraggingShape) {
        this.aCtx.strokeStyle = PALETTE[this.colors[this.selectedIndex]];
        this.aCtx.beginPath();
        this.drawSingleShape(this.aCtx, this.types[this.selectedIndex], x1, y1, x2, y2);
        this.aCtx.stroke();
      }

      // Seçim Çerçevesi (Bounding Box)
      this.aCtx.strokeStyle = '#38bdf8';
      this.aCtx.setLineDash([4 / this.camZoom, 4 / this.camZoom]);
      this.aCtx.strokeRect(
        Math.min(x1, x2) - 4 / this.camZoom,
        Math.min(y1, y2) - 4 / this.camZoom,
        Math.abs(x2 - x1) + 8 / this.camZoom,
        Math.abs(y2 - y1) + 8 / this.camZoom
      );
    }

    this.aCtx.restore();
  }

  private drawSingleShape(ctx: CanvasRenderingContext2D, type: number, x1: number, y1: number, x2: number, y2: number) {
    if (type === 1) {
      ctx.rect(Math.min(x1, x2), Math.min(y1, y2), Math.abs(x2 - x1), Math.abs(y2 - y1));
    } else if (type === 2) {
      const rx = Math.abs(x2 - x1) * 0.5, ry = Math.abs(y2 - y1) * 0.5;
      const cx = Math.min(x1, x2) + rx, cy = Math.min(y1, y2) + ry;
      ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
    } else if (type === 3) {
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
    }
  }

  private getToolTypeId(tool: ToolType): number {
    return tool === 'rect' ? 1 : tool === 'ellipse' ? 2 : 3;
  }

  private emitTelemetry(dt: number, visibleCount: number) {
    if (!this.onTelemetryUpdate) return;
    const stats: TelemetryStats = {
      total: this.shapeCount,
      visible: visibleCount,
      renderMs: Number(dt.toFixed(2)),
      zoom: Math.round(this.camZoom * 100),
    };

    if (this.onTelemetryUpdate.length === 2) {
      (this.onTelemetryUpdate as (total: number, renderMs: number) => void)(this.shapeCount, stats.renderMs);
    } else {
      (this.onTelemetryUpdate as (stats: TelemetryStats) => void)(stats);
    }
  }

  // 6. POINTER VE ETKİLEŞİM İŞLEYİCİLERİ
  private onPointerDown(e: PointerEvent) {
    this.activeCanvas.setPointerCapture(e.pointerId);

    if (e.button === 1 || this.isSpaceDown) {
      this.isPanning = true;
      this.startX = e.clientX - this.camX;
      this.startY = e.clientY - this.camY;
      return;
    }

    const wx = this.toWorldX(e.clientX);
    const wy = this.toWorldY(e.clientY);

    if (this.currentTool === 'select') {
      this.selectedIndex = this.hitTest(wx, wy);
      if (this.selectedIndex !== -1) {
        this.isDraggingShape = true;
        const i4 = this.selectedIndex * 4;
        this.dragOffsetX1 = this.coords[i4] - wx;
        this.dragOffsetY1 = this.coords[i4 + 1] - wy;
        this.dragOffsetX2 = this.coords[i4 + 2] - wx;
        this.dragOffsetY2 = this.coords[i4 + 3] - wy;
        this.renderStaticLayer();
      }
      this.renderActiveLayer();
    } else {
      this.selectedIndex = -1;
      this.isDrawing = true;
      this.startX = this.currX = wx;
      this.startY = this.currY = wy;
      this.renderActiveLayer();
    }
  }

  private onPointerMove(e: PointerEvent) {
    if (this.isPanning) {
      this.camX = e.clientX - this.startX;
      this.camY = e.clientY - this.startY;
      this.renderStaticLayer();
      this.renderActiveLayer();
      return;
    }

    const wx = this.toWorldX(e.clientX);
    const wy = this.toWorldY(e.clientY);

    if (this.isDrawing) {
      this.currX = wx;
      this.currY = wy;
      this.renderActiveLayer();
    } else if (this.isDraggingShape && this.selectedIndex !== -1) {
      const i4 = this.selectedIndex * 4;
      this.coords[i4]     = wx + this.dragOffsetX1;
      this.coords[i4 + 1] = wy + this.dragOffsetY1;
      this.coords[i4 + 2] = wx + this.dragOffsetX2;
      this.coords[i4 + 3] = wy + this.dragOffsetY2;
      this.renderActiveLayer();
    }
  }

  private onPointerUp(e: PointerEvent) {
    try {
      this.activeCanvas.releasePointerCapture(e.pointerId);
    } catch {
      // Zaten serbest bırakılmışsa göz ardı et
    }

    if (this.isPanning) {
      this.isPanning = false;
      return;
    }

    if (this.isDrawing) {
      this.isDrawing = false;
      if (Math.hypot(this.currX - this.startX, this.currY - this.startY) > 2) {
        this.addShape(
          this.getToolTypeId(this.currentTool),
          this.startX,
          this.startY,
          this.currX,
          this.currY,
          this.currentColorIndex
        );
        this.renderStaticLayer();
      }
      this.renderActiveLayer();
    } else if (this.isDraggingShape) {
      this.isDraggingShape = false;
      this.renderStaticLayer();
      this.renderActiveLayer();
    }
  }

  private onWheel(e: WheelEvent) {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.12 : 0.89;
    this.zoom(zoomFactor, e.clientX, e.clientY);
  }

  public zoom(zoomFactor: number, clientX: number = window.innerWidth / 2, clientY: number = window.innerHeight / 2) {
    const newZoom = Math.min(Math.max(this.camZoom * zoomFactor, 0.05), 20);

    this.camX = clientX - (clientX - this.camX) * (newZoom / this.camZoom);
    this.camY = clientY - (clientY - this.camY) * (newZoom / this.camZoom);
    this.camZoom = newZoom;

    this.renderStaticLayer();
    this.renderActiveLayer();
  }

  private onKeyDown(e: KeyboardEvent) {
    if (e.code === 'Space') {
      this.isSpaceDown = true;
      this.activeCanvas.style.cursor = 'grab';
    }
    if (e.key === 'r' || e.key === 'R') this.setTool('rect');
    if (e.key === 'o' || e.key === 'O') this.setTool('ellipse');
    if (e.key === 'l' || e.key === 'L') this.setTool('line');
    if (e.key === 'v' || e.key === 'V') this.setTool('select');
    if (e.key === 'Delete' || e.key === 'Backspace') {
      this.deleteSelected();
    }
  }

  private onKeyUp(e: KeyboardEvent) {
    if (e.code === 'Space') {
      this.isSpaceDown = false;
      this.activeCanvas.style.cursor = this.currentTool === 'select' ? 'default' : 'crosshair';
    }
  }

  // 7. YARDIMCI VE YÖNETİM METOTLARI
  public addStressShapes(count = 20_000) {
    const spread = 6000;
    const cx = window.innerWidth / 2;
    const cy = window.innerHeight / 2;

    for (let i = 0; i < count; i++) {
      const x = (Math.random() - 0.5) * spread + cx;
      const y = (Math.random() - 0.5) * spread + cy;
      const w = 20 + Math.random() * 80;
      const h = 20 + Math.random() * 80;
      const type = (i % 3) + 1;
      const color = i % PALETTE.length;
      this.addShape(type, x, y, x + w, y + h, color);
    }
    this.renderStaticLayer();
  }

  public clear() {
    this.shapeCount = 0;
    this.selectedIndex = -1;
    this.renderStaticLayer();
    this.renderActiveLayer();
  }

  public resetCamera() {
    this.camX = 0;
    this.camY = 0;
    this.camZoom = 1;
    this.renderStaticLayer();
    this.renderActiveLayer();
  }

  private addDefaultShapes() {
    this.addShape(1, 200, 180, 420, 320, 0);
    this.addShape(2, 480, 180, 660, 320, 1);
    this.addShape(3, 420, 250, 480, 250, 3);
    this.renderStaticLayer();
  }
}
