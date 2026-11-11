import { createSignal, onMount, onCleanup, For } from 'solid-js';
import { WhiteboardEngine, ToolType, PALETTE, TelemetryStats } from './engine/WhiteboardEngine';

export default function App() {
  let staticRef!: HTMLCanvasElement;
  let activeRef!: HTMLCanvasElement;
  let engine!: WhiteboardEngine;

  const [tool, setTool] = createSignal<ToolType>('rect');
  const [selectedColor, setSelectedColor] = createSignal(0);
  const [stats, setStats] = createSignal<TelemetryStats>({
    total: 0,
    visible: 0,
    renderMs: 0,
    zoom: 100,
  });

  onMount(() => {
    engine = new WhiteboardEngine(staticRef, activeRef, (s) => {
      setStats(s);
    });

    onCleanup(() => {
      engine?.destroy();
    });
  });

  const selectTool = (t: ToolType) => {
    setTool(t);
    engine?.setTool(t);
  };

  const selectColor = (index: number) => {
    setSelectedColor(index);
    engine?.setColor(index);
  };

  const handleStressTest = () => {
    engine?.addStressShapes(20_000);
  };

  const handleClear = () => {
    engine?.clear();
  };

  const handleResetCamera = () => {
    engine?.resetCamera();
  };

  return (
    <div class="app-container">
      <style>{`
        * { box-sizing: border-box; margin: 0; padding: 0; user-select: none; }
        body, html { width: 100%; height: 100%; overflow: hidden; background: #121212; font-family: system-ui, sans-serif; color: #eee; }
        .app-container { position: relative; width: 100vw; height: 100vh; overflow: hidden; }
        .canvas-layer { position: absolute; top: 0; left: 0; width: 100vw; height: 100vh; }
        .static-layer { z-index: 1; }
        .active-layer { z-index: 2; cursor: crosshair; }
        .toolbar {
          position: fixed; top: 16px; left: 50%; transform: translateX(-50%);
          z-index: 10; display: flex; align-items: center; gap: 8px; background: #1e1e24;
          padding: 6px 10px; border-radius: 12px; border: 1px solid #33333f;
          box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5); backdrop-filter: blur(10px);
        }
        .tool-group, .action-group { display: flex; gap: 4px; }
        .divider { width: 1px; height: 24px; background: #33333f; margin: 0 4px; }
        .btn {
          background: transparent; color: #ccc; border: none; padding: 8px 12px;
          border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 500;
          transition: all 0.15s ease-in-out;
        }
        .btn:hover { background: #2c2c35; color: #fff; }
        .btn.active { background: #4f46e5; color: #fff; box-shadow: 0 2px 8px rgba(79, 70, 229, 0.4); }
        .btn.stress { background: #059669; color: #fff; }
        .btn.stress:hover { background: #047857; }
        .btn.reset { background: #374151; color: #e5e7eb; }
        .btn.reset:hover { background: #4b5563; }
        .btn.danger { background: #dc2626; color: #fff; }
        .btn.danger:hover { background: #b91c1c; }
        .color-palette { display: flex; gap: 6px; align-items: center; }
        .color-btn {
          width: 20px; height: 20px; border-radius: 50%; border: 2px solid transparent;
          cursor: pointer; transition: transform 0.12s, border-color 0.12s;
        }
        .color-btn:hover { transform: scale(1.15); }
        .color-btn.selected { border-color: #ffffff; transform: scale(1.2); box-shadow: 0 0 8px rgba(255, 255, 255, 0.4); }
        .hud {
          position: fixed; bottom: 16px; left: 16px; z-index: 10;
          background: rgba(30, 30, 36, 0.88); padding: 10px 14px;
          border-radius: 10px; border: 1px solid #33333f;
          font-family: monospace; font-size: 12px; line-height: 1.6;
          pointer-events: none; backdrop-filter: blur(8px);
        }
        .hud span { color: #34d399; font-weight: 600; }
        .hint {
          position: fixed; bottom: 16px; right: 16px; z-index: 10;
          background: rgba(30, 30, 36, 0.8); padding: 8px 12px;
          border-radius: 8px; border: 1px solid #2a2a34;
          font-size: 12px; color: #9ca3af; pointer-events: none;
        }
        .hint b { color: #e5e7eb; }
      `}</style>

      {/* Üst Araç Çubuğu */}
      <header class="toolbar">
        <div class="tool-group">
          <button
            class="btn"
            classList={{ active: tool() === 'rect' }}
            onClick={() => selectTool('rect')}
            title="Kısayol: R"
          >
            Dikdörtgen (R)
          </button>
          <button
            class="btn"
            classList={{ active: tool() === 'ellipse' }}
            onClick={() => selectTool('ellipse')}
            title="Kısayol: O"
          >
            Elips (O)
          </button>
          <button
            class="btn"
            classList={{ active: tool() === 'line' }}
            onClick={() => selectTool('line')}
            title="Kısayol: L"
          >
            Çizgi (L)
          </button>
          <button
            class="btn"
            classList={{ active: tool() === 'select' }}
            onClick={() => selectTool('select')}
            title="Kısayol: V"
          >
            Seç / Taşı (V)
          </button>
        </div>

        <div class="divider" />

        {/* Renk Paleti */}
        <div class="color-palette">
          <For each={PALETTE}>
            {(c, i) => (
              <button
                class="color-btn"
                classList={{ selected: selectedColor() === i() }}
                style={{ 'background-color': c }}
                onClick={() => selectColor(i())}
                title={`Renk #${i() + 1}`}
              />
            )}
          </For>
        </div>

        <div class="divider" />

        <div class="action-group">
          <button class="btn stress" onClick={handleStressTest}>
            +20.000 Şekil
          </button>
          <button class="btn reset" onClick={handleResetCamera} title="Kamerayı Sıfırla">
            Kamera 100%
          </button>
          <button class="btn danger" onClick={handleClear}>
            Temizle
          </button>
        </div>
      </header>

      {/* Performans ve Telemetri Göstergesi (HUD) */}
      <aside class="hud">
        <div>Toplam Şekil: <span>{stats().total.toLocaleString()}</span></div>
        <div>Ekranda Çizilen: <span>{stats().visible.toLocaleString()}</span></div>
        <div>Statik Render: <span>{stats().renderMs.toFixed(2)} ms</span></div>
        <div>Yakınlaştırma: <span>{stats().zoom}%</span></div>
      </aside>

      {/* Kullanım İpuçları */}
      <footer class="hint">
        Kaydırma (Pan): <b>Boşluk (Space) + Sürükle</b> veya <b>Orta Tuş</b> | Yakınlaştırma: <b>Fare Tekerleği</b> | Silme: <b>Delete / Backspace</b>
      </footer>

      {/* Çift Katmanlı Tuval (Dual Canvas) */}
      <canvas ref={staticRef} class="canvas-layer static-layer" />
      <canvas ref={activeRef} class="canvas-layer active-layer" />
    </div>
  );
}
