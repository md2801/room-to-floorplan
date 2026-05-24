import { useEffect, useState } from 'react';
import { Layers, Download } from 'lucide-react';
import './FloorplanViewer.css';

export default function FloorplanViewer({ sceneName, svgUrl, pngUrl }) {
  const [svgContent, setSvgContent] = useState('');
  const [loadError, setLoadError] = useState('');

  // Fetch the raw SVG text so we can render it inline (allows CSS styling).
  useEffect(() => {
    if (!svgUrl) return;
    setSvgContent('');
    setLoadError('');

    fetch(svgUrl)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.text();
      })
      .then((text) => setSvgContent(text))
      .catch((err) => setLoadError(`Could not load SVG: ${err.message}`));
  }, [svgUrl]);

  // ── Download helpers ───────────────────────────────────────────────────
  // Fetch the file as a blob so the browser is forced to download it
  // rather than opening it in a new tab (which happens when the browser
  // recognises the MIME type and the URL is same-origin).

  const triggerBlobDownload = async (url, filename) => {
    const res = await fetch(url);
    const blob = await res.blob();
    const objectUrl = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = objectUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(objectUrl);
  };

  const handleDownloadSVG = () => {
    if (!svgUrl) return;
    triggerBlobDownload(svgUrl, 'pred_floorplan.svg');
  };

  const handleDownloadPNG = () => {
    if (!pngUrl) return;
    triggerBlobDownload(pngUrl, 'pred_floorplan_preview.png');
  };

  // ── Render ────────────────────────────────────────────────────────────

  return (
    <div className="floorplan-viewer">
      <div className="floorplan-viewer__header">
        <h2 className="floorplan-viewer__title">Floorplan Output</h2>
        <p className="floorplan-viewer__sub">
          {sceneName || 'Generated from your scene'}
        </p>
      </div>

      <div className="view-toggle">
        <button className="view-toggle__btn view-toggle__btn--active">
          <Layers size={14} />
          Predicted Floorplan
        </button>
      </div>

      <div className="floorplan-display">
        <div className="fp-canvas fp-canvas--animate">
          {loadError && (
            <p className="fp-load-error">{loadError}</p>
          )}
          {!loadError && !svgContent && svgUrl && (
            <div className="fp-loading">Loading SVG…</div>
          )}
          {svgContent && (
            <div
              className="fp-svg-wrapper"
              dangerouslySetInnerHTML={{ __html: svgContent }}
            />
          )}
        </div>
      </div>

      <div className="fp-download-row">
        <button
          className="fp-download-btn"
          onClick={handleDownloadSVG}
          disabled={!svgUrl}
        >
          <Download size={14} />
          Download SVG
        </button>
        <button
          className="fp-download-btn"
          onClick={handleDownloadPNG}
          disabled={!pngUrl}
        >
          <Download size={14} />
          Download PNG
        </button>
      </div>
    </div>
  );
}
