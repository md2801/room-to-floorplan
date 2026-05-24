import { useState } from 'react';
import SceneUploader from '../components/SceneUploader';
import ValidationPanel from '../components/ValidationPanel';
import ProcessingPanel from '../components/ProcessingPanel';
import FloorplanViewer from '../components/FloorplanViewer';
import { validateSceneZip } from '../services/api.js';
import './Studio.css';

/**
 * Steps: Upload ZIP → Validate → Generate → Results
 *
 * After generation the backend returns { svg_url, png_url, scene_name }.
 * These are stored in `generationResult` and passed to FloorplanViewer
 * which fetches and renders the real SVG inline.
 */
export default function Studio() {
  const [sceneFile, setSceneFile] = useState(null);

  const [validating, setValidating] = useState(false);
  const [validation, setValidation] = useState(null);
  const [validationError, setValidationError] = useState(null);

  // Holds { svg_url, png_url, scene_name } returned by the backend.
  const [generationResult, setGenerationResult] = useState(null);

  // Derived step for the indicator
  const step = (() => {
    if (generationResult) return 3;
    if (!sceneFile) return 0;
    if (!validation) return 1;
    return 2;
  })();

  const STEPS = ['Upload ZIP', 'Validate', 'Generate', 'Results'];

  // ── Handlers ──────────────────────────────────────────────────────────

  const handleSceneReady = async (file) => {
    if (!file) {
      setSceneFile(null);
      setValidation(null);
      setValidationError(null);
      setGenerationResult(null);
      return;
    }
    setSceneFile(file);
    setValidation(null);
    setValidationError(null);
    setGenerationResult(null);
    setValidating(true);

    try {
      const result = await validateSceneZip(file);
      setValidation(result);
    } catch (err) {
      setValidationError('Unexpected error during validation: ' + err.message);
    } finally {
      setValidating(false);
    }
  };

  /**
   * Called by ProcessingPanel when the backend inference succeeds.
   * @param {{ svg_url: string, png_url: string, scene_name: string }} result
   */
  const handleGenerationComplete = (result) => {
    setGenerationResult(result);
  };

  const handleReset = () => {
    setSceneFile(null);
    setValidation(null);
    setValidationError(null);
    setGenerationResult(null);
  };

  // ── Render ────────────────────────────────────────────────────────────

  return (
    <main className="studio">
      {/* Step indicator */}
      <div className="studio__steps">
        {STEPS.map((label, i) => {
          const done = step > i;
          const active = step === i;
          return (
            <div
              key={i}
              className={`step-item ${active ? 'step-item--active' : ''} ${done ? 'step-item--done' : ''}`}
            >
              <div className="step-item__num">{done ? '✓' : i + 1}</div>
              <span className="step-item__label">{label}</span>
              {i < STEPS.length - 1 && (
                <div className={`step-item__line ${done ? 'step-item__line--done' : ''}`} />
              )}
            </div>
          );
        })}
      </div>

      <div className="studio__layout">
        {/* ── Left sidebar ── */}
        <aside className="studio__sidebar">

          {/* Panel 01: Upload */}
          <div className="panel">
            <div className="panel__head">
              <span className="panel__tag">01</span>
              <h2 className="panel__title">Scene Input</h2>
              {sceneFile && (
                <button className="panel__reset" onClick={handleReset}>Reset</button>
              )}
            </div>
            <div className="panel__body">
              <SceneUploader onSceneReady={handleSceneReady} />
            </div>
          </div>

          {/* Panel 02: Validation */}
          {sceneFile && (
            <div className="panel panel--validation">
              <div className="panel__head">
                <span className="panel__tag">02</span>
                <h2 className="panel__title">Validation</h2>
              </div>
              <div className="panel__body">
                {validating && (
                  <div className="validation-loading">
                    <div className="validation-loading__spinner" />
                    <span>Reading ZIP structure…</span>
                  </div>
                )}
                {validationError && (
                  <p className="validation-error-msg">{validationError}</p>
                )}
                {validation && !validating && (
                  <ValidationPanel validation={validation} />
                )}
              </div>
            </div>
          )}

          {/* Panel 03: Generate */}
          {validation && !generationResult && (
            <div className="panel panel--generate">
              <div className="panel__head">
                <span className="panel__tag">03</span>
                <h2 className="panel__title">Generate</h2>
              </div>
              <div className="panel__body">
                <p className="panel__hint">
                  Scene validated. Click below to generate the combined floorplan.
                </p>
                <ProcessingPanel
                  mode="scene"
                  sceneMetadata={validation}
                  onComplete={handleGenerationComplete}
                />
              </div>
            </div>
          )}
        </aside>

        {/* ── Main display ── */}
        <section className="studio__main">
          {!generationResult ? (
            <div className="studio__empty">
              <div className="studio__empty-inner">
                <div className="studio__empty-grid" aria-hidden="true">
                  {Array.from({ length: 80 }).map((_, i) => (
                    <div key={i} className="studio__empty-cell" />
                  ))}
                </div>
                <div className="studio__empty-content">
                  <div className="studio__empty-icon">
                    <svg width="52" height="52" viewBox="0 0 52 52" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <rect x="4" y="4" width="44" height="44" stroke="currentColor" strokeWidth="1.5" fill="none" />
                      <rect x="4" y="4" width="24" height="44" stroke="currentColor" strokeWidth="1.5" fill="none" />
                      <rect x="28" y="4" width="20" height="24" stroke="currentColor" strokeWidth="1.5" fill="none" />
                      <rect x="4" y="30" width="11" height="18" stroke="currentColor" strokeWidth="1.5" fill="none" />
                      <circle cx="16" cy="16" r="3" fill="currentColor" opacity="0.3" />
                    </svg>
                  </div>
                  <p className="studio__empty-label">
                    {validating ? 'Validating scene ZIP…' : 'Floorplan will appear here'}
                  </p>
                  <p className="studio__empty-sub">
                    {!sceneFile
                      ? 'Upload a Structured3D scene ZIP to begin'
                      : validation
                      ? 'Scene validated — click Generate to produce the floorplan'
                      : validating
                      ? 'Validating ZIP contents…'
                      : 'Upload a scene ZIP to validate'}
                  </p>
                </div>
              </div>
            </div>
          ) : (
            <div className="studio__result">
              <FloorplanViewer
                sceneName={generationResult.scene_name}
                svgUrl={generationResult.svg_url}
                pngUrl={generationResult.png_url}
              />
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
