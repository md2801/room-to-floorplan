import { useState } from 'react';
import SceneUploader from '../components/SceneUploader';
import VideoUploader from '../components/VideoUploader';
import ValidationPanel from '../components/ValidationPanel';
import ProcessingPanel from '../components/ProcessingPanel';
import FloorplanViewer from '../components/FloorplanViewer';
import { validateSceneZip } from '../services/api.js';
import './Studio.css';

/**
 * Steps:
 *  Scene mode:  Upload ZIP → Validate → Select Room → Generate → Results
 *  Video mode:  Upload/Record Video → Generate → Results
 */
export default function Studio() {
  // Upload state
  const [inputMode, setInputMode] = useState(null); // null | 'scene' | 'video'
  const [sceneFile, setSceneFile] = useState(null);
  const [videoFile, setVideoFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);

  // Scene validation state
  const [validating, setValidating] = useState(false);
  const [validation, setValidation] = useState(null);
  const [validationError, setValidationError] = useState(null);

  // Room selection
  const [selectedRoomId, setSelectedRoomId] = useState(null);

  // Generation state
  const [isGenerated, setIsGenerated] = useState(false);

  const [png, setPng] = useState(null);
  const [svg, setSvg] = useState(null);

  // Derived step for the indicator
  const step = (() => {
    if (isGenerated) return 3;
    if (inputMode === 'scene') {
      if (!sceneFile) return 0;
      if (!validation) return 1;
      return 2;
    }
    if (inputMode === 'video') {
      if (!videoFile) return 0;
      return 2;
    }
    return 0;
  })();

  const STEPS = inputMode === 'video'
    ? ['Input', 'Generate', 'Results']
    : ['Upload ZIP', 'Validate', 'Generate', 'Results'];

  // ── Handlers ──────────────────────────────────────────────

  const handleSceneReady = async (file) => {
    if (!file) {
      setSceneFile(null);
      setValidation(null);
      setValidationError(null);
      setSelectedRoomId(null);
      setInputMode(null);
      return;
    }
    setInputMode('scene');
    setSceneFile(file);
    setValidation(null);
    setValidationError(null);
    setSelectedRoomId(null);
    setIsGenerated(false);
    setValidating(true);

    try {
      const result = await validateSceneZip(file);
      setValidation(result);
      // Auto-select first valid room
      const firstValid = result.rooms.find((r) => r.isValid);
      if (firstValid) setSelectedRoomId(firstValid.roomId);
    } catch (err) {
      setValidationError('Unexpected error during validation: ' + err.message);
    } finally {
      setValidating(false);
    }
  };

  const handleVideoReady = (file, url) => {
    if (!file) {
      setVideoFile(null);
      setVideoUrl(null);
      if (!sceneFile) setInputMode(null);
      return;
    }
    setInputMode('video');
    setVideoFile(file);
    setVideoUrl(url);
    setIsGenerated(false);
  };

  const handleGenerationComplete = (result) => {
    setPng(result?.png || null);
    setSvg(result?.svg || null);
    setIsGenerated(true);
  };

  const handleReset = () => {
    setInputMode(null);
    setSceneFile(null);
    setVideoFile(null);
    setVideoUrl(null);
    setValidation(null);
    setValidationError(null);
    setSelectedRoomId(null);
    setIsGenerated(false);
  };

  // ── Render ─────────────────────────────────────────────────

  return (
    <main className="studio">
      {/* Step indicator */}
      <div className="studio__steps">
        {STEPS.map((label, i) => {
          const done = step > i;
          const active = step === i;
          return (
            <div key={i} className={`step-item ${active ? 'step-item--active' : ''} ${done ? 'step-item--done' : ''}`}>
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
              {(sceneFile || videoFile) && (
                <button className="panel__reset" onClick={handleReset}>Reset</button>
              )}
            </div>
            <div className="panel__body">
              <SceneUploader
                onSceneReady={handleSceneReady}
                onVideoReady={handleVideoReady}
              />
            </div>
          </div>

          {/* Panel 02: Validation (scene mode only) */}
          {inputMode === 'scene' && sceneFile && (
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
                  <ValidationPanel
                    validation={validation}
                    selectedRoomId={selectedRoomId}
                    onRoomSelect={setSelectedRoomId}
                  />
                )}
              </div>
            </div>
          )}

          {/* Panel 03: Generate */}
          {((inputMode === 'scene' && validation) || (inputMode === 'video' && videoFile)) && (
            <div className="panel panel--generate">
              <div className="panel__head">
                <span className="panel__tag">{inputMode === 'scene' ? '03' : '02'}</span>
                <h2 className="panel__title">Generate</h2>
              </div>
              <div className="panel__body">
                {inputMode === 'scene' && (
                  <p className="panel__hint">
                    {selectedRoomId
                      ? `Ready to generate floorplan for room ${selectedRoomId} from ${validation.sceneName}.`
                      : 'Select a valid room from the validation panel above.'}
                  </p>
                )}
                {inputMode === 'video' && (
                  <p className="panel__hint">
                    Video loaded. Click below to begin AI floorplan generation.
                  </p>
                )}
                <ProcessingPanel
                  mode={inputMode}
                  sceneMetadata={validation}
                  selectedRoomId={selectedRoomId}
                  videoFile={videoFile}
                  onComplete={handleGenerationComplete}
                />
              </div>
            </div>
          )}
        </aside>

        {/* ── Main display ── */}
        <section className="studio__main">
          {!isGenerated ? (
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
                    {inputMode === null
                      ? 'Upload a Structured3D scene ZIP or a room walkthrough video'
                      : inputMode === 'scene'
                      ? validation
                        ? selectedRoomId
                          ? `Room ${selectedRoomId} selected — click Generate to begin`
                          : 'Select a valid room from the sidebar'
                        : validating
                        ? 'Validating ZIP contents…'
                        : 'Upload a scene ZIP to validate'
                      : 'Upload complete — click Generate'}
                  </p>
                </div>
              </div>
            </div>
          ) : (
            <div className="studio__result">
              <FloorplanViewer
                sceneName={validation?.sceneName}
                roomId={selectedRoomId}
                png={png}
                svg={svg}
              />
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
