import { useState } from 'react';
import { Cpu, ChevronRight } from 'lucide-react';
import './ProcessingPanel.css';

/**
 * Handles the "Generate Floorplan" button and animated progress bar.
 * On success it calls onComplete({ svg_url, png_url, scene_name }).
 *
 * Props:
 *  - sceneMetadata : result from validateSceneZip() — must include _file
 *  - onComplete    : ({ svg_url, png_url, scene_name }) => void
 */
export default function ProcessingPanel({ sceneMetadata, onComplete }) {
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMsg, setStatusMsg] = useState('');
  const [errorMsg, setErrorMsg] = useState('');

  const isDisabled = !sceneMetadata;

  const handleGenerate = async () => {
    if (isDisabled || isProcessing) return;
    setIsProcessing(true);
    setProgress(0);
    setStatusMsg('');
    setErrorMsg('');

    try {
      const { generateFloorplanFromScene } = await import('../services/api.js');
      const result = await generateFloorplanFromScene(
        sceneMetadata,
        null,
        (p, msg) => {
          setProgress(p);
          setStatusMsg(msg);
        }
      );

      setIsProcessing(false);
      onComplete({
        svg_url:    result.svg_url,
        png_url:    result.png_url,
        scene_name: result.scene_name,
      });
    } catch (err) {
      setIsProcessing(false);
      setProgress(0);
      setStatusMsg('');
      setErrorMsg(err.message || 'An unexpected error occurred.');
    }
  };

  return (
    <div className="processing-panel">
      {!isProcessing && progress === 0 && (
        <>
          <button
            className="generate-btn"
            onClick={handleGenerate}
            disabled={isDisabled}
          >
            <Cpu size={15} />
            <span>Generate Floorplan</span>
            <ChevronRight size={15} className="generate-btn__arrow" />
          </button>
          {errorMsg && (
            <p className="processing-error">{errorMsg}</p>
          )}
        </>
      )}

      {(isProcessing || progress > 0) && (
        <div className="progress-container">
          <div className="progress-header">
            <span className="progress-label">
              {progress === 100 ? 'Analysis complete' : 'Processing…'}
            </span>
            <span className="progress-pct">{progress}%</span>
          </div>
          <div className="progress-track">
            <div
              className={`progress-bar ${progress === 100 ? 'progress-bar--done' : ''}`}
              style={{ width: `${progress}%` }}
            />
          </div>
          {statusMsg && progress < 100 && (
            <p className="progress-status">{statusMsg}</p>
          )}
        </div>
      )}
    </div>
  );
}
