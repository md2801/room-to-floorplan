import { useState } from 'react';
import { Cpu, ChevronRight } from 'lucide-react';
import './ProcessingPanel.css';

/**
 * Handles the "Generate Floorplan" button and animated progress bar.
 * Supports both scene ZIP mode and legacy video mode.
 *
 * Props:
 *  - mode: 'scene' | 'video'
 *  - sceneMetadata: validation result (scene mode)
 *  - selectedRoomId: string (scene mode)
 *  - videoFile: File (video mode)
 *  - onComplete: () => void
 */
export default function ProcessingPanel({
  mode = 'video',
  sceneMetadata,
  selectedRoomId,
  sceneFile,
  videoFile,
  onComplete,
}) {
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMsg, setStatusMsg] = useState('');

  const [preview_png, setPreviewPng] = useState(null);
  const [svg, setSvg] = useState(null);

  const isDisabled = mode === 'scene'
    ? !sceneMetadata || !selectedRoomId
    : !videoFile;

  const handleGenerate = async () => {
    if (isDisabled || isProcessing) return;
    setIsProcessing(true);
    setProgress(0);
    setStatusMsg('');

    if (mode === 'scene') {
      const { generateFloorplanFromScene } = await import('../services/api.js');
      const result = await generateFloorplanFromScene(
        sceneMetadata, 
        selectedRoomId, 
        sceneFile, 
        (p, msg) => {
        setProgress(p);
        setStatusMsg(msg);
      });

      console.log("Inference result:", result);
      onComplete(result);
      return;
    } else {
      const { uploadVideo, generateFloorplan } = await import('../services/api.js');
      const upload = await uploadVideo(videoFile);
      await generateFloorplan(upload.fileId, (p, msg) => {
        setProgress(p);
        setStatusMsg(msg);
      });
    }

    setIsProcessing(false);
    onComplete();
  };

  return (
    <div className="processing-panel">
      {!isProcessing && progress === 0 && (
        <>
          {mode === 'scene' && !selectedRoomId && (
            <p className="processing-panel__hint">Select a room above to enable generation.</p>
          )}
          <button
            className="generate-btn"
            onClick={handleGenerate}
            disabled={isDisabled}
          >
            <Cpu size={15} />
            <span>Generate Floorplan</span>
            <ChevronRight size={15} className="generate-btn__arrow" />
          </button>
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
