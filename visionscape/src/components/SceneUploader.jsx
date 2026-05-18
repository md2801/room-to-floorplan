import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { FileArchive, Video, Upload, X } from 'lucide-react';
import VideoUploader from './VideoUploader';
import './SceneUploader.css';

/**
 * Top-level upload mode selector.
 * Mode A: Structured3D Scene ZIP  (primary)
 * Mode B: Video Walkthrough        (legacy)
 */
export default function SceneUploader({ onSceneReady, onVideoReady }) {
  const [mode, setMode] = useState(null); // null | 'scene' | 'video'
  const [file, setFile] = useState(null);
  const [isDragActive, setDragActive] = useState(false);

  const handleDrop = useCallback(
    (accepted) => {
      const f = accepted[0];
      if (!f) return;
      setFile(f);
      onSceneReady(f);
    },
    [onSceneReady]
  );

  const { getRootProps, getInputProps, isDragActive: dropzoneDrag } = useDropzone({
    onDrop: handleDrop,
    accept: { 'application/zip': ['.zip'], 'application/x-zip-compressed': ['.zip'] },
    maxFiles: 1,
    multiple: false,
  });

  const reset = () => {
    setFile(null);
    setMode(null);
    onSceneReady(null);
  };

  /* ── Mode selection ── */
  if (!mode) {
    return (
      <div className="scene-uploader-choice">
        <p className="scene-uploader-choice__label">Select an input type to begin</p>
        <div className="scene-uploader-choice__grid">
          <button className="mode-card mode-card--primary" onClick={() => setMode('scene')}>
            <div className="mode-card__icon">
              <FileArchive size={22} />
            </div>
            <h3 className="mode-card__title">Structured3D Scene ZIP</h3>
            <p className="mode-card__desc">
              Upload a <code>scene_XXXXX.zip</code> archive from the Structured3D dataset
            </p>
            <span className="mode-card__badge">Recommended</span>
          </button>

          <button className="mode-card" onClick={() => setMode('video')}>
            <div className="mode-card__icon">
              <Video size={22} />
            </div>
            <h3 className="mode-card__title">Video Walkthrough</h3>
            <p className="mode-card__desc">
              Upload or record a room walkthrough video (MP4, MOV, WebM)
            </p>
          </button>
        </div>
      </div>
    );
  }

  /* ── Video mode (delegate to existing component) ── */
  if (mode === 'video') {
    return (
      <div>
        <button className="scene-uploader-back" onClick={() => setMode(null)}>
          ← Change input type
        </button>
        <VideoUploader onVideoReady={onVideoReady} />
      </div>
    );
  }

  /* ── Scene ZIP mode ── */
  if (file) {
    return (
      <div className="scene-file-ready">
        <div className="scene-file-ready__header">
          <FileArchive size={14} className="scene-file-ready__icon" />
          <span className="scene-file-ready__name">{file.name}</span>
          <span className="scene-file-ready__size">
            {(file.size / 1024 / 1024).toFixed(1)} MB
          </span>
          <button className="scene-file-ready__reset" onClick={reset}>
            <X size={13} /> Change
          </button>
        </div>
        <p className="scene-file-ready__hint">Validating scene structure…</p>
      </div>
    );
  }

  return (
    <div className="scene-zip-upload">
      <button className="scene-uploader-back" onClick={() => setMode(null)}>
        ← Change input type
      </button>
      <div
        {...getRootProps()}
        className={`scene-dropzone ${dropzoneDrag ? 'scene-dropzone--active' : ''}`}
      >
        <input {...getInputProps()} />
        <div className="scene-dropzone__inner">
          <div className="scene-dropzone__icon">
            <FileArchive size={30} />
          </div>
          {dropzoneDrag ? (
            <p className="scene-dropzone__text">Release to upload ZIP</p>
          ) : (
            <>
              <p className="scene-dropzone__text">Drop your scene ZIP here</p>
              <p className="scene-dropzone__sub">
                or <span className="scene-dropzone__browse">browse files</span>
              </p>
              <p className="scene-dropzone__hint">
                Expected: <code>scene_XXXXX.zip</code> (Structured3D format)
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
