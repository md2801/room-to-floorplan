import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { FileArchive, X } from 'lucide-react';
import './SceneUploader.css';

/**
 * ZIP upload input for Structured3D scene archives.
 * Video walkthrough mode has been removed — ZIP is the only input path.
 */
export default function SceneUploader({ onSceneReady }) {
  const [file, setFile] = useState(null);

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
    onSceneReady(null);
  };

  /* ── File accepted — show summary chip ── */
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

  /* ── Dropzone ── */
  return (
    <div className="scene-zip-upload">
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
