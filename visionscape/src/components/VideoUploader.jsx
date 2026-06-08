import { useState, useRef, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Camera, X, Video, CheckCircle } from 'lucide-react';
import './VideoUploader.css';

export default function VideoUploader({ onVideoReady }) {
  const [mode, setMode] = useState(null); // 'upload' | 'record'
  const [videoFile, setVideoFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordedChunks, setRecordedChunks] = useState([]);
  const [cameraError, setCameraError] = useState(null);

  const videoPreviewRef = useRef(null);
  const livePreviewRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);

  const handleFileDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    setVideoFile(file);
    setVideoUrl(url);
    onVideoReady(file, url);
  }, [onVideoReady]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: handleFileDrop,
    accept: { 'video/*': ['.mp4', '.mov', '.webm', '.avi'] },
    maxFiles: 1,
    multiple: false,
  });

  const startRecording = async () => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      streamRef.current = stream;
      if (livePreviewRef.current) {
        livePreviewRef.current.srcObject = stream;
      }

      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;
      const chunks = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };

      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'video/webm' });
        const file = new File([blob], 'recording.webm', { type: 'video/webm' });
        const url = URL.createObjectURL(blob);
        setVideoFile(file);
        setVideoUrl(url);
        onVideoReady(file, url);
        streamRef.current?.getTracks().forEach(t => t.stop());
      };

      recorder.start();
      setIsRecording(true);
      setRecordedChunks([]);
    } catch (err) {
      setCameraError('Camera access denied. Please allow camera permissions and try again.');
    }
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    setIsRecording(false);
  };

  const resetVideo = () => {
    setVideoFile(null);
    setVideoUrl(null);
    setMode(null);
    streamRef.current?.getTracks().forEach(t => t.stop());
    onVideoReady(null, null);
  };

  if (videoFile && videoUrl) {
    return (
      <div className="video-ready">
        <div className="video-ready__header">
          <div className="video-ready__status">
            <CheckCircle size={16} className="video-ready__icon" />
            <span>Video loaded</span>
          </div>
          <button className="video-ready__reset" onClick={resetVideo}>
            <X size={14} /> Change
          </button>
        </div>
        <div className="video-preview">
          <video
            ref={videoPreviewRef}
            src={videoUrl}
            controls
            className="video-preview__player"
          />
          <div className="video-preview__meta">
            <Video size={13} />
            <span>{videoFile.name}</span>
            <span className="video-preview__size">{(videoFile.size / 1024 / 1024).toFixed(1)} MB</span>
          </div>
        </div>
      </div>
    );
  }

  if (!mode) {
    return (
      <div className="uploader-choice">
        <p className="uploader-choice__label">How would you like to provide your room video?</p>
        <div className="uploader-choice__grid">
          <button className="choice-card" onClick={() => setMode('upload')}>
            <div className="choice-card__icon">
              <Upload size={24} />
            </div>
            <h3 className="choice-card__title">Upload Video</h3>
            <p className="choice-card__desc">MP4, MOV, WebM supported</p>
          </button>
          <button className="choice-card" onClick={() => setMode('record')}>
            <div className="choice-card__icon">
              <Camera size={24} />
            </div>
            <h3 className="choice-card__title">Record Live</h3>
            <p className="choice-card__desc">Use your device camera</p>
          </button>
        </div>
      </div>
    );
  }

  if (mode === 'upload') {
    return (
      <div className="uploader-upload">
        <button className="uploader-back" onClick={() => setMode(null)}>← Back</button>
        <div
          {...getRootProps()}
          className={`dropzone ${isDragActive ? 'dropzone--active' : ''}`}
        >
          <input {...getInputProps()} />
          <div className="dropzone__inner">
            <div className="dropzone__icon">
              <Upload size={28} />
            </div>
            {isDragActive ? (
              <p className="dropzone__text">Release to upload</p>
            ) : (
              <>
                <p className="dropzone__text">Drag & drop your room video here</p>
                <p className="dropzone__sub">or <span className="dropzone__browse">browse files</span></p>
                <p className="dropzone__formats">MP4 · MOV · WebM · AVI</p>
              </>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (mode === 'record') {
    return (
      <div className="uploader-record">
        <button className="uploader-back" onClick={() => { setMode(null); streamRef.current?.getTracks().forEach(t => t.stop()); }}>← Back</button>

        {cameraError ? (
          <div className="camera-error">{cameraError}</div>
        ) : (
          <>
            <div className="camera-preview">
              {isRecording ? (
                <>
                  <video ref={livePreviewRef} autoPlay muted className="camera-preview__video" />
                  <div className="camera-preview__recording-badge">
                    <span className="recording-dot" /> REC
                  </div>
                </>
              ) : (
                <div className="camera-preview__placeholder">
                  <Camera size={32} className="camera-preview__icon" />
                  <p>Camera preview will appear here</p>
                </div>
              )}
            </div>
            <div className="camera-controls">
              {!isRecording ? (
                <button className="btn btn--record" onClick={startRecording}>
                  <span className="btn-record-dot" />
                  Start Recording
                </button>
              ) : (
                <button className="btn btn--stop" onClick={stopRecording}>
                  <span className="btn-stop-square" />
                  Stop Recording
                </button>
              )}
            </div>
          </>
        )}
      </div>
    );
  }
}
