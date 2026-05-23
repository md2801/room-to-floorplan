import { useState } from 'react';
import { Layers } from 'lucide-react';
import './FloorplanViewer.css';

const VIEWS = [
  { 
    id: 'furniture',
    label: 'Generated Floorplan', 
    icon: Layers 
  }
];

// View floorplan output generated (SVG)
// Floorplan viewer with furniture and 3d View has been removed as it is out of scope for now.
// User can view the floorplan and download the SVG and PNG outputs.
export default function FloorplanViewer({ sceneName, roomId, png, svg }) {
  const [activeView, setActiveView] = useState('floorplan');

  return (
    <div className="floorplan-viewer">
      <div className="floorplan-viewer__header">
        <h2 className="floorplan-viewer__title">AI Generated Floorplan</h2>
                <p className="floorplan-viewer__sub">{sceneName && roomId ? sceneName + ' · Room ' + roomId : sceneName || 'Generated from your room video'}</p>
      </div>

      {svg && (
        <img
          src={svg}
          alt="Debug Floorplan"
          style={{
            width: '600px',
            maxWidth: '100%',
            border: '3px solid red',
            background: 'white'
          }}
        />
      )}

      <div className="view-toggle">
        {VIEWS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            className={`view-toggle__btn ${activeView === id ? 'view-toggle__btn--active' : ''}`}
            onClick={() => setActiveView(id)}
          >
            <Icon size={14} />
            {label}
          </button>
        ))}
      </div>
 
      <div className="floorplan-download">
        {svg ? (
          <div className="fp-canvas fp-canvas--animate">
            {/* DOWNLOAD BUTTONS */}
            <div className="download-actions">
              <a
                href={svg.replace("/output/","/download/")}
                download="floorplan.svg"
                className="download-btn"
              >
                Download SVG
              </a>

              {png && (
                <a
                  href={png.replace("/output/","/download/")}
                  download="floorplan.png"
                  className="download-btn"
                >
                  Download PNG
                </a>
              )}
            </div>

          </div>
        ) : (
        <div className="fp-canvas fp-canvas--animate">
          <p style={{ color: '#999', textAlign: 'center' }}>
            No floorplan generated yet
          </p>
        </div>
        )}
      </div>

      <div className="floorplan-meta">
        <div className="meta-chip">AI Generated Floorplan</div>
        {
          sceneName && (
            <div className="meta-chip">{sceneName}</div>
          )
        }
        {
          roomId && (
            <div className="meta-chip">Room {roomId}</div>
          )
        }
      </div>
    </div>
  );
}