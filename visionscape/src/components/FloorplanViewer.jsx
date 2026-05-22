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

// View floorplan output generated
// Floorplan viewer with furniture and 3d View has been removed as it is out of scope for now.
export default function FloorplanViewer({ sceneName, roomId, preview_png, svg }) {
  const [activeView, setActiveView] = useState('floorplan');

  return (
    <div className="floorplan-viewer">
      <div className="floorplan-viewer__header">
        <h2 className="floorplan-viewer__title">AI Generated Floorplan</h2>
                <p className="floorplan-viewer__sub">{sceneName && roomId ? sceneName + ' · Room ' + roomId : sceneName || 'Generated from your room video'}</p>
      </div>

      {preview_png && (
        <img
          src={preview_png}
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

      <div className="floorplan-display">
        {preview_png ? (
          <div className="fp-canvas fp-canvas--animate">
            <img
              src={preview_png}
              alt="Generated Floorplan"
              className="fp-svg"
            />
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

/* ─── Floorplan SVG: With Furniture ─── */
//function FloorplanWithFurniture() {
//  return (
//    <div className="fp-canvas fp-canvas--animate">
//      <svg viewBox="0 0 560 400" xmlns="http://www.w3.org/2000/svg" className="fp-svg">
//        {/* Room outline */}
//        <rect x="20" y="20" width="520" height="360" fill="none" stroke="#c9a96e" strokeWidth="2" rx="2" />
//
//        {/* Room dividers */}
//        <line x1="300" y1="20" x2="300" y2="260" stroke="#c9a96e" strokeWidth="1.5" />
//        <line x1="300" y1="260" x2="540" y2="260" stroke="#c9a96e" strokeWidth="1.5" />
//        <line x1="20" y1="260" x2="220" y2="260" stroke="#c9a96e" strokeWidth="1.5" />
//
//        {/* Doors */}
//        <path d="M 290 20 A 30 30 0 0 0 260 50" fill="none" stroke="#c9a96e" strokeWidth="1" strokeDasharray="3,2" />
//        <path d="M 300 245 A 25 25 0 0 1 325 270" fill="none" stroke="#c9a96e" strokeWidth="1" strokeDasharray="3,2" />
//
//        {/* Room labels */}
//        <text x="155" y="145" textAnchor="middle" fill="#5a5855" fontSize="11" fontFamily="DM Sans" letterSpacing="0.1em">LIVING ROOM</text>
//        <text x="415" y="135" textAnchor="middle" fill="#5a5855" fontSize="11" fontFamily="DM Sans" letterSpacing="0.1em">BEDROOM</text>
//        <text x="155" y="320" textAnchor="middle" fill="#5a5855" fontSize="11" fontFamily="DM Sans" letterSpacing="0.1em">KITCHEN</text>
//        <text x="415" y="320" textAnchor="middle" fill="#5a5855" fontSize="11" fontFamily="DM Sans" letterSpacing="0.1em">BATHROOM</text>
//
//        {/* ── LIVING ROOM FURNITURE ── */}
//        {/* Sofa */}
//        <rect x="40" y="40" width="160" height="60" rx="4" fill="#1f1f23" stroke="#c9a96e" strokeWidth="1" />
//        <rect x="44" y="44" width="20" height="52" rx="3" fill="#2a2a2e" stroke="none" />
//        <rect x="176" y="44" width="20" height="52" rx="3" fill="#2a2a2e" stroke="none" />
//        <rect x="44" y="44" width="152" height="18" rx="2" fill="#252529" stroke="none" />
//
//        {/* Coffee table */}
//        <rect x="70" y="130" width="100" height="60" rx="3" fill="none" stroke="#c9a96e" strokeWidth="1" strokeDasharray="4,2" />
//        <circle cx="120" cy="160" r="4" fill="#c9a96e" opacity="0.3" />
//
//        {/* Armchair */}
//        <rect x="220" y="40" width="55" height="55" rx="4" fill="#1f1f23" stroke="#c9a96e" strokeWidth="1" />
//        <rect x="224" y="44" width="47" height="14" rx="2" fill="#252529" />
//
//        {/* TV unit */}
//        <rect x="40" y="215" width="235" height="16" rx="2" fill="#1f1f23" stroke="#c9a96e" strokeWidth="1" />
//        <rect x="60" y="219" width="195" height="8" rx="1" fill="#252529" />
//
//        {/* ── BEDROOM FURNITURE ── */}
//        {/* Bed */}
//        <rect x="330" y="40" width="140" height="110" rx="4" fill="#1f1f23" stroke="#c9a96e" strokeWidth="1" />
//        <rect x="334" y="44" width="132" height="30" rx="2" fill="#252529" />
//        <rect x="342" y="56" width="52" height="16" rx="8" fill="#1a1a1d" stroke="#c9a96e" strokeWidth="0.5" />
//        <rect x="406" y="56" width="52" height="16" rx="8" fill="#1a1a1d" stroke="#c9a96e" strokeWidth="0.5" />
//
//        {/* Nightstand left */}
//        <rect x="312" y="60" width="20" height="24" rx="2" fill="#1f1f23" stroke="#c9a96e" strokeWidth="0.75" />
//        {/* Nightstand right */}
//        <rect x="468" y="60" width="20" height="24" rx="2" fill="#1f1f23" stroke="#c9a96e" strokeWidth="0.75" />
//
//        {/* Wardrobe */}
//        <rect x="312" y="170" width="220" height="45" rx="2" fill="#1f1f23" stroke="#c9a96e" strokeWidth="1" />
//        <line x1="422" y1="170" x2="422" y2="215" stroke="#c9a96e" strokeWidth="0.5" />
//        <circle cx="415" cy="193" r="3" fill="#c9a96e" opacity="0.4" />
//        <circle cx="429" cy="193" r="3" fill="#c9a96e" opacity="0.4" />
//
//        {/* ── KITCHEN FURNITURE ── */}
//        {/* Counter top */}
//        <rect x="40" y="278" width="180" height="30" rx="2" fill="#1f1f23" stroke="#c9a96e" strokeWidth="1" />
//        <circle cx="85" cy="293" r="10" fill="none" stroke="#c9a96e" strokeWidth="0.75" />
//        <circle cx="115" cy="293" r="10" fill="none" stroke="#c9a96e" strokeWidth="0.75" />
//
//        {/* Kitchen island */}
//        <rect x="70" y="330" width="110" height="40" rx="2" fill="#1f1f23" stroke="#c9a96e" strokeWidth="1" />
//
//        {/* ── BATHROOM ── */}
//        {/* Bathtub */}
//        <rect x="320" y="278" width="90" height="55" rx="8" fill="#1f1f23" stroke="#c9a96e" strokeWidth="1" />
//        <rect x="328" y="286" width="74" height="39" rx="6" fill="#141416" />
//
//        {/* Toilet */}
//        <ellipse cx="455" cy="310" rx="20" ry="26" fill="#1f1f23" stroke="#c9a96e" strokeWidth="1" />
//        <rect x="445" y="278" width="40" height="18" rx="2" fill="#1f1f23" stroke="#c9a96e" strokeWidth="1" />
//
//        {/* Sink */}
//        <rect x="312" y="350" width="36" height="28" rx="4" fill="#1f1f23" stroke="#c9a96e" strokeWidth="1" />
//        <circle cx="330" cy="364" r="4" fill="none" stroke="#c9a96e" strokeWidth="0.75" />
//
//        {/* Scale */}
//        <line x1="30" y1="385" x2="80" y2="385" stroke="#5a5855" strokeWidth="1" />
//        <text x="55" y="395" textAnchor="middle" fill="#5a5855" fontSize="8" fontFamily="DM Sans">5m</text>
//      </svg>
//
//      <div className="fp-legend">
//        <div className="fp-legend__item">
//          <span className="fp-legend__swatch fp-legend__swatch--wall" />
//          <span>Walls</span>
//        </div>
//        <div className="fp-legend__item">
//          <span className="fp-legend__swatch fp-legend__swatch--furniture" />
//          <span>Furniture</span>
//        </div>
//        <div className="fp-legend__item">
//          <span className="fp-legend__swatch fp-legend__swatch--door" />
//          <span>Doors</span>
//        </div>
//      </div>
//    </div>
//  );
//}

/* ─── Floorplan SVG: Bare ─── */
//function FloorplanBare() {
//  return (
//    <div className="fp-canvas fp-canvas--animate">
//      <svg viewBox="0 0 560 400" xmlns="http://www.w3.org/2000/svg" className="fp-svg">
//        <rect x="20" y="20" width="520" height="360" fill="none" stroke="#c9a96e" strokeWidth="2" rx="2" />
//        <line x1="300" y1="20" x2="300" y2="260" stroke="#c9a96e" strokeWidth="1.5" />
//        <line x1="300" y1="260" x2="540" y2="260" stroke="#c9a96e" strokeWidth="1.5" />
//        <line x1="20" y1="260" x2="220" y2="260" stroke="#c9a96e" strokeWidth="1.5" />
//
//        {/* Doors */}
//        <path d="M 290 20 A 30 30 0 0 0 260 50" fill="none" stroke="#c9a96e" strokeWidth="1" strokeDasharray="3,2" />
//        <path d="M 300 245 A 25 25 0 0 1 325 270" fill="none" stroke="#c9a96e" strokeWidth="1" strokeDasharray="3,2" />
//
//        {/* Windows */}
//        <rect x="80" y="19" width="60" height="3" fill="#9b9791" />
//        <rect x="380" y="19" width="60" height="3" fill="#9b9791" />
//        <rect x="19" y="290" width="3" height="40" fill="#9b9791" />
//
//        {/* Room labels */}
//        <text x="155" y="145" textAnchor="middle" fill="#9b9791" fontSize="12" fontFamily="DM Sans" letterSpacing="0.08em">LIVING ROOM</text>
//        <text x="155" y="162" textAnchor="middle" fill="#5a5855" fontSize="10" fontFamily="DM Sans">24.2 m²</text>
//        <text x="415" y="135" textAnchor="middle" fill="#9b9791" fontSize="12" fontFamily="DM Sans" letterSpacing="0.08em">BEDROOM</text>
//        <text x="415" y="152" textAnchor="middle" fill="#5a5855" fontSize="10" fontFamily="DM Sans">17.8 m²</text>
//        <text x="155" y="318" textAnchor="middle" fill="#9b9791" fontSize="12" fontFamily="DM Sans" letterSpacing="0.08em">KITCHEN</text>
//        <text x="155" y="334" textAnchor="middle" fill="#5a5855" fontSize="10" fontFamily="DM Sans">9.3 m²</text>
//        <text x="415" y="318" textAnchor="middle" fill="#9b9791" fontSize="12" fontFamily="DM Sans" letterSpacing="0.08em">BATHROOM</text>
//        <text x="415" y="334" textAnchor="middle" fill="#5a5855" fontSize="10" fontFamily="DM Sans">5.1 m²</text>
//
//        {/* Dimensions */}
//        <line x1="20" y1="10" x2="540" y2="10" stroke="#3a3a3e" strokeWidth="1" />
//        <text x="280" y="8" textAnchor="middle" fill="#5a5855" fontSize="9" fontFamily="DM Sans">10.4m</text>
//        <line x1="550" y1="20" x2="550" y2="380" stroke="#3a3a3e" strokeWidth="1" />
//        <text x="555" y="200" textAnchor="start" fill="#5a5855" fontSize="9" fontFamily="DM Sans" transform="rotate(90,555,200)">7.2m</text>
//      </svg>
//    </div>
//  );
//}

/* ─── 3D Placeholder View ─── */
//function View3D() {
//  return (
//    <div className="view3d">
//      <div className="view3d__scene">
//        {/* Isometric room illustration via CSS */}
//        <div className="iso-room">
//          <div className="iso-floor" />
//          <div className="iso-wall iso-wall--back" />
//          <div className="iso-wall iso-wall--left" />
//
//          {/* Furniture silhouettes */}
//          <div className="iso-furniture iso-sofa" />
//          <div className="iso-furniture iso-table" />
//          <div className="iso-furniture iso-shelf" />
//        </div>
//
//        <div className="view3d__badge">
//          <Box size={13} />
//          <span>Interactive 3D — Coming Soon</span>
//        </div>
//      </div>
//      <p className="view3d__note">
//        Full Three.js 3D reconstruction would render here. The backend model generates a point cloud from the video depth maps, which is then meshed and textured.
//      </p>
//    </div>
//  );
//}
