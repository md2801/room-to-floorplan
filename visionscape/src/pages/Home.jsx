import { useNavigate } from 'react-router-dom';
import { ArrowRight, ScanLine, Layers, Box, FileArchive } from 'lucide-react';
import './Home.css';

const FEATURES = [
  {
    icon: FileArchive,
    title: 'Structured3D Import',
    desc: 'Upload a Structured3D scene ZIP — VisionScape validates structure, rooms, and panorama files automatically.',
  },
  {
    icon: Layers,
    title: 'Floorplan Output',
    desc: 'Generate an annotated floorplan from depth maps, semantic labels, and room boundaries.',
  },
  {
    icon: Box,
    title: '3D Reconstruction',
    desc: 'Visualise your space in three dimensions with furniture placement and room depth.',
  },
];

export default function Home() {
  const navigate = useNavigate();

  return (
    <main className="home">
      {/* Hero */}
      <section className="hero">
        <div className="hero__eyebrow">
          <span className="hero__eyebrow-dot" />
          AI-Powered Spatial Analysis
        </div>

        <h1 className="hero__title">
          Transform any room
          <br />
          <em>into a floorplan.</em>
        </h1>

        <p className="hero__desc">
          Point your camera, walk your space. VisionScape analyses your video
          and generates a precise architectural floorplan in seconds.
        </p>

        <div className="hero__actions">
          <button className="btn-primary" onClick={() => navigate('/studio')}>
            Start a Project
            <ArrowRight size={16} />
          </button>
          <button className="btn-ghost" onClick={() => navigate('/studio')}>
            View Demo
          </button>
        </div>

        <div className="hero__grid-bg" aria-hidden="true">
          {Array.from({ length: 200 }).map((_, i) => (
            <div key={i} className="hero__grid-cell" />
          ))}
        </div>
      </section>

      {/* Feature cards */}
      <section className="features">
        <div className="features__rule" />
        <div className="features__grid">
          {FEATURES.map(({ icon: Icon, title, desc }, i) => (
            <div key={i} className="feature-card" style={{ animationDelay: `${i * 80}ms` }}>
              <div className="feature-card__icon">
                <Icon size={20} />
              </div>
              <h3 className="feature-card__title">{title}</h3>
              <p className="feature-card__desc">{desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA strip */}
      <section className="cta-strip">
        <div className="cta-strip__inner">
          <div>
            <p className="cta-strip__label">Ready to begin?</p>
            <p className="cta-strip__sub">No account required. Just your video.</p>
          </div>
          <button className="btn-primary" onClick={() => navigate('/studio')}>
            Open Studio <ArrowRight size={15} />
          </button>
        </div>
      </section>
    </main>
  );
}
