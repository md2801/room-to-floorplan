import { Link, useLocation } from 'react-router-dom';
import './Navbar.css';

export default function Navbar() {
  const location = useLocation();
  const isHome = location.pathname === '/';

  return (
    <header className="navbar">
      <Link to="/" className="navbar__logo">
        <span className="navbar__logo-mark">V</span>
        <span className="navbar__logo-text">VisionScape</span>
      </Link>

      <nav className="navbar__links">
        <span className="navbar__tag">Room-to-Floorplan Generator</span>
      </nav>

      {!isHome && (
        <Link to="/" className="navbar__back">
          ← New Project
        </Link>
      )}
    </header>
  );
}
