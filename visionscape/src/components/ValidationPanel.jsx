import { useState, useEffect } from 'react';
import {
  CheckCircle2, AlertTriangle, XCircle,
  ChevronDown, ChevronUp, Image as ImageIcon,
  FileText, Folder, Eye
} from 'lucide-react';
import { extractPreviewImage } from '../services/api.js';
import './ValidationPanel.css';

/**
 * Displays the full validation result from validateSceneZip().
 * Also handles room selection and shows image previews.
 */
export default function ValidationPanel({
  validation,
  selectedRoomId,
  onRoomSelect,
}) {
  const [expandedRoom, setExpandedRoom] = useState(null);
  const [previews, setPreviews] = useState({});
  const [loadingPreviews, setLoadingPreviews] = useState(false);

  const { status, sceneName, roomIds, rooms, topLevelErrors, hasAnnotation3d, has2DRendering } = validation;
  const validRooms = rooms.filter((r) => r.isValid);
  const invalidRooms = rooms.filter((r) => !r.isValid);

  // Load previews when a room is expanded
  useEffect(() => {
    if (!expandedRoom || !validation._zip) return;
    if (previews[expandedRoom]) return; // already loaded

    const room = rooms.find((r) => r.roomId === expandedRoom);
    if (!room || !room.previewFiles) return;

    setLoadingPreviews(true);

    Promise.all(
      Object.entries(room.previewFiles).map(async ([key, path]) => {
        const url = await extractPreviewImage(validation._zip, path);
        return [key, url];
      })
    ).then((entries) => {
      setPreviews((prev) => ({
        ...prev,
        [expandedRoom]: Object.fromEntries(entries.filter(([, v]) => v)),
      }));
      setLoadingPreviews(false);
    });
  }, [expandedRoom, validation._zip, rooms, previews]);

  const statusConfig = {
    valid: {
      icon: CheckCircle2,
      label: 'Valid Scene',
      cls: 'badge--valid',
    },
    partial: {
      icon: AlertTriangle,
      label: 'Partially Valid',
      cls: 'badge--partial',
    },
    invalid: {
      icon: XCircle,
      label: 'Invalid Scene',
      cls: 'badge--invalid',
    },
  };

  const { icon: StatusIcon, label: statusLabel, cls: statusCls } = statusConfig[status];

  const toggleRoom = (id) => setExpandedRoom((prev) => (prev === id ? null : id));

  return (
    <div className="validation-panel">
      {/* ── Header ── */}
      <div className="validation-panel__header">
        <div>
          <h3 className="validation-panel__scene-name">{sceneName}</h3>
          <p className="validation-panel__file-meta">
            {roomIds.length} rooms detected · {validRooms.length} valid panoramas
          </p>
        </div>
        <div className={`status-badge ${statusCls}`}>
          <StatusIcon size={13} />
          {statusLabel}
        </div>
      </div>

      {/* ── Top-level structure ── */}
      <div className="validation-summary">
        <div className={`summary-row ${hasAnnotation3d ? 'summary-row--ok' : 'summary-row--fail'}`}>
          {hasAnnotation3d ? <CheckCircle2 size={13} /> : <XCircle size={13} />}
          <span>annotation_3d.json</span>
        </div>
        <div className={`summary-row ${has2DRendering ? 'summary-row--ok' : 'summary-row--fail'}`}>
          {has2DRendering ? <CheckCircle2 size={13} /> : <XCircle size={13} />}
          <span>2D_rendering/</span>
        </div>
      </div>

      {topLevelErrors.length > 0 && (
        <div className="validation-errors">
          {topLevelErrors.map((e, i) => (
            <p key={i} className="validation-errors__item">
              <XCircle size={12} /> {e}
            </p>
          ))}
        </div>
      )}

      {/* ── Room selector ── */}
      {validRooms.length > 0 && (
        <div className="room-selector">
          <label className="room-selector__label">
            Select room to generate floorplan
          </label>
          <div className="room-selector__grid">
            {validRooms.map((r) => (
              <button
                key={r.roomId}
                className={`room-chip ${selectedRoomId === r.roomId ? 'room-chip--active' : ''}`}
                onClick={() => onRoomSelect(r.roomId)}
              >
                <Folder size={11} />
                Room {r.roomId}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* ── Room details accordion ── */}
      <div className="room-accordion">
        <p className="room-accordion__heading">Room Details</p>
        {rooms.map((room) => (
          <div key={room.roomId} className={`room-item ${room.isValid ? 'room-item--valid' : 'room-item--invalid'}`}>
            <button
              className="room-item__toggle"
              onClick={() => toggleRoom(room.roomId)}
            >
              <div className="room-item__toggle-left">
                {room.isValid ? (
                  <CheckCircle2 size={13} className="room-item__icon room-item__icon--ok" />
                ) : (
                  <AlertTriangle size={13} className="room-item__icon room-item__icon--warn" />
                )}
                <span className="room-item__id">Room {room.roomId}</span>
                {room.missingRequired.length > 0 && (
                  <span className="room-item__missing-count">
                    {room.missingRequired.length} missing
                  </span>
                )}
                <span className="room-item__optional-count">
                  {room.availableOptional.length} optional
                </span>
              </div>
              {expandedRoom === room.roomId ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </button>

            {expandedRoom === room.roomId && (
              <div className="room-item__body">
                {/* Missing required files */}
                {room.missingRequired.length > 0 && (
                  <div className="room-item__section">
                    <p className="room-item__section-title room-item__section-title--error">
                      <XCircle size={11} /> Missing Required
                    </p>
                    {room.missingRequired.map((f) => (
                      <p key={f} className="room-item__file room-item__file--missing">
                        <FileText size={10} /> {f}
                      </p>
                    ))}
                  </div>
                )}

                {/* Required files confirmed present */}
                {room.availableFull.length > 0 && (
                  <div className="room-item__section">
                    <p className="room-item__section-title room-item__section-title--ok">
                      <CheckCircle2 size={11} /> Required Files Present
                    </p>
                    {room.availableFull.map((f) => (
                      <p key={f} className="room-item__file room-item__file--ok">
                        <FileText size={10} /> {f}
                      </p>
                    ))}
                  </div>
                )}

                {/* Optional folders */}
                {room.availableOptional.length > 0 && (
                  <div className="room-item__section">
                    <p className="room-item__section-title">
                      <Eye size={11} /> Optional Available
                    </p>
                    {room.availableOptional.map((f) => (
                      <p key={f} className="room-item__file">
                        <Folder size={10} /> {f}
                      </p>
                    ))}
                  </div>
                )}

                {/* Panorama image previews */}
                {room.isValid && (
                  <div className="room-item__previews">
                    <p className="room-item__section-title">
                      <ImageIcon size={11} /> Panorama Previews
                    </p>
                    {loadingPreviews && expandedRoom === room.roomId && (
                      <p className="room-item__preview-loading">Extracting images…</p>
                    )}
                    {previews[room.roomId] && (
                      <div className="preview-grid">
                        {Object.entries(previews[room.roomId]).map(([key, url]) => (
                          <div key={key} className="preview-tile">
                            <img src={url} alt={key} className="preview-tile__img" />
                            <span className="preview-tile__label">{key}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}