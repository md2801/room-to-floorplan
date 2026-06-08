// ─── VisionScape API Services ────────────────────────────────────────────────
// Connects to the Flask backend running on http://localhost:5000
// The validateSceneZip() function runs entirely in the browser (no backend needed).
// generateFloorplanFromScene() sends the ZIP to the backend and returns real URLs.

import JSZip from 'jszip';

// With the Vite proxy configured in vite.config.js, we can use relative URLs.
// This avoids CORS issues during development.
const BACKEND_URL = '';

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Returns true if a ZIP path segment is macOS metadata junk. */
function isIgnored(path) {
  const parts = path.split('/');
  return parts.some(
    (p) =>
      p === '__MACOSX' ||
      p === '.DS_Store' ||
      p.startsWith('._') ||
      (p.startsWith('.') && p !== '.')
  );
}

// ── Scene ZIP Validation (runs in browser, no backend needed) ────────────────

const REQUIRED_PANORAMA_FILES = [
  'camera_xyz.txt',
  'layout.txt',
  'full/rgb_rawlight.png',
  'full/rgb_coldlight.png',
  'full/rgb_warmlight.png',
  'full/depth.png',
  'full/semantic.png',
  'full/normal.png',
  'full/albedo.png',
];

const OPTIONAL_PANORAMA_FILES = ['empty/', 'simple/'];

export async function validateSceneZip(file) {
  const result = {
    status: 'invalid',
    sceneName: null,
    roomIds: [],
    rooms: [],
    topLevelErrors: [],
    hasAnnotation3d: false,
    has2DRendering: false,
    // Keep a reference to the raw File so we can upload it later.
    _file: file,
    _zip: null,
  };

  let zip;
  try {
    zip = await JSZip.loadAsync(file);
  } catch {
    result.topLevelErrors.push('Could not read ZIP file. Is this a valid archive?');
    return result;
  }

  const allPaths = new Set();
  zip.forEach((relativePath) => {
    if (!isIgnored(relativePath)) allPaths.add(relativePath);
  });

  // 1. Top-level scene folder (scene_XXXXX)
  const topLevelFolders = new Set();
  allPaths.forEach((p) => {
    const firstSegment = p.split('/')[0];
    if (firstSegment && /^scene_\d+$/.test(firstSegment)) {
      topLevelFolders.add(firstSegment);
    }
  });

  if (topLevelFolders.size === 0) {
    result.topLevelErrors.push('No top-level scene folder found (expected: scene_XXXXX/).');
    return result;
  }
  if (topLevelFolders.size > 1) {
    result.topLevelErrors.push(
      `Multiple top-level scene folders found: ${[...topLevelFolders].join(', ')}. Expected exactly one.`
    );
    return result;
  }

  const sceneName = [...topLevelFolders][0];
  result.sceneName = sceneName;
  const sceneRoot = `${sceneName}/`;

  // 2. annotation_3d.json
  const annotationPath = `${sceneRoot}annotation_3d.json`;
  result.hasAnnotation3d = allPaths.has(annotationPath);
  if (!result.hasAnnotation3d) {
    result.topLevelErrors.push('Missing: annotation_3d.json');
  }

  // 3. 2D_rendering/
  const renderingRoot = `${sceneRoot}2D_rendering/`;
  result.has2DRendering = allPaths.has(renderingRoot);
  if (!result.has2DRendering) {
    result.topLevelErrors.push('Missing: 2D_rendering/ folder');
    result.status = result.hasAnnotation3d ? 'partial' : 'invalid';
    return result;
  }

  // 4. Detect room IDs
  const roomIds = new Set();
  allPaths.forEach((p) => {
    if (p.startsWith(renderingRoot)) {
      const remainder = p.slice(renderingRoot.length);
      const roomId = remainder.split('/')[0];
      if (roomId && roomId !== '') roomIds.add(roomId);
    }
  });

  result.roomIds = [...roomIds].sort();

  if (result.roomIds.length === 0) {
    result.topLevelErrors.push('No room folders found under 2D_rendering/.');
    result.status = 'partial';
    return result;
  }

  // 5. Validate each room
  let validRoomCount = 0;

  for (const roomId of result.roomIds) {
    const panoramaRoot = `${renderingRoot}${roomId}/panorama/`;
    const hasPanorama = allPaths.has(panoramaRoot);

    const roomResult = {
      roomId,
      hasPanorama,
      missingRequired: [],
      availableOptional: [],
      availableFull: [],
      isValid: false,
      previewFiles: {},
    };

    if (!hasPanorama) {
      roomResult.missingRequired.push('panorama/ folder');
      result.rooms.push(roomResult);
      continue;
    }

    for (const relPath of REQUIRED_PANORAMA_FILES) {
      const fullPath = `${panoramaRoot}${relPath}`;
      if (!allPaths.has(fullPath)) roomResult.missingRequired.push(relPath);
    }

    for (const relPath of OPTIONAL_PANORAMA_FILES) {
      const fullPath = `${panoramaRoot}${relPath}`;
      if (allPaths.has(fullPath)) roomResult.availableOptional.push(relPath);
    }

    const previewMap = {
      rgb_rawlight:  `${panoramaRoot}full/rgb_rawlight.png`,
      rgb_coldlight: `${panoramaRoot}full/rgb_coldlight.png`,
      rgb_warmlight: `${panoramaRoot}full/rgb_warmlight.png`,
      depth:         `${panoramaRoot}full/depth.png`,
      semantic:      `${panoramaRoot}full/semantic.png`,
      normal:        `${panoramaRoot}full/normal.png`,
      albedo:        `${panoramaRoot}full/albedo.png`,
    };

    for (const [key, path] of Object.entries(previewMap)) {
      if (allPaths.has(path)) {
        roomResult.availableFull.push(path.split('/').pop());
        roomResult.previewFiles[key] = path;
      }
    }

    roomResult.isValid = roomResult.missingRequired.length === 0;
    if (roomResult.isValid) validRoomCount++;
    result.rooms.push(roomResult);
  }

  // 6. Overall status
  const hasTopErrors = result.topLevelErrors.length > 0;
  if (validRoomCount === result.roomIds.length && !hasTopErrors) {
    result.status = 'valid';
  } else if (validRoomCount > 0 || result.hasAnnotation3d) {
    result.status = 'partial';
  } else {
    result.status = 'invalid';
  }

  result._zip = zip;
  return result;
}

export async function extractPreviewImage(zip, path) {
  try {
    const file = zip.file(path);
    if (!file) return null;
    const blob = await file.async('blob');
    return URL.createObjectURL(blob);
  } catch {
    return null;
  }
}

// ── Real Inference API Call ──────────────────────────────────────────────────

/**
 * Upload the scene ZIP to the Flask backend, run inference, and return the
 * real SVG/PNG URLs from the server.
 *
 * @param {object}   sceneMetadata   Result from validateSceneZip() — must contain _file
 * @param {null}     _selectedRoomId  Not used (full-scene inference)
 * @param {function} onProgress       (percent: number, message: string) => void
 * @returns {{ success, scene_name, svg_url, png_url }}
 */
export async function generateFloorplanFromScene(sceneMetadata, _selectedRoomId, onProgress) {
  const file = sceneMetadata?._file;
  if (!file) {
    throw new Error('No scene file attached to sceneMetadata. Was validateSceneZip() called first?');
  }

  // ── Staged progress messages (the actual work happens on the server) ──
  const stages = [
    [5,  'Uploading scene ZIP to backend…'],
    [12, 'Extracting scene archive…'],
    [20, 'Loading model checkpoint…'],
    [30, 'Building bird\'s-eye-view observation…'],
    [45, 'Running neural inference…'],
    [60, 'Detecting room polygons…'],
    [72, 'Snapping wall axes…'],
    [82, 'Selecting best layout…'],
    [90, 'Rendering SVG floorplan…'],
    [95, 'Saving output files…'],
  ];

  // Animate progress while the fetch is in flight.
  let stageIndex = 0;
  const TICK_MS = 800;
  const progressInterval = setInterval(() => {
    if (stageIndex < stages.length) {
      const [pct, msg] = stages[stageIndex];
      onProgress?.(pct, msg);
      stageIndex++;
    }
  }, TICK_MS);

  let data;
  try {
    const formData = new FormData();
    formData.append('sceneZip', file, file.name);

    const response = await fetch(`${BACKEND_URL}/inferFloorplan`, {
      method: 'POST',
      body: formData,
    });

    data = await response.json();

    if (!response.ok || !data.success) {
      throw new Error(data.error || `Server returned HTTP ${response.status}`);
    }
  } finally {
    clearInterval(progressInterval);
  }

  onProgress?.(100, 'Complete');

  return {
    success:    data.success,
    scene_name: data.scene_name,
    svg_url:    data.svg_url,
    png_url:    data.png_url,
  };
}

// ── Legacy stubs (kept to avoid import errors in any other file) ─────────────

export async function uploadVideo(file) {
  return { success: true, fileId: `vid_${Date.now()}`, filename: file.name, size: file.size };
}

export async function generateFloorplan(_fileId, onProgress) {
  onProgress?.(100, 'Complete');
  return { success: true, floorplanId: `fp_${Date.now()}` };
}
