// ─── VisionScape Mock API Services ────────────────────────────
import JSZip from 'jszip';

// ── Helpers ──────────────────────────────────────────────────

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Returns true if a path segment should be ignored (Mac metadata) */
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

// ── Scene ZIP Validation ──────────────────────────────────────

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

const OPTIONAL_PANORAMA_FILES = [
  'empty/',
  'simple/',
];

export async function validateSceneZip(file) {
  const result = {
    status: 'invalid',
    sceneName: null,
    roomIds: [],
    rooms: [],
    topLevelErrors: [],
    hasAnnotation3d: false,
    has2DRendering: false,
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
    if (!isIgnored(relativePath)) {
      allPaths.add(relativePath);
    }
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
    result.topLevelErrors.push(`Missing: annotation_3d.json`);
  }

  // 3. 2D_rendering/
  const renderingRoot = `${sceneRoot}2D_rendering/`;
  result.has2DRendering = allPaths.has(renderingRoot);
  if (!result.has2DRendering) {
    result.topLevelErrors.push(`Missing: 2D_rendering/ folder`);
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
      if (!allPaths.has(fullPath)) {
        roomResult.missingRequired.push(relPath);
      }
    }

    for (const relPath of OPTIONAL_PANORAMA_FILES) {
      const fullPath = `${panoramaRoot}${relPath}`;
      if (allPaths.has(fullPath)) {
        roomResult.availableOptional.push(relPath);
      }
    }

    const previewMap = {
      rgb_rawlight: `${panoramaRoot}full/rgb_rawlight.png`,
      rgb_coldlight: `${panoramaRoot}full/rgb_coldlight.png`,
      rgb_warmlight: `${panoramaRoot}full/rgb_warmlight.png`,
      depth: `${panoramaRoot}full/depth.png`,
      semantic: `${panoramaRoot}full/semantic.png`,
      normal: `${panoramaRoot}full/normal.png`,
      albedo: `${panoramaRoot}full/albedo.png`,
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

export async function generateFloorplanFromScene(sceneMetadata, selectedRoomId, onProgress) {
  const steps = [
    { progress: 8,  message: 'Parsing annotation_3d.json…' },
    { progress: 20, message: `Loading room ${selectedRoomId} panorama…` },
    { progress: 35, message: 'Extracting depth map…' },
    { progress: 50, message: 'Analysing semantic labels…' },
    { progress: 65, message: 'Detecting wall planes…' },
    { progress: 78, message: 'Tracing floor boundaries…' },
    { progress: 90, message: 'Rendering floorplan…' },
    { progress: 100, message: 'Complete' },
  ];
  for (const step of steps) {
    await delay(480 + Math.random() * 350);
    onProgress?.(step.progress, step.message);
  }
  return {
    success: true,
    floorplanId: `fp_${sceneMetadata.sceneName}_${selectedRoomId}_${Date.now()}`,
    sceneName: sceneMetadata.sceneName,
    roomId: selectedRoomId,
  };
}

// ── Legacy video API ──────────────────────────────────────────
export async function uploadVideo(file) {
  await delay(1200);
  return { success: true, fileId: `vid_${Date.now()}`, filename: file.name, size: file.size };
}

export async function generateFloorplan(fileId, onProgress) {
  const steps = [
    { progress: 10, message: 'Extracting frames…' },
    { progress: 28, message: 'Detecting room boundaries…' },
    { progress: 45, message: 'Mapping spatial relationships…' },
    { progress: 62, message: 'Identifying furniture elements…' },
    { progress: 78, message: 'Rendering floorplan…' },
    { progress: 90, message: 'Generating 3D model…' },
    { progress: 100, message: 'Complete' },
  ];
  for (const step of steps) {
    await delay(600 + Math.random() * 400);
    onProgress?.(step.progress, step.message);
  }
  return { success: true, floorplanId: `fp_${Date.now()}` };
}