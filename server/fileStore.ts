/**
 * SafeDocAI Managed File Store
 * -----------------------------
 * Implements Backend Schema §04 "managed-file-store design" + §12 path-safety:
 *  - originals:   data/documents/<artifact_stem>.<ext>
 *  - stage-2:     data/output/<artifact_stem>/...
 *  - understood:  data/understood/<artifact_stem>.json
 *  - every read/unlink is validated against the managed root before use (§12)
 *  - PRD M2 size cap enforced here so every entry point shares one number
 */
import fs from 'node:fs';
import path from 'node:path';

/** PRD M2: 50 MB per file cap — single source of truth (docs: shared owned fix). */
export const MAX_FILE_BYTES = 50 * 1024 * 1024;

export interface StoreDirs {
  dataRoot: string;
  documentsDir: string;
  outputDir: string;
  understoodDir: string;
}

export function getStoreDirs(dataRoot: string): StoreDirs {
  return {
    dataRoot,
    documentsDir: path.join(dataRoot, 'documents'),
    outputDir: path.join(dataRoot, 'output'),
    understoodDir: path.join(dataRoot, 'understood')
  };
}

export function ensureStoreDirs(dataRoot: string): StoreDirs {
  const dirs = getStoreDirs(dataRoot);
  fs.mkdirSync(dirs.documentsDir, { recursive: true });
  fs.mkdirSync(dirs.outputDir, { recursive: true });
  fs.mkdirSync(dirs.understoodDir, { recursive: true });
  return dirs;
}

const SAFE_STEM = /^[a-zA-Z0-9_-]+$/;

/** §12: reject any artifact stem that could escape the managed root. */
function assertSafeStem(artifactStem: string): void {
  if (!SAFE_STEM.test(artifactStem)) {
    throw new Error(`Unsafe artifact stem rejected: "${artifactStem}"`);
  }
}

/**
 * §12: managed paths are stored project-root-relative ("data/documents/...").
 * Resolve against the project root, then prove the result stays under dataRoot.
 */
export function assertManagedPath(dataRoot: string, relativePath: string): string {
  const resolvedRoot = path.resolve(dataRoot);
  const resolved = path.resolve(process.cwd(), relativePath);
  if (resolved !== resolvedRoot && !resolved.startsWith(resolvedRoot + path.sep)) {
    throw new Error(`Path escapes managed store root: "${relativePath}"`);
  }
  return resolved;
}

export function saveOriginal(dataRoot: string, buffer: Buffer, artifactStem: string, ext: string): string {
  assertSafeStem(artifactStem);
  const relativePath = path.posix.join('data/documents', `${artifactStem}.${ext}`);
  const absolutePath = assertManagedPath(dataRoot, relativePath);
  fs.writeFileSync(absolutePath, buffer);
  return relativePath;
}

export function originalAbsolutePath(dataRoot: string, relativePath: string): string {
  return assertManagedPath(dataRoot, relativePath);
}

export function readOriginal(dataRoot: string, relativePath: string): Buffer {
  const absolutePath = originalAbsolutePath(dataRoot, relativePath);
  return fs.readFileSync(absolutePath);
}

export function originalExists(dataRoot: string, relativePath: string): boolean {
  try {
    return fs.existsSync(originalAbsolutePath(dataRoot, relativePath));
  } catch {
    return false;
  }
}

/**
 * Removal (FR-13 "Remove from SafeDocAI"): deletes ONLY managed paths —
 * original, stage-2 output dir, understood artifact. Unmanaged user paths are refused.
 */
export function deleteManagedArtifacts(dataRoot: string, relativePath: string, artifactStem: string): void {
  const dirs = getStoreDirs(dataRoot);

  try {
    const abs = originalAbsolutePath(dataRoot, relativePath);
    fs.rmSync(abs, { force: true });
  } catch (e) {
    console.warn(`[fileStore] refused to unlink non-managed path "${relativePath}":`, (e as Error).message);
  }

  if (SAFE_STEM.test(artifactStem)) {
    fs.rmSync(path.join(dirs.outputDir, artifactStem), { recursive: true, force: true });
    fs.rmSync(path.join(dirs.understoodDir, `${artifactStem}.json`), { force: true });
  }
}

export function writeUnderstoodArtifact(dataRoot: string, artifactStem: string, artifact: unknown): string {
  assertSafeStem(artifactStem);
  const dirs = getStoreDirs(dataRoot);
  const target = path.join(dirs.understoodDir, `${artifactStem}.json`);
  fs.writeFileSync(target, JSON.stringify(artifact, null, 2));
  return path.posix.join('data/understood', `${artifactStem}.json`);
}

function dirBytes(dir: string): number {
  let total = 0;
  let entries: fs.Dirent[] = [];
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return 0;
  }
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    try {
      if (entry.isDirectory()) total += dirBytes(full);
      else total += fs.statSync(full).size;
    } catch {
      // ignore transient files
    }
  }
  return total;
}

/** Real bytes on disk (documents + understood), used by honest diagnostics. */
export function bytesUsed(dataRoot: string): number {
  const dirs = getStoreDirs(dataRoot);
  return dirBytes(dirs.documentsDir) + dirBytes(dirs.understoodDir);
}

const MIME_BY_TYPE: Record<string, string> = {
  pdf: 'application/pdf',
  png: 'image/png',
  jpg: 'image/jpeg',
  jpeg: 'image/jpeg'
};

export function mimeForType(fileType: string): string {
  return MIME_BY_TYPE[fileType] || 'application/octet-stream';
}

export function toDataUrl(buffer: Buffer, fileType: string): string {
  return `data:${mimeForType(fileType)};base64,${buffer.toString('base64')}`;
}
