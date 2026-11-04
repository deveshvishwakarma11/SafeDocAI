/**
 * Cryptographic helpers for content-hash identity and collision-safe artifact stems
 * as specified in TRD Section 5 & 6
 */

export async function computeSHA256(data: ArrayBuffer | string): Promise<string> {
  const buffer = typeof data === 'string' ? new TextEncoder().encode(data) : data;
  const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

export function generateArtifactStem(fileName: string, hash: string): string {
  const cleanName = fileName.replace(/[^a-zA-Z0-9_-]/g, '_').toLowerCase();
  const shortHash = hash.substring(0, 10);
  return `${cleanName}_${shortHash}`;
}
