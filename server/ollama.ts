/**
 * Ollama local LLM client (TRD §7.2 target model: qwen2.5:3b).
 * Fully offline: talks only to the loopback Ollama server, never the internet.
 */
import os from 'node:os';
export const OLLAMA_URL = (process.env.OLLAMA_URL || 'http://127.0.0.1:11434').replace(/\/$/, '');
export const OLLAMA_MODEL = process.env.OLLAMA_MODEL || 'qwen2.5:3b';

let cachedStatus: { at: number; ready: boolean } | null = null;
const STATUS_TTL_MS = 15_000;

export async function ollamaStatus(): Promise<boolean> {
  if (cachedStatus && Date.now() - cachedStatus.at < STATUS_TTL_MS) {
    return cachedStatus.ready;
  }
  try {
    const res = await fetch(`${OLLAMA_URL}/api/tags`, { signal: AbortSignal.timeout(1500) });
    const ready = res.ok;
    cachedStatus = { at: Date.now(), ready };
    return ready;
  } catch {
    cachedStatus = { at: Date.now(), ready: false };
    return false;
  }
}

export interface OllamaGenerateOptions {
  /** Hard wall-clock limit for one generation (default OLLAMA_TIMEOUT_MS or 60s). */
  timeoutMs?: number;
  /** Caller abort (user pressed Stop / closed the tab) — cancels the generation. */
  signal?: AbortSignal;
  /** Max new tokens (default 80: short grounded answers only). */
  numPredict?: number;
}

/**
 * CPU threads Ollama may use. Defaults to half the cores so the laptop stays
 * responsive while a local model generates (full-core generation froze machines).
 */
export const OLLAMA_NUM_THREAD =
  Number(process.env.OLLAMA_NUM_THREAD) || Math.max(1, Math.floor(os.cpus().length / 2));

const DEFAULT_TIMEOUT_MS = Number(process.env.OLLAMA_TIMEOUT_MS) || 60_000;

/** Synthesis is ON unless explicitly disabled with OLLAMA_SYNTHESIS=0. */
export function synthesisEnabled(): boolean {
  return process.env.OLLAMA_SYNTHESIS !== '0';
}

/** Single-turn local generation. temperature 0 per TRD §7.2 determinism rule. */
export async function ollamaGenerate(
  prompt: string,
  options: OllamaGenerateOptions = {}
): Promise<string> {
  const signals: AbortSignal[] = [AbortSignal.timeout(options.timeoutMs ?? DEFAULT_TIMEOUT_MS)];
  if (options.signal) signals.push(options.signal);

  const res = await fetch(`${OLLAMA_URL}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: OLLAMA_MODEL,
      prompt,
      stream: false,
      keep_alive: '10m',
      options: {
        temperature: 0,
        num_ctx: 2048,
        num_predict: options.numPredict ?? 80,
        num_thread: OLLAMA_NUM_THREAD,
        stop: ['\nQUESTION:', '\nCONTEXT:']
      }
    }),
    // Closing the connection makes Ollama stop generating (frees the CPU).
    signal: AbortSignal.any(signals)
  });

  if (!res.ok) {
    throw new Error(`Ollama generate failed: HTTP ${res.status}`);
  }

  const data = (await res.json()) as { response?: string };
  return (data.response || '').trim();
}
