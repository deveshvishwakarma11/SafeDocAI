# SafeDocAI

Local-first personal document intelligence. Upload PDFs/images, ask questions in
English, Hindi or Hinglish, and get answers grounded in your own documents.
Everything runs on your machine: no cloud APIs, no telemetry.

## Run

Prerequisites: Node.js 20+, optional [Ollama](https://ollama.com) with `ollama pull qwen2.5:3b`.
OCR language data lives in `data/tessdata/` (`eng`, `hin`).

```bash
npm install
cp .env.example .env     # optional
npm run dev              # http://127.0.0.1:3000
npm test
```

## How a question is answered

1. **Exact fields** (roll number, DOB, amount...) come straight from SQLite. Instant, no model.
2. **Content search**: the server picks the document by what it *says* (BM25 over stored chunks), not only by file name. It asks "which document?" only when two files genuinely match.
3. **Local LLM** (Ollama) answers from just the top 3 chunks (~800 chars prompt). Stop cancels generation immediately.
4. **Verification**: every ID/date/amount in the answer must appear in the retrieved text. Otherwise you get the document's own passage or an honest "couldn't verify", never a guess.

No model running? Steps 1, 2 and the verbatim passage still work.

### Result sheets, statements and other multi-section documents

Documents that repeat sections (sessions, semesters, periods) and tables are parsed
into **scoped facts** and **table rows** (`server/structure.ts`). Questions like
"Semester 3 ka SGPA", "Maths-I ke marks", "kis subject me fail hua" are answered
exactly from that structure. A value that exists in several sections (e.g. the SGPA of
semester 1 in a regular and a back session) is shown for **every** section, never
collapsed to the first one. Chunks follow the sections too, so a table is never cut in half.

The browser client no longer answers from stored fields itself: the server owns every answer.

## Privacy

- Server listens on `127.0.0.1` only.
- Originals, SQLite and derived artifacts stay under `data/` (git-ignored).
- Ollama is called over loopback only.

Product docs: PRD, TRD and App Flow (kept outside this repo).
