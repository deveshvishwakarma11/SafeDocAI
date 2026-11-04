import { QueryResult, StoredDocument, ExtractedField, SourceEvidence, QueryRoute } from '../types';
import { vectorEngine } from './vectorEngine';
import { matchFieldGroup, PARENT_CANDIDATE_FIELDS, SynonymGroup } from './fieldMatcher';

export class QueryRouter {
  public static async executeQuery(
    rawQuery: string,
    options: {
      documents: StoredDocument[];
      activeDocFilter?: number;
      isCancelled?: () => boolean;
      signal?: AbortSignal;
    }
  ): Promise<QueryResult> {
    const startTime = performance.now();
    const query = rawQuery.trim();
    const lowerQuery = query.toLowerCase();

    // 1. Language Detection
    const detectedLang = this.detectLanguage(query);

    // 2. Check if inventory is empty (documents come from the API-backed repository)
    const allDocs = (options.documents || []).filter(d => d.status === 'success');
    if (allDocs.length === 0) {
      return {
        route: 'out_of_scope',
        detected_language: detectedLang,
        intent: 'empty_inventory',
        answer: detectedLang === 'hi' || detectedLang === 'hinglish'
          ? 'SafeDocAI me abhi koi documents indexed nahi hain. Kirpaya pehle PDF ya image upload karein.'
          : 'No documents are currently indexed in SafeDocAI. Please upload a PDF or image document first to query your records.',
        evidence: [],
        is_insufficient_evidence: true,
        latency_ms: Math.round(performance.now() - startTime)
      };
    }

    // One document, one chat: the file open in the viewer scopes every content
    // answer below, so fast local paths cannot answer from a different file.
    const scopedDocs = options.activeDocFilter
      ? allDocs.filter(d => d.id === options.activeDocFilter)
      : allDocs;
    const docs = scopedDocs.length > 0 ? scopedDocs : allDocs;

    // 2.5 Greeting & Conversational Check (hi, hii, hello, namaste, hey)
    if (this.isGreeting(lowerQuery)) {
      const greetingAnswer = this.generateGreetingResponse(detectedLang, allDocs);
      return {
        route: 'greeting',
        detected_language: detectedLang,
        intent: 'friendly_greeting',
        answer: greetingAnswer,
        evidence: [],
        clarification_options: allDocs.length > 0 ? [
          { label: 'Check Roll Number / Marksheet', value: 'What is my roll number?' },
          { label: 'Check Bill Due Date / Amount', value: 'What is my electricity bill amount and due date?' },
          { label: 'Check Bank Account / Balance', value: 'What is my bank account number?' }
        ] : undefined,
        latency_ms: Math.round(performance.now() - startTime)
      };
    }

    // 3. Out-of-scope check (world knowledge, weather, coding, etc.)
    if (this.isOutOfScope(lowerQuery)) {
      return {
        route: 'out_of_scope',
        detected_language: detectedLang,
        intent: 'unrelated_request',
        answer: detectedLang === 'hi' || detectedLang === 'hinglish'
          ? 'SafeDocAI kewal aapke uploaded documents se verified jawab deta hai. Yeh sawaal aapke documents ke bahar hai.'
          : 'SafeDocAI strictly answers queries about your stored personal documents. This question is outside the scope of your indexed documents.',
        evidence: [],
        is_insufficient_evidence: true,
        latency_ms: Math.round(performance.now() - startTime)
      };
    }

    // 4. Document Retrieval intent ("Show my passbook", "Marksheet dikhao", "Open electricity bill")
    const docRetrievalMatch = this.checkDocumentRetrieval(lowerQuery, allDocs);
    if (docRetrievalMatch) {
      const doc = docRetrievalMatch;
      return {
        route: 'document_retrieval',
        detected_language: detectedLang,
        intent: `open_document_${doc.id}`,
        answer: detectedLang === 'hi' || detectedLang === 'hinglish'
          ? `Maine aapka dastavej "${doc.file_name}" khoj liya hai. Aap ise neeche inspect kar sakte hain.`
          : `I found your document "${doc.file_name}" (${doc.category}). You can view the document and its verified fields below.`,
        evidence: [
          {
            document_id: doc.id,
            file_name: doc.file_name,
            snippet: doc.raw_text.substring(0, 280) + '...',
            page_number: 1,
            confidence: 1.0
          }
        ],
        latency_ms: Math.round(performance.now() - startTime)
      };
    }

    // 5. (Removed) Client-side exact-field lookup.
    //    It answered from flat fields BEFORE the server, so it returned the first
    //    value of a repeated field (SGPA of one semester for every semester) and
    //    never saw the structured/section-aware engines. The server owns every
    //    answer now; this client only renders it.

    // 6. Try Server-Side Grounded AI Query (local engines + Ollama via /api/ask-query)
    try {
      const res = await fetch('/api/ask-query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: options.signal,
        body: JSON.stringify({
          query,
          activeDocFilter: options.activeDocFilter
        })
      });

      if (res.ok) {
        const aiData = await res.json();
        if (aiData.answer) {
          return {
            route: aiData.route || 'semantic_rag',
            detected_language: aiData.detected_language || detectedLang,
            intent: aiData.intent || 'grounded_answer',
            answer: aiData.answer,
            evidence: aiData.evidence || [],
            clarification_options: aiData.clarification_options,
            is_insufficient_evidence: aiData.is_insufficient_evidence,
            latency_ms: Math.round(performance.now() - startTime)
          };
        }
      }

      // The server answered but failed (HTTP 5xx / empty): say so. Never replace a
      // failed answer with a passage found by a second, weaker search.
      return {
        route: 'out_of_scope',
        detected_language: detectedLang,
        intent: 'server_error',
        answer:
          detectedLang === 'hi' || detectedLang === 'hinglish'
            ? 'Is sawal ka jawab banane me local server se dikkat aayi. Dobara try karein.'
            : 'The local server could not answer this question. Please try again.',
        evidence: [],
        is_insufficient_evidence: true,
        latency_ms: Math.round(performance.now() - startTime)
      };
    } catch (e: any) {
      // A user-initiated Stop must stay stopped — never fall through to local
      // vector search and append an answer the user already cancelled.
      if (e?.name === 'AbortError') throw e;
      console.warn('Local server unreachable, using the offline fallback search:', e);
    }

    // 7. Offline fallback (only when the local server cannot be reached at all)
    const searchResults = vectorEngine.searchChunks(docs, query, 3, options.activeDocFilter);

    if (searchResults.length > 0) {
      const topResults = searchResults;

      const evidenceList: SourceEvidence[] = topResults.map(r => ({
        document_id: r.chunk.document_id,
        file_name: r.chunk.file_name,
        snippet: r.chunk.text,
        page_number: r.chunk.page_number,
        confidence: Math.min(0.99, Number((r.score * 1.2).toFixed(2)))
      }));

      const synthesizedAnswer = this.generateGroundedSemanticAnswer(query, topResults, detectedLang);

      return {
        route: 'semantic_rag',
        detected_language: detectedLang,
        intent: 'semantic_content_answer',
        answer: synthesizedAnswer,
        evidence: evidenceList,
        latency_ms: Math.round(performance.now() - startTime)
      };
    }

    // 8. Explicit "show all details" request only.
    //    Broad tokens ('document', 'kya hai', 'batao') used to send every
    //    unrecognised question here, so unrelated questions returned the same
    //    all-fields dump. Unknown questions now fall through honestly instead.
    const wantsSummary =
      /\b(marksheet|marks|percentage|result|details?|summary|summari[sz]e|overview)\b/.test(lowerQuery) ||
      /\b(?:10th|12th)\b/.test(lowerQuery) ||
      /\b(all\s+(?:the\s+)?(?:information|details)|poori\s*jankari|sab\s*kuch|kya\s*kya\s+hai|sab\s+batao|sab\s+details)\b/.test(
        lowerQuery
      ) ||
      /\btell\s+me\s+about\s+(?:this|the|my)\s+(?:document|file|pdf|id)\b/.test(lowerQuery);

    if (wantsSummary) {
      // Find the most relevant document (e.g. matching name or category, or latest uploaded)
      let targetDoc = docs[0] || null;

      if (!targetDoc) {
        if (lowerQuery.includes('marksheet') || lowerQuery.includes('10th') || lowerQuery.includes('12th') || lowerQuery.includes('college') || lowerQuery.includes('marks')) {
          targetDoc = allDocs.find(d => d.category === 'Marksheet / Degree' || d.file_name.toLowerCase().includes('marksheet') || d.file_name.toLowerCase().includes('10th')) || allDocs[0];
        } else if (lowerQuery.includes('bill') || lowerQuery.includes('electricity')) {
          targetDoc = allDocs.find(d => d.category === 'Utility Bill' || d.file_name.toLowerCase().includes('bill')) || allDocs[0];
        } else {
          // Latest document uploaded
          targetDoc = allDocs[allDocs.length - 1] || allDocs[0];
        }
      }

      if (targetDoc) {
        const fieldsList = (targetDoc.fields || []).map(f => `• **${f.display_label}**: \`${f.field_value}\``).join('\n');

        // No verified fields → do not paste raw OCR as "details".
        if (!fieldsList) {
          return {
            route: 'exact_field',
            detected_language: detectedLang,
            intent: 'no_verified_fields',
            answer: detectedLang === 'hi' || detectedLang === 'hinglish'
              ? `\`${targetDoc.file_name}\` se koi verified field extract nahi ho paya. Document viewer me **Re-run OCR** try karein.`
              : `No verified fields could be extracted from **${targetDoc.file_name}**. Try **Re-run OCR** in the document viewer.`,
            evidence: [],
            is_insufficient_evidence: true,
            latency_ms: Math.round(performance.now() - startTime)
          };
        }

        return {
          route: 'exact_field',
          detected_language: detectedLang,
          intent: 'field_summary',
          answer:
            `### 📄 **${targetDoc.file_name}** (${targetDoc.category})\n\n${fieldsList}\n\n` +
            (detectedLang === 'hi' || detectedLang === 'hinglish'
              ? '*Yeh values stored transcription se padhi aur verify ki gayi hain.*'
              : '*Values were read from the stored transcription and verified against it.*'),
          evidence: (targetDoc.fields || []).slice(0, 3).map(f => ({
            document_id: targetDoc.id,
            file_name: targetDoc.file_name,
            field_name: f.display_label,
            value: f.field_value,
            snippet: f.evidence_text,
            page_number: f.page_number,
            confidence: f.confidence
          })),
          latency_ms: Math.round(performance.now() - startTime)
        };
      }
    }

    // 9. Insufficient Evidence / Not Found
    return {
      route: 'exact_field',
      detected_language: detectedLang,
      intent: 'insufficient_evidence',
      answer: detectedLang === 'hi' || detectedLang === 'hinglish'
        ? `\`${docs[0].file_name}\` me yeh jankari nahi mili. SafeDocAI galat anuman nahi lagata — aap koi doosra sawaal pooch sakte hain.`
        : `This information is not recorded in **${docs[0].file_name}**. SafeDocAI does not guess missing data — try asking about something else in this file.`,
      evidence: [],
      is_insufficient_evidence: true,
      latency_ms: Math.round(performance.now() - startTime)
    };
  }

  private static detectLanguage(query: string): 'en' | 'hi' | 'hinglish' {
    if (/[\u0900-\u097F]/.test(query)) {
      return 'hi';
    }
    const hinglishMarkers = ['kya', 'hai', 'kitna', 'kaun', 'mera', 'meri', 'batao', 'dikhao', 'kaha', 'kab', 'tarikh', 'khata', 'bima', 'bijli', 'uske', 'baren', 'upload', 'kiya', 'tha'];
    const lower = query.toLowerCase();
    for (const marker of hinglishMarkers) {
      if (new RegExp(`\\b${marker}\\b`, 'i').test(lower)) {
        return 'hinglish';
      }
    }
    return 'en';
  }

  private static isGreeting(lower: string): boolean {
    const clean = lower.trim().replace(/[!?.,]+/g, '');
    const greetingWords = [
      'hi', 'hii', 'hiii', 'hello', 'helo', 'hey', 'heyy', 'namaste', 'namaskar',
      'pranam', 'halo', 'good morning', 'good afternoon', 'good evening',
      'kaise ho', 'kya haal hai', 'what can you do', 'kya kar sakte ho',
      'help', 'madad'
    ];
    return greetingWords.includes(clean) ||
      /^(hi+|hey+|hello+|namaste+)\b/i.test(clean);
  }

  private static generateGreetingResponse(lang: 'en' | 'hi' | 'hinglish', docs: StoredDocument[]): string {
    const count = docs.length;
    if (lang === 'hi' || lang === 'hinglish') {
      return `**Namaste! 🙏 Main aapka SafeDocAI personal document assistant hoon.**

Main aapke stored documents (jaise marksheet, electricity bill, passbook, aadhaar) se verified details khojne me madad karta hoon.

**Aap mujhse pooch sakte hain:**
• 🎓 *"Mera roll number kya hai?"*
• ⚡ *"Electricity bill ka amount aur due date kab hai?"*
• 🏦 *"SBI passbook me account number aur balance kitna hai?"*
• 📄 *"Meri policy ya marksheet summarize karo"*

Aap kis document ke baare mein jaankari chahte hain?`;
    }

    return `**Hello! 👋 I'm your SafeDocAI personal document assistant.**

I help you quickly find verified facts, retrieve exact numbers, and summarize your stored documents (currently **${count} documents** indexed).

**You can ask me:**
• 🎓 *"What is my roll number?"*
• ⚡ *"What is my electricity bill amount and due date?"*
• 🏦 *"What is my bank account number and balance?"*
• 📄 *"Summarize my insurance policy or marksheet"*

How can I help you today?`;
  }

  private static isOutOfScope(lower: string): boolean {
    const outOfScopePhrases = [
      'weather today', 'who is prime minister', 'capital of france', 'write python code',
      'recipe for', 'tell me a joke', 'who created you', 'translate into french', 'solve x^2',
      'cricket score', 'movie review', 'generate image of'
    ];
    return outOfScopePhrases.some(phrase => lower.includes(phrase));
  }

  private static checkDocumentRetrieval(lower: string, docs: StoredDocument[]): StoredDocument | null {
    const isDocRequest = lower.includes('show') || lower.includes('open') || lower.includes('dikhao') || lower.includes('dekho') || lower.includes('view') || lower.includes('display');
    if (!isDocRequest) return null;

    for (const doc of docs) {
      const nameParts = doc.file_name.toLowerCase().replace(/\.[^.]+$/, '').split(/\s+/);
      const categoryLower = doc.category.toLowerCase();

      const matchesName = nameParts.some(part => part.length > 3 && lower.includes(part));
      const matchesCategory = (lower.includes('bill') && categoryLower.includes('bill')) ||
                              (lower.includes('marksheet') && categoryLower.includes('marksheet')) ||
                              (lower.includes('passbook') && categoryLower.includes('passbook')) ||
                              (lower.includes('policy') && categoryLower.includes('policy')) ||
                              (lower.includes('aadhaar') && categoryLower.includes('identity'));

      if (matchesName || matchesCategory) {
        return doc;
      }
    }
    return null;
  }

  private static findExactFieldMatch(
    lower: string,
    docs: StoredDocument[],
    filterDocId?: number
  ): {
    found: boolean;
    isAmbiguous: boolean;
    field?: ExtractedField;
    document?: StoredDocument;
    candidates: { doc: StoredDocument; field: ExtractedField }[];
  } {
    // Score-based matching: longest/most-specific keyword wins (fieldMatcher.ts)
    const fieldMatch = matchFieldGroup(lower);
    const matchedGroup: SynonymGroup | null = fieldMatch?.group || null;

    if (!matchedGroup) {
      return { found: false, isAmbiguous: false, candidates: [] };
    }

    const candidates: { doc: StoredDocument; field: ExtractedField }[] = [];
    const targetDocs = filterDocId ? docs.filter(d => d.id === filterDocId) : docs;

    // A "parent / guardian" question can be answered by father's or mother's name
    const targetFieldNames =
      matchedGroup.canonical === 'parent_name' ? PARENT_CANDIDATE_FIELDS : [matchedGroup.canonical];

    for (const doc of targetDocs) {
      for (const f of doc.fields || []) {
        if (targetFieldNames.includes(f.field_name)) {
          candidates.push({ doc, field: f });
        }
      }
    }

    if (candidates.length === 0) {
      return { found: false, isAmbiguous: false, candidates: [] };
    }

    if (candidates.length === 1) {
      return {
        found: true,
        isAmbiguous: false,
        field: candidates[0].field,
        document: candidates[0].doc,
        candidates
      };
    }

    for (const c of candidates) {
      const baseName = c.doc.file_name.toLowerCase().replace(/\.[^.]+$/, '');
      if (lower.includes(baseName) || (lower.includes('marksheet') && baseName.includes('marksheet')) || (lower.includes('bill') && baseName.includes('bill'))) {
        return {
          found: true,
          isAmbiguous: false,
          field: c.field,
          document: c.doc,
          candidates
        };
      }
    }

    return {
      found: false,
      isAmbiguous: true,
      candidates
    };
  }

  private static formatExactAnswer(field: ExtractedField, doc: StoredDocument, lang: 'en' | 'hi' | 'hinglish'): string {
    if (lang === 'hi' || lang === 'hinglish') {
      return `Aapke "${doc.file_name}" ke anusar, **${field.display_label}** hai:\n\n### **${field.field_value}**\n\n(Evidence: "${field.evidence_text}")`;
    }
    return `According to your **${doc.file_name}**, the **${field.display_label}** is:\n\n### **${field.field_value}**\n\nVerified from Page ${field.page_number || 1} with ${Math.round(field.confidence * 100)}% evidence confidence.`;
  }

  private static generateGroundedSemanticAnswer(
    query: string,
    results: { chunk: { file_name: string; text: string; page_number: number } }[],
    lang: 'en' | 'hi' | 'hinglish'
  ): string {
    const docNames = Array.from(new Set(results.map(r => r.chunk.file_name)));
    const primarySnippet = results[0].chunk.text;

    if (lang === 'hi' || lang === 'hinglish') {
      return `Aapke dastavej **${docNames.join(', ')}** se mili jankari ke anusar:\n\n${primarySnippet}\n\nYeh jankari seedhe aapke verified local text se retrieve ki gayi hai.`;
    }

    return `Based on your verified records in **${docNames.join(', ')}**:\n\n${primarySnippet}\n\nThis response is grounded strictly in the retrieved document text on your local machine.`;
  }
}
