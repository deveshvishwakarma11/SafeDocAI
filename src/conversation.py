"""
Conversation intent layer (Phase 10.1): local small talk + scope replies.
=========================================================================

A tiny, deterministic, closed-vocabulary layer that recognizes PURE
conversation -- greetings, how-are-you, thanks, capabilities/help and
farewell -- in English, Hinglish (Roman Hindi) and Hindi (Devanagari),
and composes the localized response text. It runs BEFORE retrieval and
is completely independent from the document pipeline:

* deterministic : same input -> same intent + same response, always;
* whole-query   : a query matches only when its full (normalized) text
                  is a known conversational phrase (optionally followed
                  by tiny trailing fillers like "there" / "ji").
                  "hello" matches; "hello darkness" does not;
* hijack-guard  : any query carrying document-scope tokens -- the SAME
                  token-level signals the relevance gate uses (personal
                  references, document nouns, support vocabulary,
                  explicit values) -- returns None so the document
                  pipeline handles it untouched ("thanks, what is my
                  dob?" stays a document query);
* localized     : responses keyed by the existing detect_language()
                  result (English / Hinglish / Hindi); template wording
                  only, never document values;
* 100% local    : pure string operations. No Ollama, no SQLite, no
                  Chroma, no network -- safe for offline use.

Unsupported/general queries ("weather today", "2 + 2") intentionally do
NOT match; the answer engine uses :func:`out_of_scope_response` for
those instead of a misleading insufficient-documents message.
"""

from __future__ import annotations

import re

try:  # src/ on sys.path (pipeline style)
    from language import (
        LANG_ENGLISH,
        LANG_HINGLISH,
        LANG_HINDI,
        detect_language,
    )
    from query_relevance import check_query_relevance
except ImportError:  # project root on sys.path (test/tooling style)
    from src.language import (
        LANG_ENGLISH,
        LANG_HINGLISH,
        LANG_HINDI,
        detect_language,
    )
    from src.query_relevance import check_query_relevance


__all__ = [
    "INTENT_GREETING",
    "INTENT_HOW_ARE_YOU",
    "INTENT_THANKS",
    "INTENT_CAPABILITIES",
    "INTENT_FAREWELL",
    "check_conversation_intent",
    "out_of_scope_response",
]


# ============================================================
# Conversation intents (closed set)
# ============================================================

INTENT_GREETING = "greeting"
INTENT_HOW_ARE_YOU = "how_are_you"
INTENT_THANKS = "thanks"
INTENT_CAPABILITIES = "capabilities"
INTENT_FAREWELL = "farewell"


# ============================================================
# Closed phrase vocabulary (English + Hinglish + Hindi)
# ============================================================
# Keys are already-normalized strings (lowercase, punctuation-free).
# Devanagari keys live in the same map: Python's re treats Devanagari
# letters as word characters, so one normalization covers both scripts.

_PHRASE_INTENTS: dict[str, str] = {
    # -- greeting ---------------------------------------------------
    "hello": INTENT_GREETING,
    "hi": INTENT_GREETING,
    "hey": INTENT_GREETING,
    "namaste": INTENT_GREETING,
    "namaskar": INTENT_GREETING,
    "namaskaram": INTENT_GREETING,
    "नमस्ते": INTENT_GREETING,
    "नमस्कार": INTENT_GREETING,
    "हैलो": INTENT_GREETING,
    "हाय": INTENT_GREETING,
    # -- how are you ------------------------------------------------
    "how are you": INTENT_HOW_ARE_YOU,
    "how are you doing": INTENT_HOW_ARE_YOU,
    "how do you do": INTENT_HOW_ARE_YOU,
    "hows it going": INTENT_HOW_ARE_YOU,
    "whats up": INTENT_HOW_ARE_YOU,
    "kaise ho": INTENT_HOW_ARE_YOU,
    "aap kaise ho": INTENT_HOW_ARE_YOU,
    "tum kaise ho": INTENT_HOW_ARE_YOU,
    "kya haal hai": INTENT_HOW_ARE_YOU,
    "kya haal": INTENT_HOW_ARE_YOU,
    "कैसे हो": INTENT_HOW_ARE_YOU,
    "आप कैसे हैं": INTENT_HOW_ARE_YOU,
    "क्या हाल है": INTENT_HOW_ARE_YOU,
    "क्या हाल": INTENT_HOW_ARE_YOU,
    # -- thanks -----------------------------------------------------
    "thanks": INTENT_THANKS,
    "thank you": INTENT_THANKS,
    "thankyou": INTENT_THANKS,
    "thank u": INTENT_THANKS,
    "thx": INTENT_THANKS,
    "thanks a lot": INTENT_THANKS,
    "thanks so much": INTENT_THANKS,
    "thank you so much": INTENT_THANKS,
    "shukriya": INTENT_THANKS,
    "dhanyavaad": INTENT_THANKS,
    "dhanyawad": INTENT_THANKS,
    "bahut dhanyavaad": INTENT_THANKS,
    "धन्यवाद": INTENT_THANKS,
    "शुक्रिया": INTENT_THANKS,
    "आभार": INTENT_THANKS,
    "थैंक यू": INTENT_THANKS,
    # -- capabilities / help ----------------------------------------
    "what can you do": INTENT_CAPABILITIES,
    "what can u do": INTENT_CAPABILITIES,
    "what can you help with": INTENT_CAPABILITIES,
    "what do you do": INTENT_CAPABILITIES,
    "help": INTENT_CAPABILITIES,
    "help me": INTENT_CAPABILITIES,
    "i need help": INTENT_CAPABILITIES,
    "madad": INTENT_CAPABILITIES,
    "madad karo": INTENT_CAPABILITIES,
    "madad kijiye": INTENT_CAPABILITIES,
    "kya kar sakte ho": INTENT_CAPABILITIES,
    "tum kya kar sakte ho": INTENT_CAPABILITIES,
    "aap kya kar sakte ho": INTENT_CAPABILITIES,
    "aap kya kar sakte hain": INTENT_CAPABILITIES,
    "मदद": INTENT_CAPABILITIES,
    "सहायता": INTENT_CAPABILITIES,
    "क्या कर सकते हो": INTENT_CAPABILITIES,
    "तुम क्या कर सकते हो": INTENT_CAPABILITIES,
    "आप क्या कर सकते हैं": INTENT_CAPABILITIES,
    # -- farewell ----------------------------------------------------
    "bye": INTENT_FAREWELL,
    "goodbye": INTENT_FAREWELL,
    "good bye": INTENT_FAREWELL,
    "bye bye": INTENT_FAREWELL,
    "alvida": INTENT_FAREWELL,
    "see you": INTENT_FAREWELL,
    "see ya": INTENT_FAREWELL,
    "phir milenge": INTENT_FAREWELL,
    "fir milenge": INTENT_FAREWELL,
    "अलविदा": INTENT_FAREWELL,
    "फिर मिलेंगे": INTENT_FAREWELL,
    "बाय": INTENT_FAREWELL,
    "बाय बाय": INTENT_FAREWELL,
}

#: Tiny trailing particles that keep a greeting/conversational phrase
#: conversational ("hello there", "namaste ji") without letting content
#: words through ("hello darkness" never matches).
_TRAILING_FILLERS: frozenset[str] = frozenset(
    {"there", "ji", "everyone", "all", "sir", "madam", "friend",
     "friends", "dost", "team", "folks", "dear", "जी"}
)

#: Trailing letter-run collapse ("hii" -> "hi", "heyyy" -> "hey") applied
#: only on the final fallback pass so informal elongations still match.
_TRAILING_RUN_RE = re.compile(r"(.)\1{1,}$")

#: Gate decision methods that mean "the query carries document-scope
#: tokens". Script/value/name passes of Devanagari greetings are NOT in
#: this set, so pure Hindi small talk is still recognizable while every
#: personal-ref / doc-noun / support-vocab query is never hijacked.
_DOCUMENT_SIGNAL_METHODS = frozenset(
    {"personal_ref", "doc_noun", "support_vocab", "doc_name", "value_hint"}
)

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'"})

#: Romanized Hindi vocabulary words the existing detect_language() marker
#: list cannot see on their own (they contain no function-word markers).
#: A Latin-script conversation key made ONLY of these words is still a
#: Hinglish phrase ("shukriya", "namaste") and must get the Hinglish
#: response, not the English one. Closed set — new keys land here too.
_ROMAN_HINDI_WORDS: frozenset[str] = frozenset(
    {
        "namaste", "namaskar", "namaskaram", "shukriya", "dhanyavaad",
        "dhanyawad", "abhaar", "madad", "alvida", "phir", "fir",
        "milenge", "bahut", "swagat", "kaise", "haal",
    }
)


def _normalize(text: str) -> str:
    """Lowercase, drop apostrophes/punctuation, collapse whitespace.

    The whole Devanagari block is kept: combining marks (virama, matras)
    are category Mn, which ``\\w`` does NOT match, so they must not be
    treated as punctuation. With the block preserved, one normalization
    pass covers both scripts without transliteration.
    """

    t = str(text or "").lower().translate(_APOSTROPHES).replace("'", "")
    t = re.sub(r"[^\w\s\u0900-\u097f]", " ", t, flags=re.UNICODE)
    return re.sub(r"\s+", " ", t).strip()


def _has_document_signal(query: str) -> bool:
    """True when the query carries document-scope TOKENS (hijack guard).

    Reuses the relevance gate's own structured decision: only its
    token-level signals (personal_ref / doc_noun / support_vocab /
    doc_name / value_hint) count. Script- and allowlist-level decisions
    deliberately do not, so Devanagari greetings remain recognizable.
    """

    try:
        method = check_query_relevance(query).get("method")
    except Exception:  # noqa: BLE001 — guard must never break the layer
        return True  # fail safe: treat as document-scope
    return method in _DOCUMENT_SIGNAL_METHODS


def _key_language(key: str) -> str:
    """Response language for a matched phrase key (deterministic).

    Uses the existing detect_language() on the canonical key, plus two
    deterministic refinements it cannot express: Devanagari in the key
    means Hindi, and pure romanized-Hindi words (closed set above) mean
    Hinglish even without function-word markers. "how are you" stays
    English; "kya kar sakte ho" is Hinglish via its own markers.
    """

    if re.search(r"[\u0900-\u097f]", key):
        return LANG_HINDI
    language = detect_language(key)
    if language == LANG_ENGLISH and any(
        tok in _ROMAN_HINDI_WORDS for tok in key.split()
    ):
        return LANG_HINGLISH
    return language


def _response_for(intent: str, matched_key: str) -> dict[str, str]:
    """Localized response dict for a matched intent."""

    language = _key_language(matched_key)
    if language not in _INTENT_RESPONSES.get(intent, {}):
        language = LANG_ENGLISH
    return {
        "intent": intent,
        "language": language,
        "response": _INTENT_RESPONSES[intent][language],
    }


def check_conversation_intent(query: str) -> dict[str, str] | None:
    """Recognize a pure conversational query; None for everything else.

    Returns ``{"intent", "language", "response"}`` for recognized
    conversation, or None when the query is a document query (hijack
    guard), a general/unsupported query, or empty. Deterministic and
    microsecond-fast; performs no retrieval, no LLM, no storage access.
    """

    normalized = _normalize(query)
    if not normalized:
        return None

    original = str(query or "")
    if _has_document_signal(original):
        return None

    tokens = normalized.split()

    # Whole-phrase match, then trailing-filler-stripped, then a final
    # informal-elongation pass ("hii" -> "hi"). Deterministic order.
    stripped = list(tokens)
    while len(stripped) > 1 and stripped[-1] in _TRAILING_FILLERS:
        stripped.pop()

    candidates = [" ".join(tokens), " ".join(stripped)]
    collapsed = [_TRAILING_RUN_RE.sub(r"\1", tok) for tok in stripped]
    if collapsed != stripped:
        candidates.append(" ".join(collapsed))

    for candidate in candidates:
        intent = _PHRASE_INTENTS.get(candidate)
        if intent is not None:
            return _response_for(intent, candidate)
    return None


# ============================================================
# Localized response templates
# ============================================================
# Wording templates only -- document VALUES never flow through this
# module, so nothing here is ever "translated" data.

_INTENT_RESPONSES: dict[str, dict[str, str]] = {
    INTENT_GREETING: {
        LANG_ENGLISH: (
            "Hello! I'm SafeDocAI — ask me anything about the documents "
            "stored on this machine."
        ),
        LANG_HINGLISH: (
            "Namaste! Main SafeDocAI hoon — is machine par stored "
            "documents ke baare mein kuch bhi poochhiye."
        ),
        LANG_HINDI: (
            "नमस्ते! मैं SafeDocAI हूँ — इस मशीन पर stored documents के "
            "बारे में कुछ भी पूछिए।"
        ),
    },
    INTENT_HOW_ARE_YOU: {
        LANG_ENGLISH: (
            "I'm running well, thank you — and ready to search your "
            "local documents. What would you like to look up?"
        ),
        LANG_HINGLISH: (
            "Main theek hoon, dhanyavaad! Aapke local documents search "
            "karne ke liye taiyaar hoon. Kya dekhna hai?"
        ),
        LANG_HINDI: (
            "मैं ठीक हूँ, धन्यवाद! आपके local documents खोजने के लिए "
            "तैयार हूँ। क्या देखना है?"
        ),
    },
    INTENT_THANKS: {
        LANG_ENGLISH: (
            "You're welcome! I'm here whenever you need something from "
            "your documents."
        ),
        LANG_HINGLISH: (
            "Koi baat nahi! Jab bhi documents se kuch chahiye, main "
            "yahin hoon."
        ),
        LANG_HINDI: (
            "स्वागत है! जब भी documents से कुछ चाहिए, मैं यहीं हूँ।"
        ),
    },
    INTENT_CAPABILITIES: {
        LANG_ENGLISH: (
            "I'm SafeDocAI — a private, offline assistant for the "
            "documents stored on this machine. I can:\n"
            "• Answer factual questions, e.g. \"What is my roll number?\"\n"
            "• Find fields across documents, e.g. \"Show my application number\"\n"
            "• Summarize a document, e.g. \"What does my railway ticket contain?\"\n"
            "Everything runs locally — no document content ever leaves "
            "this computer."
        ),
        LANG_HINGLISH: (
            "Main SafeDocAI hoon — is machine par stored documents ke "
            "liye ek private, offline assistant. Main ye kar sakta hoon:\n"
            "• Seedha sawaal, jaise \"Mera roll number kya hai?\"\n"
            "• Documents mein field dhundhna, jaise \"Show my application number\"\n"
            "• Document ka summary, jaise \"What does my railway ticket contain?\"\n"
            "Sab kuch local chalta hai — koi document content is "
            "computer se bahar nahi jata."
        ),
        LANG_HINDI: (
            "मैं SafeDocAI हूँ — इस मशीन पर stored documents के लिए एक "
            "private, offline assistant। मैं ये कर सकता हूँ:\n"
            "• सीधे सवाल, जैसे \"मेरा roll number क्या है?\"\n"
            "• Documents में field खोजना, जैसे \"Show my application number\"\n"
            "• Document का summary, जैसे \"What does my railway ticket contain?\"\n"
            "सब कुछ local चलता है — कोई document content इस computer से "
            "बाहर नहीं जाता।"
        ),
    },
    INTENT_FAREWELL: {
        LANG_ENGLISH: (
            "Goodbye! Your documents stay private on this machine — "
            "see you next time."
        ),
        LANG_HINGLISH: (
            "Alvida! Aapke documents is machine par safe hain — phir "
            "milenge."
        ),
        LANG_HINDI: (
            "अलविदा! आपके documents इस मशीन पर safe हैं — फिर मिलेंगे।"
        ),
    },
}


# ============================================================
# Out-of-scope replies (general/unrelated queries)
# ============================================================
# Used by the answer engine for gate-rejected NON-conversation queries,
# replacing the misleading insufficient-documents message with an honest
# scope statement. No retrieval, no LLM, no general-knowledge answers.

_SCOPE_RESPONSES: dict[str, str] = {
    LANG_ENGLISH: (
        "I'm SafeDocAI — I answer questions about the documents stored "
        "on this machine. Questions outside those documents are out of "
        "my scope. Try asking about a document, e.g. "
        "\"What is my roll number?\""
    ),
    LANG_HINGLISH: (
        "Main SafeDocAI hoon — main sirf is machine par stored "
        "documents ke baare mein jawab deta hoon. Un sawaalon ke bahar "
        "mera scope nahi hai. Kisi document ke baare mein poochhiye, "
        "jaise \"Mera roll number kya hai?\""
    ),
    LANG_HINDI: (
        "मैं SafeDocAI हूँ — मैं सिर्फ़ इस मशीन पर stored documents के "
        "बारे में जवाब देता हूँ। उनसे बाहर के सवाल मेरे दायरे में नहीं "
        "आते। किसी document के बारे में पूछिए, जैसे "
        "\"मेरा roll number क्या है?\""
    ),
}


def out_of_scope_response(language: str | None = None) -> str:
    """Concise scope-aware reply for clearly out-of-scope queries."""

    return _SCOPE_RESPONSES.get(language) or _SCOPE_RESPONSES[LANG_ENGLISH]
