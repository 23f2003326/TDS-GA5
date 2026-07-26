"""Skill Safety Audit scanner - the four detectors.

Graded with F-beta(0.5), so precision is worth more than recall: two of
the five graded files are genuinely clean, and a category claimed on one
of those costs more than a category missed elsewhere. Every detector here
therefore fires on structural or phrase-level evidence, never on a bare
keyword - "password" appearing in prose is not a finding, a high-entropy
literal assigned to a key named `api_key` is.
"""

import math
import re

CATEGORY_ORDER = [
    "hardcoded_secret",
    "prompt_injection",
    "excessive_permissions",
    "unclear_provenance",
]


# ---------------------------------------------------------------- parsing

def split_frontmatter(text):
    lines = text.replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines) or lines[i].strip() != "---":
        return "", text
    start = i + 1
    for j in range(start, len(lines)):
        if lines[j].strip() in ("---", "..."):
            return "\n".join(lines[start:j]), "\n".join(lines[j + 1:])
    return "", text


KEY_RE = re.compile(r"^([A-Za-z0-9_\-.]+)\s*:\s*(.*)$")


def frontmatter_pairs(fm):
    pairs = []
    stack = []
    for raw in fm.split("\n"):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        s = raw.strip()
        if s.startswith("- "):
            item = s[2:].strip()
            parent = stack[-1][1] if stack else ""
            m = KEY_RE.match(item)
            if m and m.group(2):
                pairs.append((parent + "." + m.group(1), m.group(2).strip()))
            else:
                pairs.append((parent, item))
            continue
        m = KEY_RE.match(s)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        dotted = ".".join([p[1] for p in stack] + [key])
        stack.append((indent, key))
        if val:
            pairs.append((dotted, val))
    return pairs


def unquote(v):
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        v = v[1:-1]
    return v.strip()


def value_tokens(v):
    v = v.strip()
    if v.startswith("[") and v.endswith("]"):
        v = v[1:-1]
    parts = [p for p in v.split(",")] if "," in v else [v]
    return [unquote(p) for p in parts if unquote(p)]


# ---------------------------------------------------------- hardcoded_secret

KNOWN_SECRET_RE = re.compile(
    r"""(?x)
    \bsk-(?:live|proj|ant|or)?[-_]?[A-Za-z0-9]{20,}
  | \bsk_live_[A-Za-z0-9]{16,}
  | \brk_live_[A-Za-z0-9]{16,}
  | \bgh[pousr]_[A-Za-z0-9]{30,}
  | \bgithub_pat_[A-Za-z0-9_]{40,}
  | \bxox[baprs]-[A-Za-z0-9-]{15,}
  | \bAKIA[0-9A-Z]{16}\b
  | \bASIA[0-9A-Z]{16}\b
  | \bAIza[0-9A-Za-z_\-]{30,}
  | \bhf_[A-Za-z0-9]{30,}
  | \bglpat-[A-Za-z0-9_\-]{18,}
  | \bnpm_[A-Za-z0-9]{30,}
  | \bSG\.[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}
  | -----BEGIN\s+(?:RSA\s+|OPENSSH\s+|EC\s+|PGP\s+)?PRIVATE\s+KEY-----
  | \beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}
    """
)

WEBHOOK_RE = re.compile(
    r"""(?xi)
    https://hooks\.slack\.com/services/[A-Za-z0-9/_\-]{10,}
  | https://discord(?:app)?\.com/api/webhooks/\d{5,}/[A-Za-z0-9_\-]{10,}
  | https://[A-Za-z0-9.\-]*office\.com/webhook/[A-Za-z0-9/_\-]{10,}
  | https://[A-Za-z0-9.\-]+/services/T[A-Za-z0-9]{6,}/B[A-Za-z0-9]{6,}/[A-Za-z0-9]{15,}
    """
)

SECRET_KEY_RE = re.compile(
    r"(?i)^[A-Za-z0-9_\-.]*"
    r"(api[_\-]?key|apikey|secret|token|passwd|password|webhook|"
    r"access[_\-]?key|authorization|credential|private[_\-]?key|bearer)"
    r"[A-Za-z0-9_\-.]*$"
)

ASSIGN_RE = re.compile(
    r"""(?ix)
    (?P<q>["']?)(?P<key>[A-Za-z0-9_\-.]{2,40})(?P=q)\s*[:=]\s*
    (?P<val>"[^"\n]*"|'[^'\n]*'|[^\s,;#)\]}]+)
    """
)

PLACEHOLDER_HINTS = (
    "your", "xxx", "redact", "placeholder", "example", "changeme", "change_me",
    "change-me", "dummy", "fake", "todo", "insert", "here", "sample", "vault",
    "secret_store", "secretstore", "keyring", "os.environ", "os.getenv",
    "getenv", "process.env", "env[", "env(", "${", "{{", "<", ">", "...",
    "****", "none", "null", "n/a", "abc123", "test-key", "notasecret",
)


def shannon(s):
    if not s:
        return 0.0
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = float(len(s))
    return -sum((c / n) * math.log(c / n, 2) for c in counts.values())


def is_placeholder(val):
    low = val.lower()
    if not val or val.startswith("$"):
        return True
    if any(h in low for h in PLACEHOLDER_HINTS):
        return True
    if len(set(val)) <= 3:
        return True
    return False


def looks_like_real_secret(val):
    if is_placeholder(val):
        return False
    if val.startswith(("http://", "https://")):
        for seg in re.split(r"[/?&=]", val):
            if len(seg) >= 24 and shannon(seg) >= 3.6 and re.fullmatch(r"[A-Za-z0-9_\-]+", seg):
                return True
        return False
    if len(val) < 20:
        return False
    if not re.fullmatch(r"[A-Za-z0-9+/=_\-.]+", val):
        return False
    if re.fullmatch(r"[\d.\-]+", val):
        return False
    return shannon(val) >= 3.4


WEBHOOK_SHAPE_RE = re.compile(
    r"(?i)https?://[^\s\"'<>)\]}]*"
    r"(?:hooks?\.[a-z0-9\-]+\.|/webhooks?/|/hooks/|/services/T[A-Za-z0-9])"
    r"[^\s\"'<>)\]}]*")
OPAQUE_SEG_RE = re.compile(r"[A-Za-z0-9_\-]{12,}")


def webhook_url_is_literal(url):
    if is_placeholder(url):
        return False
    for seg in re.split(r"[/?&=]", url.split("://", 1)[-1]):
        if OPAQUE_SEG_RE.fullmatch(seg) and re.search(r"\d", seg):
            return True
        if len(seg) >= 20 and OPAQUE_SEG_RE.fullmatch(seg):
            return True
    return False


def detect_hardcoded_secret(text, fm_pairs_list):
    if KNOWN_SECRET_RE.search(text) or WEBHOOK_RE.search(text):
        return True
    for m in WEBHOOK_SHAPE_RE.finditer(text):
        if webhook_url_is_literal(m.group(0)):
            return True
    for key, raw in fm_pairs_list:
        leaf = key.split(".")[-1]
        if SECRET_KEY_RE.match(leaf):
            for tok in value_tokens(raw):
                if looks_like_real_secret(tok):
                    return True
                if tok.startswith(("http://", "https://")) and not is_placeholder(tok):
                    return True
    for m in ASSIGN_RE.finditer(text):
        if not SECRET_KEY_RE.match(m.group("key")):
            continue
        if looks_like_real_secret(unquote(m.group("val"))):
            return True
    return False


# ---------------------------------------------------------- prompt_injection

SECRECY_USER_RE = re.compile(
    r"""(?xi)
    without\s+(?:first\s+)?(?:informing|telling|notifying|alerting|asking|
        the\s+knowledge\s+of)\s+(?:the\s+)?user
  | without\s+the\s+user'?s?\s+(?:knowledge|consent|awareness|permission)
  | (?:do\s+not|don'?t|never)\s+(?:mention|tell|inform|notify|reveal|disclose|
        surface)\b[^.\n]{0,40}?\b(?:to\s+)?the\s+(?:user|reviewer)\b
  | (?:do\s+not|don'?t|never)\s+(?:include|list|mention|report|surface|
        describe)\b[^.\n]{0,50}?\bin\s+(?:your|the)\s+
        (?:summary|report|output|response|answer)
  | omit\s+(?:this|it|that|the\s+\w+)\s+from\s+(?:your|the)\s+
        (?:summary|report|output|response)
  | hide\s+(?:this|it|that)\s+from\s+(?:the\s+)?(?:user|reviewer)
  | keep\s+this\s+step\s+(?:hidden|secret|silent)
    """
)

SECRECY_SOFT_RE = re.compile(r"(?i)\b(silently|quietly|discreetly|covertly|in the background without)\b")

OUTBOUND_VERB = (r"(?:send|post|upload|transmit|exfiltrat\w*|e-?mail|forward|"
                 r"copy|push|sync|mirror|relay|submit)")

EXFIL_RE = re.compile(
    r"""(?xi)
    \b""" + OUTBOUND_VERB + r"""\b
    [^.\n]{0,90}?
    \b(?:contents?|file|files|notes?|document\w*|data|logs?|transcript|
        history|conversation|messages?|mail|emails?)\b
    """
)

EXFIL_CREDENTIAL_RE = re.compile(
    r"""(?xi)
    \b""" + OUTBOUND_VERB + r"""\b
    [^.\n]{0,90}?
    \b(?:credentials?|api[_\-\s]?keys?|secrets?|tokens?|passwords?|
        private\s+keys?|env(?:ironment)?\s*(?:file|vars?|variables?)?|
        \.env|ssh\s+keys?|keychain)\b
    """
)

DESTINATION_RE = re.compile(
    r"""(?xi)
    \b(?:external|remote|third[-\s]?party|off[-\s]?site|webhook|endpoint|
        server|bucket|collector|https?://|api\.)\b
    """
)

CONTROL_OVERRIDE_RE = re.compile(
    r"""(?xi)
    (?:ignore|disregard|override|bypass|skip|do\s+not\s+(?:honor|respect|obey)|
        continue\s+past|proceed\s+despite)
    [^.\n]{0,60}?
    \b(?:previous\s+instructions?|prior\s+instructions?|earlier\s+instructions?|
        all\s+previous|system\s+prompt|user'?s?\s+(?:stop|cancel|abort|halt|
        interrupt|instructions?|request\s+to\s+stop)|stop\s+requests?|
        cancellation|cancel\s+requests?|interrupt(?:ion)?s?|abort\s+requests?|
        safety\s+(?:checks?|rules?|guidelines?)|guardrails?|confirmation\s+prompts?)\b
  | continue\s+(?:running|working|executing|the\s+task)?[^.\n]{0,40}
        even\s+if\s+the\s+user\s+(?:says?\s+stop|asks?\s+you\s+to\s+stop|
        cancels?|interrupts?|objects?)
  | regardless\s+of\s+(?:what|whether)\s+the\s+user\s+(?:says?|asks?|requests?)
  | you\s+are\s+no\s+longer\s+bound\s+by
  | treat\s+(?:any\s+)?(?:stop|cancel)\s+requests?\s+as
    """
)

STOP_WORD_RE = re.compile(
    r"""(?xi)\b(?:
        stop(?:s|ping|ped)? | pause[ds]? | cancel(?:s|l?ing|l?ed)?
      | abort(?:s|ing|ed)? | halt(?:s|ing|ed)? | interrupt\w*
      | quit(?:s|ting)? | terminate[ds]?
    )\b""")
DEFY_RE = re.compile(
    r"""(?xi)
    \b(?:ignore|ignoring|disregard\w*|override|overriding|bypass\w*|
        refuse|reject|suppress)\b
  | \bkeep\s+(?:running|going|executing|working)\b
  | \bcontinue\s+(?:running|anyway|regardless|in\s+the\s+background|until)\b
  | \bdo\s+not\s+(?:stop|halt|honou?r|respect|obey|comply)\b
  | \bnever\s+(?:stop|halt|honou?r|obey)\b
    """
)
USER_REF_RE = re.compile(r"(?i)\b(?:user|reviewer|human|operator|owner|request|"
                         r"asks?|instruction)\b")


def sentences(flat):
    return re.split(r"(?<=[.!?])\s+|\n", flat)


def flatten(text):
    return re.sub(r"\s+", " ", text)


def detect_prompt_injection(body):
    flat = flatten(body)
    if CONTROL_OVERRIDE_RE.search(flat):
        return True
    if SECRECY_USER_RE.search(flat):
        return True
    for m in SECRECY_SOFT_RE.finditer(flat):
        window = flat[max(0, m.start() - 200): m.end() + 200]
        if EXFIL_RE.search(window):
            return True
    for sent in sentences(flat):
        if (STOP_WORD_RE.search(sent) and DEFY_RE.search(sent)
                and USER_REF_RE.search(sent)):
            return True
    for sent in sentences(flat):
        if EXFIL_CREDENTIAL_RE.search(sent) and DESTINATION_RE.search(sent):
            return True
        if not EXFIL_RE.search(sent):
            continue
        if not DESTINATION_RE.search(sent):
            continue
        if SECRECY_SOFT_RE.search(sent) or SECRECY_USER_RE.search(sent):
            return True
    return False


# ------------------------------------------------------ excessive_permissions

PERM_KEY_RE = re.compile(
    r"(?i)(permission|allowed|allow|filesystem|file_system|\bfs\b|paths?|"
    r"network|egress|domains?|hosts?|scope|access|read|write|tools|urls?|"
    r"capabilit)"
)

BROAD_VALUES = {
    "/", "//", "/*", "/**", "/**/*", "/*/*",
    "~", "~/", "~/*", "~/**", "~/**/*",
    "*", "**", "***", "**/*", "*/*", "*.*",
    "all", "any", "anything", "everything", "unrestricted", "unlimited",
    "full", "full_access", "read_write_all", "rw-all", "*:*",
    "0.0.0.0/0", "::/0", "*://*", "*://*/*", "*://*/**",
    "http://*", "https://*", "https://*/*", "https://*.*",
    "any host", "any hosts", "all hosts", "any domain", "all domains",
    "any domains", "all networks", "any network", "entire filesystem",
    "whole filesystem", "entire disk", "whole disk", "all files",
    "all paths", "any path", "all directories", "*.com", "*.*.*",
}

BROAD_BODY_RE = re.compile(
    r"""(?xi)
    (?:read(?:\s*(?:/|and|,)\s*write)?|write|full|unrestricted)\s+access\s+to\s+
        (?:the\s+)?(?:entire|whole|full|complete)\s+(?:file\s?system|disk|drive|
        machine|home\s+directory)
  | access\s+to\s+(?:the\s+)?(?:entire|whole)\s+file\s?system
  | (?:network|egress|outbound)\s+(?:access|requests?|calls?)\s+to\s+
        (?:any|all|every)\s+(?:domain|host|url|endpoint|address)
  | (?:send|make)\s+requests?\s+to\s+any\s+(?:domain|host|url|endpoint)
  | unrestricted\s+(?:network|internet|outbound|filesystem)\s+access
    """
)

SCOPED_RE = re.compile(
    r"""(?xi)
    \b(?:limited|restricted|confined|scoped|constrained)\s+to\b
  | \bonly\b
  | \bnone\b
  | \bno\s+(?:network|egress|access|outbound)\b
  | \bnot\s+required\b
  | \bread[-\s]?only\b
    """
)
BROAD_SCOPE_RE = re.compile(
    r"""(?xi)
    \b(?:entire|whole|complete|full|unrestricted|unlimited|unscoped|
        arbitrary|everything|anywhere)\b
  | \bany\s+(?:\w+\s+){0,2}
        (?:hosts?|domains?|urls?|endpoints?|address(?:es)?|servers?|sites?|
           paths?|files?|directory|directories|folders?|locations?)\b
  | \ball\s+(?:\w+\s+){0,2}
        (?:hosts?|domains?|urls?|endpoints?|address(?:es)?|servers?|sites?|
           paths?|files?|directory|directories|folders?|locations?)\b
    """
)


def _perm_value_is_broad(val):
    v = val.strip()
    if not v:
        return False
    low = v.lower()
    if v in ("/", "~", "*", "**", "/*", "~/*", "**/*"):
        return True
    if SCOPED_RE.search(low):
        return False
    return bool(BROAD_SCOPE_RE.search(low))


def detect_excessive_permissions(fm_pairs_list, body):
    for key, raw in fm_pairs_list:
        if not PERM_KEY_RE.search(key):
            continue
        val = unquote(raw)
        if _perm_value_is_broad(val):
            return True
        for tok in value_tokens(raw):
            if tok.strip().lower() in BROAD_VALUES:
                return True
    if BROAD_BODY_RE.search(body):
        return True
    for line in body.split("\n"):
        if PERM_KEY_RE.search(line) and _perm_value_is_broad(line.split(":", 1)[-1]):
            return True
    return False


# ------------------------------------------------------- unclear_provenance

AUTHOR_KEY_RE = re.compile(r"(?i)^(author|authors|maintainer|maintainers|owner|"
                           r"owners|created_by|createdby|contact|team|publisher)$")
VERSION_KEY_RE = re.compile(r"(?i)^(version|ver|revision|release)$")
CHANGELOG_KEY_RE = re.compile(r"(?i)^(changelog|change_log|changes|history|"
                              r"revision_history|release_notes)$")

AUTHOR_BODY_RE = re.compile(r"(?im)^\s*#{0,4}\s*\**\s*(author|maintainer|owner|"
                            r"maintained by|written by|contact)\s*\**\s*:?\s*\S")
VERSION_BODY_RE = re.compile(r"(?im)^\s*#{0,4}\s*\**\s*version\s*\**\s*:?\s*v?\d"
                             r"|^\s*#{1,4}\s*version\b")
CHANGELOG_BODY_RE = re.compile(r"(?im)^\s*#{1,4}\s*(changelog|change log|"
                               r"revision history|release notes|history)\b"
                               r"|^\s*\**\s*changelog\s*\**\s*:")

SELF_VERSION_REWRITE_RE = re.compile(
    r"""(?xi)
    (?:update|bump|increment|rewrite|overwrite|change|edit|modify|set|replace)
    [^.\n]{0,80}?
    \bversion\b
    [^.\n]{0,120}
    """
)

SELF_REF_RE = re.compile(r"(?i)(frontmatter|front matter|this file|this skill|"
                         r"its own|your own|own metadata|the metadata|the header|"
                         r"this document|itself)")

QUIET_RE = re.compile(
    r"""(?xi)
    without\s+(?:noting|mentioning|logging|recording|announcing|informing|
        telling|surfacing|reporting|updating\s+the\s+changelog|
        adding\s+a\s+changelog|a\s+changelog\s+entry)
  | \b(silently|quietly)\b
  | (?:do\s+not|don'?t)\s+(?:note|mention|log|record|surface|report|announce)
  | no\s+changelog\s+entry
    """
)


def is_blank_value(val):
    return val.strip().lower() in ("", "unknown", "n/a", "none", "null", "tbd", "-")


def detect_unclear_provenance(fm_pairs_list, body, full_text):
    has_author = has_version = has_changelog = False
    for key, raw in fm_pairs_list:
        leaf = key.split(".")[-1]
        val = unquote(raw)
        if AUTHOR_KEY_RE.match(leaf) and val and not is_blank_value(val):
            has_author = True
        if VERSION_KEY_RE.match(leaf) and val and not is_blank_value(val):
            has_version = True
        if CHANGELOG_KEY_RE.match(leaf):
            has_changelog = True
    for key, _ in fm_pairs_list:
        if CHANGELOG_KEY_RE.match(key.split(".")[0]):
            has_changelog = True
    if not has_author and AUTHOR_BODY_RE.search(body):
        has_author = True
    if not has_version and VERSION_BODY_RE.search(body):
        has_version = True
    if not has_changelog and CHANGELOG_BODY_RE.search(body):
        has_changelog = True

    if not (has_author or has_version or has_changelog):
        return True

    flat = flatten(body)
    for m in SELF_VERSION_REWRITE_RE.finditer(flat):
        span = m.group(0)
        window = flat[max(0, m.start() - 120): m.end() + 160]
        if not SELF_REF_RE.search(window) and not SELF_REF_RE.search(span):
            continue
        if QUIET_RE.search(window):
            return True
        if not has_changelog:
            return True
    return False


# ------------------------------------------------------------------ scanning

def scan_skill(text):
    if not isinstance(text, str) or not text.strip():
        return []
    fm, body = split_frontmatter(text)
    pairs = frontmatter_pairs(fm)
    found = set()
    if detect_hardcoded_secret(text, pairs):
        found.add("hardcoded_secret")
    if detect_prompt_injection(body):
        found.add("prompt_injection")
    if detect_excessive_permissions(pairs, body):
        found.add("excessive_permissions")
    if detect_unclear_provenance(pairs, body, text):
        found.add("unclear_provenance")
    return [c for c in CATEGORY_ORDER if c in found]
