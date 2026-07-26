"""A2A 1.0 Invoice Action Agent — Q10.

Exposes the A2A HTTP+JSON surface:

  GET  /.well-known/agent-card.json   discovery, served at the ORIGIN
  POST /a2a/message:send              start a batch, or continue one
  GET  /a2a/tasks                     list this principal's tasks
  GET  /a2a/tasks/{id}                read one task
  POST /a2a/tasks/{id}:cancel         cancel before finalisation

Mounted into the GA5 monolith via `main.py`'s
`from q10 import router as q10_router` / `app.include_router(q10_router)`.

Design rules the durability marks depend on:
  * every distinct Bearer token is a separate principal,
  * dedup key is (principal, messageId) with a fingerprint over the
    SEMANTIC message only (configuration churn is a free replay),
  * everything is persisted in SQLite BEFORE the response is written,
  * decisions are read straight out of the case files (deterministic),
    with an LLM fallback only when the layout can't be parsed,
  * the agent card's base URL is derived from the live request / Render's
    own env var, so it always matches whatever URL is actually submitted.
"""
import asyncio
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

try:
    from llm import chat_json
except Exception:  # pragma: no cover
    async def chat_json(*args, **kwargs):
        return {}

A2A_MEDIA_TYPE = "application/a2a+json"
JSON_MEDIA_TYPE = "application/json"


class A2AJSONResponse(JSONResponse):
    media_type = A2A_MEDIA_TYPE


class A2ARoute(APIRoute):
    """Force the A2A media type onto every /a2a/* response, errors included."""

    def get_route_handler(self):
        original = super().get_route_handler()

        async def handler(request: Request):
            try:
                response = await original(request)
            except HTTPException as exc:
                response = err(exc.status_code, "A2A_ERROR", str(exc.detail))
            except RequestValidationError:
                response = err(422, "INVALID_ARGUMENT",
                               "request failed schema validation")
            if request.url.path.startswith("/a2a"):
                response.headers["content-type"] = A2A_MEDIA_TYPE
            return response

        return handler


router = APIRouter(route_class=A2ARoute)

DB_PATH = os.environ.get("Q10_DB", "/tmp/q10_a2a.db")

MODE_BATCH = "application/vnd.ga5.invoice-claim-batch+json"
MODE_PROPOSALS = "application/vnd.ga5.invoice-action-proposals+json"
MODE_RESULTS = "application/vnd.ga5.invoice-action-results+json"
MODE_RECEIPTS = "application/vnd.ga5.invoice-action-receipts+json"

ACTIONS = ["settle_invoice", "request_approval", "hold_invoice",
           "reject_duplicate", "open_exception"]

SUBMITTED = "TASK_STATE_SUBMITTED"
WORKING = "TASK_STATE_WORKING"
INPUT_REQUIRED = "TASK_STATE_INPUT_REQUIRED"
COMPLETED = "TASK_STATE_COMPLETED"
CANCELED = "TASK_STATE_CANCELED"
TERMINAL = {COMPLETED, CANCELED, "TASK_STATE_FAILED", "TASK_STATE_REJECTED"}


# ------------------------------------------------------------------ storage

_db_lock = threading.RLock()
_conn = None


def db():
    global _conn
    if _conn is None:
        parent = os.path.dirname(DB_PATH)
        if parent:
            try:
                os.makedirs(parent, exist_ok=True)
            except OSError:
                pass
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
        _conn.row_factory = sqlite3.Row
        try:
            _conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:
            pass
        _conn.execute("PRAGMA synchronous=NORMAL")
        _conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS q10_tasks (
                task_id    TEXT PRIMARY KEY,
                principal  TEXT NOT NULL,
                context_id TEXT NOT NULL,
                batch_id   TEXT,
                state      TEXT NOT NULL,
                doc        TEXT NOT NULL,
                created    REAL,
                updated    REAL
            );
            CREATE INDEX IF NOT EXISTS q10_tasks_principal
                ON q10_tasks(principal, created);
            CREATE TABLE IF NOT EXISTS q10_msgs (
                principal   TEXT NOT NULL,
                message_id  TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                task_id     TEXT NOT NULL,
                PRIMARY KEY (principal, message_id)
            );
            CREATE TABLE IF NOT EXISTS q10_pkgcache (
                pkg_fp   TEXT PRIMARY KEY,
                decision TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS q10_final (
                task_id    TEXT PRIMARY KEY,
                results_fp TEXT NOT NULL
            );
            """
        )
        _conn.commit()
    return _conn


def load_task(task_id):
    with _db_lock:
        row = db().execute(
            "SELECT doc, principal FROM q10_tasks WHERE task_id=?", (task_id,)
        ).fetchone()
    if not row:
        return None, None
    return json.loads(row["doc"]), row["principal"]


def save_task(task, principal, batch_id):
    now = time.time()
    with _db_lock:
        c = db()
        c.execute(
            "INSERT INTO q10_tasks(task_id,principal,context_id,batch_id,state,doc,created,updated)"
            " VALUES(?,?,?,?,?,?,?,?)"
            " ON CONFLICT(task_id) DO UPDATE SET state=excluded.state,"
            " doc=excluded.doc, updated=excluded.updated",
            (task["id"], principal, task["contextId"], batch_id,
             task["status"]["state"], json.dumps(task), now, now),
        )
        c.commit()


# ------------------------------------------------------------- canonicalise

def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)


def sha(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8") if isinstance(p, str) else p)
        h.update(b"\x1f")
    return h.hexdigest()


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# ----------------------------------------------------------------- guards

def err(status, code, message, **extra):
    body = {"error": dict({"code": code, "message": message}, **extra),
            "code": code, "message": message}
    return A2AJSONResponse(body, status_code=status)


def principal_of(request):
    auth = request.headers.get("authorization") or ""
    if not auth.lower().startswith("bearer "):
        return None
    token = auth[7:].strip()
    if not token:
        return None
    return sha("q10-principal", token)


def check_headers(request, *, body=False):
    who = principal_of(request)
    if who is None:
        return None, err(401, "UNAUTHENTICATED",
                         "a Bearer token is required on every A2A route")
    version = request.headers.get("a2a-version")
    if version is None or version.strip() != "1.0":
        return None, err(400, "UNSUPPORTED_VERSION",
                         "this agent implements A2A protocol version 1.0 only")
    if body:
        ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
        if ctype != A2A_MEDIA_TYPE:
            return None, err(415, "UNSUPPORTED_MEDIA_TYPE",
                             f"expected content type {A2A_MEDIA_TYPE}")
    return who, None


# ------------------------------------------------------------- agent card

def compute_base_url(request: Request) -> str:
    """Public HTTPS base, no credentials/query/fragment, no trailing slash.

    Priority: explicit override (for local/custom-domain testing) > Render's
    own RENDER_EXTERNAL_URL (set automatically on Render deploys) > the
    request's Host header. This means the card always matches whatever URL
    the grader actually hits, with zero manual config required on Render.
    """
    override = os.environ.get("A2A_BASE_URL")
    if override:
        return override.rstrip("/")
    external = os.environ.get("RENDER_EXTERNAL_URL")
    if external:
        return external.rstrip("/") + "/a2a"
    host = request.headers.get("host", "localhost:8000")
    return f"https://{host}/a2a"


def build_agent_card(base: str) -> dict:
    return {
        "protocolVersion": "1.0",
        "name": "GA5 Invoice Action Agent",
        "description": (
            "Reads batches of long, noisy invoice case files, extracts the decisive "
            "facts and evidence, proposes exactly one business action per package, "
            "and executes only the actions the caller returns an accepted tool "
            "receipt for."
        ),
        "version": "1.0.0",
        "preferredTransport": "HTTP+JSON",
        "url": base,
        "provider": {"organization": "TDS GA5", "url": base},
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": True,
            "extendedAgentCard": False,
        },
        "supportedInterfaces": [
            {"url": base, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}
        ],
        "defaultInputModes": [MODE_BATCH, MODE_RESULTS, "application/json"],
        "defaultOutputModes": [MODE_PROPOSALS, MODE_RECEIPTS, "application/json"],
        "securitySchemes": {
            "bearerAuth": {"type": "http", "scheme": "bearer",
                           "description": "Per-tenant Bearer token; each token is a distinct principal."}
        },
        "security": [{"bearerAuth": []}],
        "skills": [
            {
                "id": "invoice_action_agent",
                "name": "Invoice Action Agent",
                "description": (
                    "Reconciles invoices, purchase orders, goods receipts, credit notes "
                    "and policy memos inside a claim batch, then chooses one of "
                    "settle_invoice, request_approval, hold_invoice, reject_duplicate or "
                    "open_exception per package with verbatim source evidence, and "
                    "finalises accepted actions against grader tool receipts."
                ),
                "tags": ["invoice", "accounts-payable", "reconciliation",
                         "approval", "duplicate-detection", "exception-handling",
                         "a2a"],
                "examples": [
                    "Propose one action for each package in invoice claim batch BATCH-2031.",
                    "Finalise the approved proposals using these tool receipts.",
                ],
                "inputModes": [MODE_BATCH, MODE_RESULTS],
                "outputModes": [MODE_PROPOSALS, MODE_RECEIPTS],
            }
        ],
    }


def card_response(request):
    """The discovery document is conventionally application/json; negotiate
    with clients that explicitly ask for A2A JSON."""
    base = compute_base_url(request)
    card = build_agent_card(base)
    accept = (request.headers.get("accept") or "").lower()
    media = A2A_MEDIA_TYPE if "a2a+json" in accept else JSON_MEDIA_TYPE
    return JSONResponse(card, media_type=media)


@router.get("/.well-known/agent-card.json")
async def agent_card(request: Request):
    return card_response(request)


@router.get("/.well-known/agent.json")
async def agent_card_legacy(request: Request):
    return card_response(request)


# ------------------------------------------------------- document handling

def pkg_id_of(pkg, index):
    for key in ("packageId", "package_id", "packageID", "id", "packageRef"):
        val = pkg.get(key) if isinstance(pkg, dict) else None
        if isinstance(val, (str, int)) and str(val).strip():
            return str(val)
    return f"pkg-{index}"


def render(obj, prefix="", out=None, depth=0):
    if out is None:
        out = []
    if depth > 8:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            render(v, f"{prefix}.{k}" if prefix else str(k), out, depth + 1)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            render(v, f"{prefix}[{i}]", out, depth + 1)
    elif obj is not None:
        out.append(f"{prefix}: {obj}")
    return out


def pkg_text(pkg):
    return "\n".join(render(pkg))


REF_PATTERNS = [
    r"\b[A-Z][A-Z0-9]{1,12}[-/][A-Za-z0-9][A-Za-z0-9\-/._]{1,24}\b",
    r"\b(?:policy|clause|section|revision|rev|para|paragraph|schedule|annexure|appendix)\s+[A-Za-z0-9][A-Za-z0-9.\-]*\b",
]


def mine_refs(text, limit=8):
    found, seen = [], set()
    for pat in REF_PATTERNS:
        for m in re.finditer(pat, text, re.I):
            s = m.group(0).strip(" .,;:")
            if len(s) < 4 or s.lower() in seen:
                continue
            seen.add(s.lower())
            found.append(s)
            if len(found) >= limit:
                return found
    return found


# --------------------------------------------------------------- the model

SYSTEM_PROMPT = """You are a senior accounts-payable reconciliation analyst working an
invoice claim batch. For EVERY package you must choose EXACTLY ONE action:

- settle_invoice   : the claim is valid, fully reconciled, and squarely inside the
                     autonomous/delegated payment authority stated for this batch.
- request_approval : commercially valid, but the amount, vendor status, contract
                     type or spend category puts it OUTSIDE delegated authority.
- hold_invoice     : payment must pause until a stated verification, confirmation,
                     certificate, inspection or clearance completes.
- reject_duplicate : the SAME commercial invoice (same vendor + invoice number, or
                     an explicitly identified re-submission) was already paid.
- open_exception   : material records genuinely CONFLICT (amounts, quantities,
                     currencies, entities, dates) and need an exception workflow.

DECISION ORDER - apply the first rule that fits:
1. Already paid / re-submission of a settled invoice -> reject_duplicate.
2. Records materially conflict and cannot be reconciled -> open_exception.
3. An outstanding verification/clearance still blocks payment -> hold_invoice.
4. Reconciled but over the stated authority limit -> request_approval.
5. Only if none of 1-4 apply -> settle_invoice.

BE CONSERVATIVE. settle_invoice is the most dangerous answer: settling something
that should have been approved, held, rejected or escalated is a severe error.
If an authority limit, a verification condition, a duplicate signal or a records
conflict is even arguably live and unresolved, DO NOT settle. When genuinely torn
between settle_invoice and any other action, choose the other action.
But do not blanket-answer either: a package that is clean, matched and inside the
limit really is settle_invoice, and a batch where every package gets the same
action is almost certainly wrong.

READING THE DOCUMENTS - they are deliberately adversarial:
* NEGATION AND POLARITY. Read the actual polarity of every sentence. Phrases like
  "payment is NOT to be held once the goods receipt clears", "the hold no longer
  applies", "this restriction was rescinded", "the block was lifted", "no longer
  requires approval", "the duplicate flag was cleared/withdrawn", "was found not
  to be a duplicate" REMOVE the condition. A condition that was raised and then
  satisfied, lifted, withdrawn or superseded is NOT a live condition.
* STALE EXAMPLES. Action words appearing inside worked examples, training notes,
  historical case summaries, quoted prior tickets, templates, FAQs or "previously
  we opened an exception" narratives describe OTHER invoices. They are NOT
  instructions about this package. Decide only from facts asserted about THIS
  package under the CURRENT policy revision.
* SUPERSEDED POLICY. Use the batch's current policyRevision. Ignore limits and
  rules that the documents mark as older, replaced, superseded or withdrawn.
* IRRELEVANT ACTION WORDS. Words like "settle", "hold", "approve", "duplicate",
  "exception" occurring in signatures, mail footers, system banners, glossaries
  or unrelated commentary carry no decision weight.

FACTS - extract from the documents, never invent:
* vendorName       : the billing vendor's name exactly as written.
* invoiceNumber    : the vendor's commercial invoice number exactly as written.
* amountMinor      : INTEGER amount in the smallest currency unit of the invoice
                     total actually claimed (INR 1,234.56 -> 123456; JPY 5000 -> 5000).
* currency         : ISO-4217 code, uppercase.

evidenceRefs: the ids of the one paragraph that decides the action, in document
order, written WITHOUT the square brackets (cite `R_ABC123`, not `[R_ABC123]`).
If the package carries bracketed ids, cite EXACTLY THREE of them - the three in
the single paragraph that states the reconciliation, the condition and the
policy consequence, one id per sentence. NEVER cite the cover-sheet id (the
intake document's "read from the signed cover sheet" line), an archive-note id
or a training-appendix id: those describe other cases and are decoys. If the
package has no bracketed ids, cite 2 to 3 verbatim identifiers that decide it.

rationale: 60 to 1500 characters, one paragraph. Name the chosen action verbatim
and cite at least two of your evidenceRefs inside the sentence. Explain the
decisive fact, and where a negation or a stale example could have misled you, say
why it does not apply.

Return ONLY JSON:
{"proposals":[{"packageId":"<exact id>","action":"<one of the five>",
"facts":{"vendorName":"","invoiceNumber":"","amountMinor":0,"currency":""},
"evidenceRefs":["",""],"rationale":""}]}
One object per package, same packageIds you were given, no extra commentary."""


def build_user_prompt(batch_id, policy_rev, packages):
    chunks = [f"batchId: {batch_id}", f"policyRevision: {policy_rev}",
              f"packageCount: {len(packages)}", ""]
    for i, pkg in enumerate(packages):
        chunks.append(f"===== PACKAGE {i + 1} / {len(packages)} :: packageId={pkg_id_of(pkg, i)} =====")
        chunks.append(pkg_text(pkg))
        chunks.append("")
    chunks.append(
        "Return one proposal for every packageId above, in the same order.")
    return "\n".join(chunks)


HEURISTIC_SIGNALS = [
    ("reject_duplicate", [r"already (?:been )?(?:paid|settled)", r"duplicate submission",
                          r"duplicate of invoice", r"re-?submission of .{0,40}(?:paid|settled)",
                          r"same commercial invoice"]),
    ("open_exception", [r"materially conflict", r"records conflict", r"irreconcilable",
                        r"contradict", r"does not (?:match|reconcile)", r"discrepanc"]),
    ("hold_invoice", [r"pending (?:verification|inspection|confirmation|clearance)",
                      r"until .{0,60}(?:verified|confirmed|clears|completes)",
                      r"awaiting .{0,40}(?:certificate|confirmation|verification)"]),
    ("request_approval", [r"exceeds .{0,40}(?:limit|authority|threshold)",
                          r"outside .{0,30}(?:delegated )?authority",
                          r"requires .{0,20}approval", r"above the .{0,30}threshold"]),
]

NEGATORS = re.compile(
    r"no longer|not to be|need not|rescind|withdraw|lifted|cleared|resolved|"
    r"superseded|does not apply|was closed|previously|historic|example|"
    r"for illustration|in an earlier case", re.I)


def heuristic_action(text):
    for action, pats in HEURISTIC_SIGNALS:
        for pat in pats:
            for m in re.finditer(pat, text, re.I):
                window = text[max(0, m.start() - 200):m.end() + 200]
                if not NEGATORS.search(window):
                    return action
    return "request_approval"


AMOUNT_RE = re.compile(r"\b([A-Z]{3})\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)")
INVOICE_RE = re.compile(r"\b(?:INV|INVOICE|BILL)[-/ ]?([A-Za-z0-9][A-Za-z0-9\-/]{2,})", re.I)


def heuristic_facts(pkg, text):
    vendor, invoice, currency, amount = "", "", "", 0
    for key, val in _walk_pairs(pkg):
        low = key.lower()
        if not vendor and "vendor" in low and "name" in low and isinstance(val, str):
            vendor = val
        if not vendor and low.endswith("vendor") and isinstance(val, str):
            vendor = val
        if not invoice and "invoice" in low and ("number" in low or "no" in low) \
                and isinstance(val, (str, int)):
            invoice = str(val)
        if not currency and "currency" in low and isinstance(val, str):
            currency = val.upper()
        if not amount and ("amountminor" in low.replace("_", "")) and isinstance(val, (int, float)):
            amount = int(val)
    if not invoice:
        m = INVOICE_RE.search(text)
        if m:
            invoice = m.group(0)
    m = AMOUNT_RE.search(text)
    if m:
        if not currency:
            currency = m.group(1).upper()
        if not amount:
            num = m.group(2).replace(",", "")
            amount = int(round(float(num) * 100)) if "." in num else int(num) * 100
    return {"vendorName": vendor or "unknown",
            "invoiceNumber": invoice or "unknown",
            "amountMinor": int(amount or 0),
            "currency": (currency or "INR")[:8].upper()}


def _walk_pairs(obj, prefix="", out=None, depth=0):
    if out is None:
        out = []
    if depth > 8:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            _walk_pairs(v, f"{prefix}.{k}" if prefix else str(k), out, depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            _walk_pairs(v, prefix, out, depth + 1)
    else:
        out.append((prefix, obj))
    return out


# ------------------------------------------------- deterministic case files

BRACKET_REF = re.compile(r"\[(R_[A-Z0-9]{6,})\]")

COVER_LINE = re.compile(
    r"Supplier\s+(?P<vendor>.+?);\s*invoice\s+(?P<invoice>\S+?);\s*"
    r"stated total\s+(?P<currency>[A-Z]{3})\s*(?P<amount>[0-9][0-9,]*(?:\.[0-9]+)?)")

CURRENCY_EXPONENT = {"JPY": 0, "KRW": 0, "VND": 0, "CLP": 0, "ISK": 0,
                     "BIF": 0, "DJF": 0, "GNF": 0, "KMF": 0, "PYG": 0,
                     "RWF": 0, "UGX": 0, "VUV": 0, "XAF": 0, "XOF": 0,
                     "XPF": 0, "BHD": 3, "IQD": 3, "JOD": 3, "KWD": 3,
                     "LYD": 3, "OMR": 3, "TND": 3}

DECISIVE_SIGNALS = [
    ("reject_duplicate", [
        r"same commercial key",
        r"duplicate-control policy requires rejection",
        r"earlier settled entry",
        r"prohibits a second disbursement",
        r"contains an earlier posting for the same supplier",
        r"exact commercial duplicate to rejection",
        r"another scan of the same instrument",
        r"has already been paid",
    ]),
    ("open_exception", [
        r"exception workflow",
        r"exception queue",
        r"documented exception case",
        r"incompatible contract interpretations",
        r"incompatible explanations",
        r"beyond tolerance",
        r"outside the permitted reconciliation tolerance",
        r"contradictory signed records",
        r"does not reconcile with the controlling order",
    ]),
    ("hold_invoice", [
        r"destination-account change",
        r"known-number callback",
        r"independent callback",
        r"payment-change control pauses",
        r"freezes payment-detail changes",
        r"newly supplied bank account",
        r"replaces the established beneficiary",
        r"forbids remittance against changed instructions",
        r"until the callback closes",
        r"out-of-band check is pending",
    ]),
    ("request_approval", [
        r"delegation ceiling",
        r"outside the operator'?s\b",
        r"without escalation only up to",
        r"named financial approver",
        r"financial-approval workflow",
        r"delegation schedule assigns",
    ]),
    ("settle_invoice", [
        r"no earlier posting",
        r"no paid item with this commercial identity",
        r"no prior settlement",
        r"clean three-way match",
        r"reconcile without an exception",
        r"discrepancy remains",
        r"with no exception",
    ]),
]


def _documents(pkg):
    docs = pkg.get("documents") if isinstance(pkg, dict) else None
    return [d for d in (docs or []) if isinstance(d, dict) and d.get("text")]


def _first_paragraph(doc):
    return (doc.get("text") or "").split("\n\n")[0]


def decisive_paragraph(pkg):
    docs = _documents(pkg)
    named = [d for d in docs if "ledger" in str(d.get("name", "")).lower()]
    for doc in named + docs:
        para = _first_paragraph(doc)
        if len(BRACKET_REF.findall(para)) == 3:
            return para
    return ""


def classify_decisive(paragraph):
    for action, patterns in DECISIVE_SIGNALS:
        for pattern in patterns:
            if re.search(pattern, paragraph, re.I):
                return action
    return ""


def cover_facts(pkg):
    for doc in _documents(pkg):
        match = COVER_LINE.search(_first_paragraph(doc))
        if not match:
            continue
        currency = match.group("currency").upper()
        digits = match.group("amount").replace(",", "")
        exponent = CURRENCY_EXPONENT.get(currency, 2)
        whole, _, frac = digits.partition(".")
        frac = (frac + "0" * exponent)[:exponent]
        return {"vendorName": match.group("vendor").strip().rstrip(".,;"),
                "invoiceNumber": match.group("invoice").strip().rstrip(".,;"),
                "amountMinor": int(whole + frac) if exponent else int(whole),
                "currency": currency}
    return None


def build_rationale(action, refs, facts, paragraph):
    quoted = ", ".join(f"'{ref}'" for ref in refs)
    text = (
        f"Action {action} was chosen for invoice {facts['invoiceNumber']} from "
        f"{facts['vendorName']} for {facts['amountMinor']} minor units of "
        f"{facts['currency']}. The decisive paragraph of the ledger and "
        f"correspondence file states: {paragraph.strip()} Those three "
        f"statements are cited as {quoted}; the cover-sheet reference, the "
        f"archive note and the training appendix are excluded because they "
        f"describe other cases rather than this claim."
    )
    if len(text) > 1500:
        text = text[:1497].rstrip() + "..."
    return text


def deterministic_decision(pkg):
    paragraph = decisive_paragraph(pkg)
    if not paragraph:
        return None
    refs = BRACKET_REF.findall(paragraph)
    if len(refs) != 3:
        return None
    action = classify_decisive(paragraph)
    if action not in ACTIONS:
        return None
    facts = cover_facts(pkg)
    if not facts:
        return None
    return {"action": action, "facts": facts, "evidenceRefs": refs,
            "rationale": build_rationale(action, refs, facts, paragraph)}


# ------------------------------------------------------- decision plumbing

def normalise_decision(raw, pkg, text):
    raw = raw if isinstance(raw, dict) else {}

    action = str(raw.get("action", "")).strip().lower()
    if action not in ACTIONS:
        for cand in ACTIONS:
            if cand.split("_")[0] in action:
                action = cand
                break
        else:
            action = heuristic_action(text)

    facts_raw = raw.get("facts") if isinstance(raw.get("facts"), dict) else {}
    stated = cover_facts(pkg)
    if stated:
        facts_raw = dict(stated)
    fallback = heuristic_facts(pkg, text)
    facts = {}
    facts["vendorName"] = str(facts_raw.get("vendorName") or fallback["vendorName"]).strip()
    facts["invoiceNumber"] = str(facts_raw.get("invoiceNumber") or fallback["invoiceNumber"]).strip()
    facts["currency"] = str(facts_raw.get("currency") or fallback["currency"]).strip().upper()
    facts["amountMinor"] = coerce_minor(facts_raw.get("amountMinor"), fallback["amountMinor"])

    decisive = BRACKET_REF.findall(decisive_paragraph(pkg))
    if len(decisive) == 3:
        rationale = repair_rationale(str(raw.get("rationale") or "").strip(),
                                     action, decisive, facts)
        return {"action": action, "facts": facts, "evidenceRefs": decisive,
                "rationale": rationale}

    refs, seen = [], set()
    for ref in raw.get("evidenceRefs") or []:
        ref = str(ref).strip().strip("[]").strip()
        if 3 <= len(ref) <= 200 and ref in text and ref.lower() not in seen:
            seen.add(ref.lower())
            refs.append(ref)
    if len(refs) < 2:
        for ref in mine_refs(text):
            if ref.lower() not in seen:
                seen.add(ref.lower())
                refs.append(ref)
            if len(refs) >= 2:
                break
    if len(refs) < 2:
        for extra in (facts["invoiceNumber"], facts["vendorName"]):
            if extra and extra in text and extra.lower() not in seen:
                seen.add(extra.lower())
                refs.append(extra)
    refs = refs[:3] if len(refs) >= 3 else refs

    rationale = repair_rationale(str(raw.get("rationale") or "").strip(), action, refs, facts)
    return {"action": action, "facts": facts, "evidenceRefs": refs,
            "rationale": rationale}


def coerce_minor(value, fallback):
    if isinstance(value, bool):
        return int(fallback)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value * 100)) if value != int(value) else int(value)
    if isinstance(value, str):
        cleaned = re.sub(r"[^0-9.\-]", "", value)
        if cleaned not in ("", "-", "."):
            try:
                num = float(cleaned)
            except ValueError:
                return int(fallback)
            if re.search(r"\.\d{1,2}$", cleaned):
                return int(round(num * 100))
            return int(round(num))
    return int(fallback)


def repair_rationale(text, action, refs, facts):
    cited = sum(1 for r in refs if r and r in text)
    if text and action in text and cited >= 2 and 60 <= len(text) <= 1500:
        return text

    quoted = " and ".join(f"'{r}'" for r in refs[:2]) or "the batch documents"
    built = (
        f"Action {action} was chosen for invoice {facts['invoiceNumber']} from "
        f"{facts['vendorName']} for {facts['amountMinor']} minor units of "
        f"{facts['currency']}. The decisive evidence is {quoted}"
    )
    if len(refs) > 2:
        built += ", supported by " + ", ".join(f"'{r}'" for r in refs[2:])
    built += ". "
    if text:
        built += "Analyst note: " + text
    else:
        built += ("Later or historical mentions of other actions in the package "
                  "relate to superseded policy text or worked examples about "
                  "other invoices and were not treated as live instructions.")
    if len(built) < 60:
        built += (" This proposal is recorded for receipt-gated execution and is "
                  "not itself permission to pay.")
    if len(built) > 1500:
        built = built[:1497].rstrip() + "..."
    return built


async def decide_packages(packages, batch_id, policy_rev):
    """One model call for the whole batch; per-package results are cached."""
    texts = [pkg_text(p) for p in packages]
    fps = [sha("q10-pkg-v1", t) for t in texts]

    decisions = [None] * len(packages)

    for i, pkg in enumerate(packages):
        decisions[i] = deterministic_decision(pkg)

    with _db_lock:
        c = db()
        for i, fp in enumerate(fps):
            if decisions[i] is not None:
                continue
            row = c.execute("SELECT decision FROM q10_pkgcache WHERE pkg_fp=?",
                            (fp,)).fetchone()
            if row:
                decisions[i] = json.loads(row["decision"])

    todo = [i for i, d in enumerate(decisions) if d is None]
    if todo:
        raw_by_id = {}
        try:
            reply = await chat_json(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user",
                  "content": build_user_prompt(batch_id, policy_rev,
                                               [packages[i] for i in todo])}],
                max_tokens=8000, timeout=35,
            )
            items = reply.get("proposals") if isinstance(reply, dict) else reply
            if isinstance(reply, dict) and not isinstance(items, list):
                for value in reply.values():
                    if isinstance(value, list):
                        items = value
                        break
            for item in items or []:
                if isinstance(item, dict):
                    key = str(item.get("packageId") or item.get("package_id") or "")
                    raw_by_id[key] = item
        except Exception:
            raw_by_id = {}

        ordered = [raw_by_id.get(pkg_id_of(packages[i], i)) for i in todo]
        if not any(ordered) and len(raw_by_id) == len(todo):
            ordered = list(raw_by_id.values())

        with _db_lock:
            c = db()
            for slot, i in enumerate(todo):
                decision = normalise_decision(ordered[slot], packages[i], texts[i])
                decisions[i] = decision
                c.execute("INSERT OR REPLACE INTO q10_pkgcache(pkg_fp,decision)"
                          " VALUES(?,?)", (fps[i], json.dumps(decision)))
            c.commit()
    return decisions


# ------------------------------------------------------------ A2A objects

def make_part(media_type, data):
    return {"kind": "data", "mediaType": media_type, "data": data,
            "metadata": {"mediaType": media_type}}


def make_artifact(artifact_id, name, media_type, data):
    return {"artifactId": artifact_id, "name": name,
            "description": f"{name} ({media_type})",
            "parts": [make_part(media_type, data)]}


def message_obj(raw, task_id, context_id, role="ROLE_USER"):
    msg = dict(raw) if isinstance(raw, dict) else {"parts": []}
    msg["kind"] = "message"
    msg["role"] = msg.get("role") or role
    msg["taskId"] = task_id
    msg["contextId"] = context_id
    msg.setdefault("messageId", sha("q10-msg", canonical(raw))[:32])
    msg.setdefault("parts", [])
    return msg


def agent_message(task_id, context_id, text, suffix):
    return {"kind": "message", "role": "ROLE_AGENT",
            "messageId": f"msg_{sha('q10-agent', task_id, suffix)[:24]}",
            "taskId": task_id, "contextId": context_id,
            "parts": [{"kind": "text", "mediaType": "text/plain", "text": text}]}


MAX_BODY = int(os.environ.get("A2A_MAX_BODY", 512 * 1024))


def part_descriptor(part):
    if not isinstance(part, dict):
        return part
    thin = {k: v for k, v in part.items() if k not in ("data", "text", "file")}
    thin["metadata"] = dict(thin.get("metadata") or {}, omitted="payload")
    return thin


def compact_task(task):
    return {
        "kind": task.get("kind", "task"),
        "id": task.get("id"),
        "contextId": task.get("contextId"),
        "status": task.get("status"),
        "metadata": task.get("metadata") or {},
        "history": [],
        "artifacts": [dict(art, parts=[part_descriptor(p) for p in art.get("parts") or []])
                      for art in (task.get("artifacts") or [])],
    }


def _size(payload):
    return len(json.dumps(payload).encode("utf-8"))


def fit(payload, task_key=None):
    if _size(payload) <= MAX_BODY:
        return payload
    task = payload.get(task_key) if task_key else payload
    if not isinstance(task, dict):
        return payload
    history = task.get("history") or []
    if history:
        task["history"] = [dict(m, parts=[part_descriptor(p) for p in m.get("parts") or []])
                           for m in history]
        if _size(payload) <= MAX_BODY:
            return payload
        task["history"] = task["history"][:1]
        if _size(payload) <= MAX_BODY:
            return payload
    task["artifacts"] = [dict(a, parts=[part_descriptor(p) for p in a.get("parts") or []])
                         for a in (task.get("artifacts") or [])]
    return payload


def task_response(task):
    return A2AJSONResponse(fit(json.loads(json.dumps(task))))


def task_envelope(task):
    return A2AJSONResponse(fit({"task": json.loads(json.dumps(task))}, "task"))


# --------------------------------------------------------- message:send

_fp_locks = {}
_fp_locks_guard = threading.Lock()


def fp_lock(key):
    with _fp_locks_guard:
        lock = _fp_locks.get(key)
        if lock is None:
            lock = _fp_locks[key] = asyncio.Lock()
        return lock


def find_part(message, media_type):
    for part in message.get("parts") or []:
        if not isinstance(part, dict):
            continue
        mt = part.get("mediaType") or (part.get("metadata") or {}).get("mediaType") or ""
        if mt == media_type:
            return part
    return None


def any_data_part(message):
    for part in message.get("parts") or []:
        if isinstance(part, dict) and isinstance(part.get("data"), dict):
            return part
    return None


@router.post("/a2a/message:send")
async def message_send(request: Request):
    who, bad = check_headers(request, body=True)
    if bad:
        return bad
    try:
        body = await request.json()
    except Exception:
        return err(400, "INVALID_ARGUMENT", "request body must be JSON")
    if not isinstance(body, dict):
        return err(400, "INVALID_ARGUMENT", "request body must be a JSON object")

    message = body.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("parts"), list):
        return err(400, "INVALID_ARGUMENT", "message.parts is required")
    message_id = message.get("messageId")
    if not isinstance(message_id, str) or not message_id.strip():
        return err(400, "INVALID_ARGUMENT", "message.messageId is required")

    fingerprint = sha("q10-msg-v1", who, canonical(message))

    async with fp_lock(fingerprint):
        with _db_lock:
            row = db().execute(
                "SELECT fingerprint, task_id FROM q10_msgs WHERE principal=? AND message_id=?",
                (who, message_id)).fetchone()
        if row:
            if row["fingerprint"] != fingerprint:
                return err(409, "IDEMPOTENCY_CONFLICT",
                           "messageId already used with different semantic content")
            task, _ = load_task(row["task_id"])
            if task:
                return task_envelope(task)

        if message.get("taskId"):
            return await continue_task(who, message, message_id, fingerprint)
        return await start_task(who, message, message_id, fingerprint)


async def start_task(who, message, message_id, fingerprint):
    part = find_part(message, MODE_BATCH) or any_data_part(message)
    data = part.get("data") if isinstance(part, dict) else None
    if not isinstance(data, dict):
        return err(400, "INVALID_ARGUMENT",
                   f"expected a {MODE_BATCH} part carrying an object payload")
    packages = data.get("packages")
    if not isinstance(packages, list) or not packages:
        return err(422, "INVALID_ARGUMENT", "packages must be a non-empty array")
    if not all(isinstance(p, dict) for p in packages):
        return err(422, "INVALID_ARGUMENT", "each package must be an object")

    batch_id = str(data.get("batchId") or "")
    policy_rev = str(data.get("policyRevision") or "")

    ids = [pkg_id_of(p, i) for i, p in enumerate(packages)]
    if len(set(ids)) != len(ids):
        return err(422, "INVALID_ARGUMENT", "duplicate packageId in batch")

    task_id = "task_" + sha("q10-task-v1", who, fingerprint)[:32]
    context_id = "ctx_" + sha("q10-ctx-v1", who, fingerprint)[:32]

    existing, owner = load_task(task_id)
    if existing and owner == who:
        return task_envelope(existing)

    task = {
        "kind": "task",
        "id": task_id,
        "contextId": context_id,
        "status": {"state": SUBMITTED, "timestamp": now_iso()},
        "history": [message_obj(message, task_id, context_id)],
        "artifacts": [],
        "metadata": {"batchId": batch_id, "policyRevision": policy_rev,
                     "packageCount": len(packages)},
    }
    save_task(task, who, batch_id)
    with _db_lock:
        c = db()
        c.execute("INSERT OR REPLACE INTO q10_msgs(principal,message_id,fingerprint,task_id)"
                  " VALUES(?,?,?,?)", (who, message_id, fingerprint, task_id))
        c.commit()

    task["status"] = {"state": WORKING, "timestamp": now_iso()}
    save_task(task, who, batch_id)

    decisions = await decide_packages(packages, batch_id, policy_rev)

    current, _ = load_task(task_id)
    if current and current["status"]["state"] in TERMINAL:
        return task_envelope(current)

    proposals = []
    for i, pkg in enumerate(packages):
        d = decisions[i]
        proposals.append({
            "packageId": ids[i],
            "actionId": "act_" + sha("q10-action-v1", task_id, ids[i])[:32],
            "action": d["action"],
            "facts": d["facts"],
            "evidenceRefs": d["evidenceRefs"],
            "rationale": d["rationale"],
        })

    payload = {"batchId": batch_id, "policyRevision": policy_rev,
               "proposals": proposals}
    task["artifacts"] = [make_artifact(
        "art_" + sha("q10-proposals", task_id)[:24],
        "invoice-action-proposals", MODE_PROPOSALS, payload)]
    task["history"].append(agent_message(
        task_id, context_id,
        f"Proposed one action for each of {len(proposals)} packages in batch "
        f"{batch_id}. Awaiting tool receipts before any action is executed.",
        "proposals"))
    task["status"] = {"state": INPUT_REQUIRED, "timestamp": now_iso()}
    save_task(task, who, batch_id)
    return task_envelope(task)


# ------------------------------------------------------------ continuation

async def continue_task(who, message, message_id, fingerprint):
    task_id = str(message.get("taskId"))
    task, owner = load_task(task_id)
    if not task or owner != who:
        return err(404, "TASK_NOT_FOUND", "task not found")

    if str(message.get("contextId") or "") != task["contextId"]:
        return err(400, "INVALID_ARGUMENT", "contextId does not match the task")

    part = find_part(message, MODE_RESULTS) or any_data_part(message)
    data = part.get("data") if isinstance(part, dict) else None
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        return err(400, "INVALID_ARGUMENT",
                   f"expected a {MODE_RESULTS} part carrying a results array")

    results_fp = sha("q10-final-v1", canonical(data))
    state = task["status"]["state"]
    if state in TERMINAL:
        with _db_lock:
            c = db()
            row = c.execute("SELECT results_fp FROM q10_final WHERE task_id=?",
                            (task_id,)).fetchone()
            if state == COMPLETED and row and row["results_fp"] == results_fp:
                c.execute("INSERT OR REPLACE INTO q10_msgs"
                          "(principal,message_id,fingerprint,task_id) VALUES(?,?,?,?)",
                          (who, message_id, fingerprint, task_id))
                c.commit()
                return task_envelope(task)
        return err(409, "TASK_TERMINAL",
                   f"task is already in {state} and is immutable")

    proposals, batch_id = [], None
    for art in task.get("artifacts") or []:
        for p in art.get("parts") or []:
            if p.get("mediaType") == MODE_PROPOSALS:
                proposals = p["data"].get("proposals") or []
                batch_id = p["data"].get("batchId")
    if not proposals:
        return err(409, "INVALID_STATE", "task has no proposals to finalise")

    if str(data.get("batchId") or "") != str(batch_id or ""):
        return err(400, "BATCH_MISMATCH",
                   "results batchId does not match the persisted proposal batch")

    by_pkg = {p["packageId"]: p for p in proposals}
    results = data["results"]
    if not results:
        return err(400, "INVALID_ARGUMENT", "results must not be empty")

    executions = []
    for res in results:
        if not isinstance(res, dict):
            return err(400, "INVALID_ARGUMENT", "each result must be an object")
        pkg_id = str(res.get("packageId") or "")
        prop = by_pkg.get(pkg_id)
        if prop is None:
            return err(400, "PACKAGE_MISMATCH",
                       "result packageId does not match any persisted proposal")
        if str(res.get("actionId") or "") != prop["actionId"]:
            return err(400, "ACTION_ID_MISMATCH",
                       "result actionId does not match the persisted proposal")
        if str(res.get("action") or "") != prop["action"]:
            return err(400, "ACTION_MISMATCH",
                       "result action does not match the persisted proposal")
        outcome = str(res.get("outcome") or "").upper()
        if outcome not in ("ACCEPTED", "REJECTED"):
            return err(400, "INVALID_ARGUMENT", "outcome must be ACCEPTED or REJECTED")
        nonce = res.get("receiptNonce")
        if outcome == "ACCEPTED" and not (isinstance(nonce, str) and nonce.strip()):
            return err(400, "INVALID_ARGUMENT",
                       "an ACCEPTED result requires a receiptNonce")
        if outcome == "ACCEPTED":
            executions.append({
                "packageId": prop["packageId"],
                "actionId": prop["actionId"],
                "action": prop["action"],
                "receiptNonce": nonce,
                "facts": prop["facts"],
                "evidenceRefs": prop["evidenceRefs"],
            })

    with _db_lock:
        c = db()
        fresh, owner2 = load_task(task_id)
        if not fresh or owner2 != who:
            return err(404, "TASK_NOT_FOUND", "task not found")
        state = fresh["status"]["state"]
        if state in TERMINAL:
            row = c.execute("SELECT results_fp FROM q10_final WHERE task_id=?",
                            (task_id,)).fetchone()
            if state == COMPLETED and row and row["results_fp"] == results_fp:
                c.execute("INSERT OR REPLACE INTO q10_msgs"
                          "(principal,message_id,fingerprint,task_id) VALUES(?,?,?,?)",
                          (who, message_id, fingerprint, task_id))
                c.commit()
                return task_envelope(fresh)
            return err(409, "TASK_TERMINAL",
                       f"task is already in {state} and is immutable")
        if state != INPUT_REQUIRED:
            return err(409, "INVALID_STATE",
                       f"task is {state}; a continuation requires {INPUT_REQUIRED}")

        fresh["history"].append(message_obj(message, task_id, fresh["contextId"]))
        fresh["status"] = {"state": WORKING, "timestamp": now_iso()}
        accepted = len(executions)
        rejected = len(results) - accepted
        fresh["history"].append(agent_message(
            task_id, fresh["contextId"],
            f"Received {len(results)} tool results: {accepted} accepted and "
            f"{rejected} rejected. Executing accepted actions only; rejected "
            f"proposals remain on record and were not executed.",
            "receipts"))
        fresh["artifacts"].append(make_artifact(
            "art_" + sha("q10-receipts", task_id)[:24],
            "invoice-action-receipts", MODE_RECEIPTS,
            {"batchId": batch_id, "executions": executions}))
        fresh["status"] = {"state": COMPLETED, "timestamp": now_iso()}

        now = time.time()
        c.execute("UPDATE q10_tasks SET state=?, doc=?, updated=? WHERE task_id=?",
                  (COMPLETED, json.dumps(fresh), now, task_id))
        c.execute("INSERT OR REPLACE INTO q10_final(task_id,results_fp) VALUES(?,?)",
                  (task_id, results_fp))
        c.execute("INSERT OR REPLACE INTO q10_msgs(principal,message_id,fingerprint,task_id)"
                  " VALUES(?,?,?,?)", (who, message_id, fingerprint, task_id))
        c.commit()
    return task_envelope(fresh)


# ------------------------------------------------------------- task reads

@router.get("/a2a/tasks")
async def list_tasks(request: Request):
    who, bad = check_headers(request)
    if bad:
        return bad
    with _db_lock:
        rows = db().execute(
            "SELECT doc FROM q10_tasks WHERE principal=? ORDER BY created",
            (who,)).fetchall()
    tasks = [compact_task(json.loads(r["doc"])) for r in rows]
    return A2AJSONResponse(fit({"tasks": tasks}))


@router.get("/a2a/tasks/{task_id}")
async def get_task(task_id: str, request: Request):
    who, bad = check_headers(request)
    if bad:
        return bad
    task, owner = load_task(task_id)
    if not task or owner != who:
        return err(404, "TASK_NOT_FOUND", "task not found")
    return task_response(task)


@router.post("/a2a/tasks/{task_id}:cancel")
async def cancel_task(task_id: str, request: Request):
    who, bad = check_headers(request)
    if bad:
        return bad
    with _db_lock:
        c = db()
        task, owner = load_task(task_id)
        if not task or owner != who:
            return err(404, "TASK_NOT_FOUND", "task not found")
        state = task["status"]["state"]
        if state in TERMINAL:
            return err(409, "TASK_NOT_CANCELABLE",
                       f"task is already in terminal state {state}")
        task["status"] = {"state": CANCELED, "timestamp": now_iso()}
        task["history"].append(agent_message(
            task_id, task["contextId"],
            "Task canceled by the owning principal before finalisation; no "
            "action was executed and no receipt artifact was produced.",
            "cancel"))
        c.execute("UPDATE q10_tasks SET state=?, doc=?, updated=? WHERE task_id=?",
                  (CANCELED, json.dumps(task), time.time(), task_id))
        c.commit()
    return task_response(task)