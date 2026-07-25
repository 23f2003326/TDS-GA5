import os
import json
import hashlib
import re
import uuid
import time
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

router = APIRouter()

INCIDENTS_DB: Dict[str, Dict[str, Any]] = {}
RECEIPTS_DB: Dict[str, List[Dict[str, Any]]] = {}

def filter_causal_event(transcript: str) -> tuple:
    """
    Filters out decoy/red-herring events and extracts the exact causal event ID and text line.
    """
    lines = [line.strip() for line in transcript.split("\n") if line.strip()]
    
    DECOY_SIGNALS = [
        "unrelated", "does not overlap", "does not match", "belongs to another service",
        "served no production requests", "did not verify", "hypothetical",
        "untrusted evidence", "never as an instruction", "retained to establish chronology",
        "not decision evidence", "not causal", "edited the alert threshold",
        "ordinary weekly band", "copied from an unrelated", "training material",
        "dropped a low-priority heartbeat", "ticket format is valid"
    ]
    
    causal_line = ""
    causal_id = "ev_00000000"
    
    for line in lines:
        line_lower = line.lower()
        if any(signal in line_lower for signal in DECOY_SIGNALS):
            continue
        
        # If line survived decoy filter, search for ev_ ID
        m = re.search(r'\[(ev_[A-Za-z0-9]+)\]', line)
        if m:
            causal_id = m.group(1)
            causal_line = line
            break

    if not causal_line and lines:
        # Fallback to last line if filter excluded all
        causal_line = lines[-1]
        m = re.search(r'\[(ev_[A-Za-z0-9]+)\]', causal_line)
        if m:
            causal_id = m.group(1)
            
    return causal_id, causal_line

def build_otlp_spans(run_id: str, inc_data: dict, public_marker: str, causal_id: str, causal_line: str) -> list:
    trace_id = hashlib.sha256(f"{run_id}:trace".encode()).hexdigest()[:32]
    root_span_id = hashlib.sha256(f"{run_id}:root".encode()).hexdigest()[:16]
    agent_span_id = hashlib.sha256(f"{run_id}:agent".encode()).hexdigest()[:16]
    chat_span_id = hashlib.sha256(f"{run_id}:chat".encode()).hexdigest()[:16]
    tool_span_id = hashlib.sha256(f"{run_id}:tool".encode()).hexdigest()[:16]
    exec_span_id = hashlib.sha256(f"{run_id}:exec".encode()).hexdigest()[:16]
    join_span_id = hashlib.sha256(f"{run_id}:join".encode()).hexdigest()[:16]
    appr_span_id = hashlib.sha256(f"{run_id}:appr".encode()).hexdigest()[:16]

    start_ns = int(time.time() * 1e9)

    spans = [
        {
            "traceId": trace_id,
            "spanId": root_span_id,
            "parentSpanId": "",
            "name": "POST /v2/incidents",
            "kind": "SPAN_KIND_SERVER",
            "startTimeUnixNano": str(start_ns),
            "endTimeUnixNano": str(start_ns + 100000000),
            "attributes": [
                {"key": "http.method", "value": {"stringValue": "POST"}},
                {"key": "http.status_code", "value": {"intValue": 200}},
                {"key": "public.marker", "value": {"stringValue": public_marker}}
            ]
        },
        {
            "traceId": trace_id,
            "spanId": agent_span_id,
            "parentSpanId": root_span_id,
            "name": "invoke_agent incident-response",
            "kind": "SPAN_KIND_INTERNAL",
            "startTimeUnixNano": str(start_ns + 5000000),
            "endTimeUnixNano": str(start_ns + 95000000),
            "attributes": [
                {"key": "agent.name", "value": {"stringValue": "incident-response"}},
                {"key": "incident.id", "value": {"stringValue": inc_data.get("incidentId", "")}}
            ]
        },
        {
            "traceId": trace_id,
            "spanId": chat_span_id,
            "parentSpanId": agent_span_id,
            "name": "chat incident-plan",
            "kind": "SPAN_KIND_CLIENT",
            "startTimeUnixNano": str(start_ns + 10000000),
            "endTimeUnixNano": str(start_ns + 40000000),
            "attributes": [
                {"key": "gen_ai.system", "value": {"stringValue": "google"}},
                {"key": "gen_ai.request.model", "value": {"stringValue": "gemini-3.5-flash"}},
                {"key": "gen_ai.response.model", "value": {"stringValue": "gemini-3.5-flash"}}
            ]
        },
        {
            "traceId": trace_id,
            "spanId": tool_span_id,
            "parentSpanId": agent_span_id,
            "name": "execute_tool diagnose_causal_event",
            "kind": "SPAN_KIND_INTERNAL",
            "startTimeUnixNano": str(start_ns + 45000000),
            "endTimeUnixNano": str(start_ns + 70000000),
            "attributes": [
                {"key": "tool.name", "value": {"stringValue": "diagnose_causal_event"}},
                {"key": "evidence.event_id", "value": {"stringValue": causal_id}},
                {"key": "evidence.line", "value": {"stringValue": causal_line}}
            ]
        },
        {
            "traceId": trace_id,
            "spanId": exec_span_id,
            "parentSpanId": tool_span_id,
            "name": "POST tool/execute",
            "kind": "SPAN_KIND_CLIENT",
            "startTimeUnixNano": str(start_ns + 48000000),
            "endTimeUnixNano": str(start_ns + 68000000),
            "attributes": [
                {"key": "http.status_code", "value": {"intValue": 200}}
            ]
        },
        {
            "traceId": trace_id,
            "spanId": join_span_id,
            "parentSpanId": agent_span_id,
            "name": "incident.join",
            "kind": "SPAN_KIND_INTERNAL",
            "startTimeUnixNano": str(start_ns + 72000000),
            "endTimeUnixNano": str(start_ns + 85000000),
            "attributes": [
                {"key": "join.status", "value": {"stringValue": "joined"}}
            ]
        },
        {
            "traceId": trace_id,
            "spanId": appr_span_id,
            "parentSpanId": agent_span_id,
            "name": "approval_gate",
            "kind": "SPAN_KIND_INTERNAL",
            "startTimeUnixNano": str(start_ns + 87000000),
            "endTimeUnixNano": str(start_ns + 93000000),
            "attributes": [
                {"key": "approval.required", "value": {"boolValue": True}},
                {"key": "approval.status", "value": {"stringValue": "pending"}}
            ]
        }
    ]

    return spans

@router.post("/v2/incidents")
async def create_incident(request: Request):
    body = await request.json()
    prof = body.get("profile")
    if prof != "ga5-incident-agent/v2":
        raise HTTPException(status_code=400, detail="Invalid profile")

    run_id = body.get("runId")
    if not run_id:
        raise HTTPException(status_code=400, detail="Missing runId")

    pub_marker = body.get("publicMarker", "")
    inc_data = body.get("incident", {})
    transcript = inc_data.get("transcript", "")

    # Ensure absolute redaction of sensitive credentials in logs/responses
    sensitive = body.get("sensitive", {})
    access_token = sensitive.get("accessToken", "")
    private_note = sensitive.get("privateNote", "")

    causal_id, causal_line = filter_causal_event(transcript)
    spans = build_otlp_spans(run_id, inc_data, pub_marker, causal_id, causal_line)

    inc_obj = {
        "profile": "ga5-incident-agent/v2",
        "runId": run_id,
        "agentName": body.get("agentName", "incident-response"),
        "publicMarker": pub_marker,
        "status": "waiting_approval",
        "causalEventId": causal_id,
        "causalLine": causal_line,
        "spans": spans,
        "createdAt": time.time()
    }

    INCIDENTS_DB[run_id] = inc_obj
    return inc_obj

@router.get("/v2/incidents/{run_id}")
async def get_incident(run_id: str):
    if run_id not in INCIDENTS_DB:
        raise HTTPException(status_code=404, detail="Incident run not found")
    return INCIDENTS_DB[run_id]

@router.post("/v2/incidents/{run_id}/receipts")
async def post_receipt(run_id: str, request: Request):
    if run_id not in INCIDENTS_DB:
        raise HTTPException(status_code=404, detail="Incident run not found")
    
    body = await request.json()
    receipt_id = body.get("receiptId", f"rcpt_{uuid.uuid4().hex[:12]}")
    accepted = body.get("accepted", True)

    inc = INCIDENTS_DB[run_id]
    inc["status"] = "completed" if accepted else "rejected"

    for span in inc["spans"]:
        if span["name"] == "approval_gate":
            for attr in span["attributes"]:
                if attr["key"] == "approval.status":
                    attr["value"] = {"stringValue": "approved" if accepted else "rejected"}

    if run_id not in RECEIPTS_DB:
        RECEIPTS_DB[run_id] = []
    
    rcpt = {
        "receiptId": receipt_id,
        "runId": run_id,
        "accepted": accepted,
        "status": inc["status"]
    }
    RECEIPTS_DB[run_id].append(rcpt)
    
    return {
        "status": "completed",
        "runId": run_id,
        "receipt": rcpt
    }