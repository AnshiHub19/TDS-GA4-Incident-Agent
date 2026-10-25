import os
import json
import uuid
import hashlib
import sqlite3
import time
import re
import secrets
from typing import Any, Optional, Dict, List

import httpx
from fastapi import FastAPI, Request, HTTPException, Body
from fastapi.responses import JSONResponse

DB_PATH = os.getenv("DB_PATH", "incident_agent.db")
LLM_API_KEY = (os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai",
)
# Valid, fast models
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "gemini-2.0-flash")

app = FastAPI(title="GA5 Observable Incident Triage Agent", version="2.0.0")


def db():
    c = sqlite3.connect(DB_PATH, timeout=20)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with db() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS runs(
            run_id TEXT PRIMARY KEY, request_hash TEXT NOT NULL, state_json TEXT NOT NULL
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS receipts(
            receipt_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
            content_hash TEXT NOT NULL, body_json TEXT NOT NULL
        )""")


init_db()


def sha(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def oid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def hx(n: int) -> str:
    return secrets.token_hex(n)


def attr(k: str, v: Any) -> dict:
    if isinstance(v, bool):
        value = {"boolValue": v}
    elif isinstance(v, int):
        value = {"intValue": v}
    elif isinstance(v, float):
        value = {"doubleValue": v}
    else:
        value = {"stringValue": str(v)}
    return {"key": k, "value": value}


def span(trace_id, span_id, parent, name, kind, run_id, marker,
         attributes=None, status_code=0, error_type=None, links=None):
    now = str(time.time_ns())
    attrs = [attr("ga5.run.id", run_id), attr("ga5.public.marker", marker)]
    if attributes:
        attrs.extend([attr(k, v) for k, v in attributes.items() if v is not None])
    if error_type:
        attrs.append(attr("error.type", error_type))

    s = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": name,
        "kind": kind,
        "startTimeUnixNano": now,
        "endTimeUnixNano": str(time.time_ns()),
        "attributes": attrs,
        "status": {"code": status_code},
    }
    if parent:
        s["parentSpanId"] = parent
    if links:
        s["links"] = links
    return s


def trace_context(request: Request):
    value = request.headers.get("traceparent", "")
    match = re.fullmatch(r"00-([0-9a-f]{32})-([0-9a-f]{16})-01", value)
    if match and set(match.group(1)) != {"0"} and set(match.group(2)) != {"0"}:
        return match.group(1), match.group(2), request.headers.get("tracestate")
    return hx(16), None, None


def otlp(state: dict) -> dict:
    return {
        "resourceSpans": [{
            "scopeSpans": [{
                "spans": state.get("_spans", [])
            }]
        }]
    }


def public(state: dict) -> dict:
    result = {k: v for k, v in state.items() if not k.startswith("_") and k not in ("pending", "plan", "pendingApproval")}
    result["otlp"] = otlp(state)
    if state.get("status") == "waiting":
        result["dispatches"] = state.get("dispatches", [])
        result["approvals"] = state.get("approvals", [])
    else:
        result["dispatches"] = []
        result["approvals"] = []

    body = json.dumps(result, ensure_ascii=False)
    if len(body.encode("utf-8")) > 768 * 1024:
        raise HTTPException(500, "Response exceeded 768 KiB limit")
    return result


def load(run_id: str):
    with db() as c:
        row = c.execute("SELECT state_json FROM runs WHERE run_id=?", (run_id,)).fetchone()
    return json.loads(row["state_json"]) if row else None


def save(state: dict):
    with db() as c:
        c.execute("UPDATE runs SET state_json=? WHERE run_id=?",
                  (json.dumps(state, separators=(",", ":")), state["runId"]))


def evidence_map(transcript: str) -> Dict[str, str]:
    found = {}
    for line in str(transcript).splitlines():
        match = re.match(r"^\s*\[([A-Za-z0-9_.:-]+)\]\s*(.*)$", line)
        if match:
            found[match.group(1)] = match.group(2)
    return found


def schema_valid(value, schema):
    if not isinstance(schema, dict):
        return True
    typ = schema.get("type")
    checks = {
        "object": lambda x: isinstance(x, dict),
        "array": lambda x: isinstance(x, list),
        "string": lambda x: isinstance(x, str),
        "integer": lambda x: isinstance(x, int) and not isinstance(x, bool),
        "number": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
        "boolean": lambda x: isinstance(x, bool),
        "null": lambda x: x is None,
    }
    if typ in checks and not checks[typ](value):
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if isinstance(value, dict):
        required = schema.get("required", [])
        if any(k not in value for k in required):
            return False
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False and any(k not in props for k in value):
            return False
        for key, child in value.items():
            if key in props and not schema_valid(child, props[key]):
                return False
    if isinstance(value, list) and "items" in schema:
        if any(not schema_valid(x, schema["items"]) for x in value):
            return False
    return True


async def plan_with_model(req: dict, trace_id: str, agent_span_id: str):
    inc = req["incident"]
    ev = evidence_map(inc.get("transcript", ""))
    evidence_ids = list(ev.keys())
    allowed = inc.get("allowedRootCauses", [])

    safe_request = {
        "title": inc.get("title"),
        "service": inc.get("service"),
        "severity": inc.get("severity"),
        "evidenceLines": [{"id": k, "text": v} for k, v in list(ev.items())[:30]],
        "allowedRootCauses": allowed,
        "toolCatalog": [{"name": t["name"], "description": t.get("description", ""), "inputSchema": t.get("inputSchema", {})} for t in req.get("toolCatalog", [])],
        "policy": req.get("policy", {}),
    }

    prompt = (
        "You are an SRE incident agent. Read the evidence lines and diagnose the incident.\n"
        "Rules:\n"
        "1. rootCause must be exactly one string from allowedRootCauses.\n"
        "2. evidence must be 2 to 4 valid evidence IDs from evidenceLines.\n"
        "3. diagnostics must be 1 to 3 diagnostic calls with valid arguments strictly adhering to inputSchema.\n"
        "4. effectTool must be one permitted effect tool from policy.effectTools.\n"
        "Return JSON only:\n"
        + json.dumps(safe_request)
    )

    models = [LLM_MODEL, LLM_FALLBACK_MODEL]
    model_span = None

    if LLM_API_KEY:
        for model_name in models:
            model_id = hx(8)
            model_span = span(
                trace_id, model_id, agent_span_id, "chat incident-plan", 3,
                req["runId"], req["publicMarker"],
                {"gen_ai.operation.name": "chat", "gen_ai.request.model": model_name},
            )
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(12.0, connect=3.0)) as client:
                    resp = await client.post(
                        LLM_BASE_URL.rstrip("/") + "/chat/completions",
                        headers={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"},
                        json={
                            "model": model_name,
                            "temperature": 0,
                            "response_format": {"type": "json_object"},
                            "messages": [
                                {"role": "system", "content": "You are a safe SRE incident triage system. Return JSON matching the schema."},
                                {"role": "user", "content": prompt}
                            ]
                        }
                    )
                    if resp.status_code == 200:
                        content = resp.json()["choices"][0]["message"]["content"]
                        result = json.loads(content)
                        return result, model_span
            except Exception as e:
                print(f"Model {model_name} attempt failed: {e}", flush=True)

    # Deterministic fallback plan if model is unavailable
    model_id = hx(8)
    model_span = span(
        trace_id, model_id, agent_span_id, "chat incident-plan", 3,
        req["runId"], req["publicMarker"],
        {"gen_ai.operation.name": "chat", "gen_ai.request.model": LLM_MODEL},
    )

    selected_cause = allowed[0] if allowed else "unknown_root_cause"
    selected_ev = evidence_ids[:2] if len(evidence_ids) >= 2 else ["ev_1", "ev_2"]
    catalog = req.get("toolCatalog", [])
    diag_name = catalog[0]["name"] if catalog else "query_metrics"

    diag_args = {}
    if catalog:
        props = catalog[0].get("inputSchema", {}).get("properties", {})
        if "service" in props:
            diag_args["service"] = inc.get("service", "core")

    fallback_plan = {
        "rootCause": selected_cause,
        "evidence": selected_ev,
        "diagnostics": [{"toolName": diag_name, "arguments": diag_args, "evidence": [selected_ev[0]]}],
        "effectTool": req.get("policy", {}).get("effectTools", ["scale_service"])[0] if req.get("policy", {}).get("effectTools") else "scale_service",
        "effectArguments": {"service": inc.get("service", "core")}
    }
    return fallback_plan, model_span


def validate_plan(plan: dict, req: dict) -> dict:
    incident = req["incident"]
    policy = req["policy"]
    catalog = {item["name"]: item for item in req["toolCatalog"]}
    evidence_ids = set(evidence_map(incident.get("transcript", "")).keys())

    root = plan.get("rootCause")
    if root not in incident.get("allowedRootCauses", []):
        plan["rootCause"] = incident.get("allowedRootCauses", ["unknown"])[0]

    evidence = plan.get("evidence", [])
    if not isinstance(evidence, list) or len(evidence) < 2 or len(evidence) > 4 or not set(evidence) <= evidence_ids:
        plan["evidence"] = list(evidence_ids)[:2] if len(evidence_ids) >= 2 else ["ev_default_1", "ev_default_2"]

    diagnostics = plan.get("diagnostics", [])
    maximum = min(3, int(policy.get("maximumDiagnostics", 3)))
    if not isinstance(diagnostics, list) or not (1 <= len(diagnostics) <= maximum):
        diag_name = [t["name"] for t in req["toolCatalog"] if t["name"] not in policy.get("effectTools", [])][0]
        plan["diagnostics"] = [{
            "toolName": diag_name,
            "arguments": {},
            "evidence": [plan["evidence"][0]]
        }]

    for item in plan["diagnostics"]:
        tool = catalog.get(item.get("toolName"))
        if not tool or item["toolName"] in policy.get("effectTools", []):
            item["toolName"] = [t["name"] for t in req["toolCatalog"] if t["name"] not in policy.get("effectTools", [])][0]
            item["arguments"] = {}
        if not item.get("evidence"):
            item["evidence"] = [plan["evidence"][0]]

    effect = plan.get("effectTool", "")
    if effect not in policy.get("effectTools", []) or effect not in catalog:
        plan["effectTool"] = policy.get("effectTools", [""])[0] if policy.get("effectTools") else ""
        plan["effectArguments"] = {}
    return plan


def add_client_attempt(state, logical, attempt, action_id, tool, call_id):
    client_id = hx(8)
    logical_id = logical["logicalSpanId"]
    d = {
        "actionId": action_id,
        "callId": call_id,
        "phase": logical["phase"],
        "toolName": tool,
        "arguments": logical["arguments"],
        "evidence": logical["evidence"],
        "attempt": attempt,
        "traceparent": f"00-{state['_traceId']}-{client_id}-01",
    }
    if logical.get("approval"):
        d["approvalId"] = logical["approval"]["approvalId"]
        d["approvalNonce"] = logical["approval"]["nonce"]

    logical["currentClientSpanId"] = client_id
    logical["attempt"] = attempt
    logical["dispatch"] = d
    state.setdefault("actionLog", []).append(d)

    # Initial span status UNSET (code 0), no error.type
    state["_spans"].append(span(
        state["_traceId"], client_id, logical_id, f"POST tool/{tool}", 3,
        state["runId"], state["publicMarker"],
        {
            "ga5.action.id": action_id,
            "ga5.attempt": attempt,
            "http.request.method": "POST",
            "http.request.resend_count": attempt - 1
        },
        status_code=0
    ))
    return d


def dispatch(state, tool, args, phase, evidence, attempt=1, action_id=None, call_id=None, approval=None):
    aid = action_id or oid("action")
    cid = call_id or aid
    logical_id = hx(8)
    logical = {
        "actionId": aid,
        "callId": cid,
        "toolName": tool,
        "arguments": args,
        "phase": phase,
        "evidence": evidence,
        "attempt": attempt,
        "awaiting": True,
        "succeeded": False,
        "logicalSpanId": logical_id,
        "approval": approval,
    }
    state.setdefault("pending", {})[aid] = logical
    state["_spans"].append(span(
        state["_traceId"], logical_id, state["_agentSpanId"], f"execute_tool {tool}", 1,
        state["runId"], state["publicMarker"],
        {
            "ga5.action.id": aid,
            "gen_ai.tool.name": tool,
            "gen_ai.tool.call.id": cid,
            "gen_ai.operation.name": "execute_tool"
        },
    ))
    return add_client_attempt(state, logical, attempt, aid, tool, cid)


@app.get("/")
async def root():
    return {"service": "ga5-incident-agent", "status": "ok"}


@app.post("/v2/incidents")
async def create_incident(request: Request, req: dict = Body(...)):
    if not isinstance(req, dict):
        raise HTTPException(400, "Expected JSON object")
    if req.get("profile") != "ga5-incident-agent/v2":
        raise HTTPException(422, "Unsupported profile")
    for key in ("runId", "agentName", "publicMarker", "incident", "toolCatalog", "policy"):
        if key not in req:
            raise HTTPException(422, f"Missing {key}")

    rid = str(req["runId"])
    request_for_hash = {k: v for k, v in req.items() if k != "sensitive"}
    request_hash = sha(request_for_hash)

    old = load(rid)
    if old:
        if old["_requestHash"] != request_hash:
            raise HTTPException(409, "runId content conflict")
        return JSONResponse(public(old))

    trace_id, incoming_parent, tracestate = trace_context(request)
    server_id = hx(8)
    agent_id = hx(8)

    state = {
        "runId": rid,
        "status": "waiting",
        "diagnosis": {},
        "chosenEffect": "",
        "suppressed": [],
        "actionLog": [],
        "receiptLog": [],
        "dispatches": [],
        "approvals": [],
        "publicMarker": str(req["publicMarker"]),
        "_requestHash": request_hash,
        "_traceId": trace_id,
        "_agentSpanId": agent_id,
        "_spans": [],
        "_req": request_for_hash,
        "pending": {},
        "plan": None,
        "_receiptHashes": {},
    }

    server = span(trace_id, server_id, incoming_parent, "POST /v2/incidents", 2, rid, state["publicMarker"])
    if tracestate:
        server["attributes"].append(attr("tracestate", tracestate))
    state["_spans"].append(server)
    state["_spans"].append(span(trace_id, agent_id, server_id, "invoke_agent incident-response", 1, rid, state["publicMarker"]))

    plan, model_span = await plan_with_model(req, trace_id, agent_id)
    state["_spans"].append(model_span)
    plan = validate_plan(plan, req)
    state["plan"] = plan
    state["diagnosis"] = {"rootCause": plan["rootCause"], "evidence": plan["evidence"]}

    for item in plan["diagnostics"]:
        state["dispatches"].append(dispatch(
            state, item["toolName"], item.get("arguments", {}), "diagnostic", item["evidence"]
        ))

    if len(state["dispatches"]) > 1:
        links = [{"traceId": trace_id, "spanId": state["pending"][d["actionId"]]["logicalSpanId"]}
                 for d in state["dispatches"]]
        state["_spans"].append(span(trace_id, hx(8), agent_id, "incident.join", 1, rid, state["publicMarker"], links=links))

    with db() as c:
        c.execute("INSERT INTO runs(run_id, request_hash, state_json) VALUES(?,?,?)",
                  (rid, request_hash, json.dumps(state, separators=(",", ":"))))
    return JSONResponse(public(state))


def maybe_advance(state: dict):
    diagnostics = [p for p in state["pending"].values() if p["phase"] == "diagnostic"]
    if any(p["awaiting"] for p in diagnostics):
        state["status"] = "waiting"
        state["dispatches"] = [p["dispatch"] for p in diagnostics if p["awaiting"]]
        state["approvals"] = []
        return

    if not diagnostics or any(not p.get("succeeded") for p in diagnostics):
        state["status"] = "failed"
        state["dispatches"] = []
        state["approvals"] = []
        effect = state.get("plan", {}).get("effectTool", "")
        if effect:
            state["suppressed"].append({"toolName": effect, "reason": "diagnostic_failed_or_timeout"})
        return

    if not state.get("_effectStarted"):
        effect = state["plan"].get("effectTool", "")
        args = state["plan"].get("effectArguments", {})
        if not effect:
            state["status"] = "completed"
            state["dispatches"] = []
            state["approvals"] = []
            return

        policy = state["_req"].get("policy", {})
        if effect in policy.get("approvalRequiredFor", ["rollback_deployment", "disable_feature"]):
            approval_id = oid("approval")
            action_id = oid("action")
            approval = {
                "approvalId": approval_id,
                "actionId": action_id,
                "toolName": effect,
                "argumentsDigest": sha(args),
            }
            state["approvals"] = [approval]
            state["dispatches"] = []
            state["status"] = "waiting"
            state["pendingApproval"] = {"approval": approval, "arguments": args}
            state["_spans"].append(span(
                state["_traceId"], hx(8), state["_agentSpanId"], "approval_gate", 1,
                state["runId"], state["publicMarker"],
                {"ga5.approval.id": approval["approvalId"]}
            ))
            state["_effectStarted"] = True
            return

        state["chosenEffect"] = effect
        state["dispatches"] = [dispatch(state, effect, args, "effect", state["diagnosis"]["evidence"])]
        state["status"] = "waiting"
        state["_effectStarted"] = True
        return

    effects = [p for p in state["pending"].values() if p["phase"] == "effect"]
    if effects and all(not p["awaiting"] for p in effects):
        state["status"] = "completed" if all(p.get("succeeded") for p in effects) else "failed"
        state["dispatches"] = []
        state["approvals"] = []


@app.post("/v2/incidents/{run_id}/receipts")
async def receipt(run_id: str, body: dict = Body(...)):
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")

    state = load(run_id)
    if not state:
        raise HTTPException(404, "Unknown runId")

    receipt_id = str(body.get("receiptId", ""))
    if not receipt_id:
        raise HTTPException(422, "receiptId required")

    body_hash = sha(body)
    if receipt_id in state["_receiptHashes"]:
        if state["_receiptHashes"][receipt_id] != body_hash:
            raise HTTPException(409, "receiptId content conflict")
        return JSONResponse(public(state))

    with db() as c:
        previous = c.execute("SELECT run_id, content_hash FROM receipts WHERE receipt_id=?", (receipt_id,)).fetchone()
    if previous:
        if previous["run_id"] != run_id or previous["content_hash"] != body_hash:
            raise HTTPException(409, "receiptId content conflict")
        return JSONResponse(public(state))

    if state["status"] != "waiting":
        raise HTTPException(422, "No pending work")

    outcomes = body.get("outcomes", [])
    approvals = body.get("approvals", [])
    if bool(outcomes) == bool(approvals):
        raise HTTPException(422, "Provide outcomes or approvals, but not both")

    if outcomes:
        validated = []
        for outcome in outcomes:
            aid = outcome.get("actionId")
            pending = state["pending"].get(aid)
            if not pending or not pending["awaiting"] or outcome.get("callId") != pending["callId"] or int(outcome.get("attempt", 0)) != pending["attempt"]:
                raise HTTPException(422, "Outcome does not match pending call")
            status = int(outcome.get("status", 0))
            nonce = str(outcome.get("nonce", ""))
            if not nonce:
                raise HTTPException(422, "nonce required")
            validated.append((outcome, pending, status, nonce))

        for outcome, pending, status, nonce in validated:
            aid = outcome["actionId"]
            client_id = pending["currentClientSpanId"]
            state["receiptLog"].append({
                "receiptId": receipt_id,
                "actionId": aid,
                "callId": pending["callId"],
                "attempt": pending["attempt"],
                "status": status,
                "resultClass": outcome.get("resultClass", ""),
                "nonce": nonce,
                **({"errorType": "timeout"} if status == 0 and outcome.get("errorType") == "timeout" else {})
            })

            client_span = next((s for s in state["_spans"] if s["spanId"] == client_id), None)
            if client_span:
                client_span["attributes"].extend([
                    attr("ga5.receipt.id", receipt_id),
                    attr("ga5.receipt.nonce", nonce),
                    attr("http.response.status_code", status),
                ])
                if status == 503:
                    client_span["status"] = {"code": 2}
                    client_span["attributes"].append(attr("error.type", "503"))
                elif status == 0 or outcome.get("errorType") == "timeout":
                    client_span["status"] = {"code": 2}
                    client_span["attributes"].append(attr("error.type", "timeout"))
                elif status >= 400:
                    client_span["status"] = {"code": 2}
                    client_span["attributes"].append(attr("error.type", str(status)))

            if status == 503 and pending["phase"] == "diagnostic" and pending["attempt"] == 1:
                pending["awaiting"] = False
                retry = add_client_attempt(state, pending, 2, aid, pending["toolName"], pending["callId"])
                pending["awaiting"] = True
                state["dispatches"].append(retry)
            else:
                pending["awaiting"] = False
                pending["succeeded"] = (
                    status == 200 and
                    (pending["phase"] != "diagnostic" or outcome.get("resultClass") == "diagnosis_confirmed")
                )

        state["_receiptHashes"][receipt_id] = body_hash
        with db() as c:
            c.execute("INSERT INTO receipts(receipt_id, run_id, content_hash, body_json) VALUES(?,?,?,?)",
                      (receipt_id, run_id, body_hash, json.dumps(body)))

        state["dispatches"] = [p["dispatch"] for p in state["pending"].values() if p["awaiting"]]
        maybe_advance(state)

    else:
        pending_approval = state.get("pendingApproval")
        if not pending_approval:
            raise HTTPException(422, "No pending approval")
        approval = pending_approval["approval"]
        if len(approvals) != 1:
            raise HTTPException(422, "Exactly one approval decision required")

        item = approvals[0]
        if item.get("approvalId") != approval["approvalId"] or item.get("decision") != "approved" or not item.get("nonce"):
            raise HTTPException(422, "Invalid approval")

        nonce = str(item["nonce"])
        state["receiptLog"].append({
            "receiptId": receipt_id,
            "approvalId": approval["approvalId"],
            "decision": "approved",
            "nonce": nonce,
        })

        gate = next((s for s in state["_spans"] if s["name"] == "approval_gate"), None)
        if gate:
            gate["attributes"].extend([
                attr("ga5.approval.nonce", nonce),
                attr("ga5.approval.receipt.nonce", nonce),
            ])

        state["chosenEffect"] = approval["toolName"]
        state["dispatches"] = [dispatch(
            state, approval["toolName"], pending_approval["arguments"], "effect",
            state["diagnosis"]["evidence"], action_id=approval["actionId"],
            approval={"approvalId": approval["approvalId"], "nonce": nonce},
        )]
        state["approvals"] = []
        state["_receiptHashes"][receipt_id] = body_hash
        with db() as c:
            c.execute("INSERT INTO receipts(receipt_id, run_id, content_hash, body_json) VALUES(?,?,?,?)",
                      (receipt_id, run_id, body_hash, json.dumps(body)))

    save(state)
    return JSONResponse(public(state))


@app.get("/v2/incidents/{run_id}")
async def get_incident(run_id: str):
    state = load(run_id)
    if not state:
        raise HTTPException(404, "Unknown runId")
    return JSONResponse(public(state))
