import os
import json
import hashlib
import sqlite3
import time
import re
import secrets
from typing import Any, Optional, Dict, List

import httpx
from fastapi import FastAPI, Request, HTTPException, Body
from fastapi.responses import JSONResponse

DB_PATH = os.getenv("DB_PATH", "incident_v8.db")
LLM_API_KEY = (os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")

app = FastAPI(title="GA5 Observable Incident Triage Agent", version="3.6.0")


def db():
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with db() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS runs(
            run_id TEXT PRIMARY KEY,
            request_hash TEXT NOT NULL,
            state_json TEXT NOT NULL,
            initial_response TEXT NOT NULL,
            current_response TEXT NOT NULL
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS receipts(
            receipt_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            response_json TEXT NOT NULL
        )""")


init_db()


def canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(obj: Any) -> str:
    return hashlib.sha256(canon(obj).encode("utf-8")).hexdigest()


def hx_span() -> str:
    return secrets.token_hex(8)  # 16 lowercase hex characters


def hx_trace() -> str:
    return secrets.token_hex(16)  # 32 lowercase hex characters


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
        "kind": int(kind),
        "startTimeUnixNano": now,
        "endTimeUnixNano": str(time.time_ns()),
        "attributes": attrs,
        "status": {"code": int(status_code)},
    }
    if parent:
        s["parentSpanId"] = parent
    if links:
        s["links"] = links
    return s


def parse_traceparent(tp: str):
    if not tp:
        return None, None
    m = re.fullmatch(r"00-([0-9a-f]{32})-([0-9a-f]{16})-01", tp.strip().lower())
    if m and set(m.group(1)) != {"0"} and set(m.group(2)) != {"0"}:
        return m.group(1), m.group(2)
    return None, None


def otlp(state: dict) -> dict:
    return {
        "resourceSpans": [{
            "scopeSpans": [{
                "spans": state.get("_spans", [])
            }]
        }]
    }


def format_response(state: dict) -> dict:
    status = state.get("status", "waiting")
    if status == "waiting":
        return {
            "runId": state["runId"],
            "status": "waiting",
            "diagnosis": state.get("diagnosis", {}),
            "dispatches": state.get("dispatches", []),
            "approvals": state.get("approvals", [])
        }
    else:
        return {
            "runId": state["runId"],
            "status": status,
            "diagnosis": state.get("diagnosis", {}),
            "chosenEffect": state.get("chosenEffect", ""),
            "suppressed": state.get("suppressed", []),
            "actionLog": state.get("actionLog", []),
            "receiptLog": state.get("receiptLog", []),
            "otlp": otlp(state),
            "dispatches": [],
            "approvals": []
        }


def extract_evidence_lines(transcript: str) -> List[tuple]:
    lines = []
    for raw in str(transcript).splitlines():
        m = re.search(r"\[([A-Za-z0-9_.:-]+)\]\s*(.*)", raw)
        if m:
            lines.append((m.group(1), m.group(2).strip()))
    return lines


def build_tool_arguments(schema: dict, inc: dict, transcript: str) -> dict:
    args = {}
    props = schema.get("properties", {})
    required = schema.get("required", [])
    service_name = str(inc.get("service", "payment-service"))

    for key, spec in props.items():
        stype = spec.get("type", "string")
        if key in ("service", "target_service", "target"):
            args[key] = service_name
        elif key in ("incidentId", "incident_id"):
            args[key] = str(inc.get("incidentId", "inc-1"))
        elif key == "severity":
            args[key] = str(inc.get("severity", "SEV-1"))
        elif "metric" in key:
            candidates = spec.get("enum", ["cpu_utilization", "latency_p99", "error_rate", "memory_usage"])
            matched = next((c for c in candidates if c.lower() in transcript.lower()), candidates[0])
            args[key] = matched
        elif key == "replicas":
            m_rep = re.search(r"(\d+)\s*(?:replicas|instances|pods)", transcript)
            args[key] = int(m_rep.group(1)) if m_rep else 3
        elif key == "reason":
            args[key] = "Remediating verified incident"
        elif stype == "string":
            if "enum" in spec:
                matched = next((c for c in spec["enum"] if str(c).lower() in transcript.lower()), spec["enum"][0])
                args[key] = matched
            else:
                args[key] = service_name
        elif stype == "integer":
            args[key] = 1
        elif stype == "boolean":
            args[key] = True

    for req_k in required:
        if req_k not in args:
            args[req_k] = service_name
    return args


async def plan_incident(req: dict, trace_id: str, agent_span_id: str):
    inc = req["incident"]
    transcript = str(inc.get("transcript", ""))
    t_lower = transcript.lower()
    allowed_causes = inc.get("allowedRootCauses", [])
    catalog = req.get("toolCatalog", [])
    policy = req.get("policy", {})
    effect_tools = set(policy.get("effectTools", []))
    diag_catalog = [t for t in catalog if t["name"] not in effect_tools]
    max_diag = int(policy.get("maximumDiagnostics", 3))

    model_id = hx_span()
    model_span = span(
        trace_id, model_id, agent_span_id, "chat incident-plan", 3,
        req["runId"], req["publicMarker"],
        {"gen_ai.operation.name": "chat", "gen_ai.request.model": LLM_MODEL},
    )

    ev_lines = extract_evidence_lines(transcript)
    all_ev_ids = [lid for lid, _ in ev_lines]

    # Best root cause determination
    best_cause = allowed_causes[0] if allowed_causes else "unknown"
    best_score = -1
    for cause in allowed_causes:
        tokens = [w for w in re.split(r"[_ -]+", cause.lower()) if len(w) > 3]
        score = sum(t_lower.count(tok) for tok in tokens)
        if score > best_score:
            best_score = score
            best_cause = cause

    # 2 to 4 genuine bracketed IDs from transcript
    cause_tokens = [w for w in re.split(r"[_ -]+", best_cause.lower()) if len(w) > 3]
    relevant_ids = []
    for lid, txt in ev_lines:
        tl = txt.lower()
        if any(tok in tl for tok in cause_tokens) or any(w in tl for w in ["error", "failure", "timeout", "latency", "exhaust", "high", "alert"]):
            relevant_ids.append(lid)

    if len(relevant_ids) < 2:
        relevant_ids = (relevant_ids + all_ev_ids)[:2]
    selected_evidence = relevant_ids[:3] if relevant_ids else all_ev_ids[:2]

    # Rank diagnostic tools
    def rank_tool(tool):
        name = tool["name"].lower()
        score = 0
        if "metric" in name and any(k in t_lower for k in ["metric", "cpu", "memory", "utilization"]):
            score += 5
        if "log" in name and any(k in t_lower for k in ["log", "error", "stack", "trace"]):
            score += 5
        if "trace" in name and any(k in t_lower for k in ["latency", "duration", "p99"]):
            score += 4
        if "health" in name or "status" in name:
            score += 2
        return score

    sorted_diags = sorted(diag_catalog, key=rank_tool, reverse=True)
    chosen_diag = sorted_diags[:min(1, max_diag)] if sorted_diags else catalog[:1]

    diagnostics = []
    for t in chosen_diag:
        targs = build_tool_arguments(t.get("inputSchema", {}), inc, transcript)
        diagnostics.append({
            "toolName": t["name"],
            "arguments": targs,
            "evidence": [selected_evidence[0]]
        })

    # Scaling limit check
    is_at_scale_limit = any(w in t_lower for w in ["scaling limit", "limit reached", "max replicas", "already at maximum", "cannot scale"])
    if is_at_scale_limit and "page_oncall" in effect_tools:
        effect_tool = "page_oncall"
    elif is_at_scale_limit and "page_owner" in effect_tools:
        effect_tool = "page_owner"
    else:
        effect_tool = list(effect_tools)[0] if effect_tools else "scale_service"

    eff_spec = next((t for t in catalog if t["name"] == effect_tool), None)
    eff_args = build_tool_arguments(eff_spec.get("inputSchema", {}), inc, transcript) if eff_spec else {"service": inc.get("service", "core")}

    plan = {
        "rootCause": best_cause,
        "evidence": selected_evidence,
        "diagnostics": diagnostics,
        "effectTool": effect_tool,
        "effectArguments": eff_args
    }

    # Model refine via Gemini if credentials exist
    if LLM_API_KEY:
        try:
            diag_specs = [{"name": t["name"], "schema": t.get("inputSchema", {})} for t in diag_catalog]
            prompt = (
                f"Incident: {inc.get('title')} on service {inc.get('service')}.\n"
                f"Allowed Root Causes: {json.dumps(allowed_causes)}\n"
                f"Diagnostic Tools: {json.dumps(diag_specs)}\n"
                f"Transcript snippet:\n{transcript[:5000]}\n\n"
                f"Select 1 root cause from Allowed Root Causes, 2 to 4 bracketed IDs from transcript, "
                f"and 1 diagnostic tool with exact schema-valid arguments.\n"
                f"Return JSON: {{\"rootCause\":\"...\", \"evidence\":[\"id1\",\"id2\"], "
                f"\"diagnosticTool\":\"<name>\", \"diagnosticArgs\":{{}}}}"
            )
            async with httpx.AsyncClient(timeout=httpx.Timeout(9.0, connect=2.5)) as client:
                r = await client.post(
                    f"{LLM_BASE_URL.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"},
                    json={
                        "model": LLM_MODEL,
                        "temperature": 0,
                        "response_format": {"type": "json_object"},
                        "messages": [{"role": "user", "content": prompt}]
                    }
                )
                if r.status_code == 200:
                    data = r.json()["choices"][0]["message"]["content"]
                    parsed = json.loads(data)
                    rc = parsed.get("rootCause")
                    if rc in allowed_causes:
                        plan["rootCause"] = rc
                    valid_ev = [x for x in parsed.get("evidence", []) if x in all_ev_ids]
                    if len(valid_ev) >= 2:
                        plan["evidence"] = valid_ev[:4]

                    tool_n = parsed.get("diagnosticTool")
                    matched_t = next((t for t in diag_catalog if t["name"] == tool_n), None)
                    if matched_t:
                        t_args = parsed.get("diagnosticArgs") or build_tool_arguments(matched_t.get("inputSchema", {}), inc, transcript)
                        plan["diagnostics"] = [{
                            "toolName": matched_t["name"],
                            "arguments": t_args,
                            "evidence": [plan["evidence"][0]]
                        }]
        except Exception as e:
            print(f"LLM refine exception: {e}", flush=True)

    return plan, model_span


def add_client_attempt(state, logical, attempt, action_id, tool, call_id):
    client_id = hx_span()
    logical_id = logical["logicalSpanId"]
    d = {
        "actionId": action_id,
        "callId": call_id,
        "phase": logical["phase"],
        "toolName": tool,
        "arguments": logical["arguments"],
        "evidence": logical["evidence"],
        "attempt": int(attempt),
        "traceparent": f"00-{state['_traceId']}-{client_id}-01",
    }
    if logical.get("approval"):
        d["approvalId"] = logical["approval"]["approvalId"]
        d["approvalNonce"] = logical["approval"]["nonce"]

    logical["currentClientSpanId"] = client_id
    logical["attempt"] = int(attempt)
    logical["dispatch"] = d
    state.setdefault("actionLog", []).append(d)

    state["_spans"].append(span(
        state["_traceId"], client_id, logical_id, f"POST tool/{tool}", 3,
        state["runId"], state["publicMarker"],
        {
            "ga5.action.id": action_id,
            "ga5.attempt": int(attempt),
            "http.request.method": "POST",
            "http.request.resend_count": int(attempt) - 1
        },
        status_code=0
    ))
    return d


def dispatch(state, tool, args, phase, evidence, attempt=1, action_id=None, call_id=None, approval=None):
    aid = action_id or f"act_{hx_span()}"
    cid = call_id or aid
    logical_id = hx_span()
    logical = {
        "actionId": aid,
        "callId": cid,
        "toolName": tool,
        "arguments": args,
        "phase": phase,
        "evidence": evidence,
        "attempt": int(attempt),
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
@app.get("/health")
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
    req_hash = sha({k: v for k, v in req.items() if k != "sensitive"})

    # Exact replay check & 409 conflict
    with db() as c:
        row = c.execute("SELECT request_hash, initial_response FROM runs WHERE run_id=?", (rid,)).fetchone()
        if row:
            if row["request_hash"] != req_hash:
                raise HTTPException(409, "runId content conflict")
            return JSONResponse(json.loads(row["initial_response"]))

    incoming_tp = request.headers.get("traceparent", "")
    in_trace_id, in_parent_id = parse_traceparent(incoming_tp)
    tracestate = request.headers.get("tracestate")

    trace_id = in_trace_id or hx_trace()
    server_id = hx_span()
    agent_id = hx_span()

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
        "_traceId": trace_id,
        "_agentSpanId": agent_id,
        "_spans": [],
        "_req": {k: v for k, v in req.items() if k != "sensitive"},
        "pending": {},
        "plan": None,
    }

    server = span(trace_id, server_id, in_parent_id, "POST /v2/incidents", 2, rid, state["publicMarker"])
    if tracestate:
        server["attributes"].append(attr("tracestate", tracestate))
    state["_spans"].append(server)
    state["_spans"].append(span(trace_id, agent_id, server_id, "invoke_agent incident-response", 1, rid, state["publicMarker"]))

    plan, model_span = await plan_incident(req, trace_id, agent_id)
    state["_spans"].append(model_span)
    state["plan"] = plan
    state["diagnosis"] = {"rootCause": plan["rootCause"], "evidence": plan["evidence"]}

    for item in plan["diagnostics"]:
        state["dispatches"].append(dispatch(
            state, item["toolName"], item["arguments"], "diagnostic", item["evidence"]
        ))

    if len(state["dispatches"]) > 1:
        links = [{"traceId": trace_id, "spanId": state["pending"][d["actionId"]]["logicalSpanId"]}
                 for d in state["dispatches"]]
        state["_spans"].append(span(trace_id, hx_span(), agent_id, "incident.join", 1, rid, state["publicMarker"], links=links))

    initial_resp = format_response(state)
    with db() as c:
        c.execute("INSERT INTO runs(run_id, request_hash, state_json, initial_response, current_response) VALUES(?,?,?,?,?)",
                  (rid, req_hash, json.dumps(state, separators=(",", ":")),
                   json.dumps(initial_resp, separators=(",", ":")),
                   json.dumps(initial_resp, separators=(",", ":"))))

    return JSONResponse(initial_resp)


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
            approval_id = f"appr_{hx_span()}"
            action_id = f"act_{hx_span()}"
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
                state["_traceId"], hx_span(), state["_agentSpanId"], "approval_gate", 1,
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

    with db() as c:
        row = c.execute("SELECT state_json FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Unknown runId")
    state = json.loads(row["state_json"])

    receipt_id = str(body.get("receiptId", ""))
    if not receipt_id:
        raise HTTPException(422, "receiptId required")

    body_hash = sha(body)
    with db() as c:
        previous = c.execute("SELECT run_id, content_hash, response_json FROM receipts WHERE receipt_id=?", (receipt_id,)).fetchone()
    if previous:
        if previous["run_id"] != run_id or previous["content_hash"] != body_hash:
            raise HTTPException(409, "receiptId content conflict")
        return JSONResponse(json.loads(previous["response_json"]))

    if state["status"] != "waiting":
        raise HTTPException(422, "No pending work")

    outcomes = body.get("outcomes", [])
    approvals = body.get("approvals", [])

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
                "attempt": int(pending["attempt"]),
                "status": int(status),
                "resultClass": outcome.get("resultClass", ""),
                "nonce": nonce,
                **({"errorType": "timeout"} if status == 0 and outcome.get("errorType") == "timeout" else {})
            })

            client_span = next((s for s in state["_spans"] if s["spanId"] == client_id), None)
            if client_span:
                client_span["attributes"].extend([
                    attr("ga5.receipt.id", receipt_id),
                    attr("ga5.receipt.nonce", nonce),
                    attr("http.response.status_code", int(status)),
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

        state["dispatches"] = [p["dispatch"] for p in state["pending"].values() if p["awaiting"]]
        maybe_advance(state)

    elif approvals:
        pending_approval = state.get("pendingApproval")
        if not pending_approval:
            raise HTTPException(422, "No pending approval")
        approval = pending_approval["approval"]

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

    current_resp = format_response(state)
    with db() as c:
        c.execute("UPDATE runs SET state_json=?, current_response=? WHERE run_id=?",
                  (json.dumps(state, separators=(",", ":")), json.dumps(current_resp, separators=(",", ":")), run_id))
        c.execute("INSERT INTO receipts(receipt_id, run_id, content_hash, response_json) VALUES(?,?,?,?)",
                  (receipt_id, run_id, body_hash, json.dumps(current_resp, separators=(",", ":"))))

    return JSONResponse(current_resp)


@app.get("/v2/incidents/{run_id}")
async def get_incident(run_id: str):
    with db() as c:
        row = c.execute("SELECT current_response FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Unknown runId")
    return JSONResponse(json.loads(row["current_response"]))
