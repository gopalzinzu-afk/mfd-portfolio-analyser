"""
MFD Portfolio Analyser — Backend Server
Python 3.9+ · Flask · Runs on port 8000

This server acts as a secure proxy between your frontend and the
Anthropic Claude API. The API key never leaves this server —
the browser only calls /api/analyse on your own domain.

IMPORTANT — file handling:
The Claude API's `document` content block only accepts application/pdf
as a base64 media type. Spreadsheets (.xlsx, .xls, .csv) are NOT a
supported document media type and will be rejected with a 400 error
if sent that way (this is documented Anthropic behaviour, not a bug
on our side).

So this server does the conversion itself:
  - .xlsx / .xls  -> read every sheet with pandas, convert to a
                      tab-separated text dump, send as a `text` block
  - .csv          -> read with pandas, convert to text, send as a
                      `text` block
  - .txt / .md    -> read as plain text, send as a `text` block
  - .pdf          -> sent through unchanged as a `document` block
                      (this is the one format the API accepts as base64)

IMPORTANT — JSON reliability:
For large portfolios (many investors, 50-70+ holdings), the model
occasionally drops a comma between consecutive array elements when
free-generating a long JSON response — a known LLM failure mode, not
a truncation issue. This is fixed at the source using Anthropic's
Structured Outputs feature (output_config.format = json_schema),
which compiles REVIEW_SCHEMA below into a generation grammar. With
this enabled, the model is structurally incapable of producing
syntactically invalid JSON — malformed commas/brackets become
impossible rather than just unlikely.

Usage:
  pip install flask flask-cors anthropic pandas openpyxl
  export ANTHROPIC_API_KEY="sk-ant-..."
  python server.py
"""

import os
import io
import json
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
import anthropic
import pandas as pd

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)  # allow requests from the same origin / dev localhost

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Max rows per sheet sent to Claude. Portfolio valuation files are
# normally well under this; this guard just stops a runaway file from
# blowing the context window.
MAX_ROWS_PER_SHEET = 2000

# Reused status/verdict enums — kept as named constants so the schema
# stays readable and the same enum can't drift between sections.
TRAFFIC_LIGHT = {"type": "string", "enum": ["GREEN", "AMBER", "RED"]}
VERDICT = {"type": "string", "enum": ["OUTPERFORM", "UNDERPERFORM", "NOT_AVAILABLE"]}
PRIORITY = {"type": "string", "enum": ["MUST", "SHOULD", "CONSIDER"]}

# JSON Schema for the full review. Deliberately flat (max 3 levels of
# nesting: root -> sectionX -> array item) and every property is
# listed in "required" with additionalProperties: false, per
# Anthropic's structured-outputs guidance — optional properties roughly
# double the grammar's state space, and unconstrained objects are
# rejected outright in strict mode.
REVIEW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["client", "reviewDate", "refDate", "arnHolder", "dataSource",
                 "sectionA", "sectionB", "sectionC", "sectionD", "sectionE", "disclaimer"],
    "properties": {
        "client": {"type": "string"},
        "reviewDate": {"type": "string"},
        "refDate": {"type": "string"},
        "arnHolder": {"type": "string"},
        "dataSource": {"type": "string"},
        "sectionA": {
            "type": "object",
            "additionalProperties": False,
            "required": ["overallVerdict", "trafficLight", "investors"],
            "properties": {
                "overallVerdict": {"type": "string"},
                "trafficLight": TRAFFIC_LIGHT,
                "investors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["name", "verdict", "status"],
                        "properties": {
                            "name": {"type": "string"},
                            "verdict": {"type": "string"},
                            "status": TRAFFIC_LIGHT,
                        },
                    },
                },
            },
        },
        "sectionB": {
            "type": "object",
            "additionalProperties": False,
            "required": ["allocation"],
            "properties": {
                "allocation": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["investor", "equity", "hybrid", "debt", "gold",
                                     "other", "totalValue", "note", "status"],
                        "properties": {
                            "investor": {"type": "string"},
                            "equity": {"type": "number"},
                            "hybrid": {"type": "number"},
                            "debt": {"type": "number"},
                            "gold": {"type": "number"},
                            "other": {"type": "number"},
                            "totalValue": {"type": "number"},
                            "note": {"type": "string"},
                            "status": TRAFFIC_LIGHT,
                        },
                    },
                },
            },
        },
        "sectionC": {
            "type": "object",
            "additionalProperties": False,
            "required": ["concentration", "rollingReturns", "overlap", "tax"],
            "properties": {
                "concentration": {"type": "array", "items": {"type": "string"}},
                "rollingReturns": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["fund", "inception",
                                     "returnFund3y", "returnBench3y", "verdict3y",
                                     "returnFund5y", "returnBench5y", "verdict5y",
                                     "returnFund10y", "returnBench10y", "verdict10y",
                                     "benchmark", "styleBox", "dataGap"],
                        "properties": {
                            "fund": {"type": "string"},
                            "inception": {"type": "string"},
                            "returnFund3y": {"type": "number"},
                            "returnBench3y": {"type": "number"},
                            "verdict3y": VERDICT,
                            "returnFund5y": {"type": "number"},
                            "returnBench5y": {"type": "number"},
                            "verdict5y": VERDICT,
                            "returnFund10y": {"type": "number"},
                            "returnBench10y": {"type": "number"},
                            "verdict10y": VERDICT,
                            "benchmark": {"type": "string"},
                            "styleBox": {"type": "string"},
                            "dataGap": {"type": "string"},
                        },
                    },
                },
                "overlap": {"type": "array", "items": {"type": "string"}},
                "tax": {"type": "array", "items": {"type": "string"}},
            },
        },
        "sectionD": {
            "type": "object",
            "additionalProperties": False,
            "required": ["actions"],
            "properties": {
                "actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["priority", "investor", "action", "rationale"],
                        "properties": {
                            "priority": PRIORITY,
                            "investor": {"type": "string"},
                            "action": {"type": "string"},
                            "rationale": {"type": "string"},
                        },
                    },
                },
            },
        },
        "sectionE": {
            "type": "object",
            "additionalProperties": False,
            "required": ["agenda"],
            "properties": {
                "agenda": {"type": "array", "items": {"type": "string"}},
            },
        },
        "disclaimer": {"type": "string"},
    },
}

SYSTEM_PROMPT = """# ROLE
You are an experienced Mutual Fund Distributor working since 1988.

# CONTEXT
You work with Mutual Funds, Life Insurance, Health Insurance, Fixed Deposits, Bonds, Equity-Commodity Trading, PMS, AIF and GIFT City products.

# TASK
Analyse the uploaded portfolio holding file and produce a structured JSON review. Be calm, factual, and professional. Never use panic or alarmist language. Reviews are calm even when findings are red.

# ANALYSIS PARAMETERS
1. Rolling returns: 3Y, 5Y, 10Y with 36-month rolling frequency from FUND INCEPTION DATE (NOT investor purchase date).
2. Reference date: last working day of prior month.
3. Compare fund rolling returns against benchmark TRI indices.
4. Rule: any fund that OUTPERFORMS its benchmark on the 10Y rolling window cannot be classified as an underperformer regardless of shorter-window results.
5. Note fund Morningstar style box and category overlap between equity funds.
6. Concentration: flag single scheme > 25% of investor portfolio; single AMC > 40%.
7. Tax: flag equity LTCG harvest candidates (> 1 year holding, partial booking under INR 1.25L annual exemption); ELSS lock-in status; debt pre-Apr-2023 units with indexation.
8. Underperformance flag is for DISCUSSION, not auto-replacement. Manager change / mandate drift / market regime must be considered before any exit.
9. Every tax observation must end with the phrase: consult your CA for personal tax position.
10. If a numeric figure is genuinely unavailable (data gap), use 0 for the number field and explain fully in dataGap — do not omit the field.

Respond with the portfolio review matching the required JSON structure. Use clear, professional language throughout. Cap MUST-priority actions at exactly 5. The disclaimer field must contain this exact text: "This portfolio review workbook is prepared by an ARN-registered Mutual Fund Distributor for discussion purposes only. It does not constitute investment advice, a solicitation to buy or sell any security, or a guarantee of returns. Mutual fund investments are subject to market risk. Past performance is not indicative of future results. Rolling returns are computed from fund inception dates using ACE MF data. All tax observations are for discussion only — consult your CA for personal tax position before taking any action. Underperformance flags are discussion triggers only and should not be treated as sell signals without considering manager change, mandate drift, and market regime." """


def spreadsheet_to_text(file_bytes: bytes, filename: str) -> str:
    """
    Convert an .xlsx/.xls/.csv file into a plain-text representation
    Claude can read as a `text` content block. Excel is not a supported
    `document` media type on the API, so this conversion is mandatory,
    not optional.
    """
    ext = filename.rsplit(".", 1)[-1].lower()
    parts = [f"=== FILE: {filename} ==="]

    if ext == "csv":
        df = pd.read_csv(io.BytesIO(file_bytes))
        if len(df) > MAX_ROWS_PER_SHEET:
            df = df.head(MAX_ROWS_PER_SHEET)
            parts.append(f"(truncated to first {MAX_ROWS_PER_SHEET} rows)")
        parts.append(df.to_csv(index=False))

    else:  # xlsx / xls
        sheets = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, engine="openpyxl")
        for sheet_name, df in sheets.items():
            parts.append(f"\n--- Sheet: {sheet_name} ---")
            if df.empty:
                parts.append("(empty sheet)")
                continue
            if len(df) > MAX_ROWS_PER_SHEET:
                df = df.head(MAX_ROWS_PER_SHEET)
                parts.append(f"(truncated to first {MAX_ROWS_PER_SHEET} rows)")
            parts.append(df.to_csv(index=False))

    return "\n".join(parts)


def build_content_blocks(files, extra_text: str):
    """
    Build the Claude API `content` array from uploaded files.

    - .pdf  -> document block (base64, application/pdf) — the only
               media type the API's document block actually accepts
    - everything else (.xlsx/.xls/.csv/.txt/.md) -> converted to plain
      text server-side and sent as a `text` block
    """
    blocks = []

    for f in files:
        filename = f.filename
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        raw = f.read()

        if ext == "pdf":
            b64 = base64.standard_b64encode(raw).decode("utf-8")
            blocks.append({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": b64,
                },
            })

        elif ext in ("xlsx", "xls"):
            text = spreadsheet_to_text(raw, filename)
            blocks.append({"type": "text", "text": text})

        elif ext == "csv":
            text = spreadsheet_to_text(raw, filename)
            blocks.append({"type": "text", "text": text})

        elif ext in ("txt", "md"):
            text = raw.decode("utf-8", errors="replace")
            blocks.append({"type": "text", "text": f"=== FILE: {filename} ===\n{text}"})

        else:
            blocks.append({
                "type": "text",
                "text": f"=== FILE: {filename} ===\n[Unsupported file type '.{ext}' — skipped. Supported: .xlsx, .xls, .csv, .pdf, .txt, .md]"
            })

    blocks.append({"type": "text", "text": extra_text})
    return blocks


@app.route("/")
def index():
    """
    Serve the frontend.

    Cache-Control is set explicitly to no-store rather than relying
    on Flask's default (no-cache, which still permits storage and
    only requires revalidation). no-store is unambiguous: nothing in
    the request path is allowed to retain a copy at all, so a
    redeploy is guaranteed to actually reach the browser on the next
    request rather than risk being served from any intermediate
    cache. This is paired with the bfcache guard in index.html's own
    pageshow handler, since back/forward cache restoration bypasses
    HTTP cache headers entirely and needs a separate fix.
    """
    response = app.send_static_file("index.html")
    response.headers["Cache-Control"] = "no-store, must-revalidate"
    return response


@app.route("/api/analyse", methods=["POST"])
def analyse():
    """
    Accepts multipart/form-data:
      - one or more files under the 'files' field
      - a 'meta' field containing JSON metadata (client name, dates, etc.)

    Converts any spreadsheet files to text server-side (the API does not
    accept .xlsx as a document block), then calls Claude.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return jsonify({"error": "ANTHROPIC_API_KEY not set on server"}), 500

    uploaded = request.files.getlist("files")
    if not uploaded:
        return jsonify({"error": "No files uploaded"}), 400

    meta_text = request.form.get("meta", "")

    try:
        content_blocks = build_content_blocks(uploaded, meta_text)
    except Exception as e:
        # A parsing failure here means the file itself is malformed —
        # this is a clear, actionable error rather than letting it
        # surface as an opaque Claude API 400 later.
        return jsonify({"error": f"Could not read uploaded file: {str(e)}"}), 400

    try:
        # 16000 tokens comfortably covers a full review for a large
        # family group (multiple investors, 50-70+ holdings, full
        # rolling-return tables per fund, and a detailed action list).
        # The model supports up to 64000 on the synchronous API, so
        # there is wide headroom here even for unusually large files.
        #
        # output_config forces Claude's response through REVIEW_SCHEMA
        # via constrained decoding, which makes syntactically invalid
        # JSON (missing commas, unclosed brackets, wrong types)
        # essentially impossible for the model to emit.
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content_blocks}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": REVIEW_SCHEMA,
                }
            },
        )

        if response.stop_reason == "max_tokens":
            return jsonify({
                "error": (
                    "Claude's response was cut off before completing "
                    "(portfolio may be very large). Try again — if this "
                    "repeats, consider splitting the holding file by "
                    "investor and analysing each separately."
                ),
                "truncated": True,
                "partial_output_tokens": response.usage.output_tokens,
            }), 200

        if response.stop_reason == "refusal":
            return jsonify({
                "error": "Claude declined to process this request. Please check the uploaded file and try again.",
                "truncated": True,
            }), 200

        # Pull the text block out of the response.
        text_block = next((b for b in response.content if b.type == "text"), None)
        if text_block is None:
            return jsonify({
                "error": "Claude's response contained no text content to parse.",
                "truncated": True,
            }), 200

        raw_text = text_block.text

        # PARSE SERVER-SIDE, with the real Python json module — not a
        # regex, and not deferred to the browser.
        #
        # With output_config active, raw_text should be EXACTLY one
        # JSON object matching REVIEW_SCHEMA, nothing more. Parsing it
        # directly (rather than regex-extracting "everything between
        # the first { and the last }") is both simpler and more
        # correct: if structured outputs is genuinely working, the
        # direct parse just succeeds. If something unexpected slipped
        # through (a known edge case in some Anthropic SDK versions,
        # see anthropic-sdk-python#1204), the direct parse fails
        # immediately and we can log/report the exact raw text rather
        # than silently shipping a broken string to the browser for
        # it to fail on with a cryptic position number.
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as e:
            # Log the full raw text server-side (Render's log viewer)
            # so this is actually diagnosable next time, instead of
            # us going back and forth guessing again.
            print("=== JSON PARSE FAILURE — RAW CLAUDE OUTPUT ===")
            print(raw_text)
            print(f"=== END RAW OUTPUT ({len(raw_text)} chars) ===")
            print(f"Parse error: {e}")

            return jsonify({
                "error": (
                    f"Claude's response could not be parsed as JSON even with "
                    f"structured outputs enabled ({e}). This has been logged "
                    f"server-side for diagnosis. Please try again — if it "
                    f"persists, this needs investigation rather than another "
                    f"prompt tweak."
                ),
                "truncated": True,
                "debug_raw_length": len(raw_text),
                "debug_parse_error": str(e),
            }), 200

        # Success: send the already-parsed object directly. The
        # frontend no longer needs to parse anything — see the
        # matching index.html change.
        return jsonify({
            "id": response.id,
            "model": response.model,
            "stop_reason": response.stop_reason,
            "review": parsed,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        })

    except anthropic.AuthenticationError:
        return jsonify({"error": "Invalid API key. Check ANTHROPIC_API_KEY."}), 401
    except anthropic.RateLimitError:
        return jsonify({"error": "Rate limit hit. Wait a moment and try again."}), 429
    except anthropic.BadRequestError as e:
        msg = str(e)
        if "too complex" in msg.lower() or "compilation" in msg.lower():
            return jsonify({
                "error": "The review schema is too complex for the API to process. This is a configuration issue, not a problem with your file — please report this."
            }), 400
        return jsonify({"error": f"Bad request: {msg}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    print(f"MFD Analyser running at http://localhost:{port}")
    print(f"API key set: {'YES' if os.environ.get('ANTHROPIC_API_KEY') else 'NO — set ANTHROPIC_API_KEY'}")
    app.run(host="0.0.0.0", port=port, debug=debug)
