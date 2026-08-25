"""The diagnostic contract a served system offers the platform.

The platform's own pipeline is a
template that real systems are grown from, so whatever the template declares and records, every derived system
inherits for free. That is the whole reason this lives in `core/` rather
than in the gateway: a system built from the template is transparent to the
platform by construction, instead of having to be retrofitted afterwards.

Two pieces. The contract says what a system *can* report, so the platform
asks for nothing it will not get and never mistakes an unsupported feature
for a broken one. The trace log records what actually happened per query, so
a production failure can later become a golden question (phase 7) without
the platform having to be in the request path to see it.
"""
from core.diagnostics.contract import DiagnosticContract, contract_for_pipeline
from core.diagnostics.trace_log import QueryTrace, TraceLog

__all__ = ["DiagnosticContract", "contract_for_pipeline", "QueryTrace", "TraceLog"]
