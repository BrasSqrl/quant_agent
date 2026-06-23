from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from json import JSONDecodeError
from pathlib import Path
from typing import Any
from uuid import uuid4

from quant_agent_runtime.contracts import QuantSuiteContractLoader
from quant_agent_runtime.ledger import InMemoryLedger
from quant_agent_runtime.models import (
    GovernanceSummary,
    PlanValidationResult,
    SeparationOfDutiesSummary,
    ValidationIssue,
)
from quant_agent_runtime.redaction import find_unsafe_payload_issues
from quant_agent_runtime.validation.errors import RuntimeValidationError


GOVERNANCE_POLICY_PACK_EXAMPLE = "agent_governance_policy_pack.v1.example.json"
GOVERNANCE_POLICY_PACK_SCHEMA = "agent_governance_policy_pack.v1.schema.json"

READ_ROUTES = [
    "GET /health",
    "GET /runtime/manifest",
    "GET /runs",
    "GET /runs/{run_id}",
    "GET /runs/{run_id}/orchestration",
    "GET /runs/{run_id}/ledger",
    "GET /runs/{run_id}/demo-narrative",
]

MUTATING_ROUTES = [
    "POST /plans",
    "POST /preflights",
    "POST /confirmations",
    "POST /action-requests",
    "POST /executions",
    "POST /retries",
    "POST /cancellations",
    "POST /pauses",
    "POST /resumptions",
    "POST /plan-revisions",
    "POST /plan-revision-activations",
    "POST /run-revalidations",
    "POST /autopilot-previews",
    "POST /autopilot-steps",
    "POST /sample-reset-previews",
    "POST /sample-resets",
    "POST /user-plan-reviews",
    "POST /user-plan-approvals",
    "POST /user-workflow-readiness",
    "POST /user-workflow-consents",
]

ALL_ROUTES = [*READ_ROUTES, *MUTATING_ROUTES]
SOD_DENIAL_EVENT_TYPE = "governance_separation_of_duties_denied"
SOD_SUPPORT_LEVEL = "role_aware_blocking_with_local_exemption"


@dataclass(frozen=True)
class GovernanceDecision:
    allowed: bool
    route: str
    capability_id: str | None
    actor_role: str
    effective_actor_role: str
    policy_pack_id: str
    reason: str


@dataclass(frozen=True)
class SeparationOfDutiesDecision:
    allowed: bool
    route: str
    capability_id: str | None
    actor_id: str
    actor_role: str
    effective_actor_role: str
    policy_pack_id: str
    reason: str
    rule_id: str | None = None
    prior_event_type: str | None = None
    prior_actor_id: str | None = None


class GovernanceService:
    def __init__(
        self,
        *,
        ledger: InMemoryLedger,
        policy_pack: dict[str, Any],
        source: str,
        actor_id: str,
        actor_role: str,
        environment: str,
        fallback_active: bool = False,
        fallback_reason: str | None = None,
        diagnostics: list[dict[str, Any]] | None = None,
    ) -> None:
        self._ledger = ledger
        self._policy_pack = policy_pack
        self._source = source
        self._environment = environment
        self._actor_id = _safe_actor_label(actor_id, fallback="local_user")
        self._requested_actor_role = actor_role
        self._fallback_active = fallback_active
        self._fallback_reason = fallback_reason
        self._diagnostics = diagnostics or []
        self._effective_actor_role = self._resolve_effective_role(actor_role)

    @classmethod
    def from_contracts(
        cls,
        *,
        ledger: InMemoryLedger,
        contract_loader: QuantSuiteContractLoader,
    ) -> "GovernanceService":
        override_path = os.environ.get("QUANT_AGENT_GOVERNANCE_POLICY_PACK_PATH")
        environment = os.environ.get("QUANT_AGENT_GOVERNANCE_ENVIRONMENT", "local_development")
        source = "QUANT_AGENT_GOVERNANCE_POLICY_PACK_PATH" if override_path else contract_loader.source_label
        path = Path(override_path) if override_path else contract_loader.contract_file(GOVERNANCE_POLICY_PACK_EXAMPLE)
        payload, fallback_reason, diagnostics = _load_policy_pack(path)
        fallback_active = payload is None
        if payload is None:
            payload = _fallback_policy_pack()
        else:
            try:
                contract_loader.validate_agent_contract_payload(payload, GOVERNANCE_POLICY_PACK_SCHEMA)
            except Exception:
                payload = _fallback_policy_pack()
                fallback_active = True
                fallback_reason = "invalid_policy_pack"
                diagnostics.append(
                    {
                        "code": "invalid_governance_policy_pack",
                        "message": "The governance policy pack could not be validated; local fallback is active.",
                    }
                )
            unsafe_issues = find_unsafe_payload_issues(payload, root="governance_policy_pack")
            if unsafe_issues:
                payload = _fallback_policy_pack()
                fallback_active = True
                fallback_reason = "unsafe_policy_pack"
                diagnostics.append(
                    {
                        "code": "unsafe_governance_policy_pack",
                        "message": "The governance policy pack contained unsafe fields or values; local fallback is active.",
                    }
                )

        actor_role = os.environ.get(
            "QUANT_AGENT_ACTOR_ROLE",
            str(payload.get("default_actor_role") or "local_developer_operator"),
        )
        actor_id = os.environ.get("QUANT_AGENT_ACTOR_ID", "local_user")
        service = cls(
            ledger=ledger,
            policy_pack=payload,
            source=source,
            actor_id=actor_id,
            actor_role=actor_role,
            environment=environment,
            fallback_active=fallback_active,
            fallback_reason=fallback_reason,
            diagnostics=diagnostics,
        )
        if service._effective_actor_role != actor_role:
            service._fallback_active = True
            service._fallback_reason = "unknown_role"
            service._diagnostics = [
                *service._diagnostics,
                {
                    "code": "unknown_governance_actor_role",
                    "message": "The configured actor role was not found; default local role permissions are active.",
                },
            ]
        return service

    @classmethod
    def local_fallback(cls, *, ledger: InMemoryLedger) -> "GovernanceService":
        return cls(
            ledger=ledger,
            policy_pack=_fallback_policy_pack(),
            source="internal_local_fallback",
            actor_id=os.environ.get("QUANT_AGENT_ACTOR_ID", "local_user"),
            actor_role=os.environ.get("QUANT_AGENT_ACTOR_ROLE", "local_developer_operator"),
            environment=os.environ.get("QUANT_AGENT_GOVERNANCE_ENVIRONMENT", "local_development"),
            fallback_active=True,
            fallback_reason="not_configured",
            diagnostics=[
                {
                    "code": "governance_local_fallback",
                    "message": "Local fallback governance policy is active.",
                }
            ],
        )

    @property
    def support_level(self) -> str:
        return "role_aware_policy_pack_enforced"

    @property
    def policy_pack_id(self) -> str:
        return str(self._policy_pack.get("policy_pack_id") or "unknown_policy_pack")

    @property
    def actor_role(self) -> str:
        return self._requested_actor_role

    @property
    def actor_id(self) -> str:
        return self._actor_id

    @property
    def effective_actor_role(self) -> str:
        return self._effective_actor_role

    def manifest_summary(self) -> GovernanceSummary:
        route_permissions = _permissions_for_role(
            self._policy_pack.get("route_permissions"),
            self._effective_actor_role,
        )
        capability_permissions = _permissions_for_role(
            self._policy_pack.get("capability_permissions"),
            self._effective_actor_role,
        )
        return GovernanceSummary(
            support_level=self.support_level,
            policy_pack_id=self.policy_pack_id,
            environment=self._environment,
            actor_id=self._actor_id,
            actor_role=self._requested_actor_role,
            effective_actor_role=self._effective_actor_role,
            source=self._source,
            fallback_active=self._fallback_active,
            fallback_reason=self._fallback_reason,
            allowed_routes=list(route_permissions.get("allowed_routes", [])),
            denied_routes=list(route_permissions.get("denied_routes", [])),
            allowed_capability_ids=list(capability_permissions.get("allowed_capability_ids", [])),
            denied_capability_ids=list(capability_permissions.get("denied_capability_ids", [])),
            diagnostics=self._diagnostics,
        )

    def run_summary(self, run_id: str | None = None) -> GovernanceSummary:
        return self.manifest_summary().model_copy(update={"run_id": run_id})

    @property
    def separation_of_duties_support_level(self) -> str:
        return SOD_SUPPORT_LEVEL

    def actor_summary(self) -> dict[str, Any]:
        return current_governance_actor(
            actor_id=self._actor_id,
            actor_role=self._requested_actor_role,
            effective_actor_role=self._effective_actor_role,
            policy_pack_id=self.policy_pack_id,
        )

    def separation_of_duties_manifest_summary(self) -> SeparationOfDutiesSummary:
        return self.separation_of_duties_run_summary()

    def separation_of_duties_run_summary(self, run_id: str | None = None) -> SeparationOfDutiesSummary:
        blocked_routes: list[str] = []
        blocker_reason: str | None = None
        if run_id:
            for route in self._sod_protected_routes():
                decision = self.evaluate_separation_of_duties(route=route, run_id=run_id)
                if not decision.allowed:
                    blocked_routes.append(route)
                    blocker_reason = blocker_reason or decision.reason
        return SeparationOfDutiesSummary(
            support_level=self.separation_of_duties_support_level,
            run_id=run_id,
            actor_id=self._actor_id,
            actor_role=self._requested_actor_role,
            effective_actor_role=self._effective_actor_role,
            actor_exempt=self._actor_exempt_from_sod(),
            active_rule_ids=[_safe_string(rule.get("rule_id")) for rule in self._active_sod_rules()],
            exempt_roles=self._sod_exempt_roles(),
            protected_routes=self._sod_protected_routes(),
            blocked_routes=blocked_routes,
            blocked=bool(blocked_routes),
            blocker_reason=blocker_reason,
            latest_denial=self._latest_sod_denial(run_id) if run_id else None,
            diagnostics=[],
        )

    def require_allowed(
        self,
        *,
        route: str,
        run_id: str | None = None,
        step_id: str | None = None,
        capability_id: str | None = None,
    ) -> None:
        resolved_capability_id = capability_id
        if resolved_capability_id is None and run_id and step_id:
            resolved_capability_id = self._capability_id_for_step(run_id, step_id)
        decision = self.evaluate(route=route, capability_id=resolved_capability_id)
        if not decision.allowed:
            if run_id:
                self._ledger_denial(run_id, decision, step_id=step_id)
            raise _permission_denied(decision, step_id=step_id)
        sod_decision = self.evaluate_separation_of_duties(
            route=route,
            run_id=run_id,
            step_id=step_id,
            capability_id=resolved_capability_id,
        )
        if not sod_decision.allowed:
            if run_id:
                self._ledger_sod_denial(run_id, sod_decision, step_id=step_id)
            raise _separation_of_duties_denied(sod_decision, step_id=step_id)

    def evaluate(self, *, route: str, capability_id: str | None = None) -> GovernanceDecision:
        route_allowed, route_reason = self._route_allowed(route)
        if not route_allowed:
            return GovernanceDecision(
                allowed=False,
                route=route,
                capability_id=capability_id,
                actor_role=self._requested_actor_role,
                effective_actor_role=self._effective_actor_role,
                policy_pack_id=self.policy_pack_id,
                reason=route_reason,
            )
        if capability_id:
            capability_allowed, capability_reason = self._capability_allowed(capability_id)
            if not capability_allowed:
                return GovernanceDecision(
                    allowed=False,
                    route=route,
                    capability_id=capability_id,
                    actor_role=self._requested_actor_role,
                    effective_actor_role=self._effective_actor_role,
                    policy_pack_id=self.policy_pack_id,
                    reason=capability_reason,
                )
        return GovernanceDecision(
            allowed=True,
            route=route,
            capability_id=capability_id,
            actor_role=self._requested_actor_role,
            effective_actor_role=self._effective_actor_role,
            policy_pack_id=self.policy_pack_id,
            reason="allowed",
        )

    def evaluate_separation_of_duties(
        self,
        *,
        route: str,
        run_id: str | None = None,
        step_id: str | None = None,
        capability_id: str | None = None,
    ) -> SeparationOfDutiesDecision:
        protected_routes = self._sod_protected_routes()
        if route not in protected_routes:
            return self._sod_decision(True, route, capability_id, "route_not_protected_by_separation_of_duties")
        active_rules = [rule for rule in self._active_sod_rules() if route in _safe_string_list(rule.get("protected_routes"))]
        if not active_rules:
            return self._sod_decision(True, route, capability_id, "no_active_separation_of_duties_rule")
        if self._actor_exempt_from_sod():
            return self._sod_decision(True, route, capability_id, "actor_role_exempt_from_separation_of_duties")
        if not run_id:
            return self._sod_decision(True, route, capability_id, "run_not_available_for_separation_of_duties")
        entry = self._ledger.get(run_id)
        if entry is None:
            return self._sod_decision(True, route, capability_id, "run_not_found_for_separation_of_duties")
        violation = self._sod_violation(entry, step_id=step_id, capability_id=capability_id)
        if violation is None:
            return self._sod_decision(True, route, capability_id, "separation_of_duties_satisfied")
        rule_id = _safe_string(active_rules[0].get("rule_id"))
        return self._sod_decision(
            False,
            route,
            capability_id,
            violation["reason"],
            rule_id=rule_id,
            prior_event_type=violation["prior_event_type"],
            prior_actor_id=violation.get("prior_actor_id"),
        )

    def _route_allowed(self, route: str) -> tuple[bool, str]:
        permissions = _permissions_for_role(
            self._policy_pack.get("route_permissions"),
            self._effective_actor_role,
        )
        allowed_routes = set(permissions.get("allowed_routes", []))
        denied_routes = set(permissions.get("denied_routes", []))
        if "*" in denied_routes or route in denied_routes:
            return False, "route_denied_by_governance_policy"
        if "*" in allowed_routes or route in allowed_routes:
            return True, "route_allowed_by_governance_policy"
        return False, "route_not_allowed_by_governance_policy"

    def _capability_allowed(self, capability_id: str) -> tuple[bool, str]:
        permissions = _permissions_for_role(
            self._policy_pack.get("capability_permissions"),
            self._effective_actor_role,
        )
        allowed_capabilities = set(permissions.get("allowed_capability_ids", []))
        denied_capabilities = set(permissions.get("denied_capability_ids", []))
        if "*" in denied_capabilities or capability_id in denied_capabilities:
            return False, "capability_denied_by_governance_policy"
        if "*" in allowed_capabilities or capability_id in allowed_capabilities:
            return True, "capability_allowed_by_governance_policy"
        return False, "capability_not_allowed_by_governance_policy"

    def _sod_decision(
        self,
        allowed: bool,
        route: str,
        capability_id: str | None,
        reason: str,
        *,
        rule_id: str | None = None,
        prior_event_type: str | None = None,
        prior_actor_id: str | None = None,
    ) -> SeparationOfDutiesDecision:
        return SeparationOfDutiesDecision(
            allowed=allowed,
            route=route,
            capability_id=capability_id,
            actor_id=self._actor_id,
            actor_role=self._requested_actor_role,
            effective_actor_role=self._effective_actor_role,
            policy_pack_id=self.policy_pack_id,
            reason=reason,
            rule_id=rule_id,
            prior_event_type=prior_event_type,
            prior_actor_id=prior_actor_id,
        )

    def _active_sod_rules(self) -> list[dict[str, Any]]:
        rules = self._policy_pack.get("separation_of_duties_rules")
        if not isinstance(rules, list):
            return []
        return [
            rule
            for rule in rules
            if isinstance(rule, dict)
            and rule.get("enforcement_mode") == "blocking"
            and _safe_string(rule.get("rule_id"))
        ]

    def _sod_exempt_roles(self) -> list[str]:
        roles: list[str] = []
        for rule in self._active_sod_rules():
            for role in _safe_string_list(rule.get("exempt_roles")):
                if role not in roles:
                    roles.append(role)
        return roles

    def _sod_protected_routes(self) -> list[str]:
        routes: list[str] = []
        for rule in self._active_sod_rules():
            for route in _safe_string_list(rule.get("protected_routes")):
                if route not in routes:
                    routes.append(route)
        return routes

    def _actor_exempt_from_sod(self) -> bool:
        exempt_roles = set(self._sod_exempt_roles())
        return self._effective_actor_role in exempt_roles or self._requested_actor_role in exempt_roles

    def _sod_violation(
        self,
        entry: Any,
        *,
        step_id: str | None,
        capability_id: str | None,
    ) -> dict[str, str | None] | None:
        active_plan_id = _active_plan_id(entry)
        for event in reversed(entry.recovery_events):
            if not isinstance(event, dict):
                continue
            event_type = event.get("event_type")
            if (
                event_type == "user_plan_approval"
                and event.get("status") == "approved"
                and _safe_string(event.get("plan_id")) == active_plan_id
            ):
                violation = self._sod_actor_violation(event, "user_plan_approval")
                if violation is not None:
                    return violation
            if event_type == "user_workflow_consent" and event.get("status") == "consented":
                violation = self._sod_actor_violation(event, "user_workflow_consent")
                if violation is not None:
                    return violation
        if step_id:
            for record in reversed(entry.confirmation_records):
                if not isinstance(record, dict):
                    continue
                if (
                    record.get("step_id") == step_id
                    and (capability_id is None or record.get("capability_id") == capability_id)
                    and record.get("status") == "confirmed"
                ):
                    violation = self._sod_actor_violation(record, "confirmation")
                    if violation is not None:
                        return violation
        return None

    def _sod_actor_violation(self, record: dict[str, Any], prior_event_type: str) -> dict[str, str | None] | None:
        actor = record.get("governance_actor")
        prior_actor_id = actor.get("actor_id") if isinstance(actor, dict) else None
        if not isinstance(prior_actor_id, str) or not prior_actor_id:
            return {
                "reason": "prior_gate_actor_unknown",
                "prior_event_type": prior_event_type,
                "prior_actor_id": None,
            }
        if prior_actor_id == self._actor_id:
            return {
                "reason": "same_actor_performed_prior_governance_gate",
                "prior_event_type": prior_event_type,
                "prior_actor_id": prior_actor_id,
            }
        return None

    def _latest_sod_denial(self, run_id: str | None) -> dict[str, Any] | None:
        if not run_id:
            return None
        entry = self._ledger.get(run_id)
        if entry is None:
            return None
        for event in reversed(entry.recovery_events):
            if isinstance(event, dict) and event.get("event_type") == SOD_DENIAL_EVENT_TYPE:
                return event
        return None

    def _capability_id_for_step(self, run_id: str, step_id: str) -> str | None:
        entry = self._ledger.get(run_id)
        if entry is None:
            return None
        plan = entry.plan_snapshot if isinstance(entry.plan_snapshot, dict) else {}
        steps = plan.get("proposed_steps")
        if not isinstance(steps, list):
            return None
        for step in steps:
            if not isinstance(step, dict):
                continue
            if step.get("step_id") == step_id and isinstance(step.get("capability_id"), str):
                return step["capability_id"]
        return None

    def _ledger_denial(
        self,
        run_id: str,
        decision: GovernanceDecision,
        *,
        step_id: str | None,
    ) -> None:
        if self._ledger.get(run_id) is None:
            return
        record = {
            "recovery_event_id": f"recovery_{uuid4().hex[:12]}",
            "event_type": "governance_permission_denied",
            "status": "denied",
            "actor": "local_user",
            "governance_actor": self.actor_summary(),
            "actor_role": decision.actor_role,
            "effective_actor_role": decision.effective_actor_role,
            "actor_id": self._actor_id,
            "policy_pack_id": decision.policy_pack_id,
            "denied_route": decision.route,
            "denied_action": decision.route,
            "step_id": step_id,
            "capability_id": decision.capability_id,
            "reason": decision.reason,
            "denied_at_utc": _utc_now_label(),
            "execution_permitted": False,
        }
        unsafe_issues = find_unsafe_payload_issues(record, root="governance_denial")
        if unsafe_issues:
            return
        try:
            self._ledger.append_recovery_event(run_id, record)
        except (KeyError, ValueError):
            return

    def _ledger_sod_denial(
        self,
        run_id: str,
        decision: SeparationOfDutiesDecision,
        *,
        step_id: str | None,
    ) -> None:
        if self._ledger.get(run_id) is None:
            return
        record = {
            "recovery_event_id": f"recovery_{uuid4().hex[:12]}",
            "event_type": SOD_DENIAL_EVENT_TYPE,
            "status": "denied",
            "actor": "local_user",
            "governance_actor": self.actor_summary(),
            "actor_id": decision.actor_id,
            "actor_role": decision.actor_role,
            "effective_actor_role": decision.effective_actor_role,
            "policy_pack_id": decision.policy_pack_id,
            "rule_id": decision.rule_id,
            "denied_route": decision.route,
            "denied_action": decision.route,
            "step_id": step_id,
            "capability_id": decision.capability_id,
            "reason": decision.reason,
            "prior_event_type": decision.prior_event_type,
            "prior_actor_id": decision.prior_actor_id,
            "denied_at_utc": _utc_now_label(),
            "execution_permitted": False,
        }
        unsafe_issues = find_unsafe_payload_issues(record, root="governance_separation_of_duties_denial")
        if unsafe_issues:
            return
        try:
            self._ledger.append_recovery_event(run_id, record)
        except (KeyError, ValueError):
            return

    def _resolve_effective_role(self, actor_role: str) -> str:
        roles = _role_ids(self._policy_pack.get("roles"))
        if actor_role in roles:
            return actor_role
        default_role = str(self._policy_pack.get("default_actor_role") or "local_developer_operator")
        if default_role in roles:
            return default_role
        return "local_developer_operator"


def _load_policy_pack(path: Path) -> tuple[dict[str, Any] | None, str | None, list[dict[str, Any]]]:
    if not path.is_file():
        return (
            None,
            "missing_policy_pack",
            [
                {
                    "code": "missing_governance_policy_pack",
                    "message": "The governance policy pack was not found; local fallback is active.",
                }
            ],
        )
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, JSONDecodeError):
        return (
            None,
            "unreadable_policy_pack",
            [
                {
                    "code": "unreadable_governance_policy_pack",
                    "message": "The governance policy pack could not be read; local fallback is active.",
                }
            ],
        )
    if not isinstance(payload, dict):
        return (
            None,
            "invalid_policy_pack",
            [
                {
                    "code": "invalid_governance_policy_pack",
                    "message": "The governance policy pack must be a JSON object; local fallback is active.",
                }
            ],
        )
    return payload, None, []


def _fallback_policy_pack() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "data_policy": "summaries_and_references_only",
        "policy_pack_id": "local_governance_fallback_policy_pack_v1",
        "environment_label": "local_development",
        "default_actor_role": "local_developer_operator",
        "roles": [
            {
                "role_id": "local_developer_operator",
                "display_name": "Local developer operator",
                "description": "Fallback role that preserves existing local behavior.",
                "actor_kind": "local_user",
            }
        ],
        "route_permissions": [
            {
                "role_id": "local_developer_operator",
                "allowed_routes": ["*"],
                "denied_routes": [],
            }
        ],
        "capability_permissions": [
            {
                "role_id": "local_developer_operator",
                "allowed_capability_ids": ["*"],
                "denied_capability_ids": [],
            }
        ],
        "audit_requirements": {
            "ledger_denials": True,
            "redaction_policy": "summaries_and_references_only",
            "actor_label_policy": "local_user_with_configured_role",
        },
        "fallback_behavior": {
            "missing_pack": "allow_existing_local_behavior_with_diagnostics",
            "invalid_pack": "allow_existing_local_behavior_with_diagnostics",
            "unknown_role": "allow_existing_local_behavior_with_diagnostics",
        },
        "separation_of_duties_rules": [],
    }


def _permissions_for_role(payload: Any, role_id: str) -> dict[str, Any]:
    if not isinstance(payload, list):
        return {}
    for item in payload:
        if isinstance(item, dict) and item.get("role_id") == role_id:
            return item
    return {}


def _role_ids(payload: Any) -> set[str]:
    if not isinstance(payload, list):
        return set()
    return {
        str(item.get("role_id"))
        for item in payload
        if isinstance(item, dict) and isinstance(item.get("role_id"), str)
    }


def current_governance_actor(
    *,
    actor_id: str | None = None,
    actor_role: str | None = None,
    effective_actor_role: str | None = None,
    policy_pack_id: str | None = None,
) -> dict[str, Any]:
    configured_actor_role = os.environ.get("QUANT_AGENT_ACTOR_ROLE", "local_developer_operator")
    return {
        "actor_id": _safe_actor_label(actor_id or os.environ.get("QUANT_AGENT_ACTOR_ID"), fallback="local_user"),
        "actor_role": _safe_actor_label(actor_role or configured_actor_role, fallback="local_developer_operator"),
        "effective_actor_role": _safe_actor_label(
            effective_actor_role or actor_role or configured_actor_role,
            fallback="local_developer_operator",
        ),
        "actor_kind": "local_user",
        "policy_pack_id": _safe_actor_label(policy_pack_id, fallback="unknown_policy_pack") if policy_pack_id else None,
    }


def _active_plan_id(entry: Any) -> str | None:
    snapshot = entry.plan_snapshot if isinstance(getattr(entry, "plan_snapshot", None), dict) else {}
    return _safe_string(snapshot.get("plan_id")) if isinstance(snapshot, dict) else None


def _safe_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _safe_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _safe_actor_label(value: Any, *, fallback: str) -> str:
    if not isinstance(value, str):
        return fallback
    stripped = value.strip()
    if not stripped or len(stripped) > 80:
        return fallback
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.:")
    if any(character not in allowed for character in stripped):
        return fallback
    return stripped


def _permission_denied(
    decision: GovernanceDecision,
    *,
    step_id: str | None,
) -> RuntimeValidationError:
    return RuntimeValidationError(
        PlanValidationResult(
            status="rejected",
            errors=[
                ValidationIssue(
                    code="governance_permission_denied",
                    message="The active governance policy does not permit this actor role to use the requested route or capability.",
                    step_id=step_id,
                    capability_id=decision.capability_id,
                )
            ],
        )
    )


def _separation_of_duties_denied(
    decision: SeparationOfDutiesDecision,
    *,
    step_id: str | None,
) -> RuntimeValidationError:
    message = "The active governance policy requires a different actor for this execution or retry step."
    if decision.reason == "prior_gate_actor_unknown":
        message = "The active governance policy cannot verify the prior gate actor, so execution or retry is blocked."
    return RuntimeValidationError(
        PlanValidationResult(
            status="rejected",
            errors=[
                ValidationIssue(
                    code="governance_separation_of_duties_denied",
                    message=message,
                    step_id=step_id,
                    capability_id=decision.capability_id,
                )
            ],
        )
    )


def _utc_now_label() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
