"""Read-only tools the specialist agents can call.

Design rules:
- Tools never change anything. The only way to change the world is to *propose* an action via
  `submit_resolution`, which goes through the policy gate. (Least privilege + auditability.)
- Every lookup is scoped to the verified sender. Asking about someone else's order returns
  "not found", exactly like a non-existent order - no data leaks through prompt injection.
- Flaky dependencies are retried inside the tool; if they stay down, the tool returns a structured
  error the model can reason about ("carrier API unavailable") instead of crashing the workflow.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from support_agent.llm.base import ToolSpec
from support_agent.resilience import TransientError, retry_call
from support_agent.services import Services
from support_agent.util import dollars, parse_iso


@dataclass
class ToolContext:
    services: Services
    customer: dict[str, Any] | None
    case_id: str


class NoArgs(BaseModel):
    pass


class OrderArgs(BaseModel):
    order_id: str = Field(description="Order id, e.g. A1001.")


class SearchArgs(BaseModel):
    query: str = Field(description="What to look up, in plain words.")


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[ToolContext, Any], dict[str, Any]]

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.args_model.model_json_schema())

    def run(self, ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.fn(ctx, self.args_model.model_validate(arguments))


NO_ACCOUNT = {"error": "No customer account matches the sender's email address. Order and account data are unavailable."}


def _owned_order(ctx: ToolContext, order_id: str) -> dict[str, Any] | None:
    if ctx.customer is None:
        return None
    order = ctx.services.repo.order(order_id)
    if order is None or order["customer_id"] != ctx.customer["id"]:
        return None
    return order


def _not_found(order_id: str) -> dict[str, Any]:
    return {"error": f"Order {order_id} was not found on this customer's account."}


def get_customer_profile(ctx: ToolContext, _: NoArgs) -> dict[str, Any]:
    c = ctx.customer
    if c is None:
        return NO_ACCOUNT
    return {
        "name": c["name"],
        "email": c["email"],
        "tier": c["tier"],
        "member_since": c["created_at"][:10],
        "account_status": c["status"],
    }


def list_orders(ctx: ToolContext, _: NoArgs) -> dict[str, Any]:
    if ctx.customer is None:
        return NO_ACCOUNT
    orders = ctx.services.repo.list_orders(ctx.customer["id"])
    return {
        "orders": [
            {"order_id": o["id"], "placed_at": o["placed_at"][:10], "status": o["status"], "total": dollars(o["total_cents"])}
            for o in orders
        ]
    }


def get_order(ctx: ToolContext, args: OrderArgs) -> dict[str, Any]:
    if ctx.customer is None:
        return NO_ACCOUNT
    order = _owned_order(ctx, args.order_id)
    if order is None:
        return _not_found(args.order_id)
    now = ctx.services.clock()
    return {
        "order_id": order["id"],
        "placed_at": order["placed_at"][:10],
        "status": order["status"],
        "shipping_address": order["shipping_address"],
        "items": [
            {
                "sku": i["sku"],
                "name": i["name"],
                "qty": i["qty"],
                "unit_price": dollars(i["unit_price_cents"]),
                "final_sale": bool(i["final_sale"]),
            }
            for i in order["items"]
        ],
        "subtotal": dollars(order["subtotal_cents"]),
        "shipping_charge": dollars(order["shipping_cents"]),
        "total": dollars(order["total_cents"]),
        "payments": [
            {"payment_id": p["id"], "amount": dollars(p["amount_cents"]), "charged_at": p["created_at"][:10]}
            for p in order["payments"]
        ],
        "total_charged": dollars(order["paid_cents"]),
        "refunds": [
            {"amount": dollars(r["amount_cents"]), "refunded_at": r["created_at"][:10], "reason": r["reason"]}
            for r in order["refunds"]
        ],
        "total_refunded": dollars(order["refunded_cents"]),
        "shipments": [
            {
                "carrier": s["carrier"],
                "tracking_number": s["tracking_number"],
                "status": s["status"],
                "shipped_at": (s["shipped_at"] or "")[:10] or None,
                "delivered_at": (s["delivered_at"] or "")[:10] or None,
                "estimated_delivery": (s["eta"] or "")[:10] or None,
                "is_replacement": bool(s["is_replacement"]),
            }
            for s in order["shipments"]
        ],
        "days_since_delivery": (now - parse_iso(order["delivered_at"])).days if order["delivered_at"] else None,
    }


def track_shipment(ctx: ToolContext, args: OrderArgs) -> dict[str, Any]:
    if ctx.customer is None:
        return NO_ACCOUNT
    order = _owned_order(ctx, args.order_id)
    if order is None:
        return _not_found(args.order_id)
    if not order["shipments"]:
        return {"order_id": order["id"], "status": "not_shipped", "detail": "Order has not shipped yet."}
    results = []
    for s in order["shipments"]:
        try:
            tracking = retry_call(
                lambda tn=s["tracking_number"]: ctx.services.carrier.track(tn),
                retry_on=(TransientError,),
                attempts=3,
                base_delay_s=0.2,
                sleep=ctx.services.sleep,
            )
        except TransientError:
            return {
                "error": "The carrier tracking service is temporarily unavailable. Use the shipment status from "
                "get_order instead, and do not guess."
            }
        results.append({**tracking, "is_replacement": bool(s["is_replacement"])})
    return {"order_id": order["id"], "shipments": results}


def check_refund_eligibility(ctx: ToolContext, args: OrderArgs) -> dict[str, Any]:
    if ctx.customer is None:
        return NO_ACCOUNT
    order = _owned_order(ctx, args.order_id)
    if order is None:
        return _not_found(args.order_id)
    return ctx.services.policy.refund_eligibility(order)


def search_help_center(ctx: ToolContext, args: SearchArgs) -> dict[str, Any]:
    hits = ctx.services.kb.search(args.query)
    if not hits:
        return {"results": [], "note": "No help-center article matches. Do not invent policy."}
    return {"results": [{"article": h.article, "section": h.heading, "text": h.text} for h in hits]}


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool("get_customer_profile", "Profile of the customer who sent this ticket (verified by sender email).",
             NoArgs, get_customer_profile),
        Tool("list_orders", "The customer's 10 most recent orders with status and totals.", NoArgs, list_orders),
        Tool("get_order", "Full details of one of the customer's orders: items, prices, final-sale flags, payments, "
             "refunds, shipments and days since delivery.", OrderArgs, get_order),
        Tool("track_shipment", "Live carrier tracking for an order's shipments.", OrderArgs, track_shipment),
        Tool("check_refund_eligibility", "Refund facts for an order: return-window status, refundable balance, "
             "prior refunds, final-sale items.", OrderArgs, check_refund_eligibility),
        Tool("search_help_center", "Search the company help center (policies: returns, shipping, account, billing).",
             SearchArgs, search_help_center),
    ]
}
