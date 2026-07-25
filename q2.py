from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

# ==============================================================================
# Q2 - Spec-Driven Development: The Proration Bug
# ==============================================================================

class ProrationRequest(BaseModel):
    old_price: float
    new_price: float
    days_remaining: float
    days_in_actual_month: float
    spec: str


def compute_charge(req: ProrationRequest) -> float:
    if req.spec == "v1":
        # Legacy rule: divisor is always exactly 30, regardless of actual month length
        charge = (req.new_price - req.old_price) * (req.days_remaining / 30.0)
    elif req.spec == "v2":
        # Corrected rule: divisor is the actual number of days in the billing month
        if not req.days_in_actual_month:
            raise HTTPException(
                status_code=400,
                detail="days_in_actual_month is required for spec v2"
            )
        charge = (req.new_price - req.old_price) * (
            req.days_remaining / req.days_in_actual_month
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid spec version")

    return round(charge, 2)


# This is the exact URL the grader submits to: /ga5/{email}/proration
@router.post("/ga5/{email}/proration")
def calculate_proration_ga5(email: str, req: ProrationRequest):
    return {"charge": compute_charge(req)}


# Extra convenience routes (not required by grader, but harmless to keep)
@router.post("/q2/charge")
@router.post("/charge")
def calculate_proration(req: ProrationRequest):
    return {"charge": compute_charge(req)}