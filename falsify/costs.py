"""The cost model. Specified by 02-ENGINE-SPEC.md Part E.

Two units live here and they are not the same, which is worth stating loudly
because the spec mixes them deliberately and a silent conversion would be
invisible in every downstream number:

    *_bps            basis points, so 5.0 means 5 bps = 0.0005
    cash_yield_annual   a plain annual rate, so 0.05 means 5%

`borrow_bps_annual` is annual basis points; `cash_yield_annual` is an annual
decimal. Part E writes them that way and both engines follow it exactly rather
than normalising, so the equations in the code read the same as the equations in
the specification.
"""

from __future__ import annotations

from dataclasses import dataclass

BPS = 10_000.0


@dataclass(frozen=True, slots=True)
class CostModel:
    """Frozen (B7). All fields default to zero so a zero-cost run is explicit.

    The cash yield term is the one everybody drops. A long/cash strategy sitting
    in cash half the time through a 5% rate environment forgoes roughly 2.5% a
    year, and omitting it understates the strategy rather than flattering it --
    the opposite of the usual direction of error.
    """

    commission_bps: float = 0.0
    half_spread_bps: float = 0.0
    slippage_bps: float = 0.0
    borrow_bps_annual: float = 0.0
    cash_yield_annual: float = 0.0

    def __post_init__(self) -> None:
        for field in (
            "commission_bps",
            "half_spread_bps",
            "slippage_bps",
            "borrow_bps_annual",
            "cash_yield_annual",
        ):
            value = getattr(self, field)
            if not (value == value) or value < 0.0:  # NaN-safe non-negativity
                raise ValueError(f"{field} must be a non-negative number, got {value!r}")

    @property
    def total_bps(self) -> float:
        """Per-side cost charged on traded notional, in basis points."""
        return self.commission_bps + self.half_spread_bps + self.slippage_bps

    def cost_rate(self) -> float:
        """`total_bps` as a fraction, i.e. what multiplies traded notional."""
        return self.total_bps / BPS

    def cash_rate_per_bar(self, bars_per_year: int) -> float:
        return self.cash_yield_annual / bars_per_year

    def borrow_rate_per_bar(self, bars_per_year: int) -> float:
        return self.borrow_bps_annual / BPS / bars_per_year


ZERO_COST = CostModel()
