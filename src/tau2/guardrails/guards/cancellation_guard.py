from datetime import datetime

from tau2.data_model.message import Message, ToolCall
from tau2.environment.environment import Environment
from tau2.guardrails.guard import Guard, GuardVerdict

# Matches the "current time" declared in the airline policy header
SIMULATED_NOW = datetime(2024, 5, 15, 15, 0, 0)


class CancellationEligibilityGuard(Guard):
    """
    Blocks cancel_reservation when none of the four policy conditions are met:

      1. Business class — always cancellable with refund
      2. Booked within last 24 hours — always cancellable with refund
      3. Airline cancelled a flight in the reservation — always cancellable with refund
      4. Has travel insurance — cancellable when reason is covered

    For condition 4, this guard checks whether the reservation has insurance.
    Verifying that the cancellation reason is covered by insurance requires
    conversational context — that check is deferred to LLMPolicyGuard if
    it is included in the middleware chain.

    The conversation history parameter is accepted for interface consistency
    but is not used by this guard (all checks are deterministic).
    """

    @property
    def name(self) -> str:
        return "CancellationEligibilityGuard"

    def applies_to(self, tool_call: ToolCall) -> bool:
        return tool_call.name == "cancel_reservation"

    def check(
        self,
        tool_call: ToolCall,
        env: Environment,
        history: list[Message],
    ) -> GuardVerdict:
        reservation_id = tool_call.arguments.get("reservation_id")
        reservation = env.use_tool(
            "get_reservation_details", reservation_id=reservation_id
        )

        # Condition 1: business class
        if reservation.cabin == "business":
            return GuardVerdict(allowed=True)

        # Condition 2: booked within last 24 hours
        created_at = datetime.fromisoformat(reservation.created_at)
        hours_since = (SIMULATED_NOW - created_at).total_seconds() / 3600
        if hours_since <= 24:
            return GuardVerdict(allowed=True)

        # Condition 3: any flight was cancelled by the airline
        for flight in reservation.flights:
            status = env.use_tool(
                "get_flight_status",
                flight_number=flight.flight_number,
                date=flight.date,
            )
            if status == "cancelled":
                return GuardVerdict(allowed=True)

        # Condition 4: has travel insurance (reason-coverage check deferred to LLM)
        if reservation.insurance == "yes":
            return GuardVerdict(allowed=True)

        return GuardVerdict(
            allowed=False,
            reason=(
                f"Reservation {reservation_id} does not meet any cancellation eligibility criteria:\n"
                f"  - Business class: NO (cabin is '{reservation.cabin}')\n"
                f"  - Booked within 24h: NO (booked {hours_since:.0f}h ago at {reservation.created_at})\n"
                f"  - Airline-cancelled flight: NO (no flights are cancelled by the airline)\n"
                f"  - Travel insurance: NO (reservation has no insurance)\n\n"
                f"Cancellation with a full refund requires one of the following:\n"
                f"  • Business class reservation\n"
                f"  • Booking was made within the last 24 hours\n"
                f"  • The airline cancelled a flight in the reservation\n"
                f"  • The reservation has travel insurance and the cancellation reason is covered"
            ),
        )
