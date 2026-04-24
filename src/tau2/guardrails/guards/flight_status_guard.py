from tau2.data_model.message import Message, ToolCall
from tau2.environment.environment import Environment
from tau2.guardrails.guard import Guard, GuardVerdict




class FlightStatusGuard(Guard):
    """
    TODO: IMPLEMENT!

    Blocks update_reservation_flights when any flight fails its status requirement:

      - New flights (not in the current reservation) must have status "available".
      - All flights (existing and new) must not be "flying" — a flight already
        in the air means a portion of the reservation has been flown, which
        prevents both flight changes and cabin changes per policy.

    This guard calls get_flight_status() for every flight in the arguments,
    so it does a small number of extra reads before allowing a write.

    The conversation history parameter is accepted but not used — this guard
    makes a fully deterministic decision from live flight data.
    """

    @property
    def name(self) -> str:
        return "FlightStatusGuard"

    def applies_to(self, tool_call: ToolCall) -> bool:
        return tool_call.name == "update_reservation_flights"

    def check(
        self,
        tool_call: ToolCall,
        env: Environment,
        history: list[Message],
    ) -> GuardVerdict:
        """ 
        args = tool_call.arguments
        reservation_id = args.get("reservation_id")
        requested_flights = args.get("flights", [])

        reservation = env.use_tool(
            "get_reservation_details", reservation_id=reservation_id
        )
        existing = {
            (f.flight_number, f.date) for f in reservation.flights
        }

        violations = []
        for f in requested_flights:
            if isinstance(f, dict):
                fn, date = f["flight_number"], f["date"]
            else:
                fn, date = f.flight_number, f.date

            status = env.use_tool("get_flight_status", flight_number=fn, date=date)

            is_new = (fn, date) not in existing
            if is_new and status != "available":
                violations.append(
                    f"  - {fn} on {date}: status '{status}' "
                    f"(new flights must have status 'available')"
                )
            elif status == "flying":
                violations.append(
                    f"  - {fn} on {date}: status '{status}' "
                    f"(flight is already in the air — reservation cannot be modified)"
                )

        if violations:
            return GuardVerdict(
                allowed=False,
                reason=(
                    "One or more flights fail the required status check:\n"
                    + "\n".join(violations)
                    + "\n\nNew flights must be 'available'. "
                    "Use search_direct_flight or search_onestop_flight to find valid options."
                ),
            )
        """
        return GuardVerdict(allowed=True)
