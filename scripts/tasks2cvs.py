import json
import csv
import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def convert(json_path, csv_path):
    with open(json_path) as f:
        raw = f.read()
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    data = json.loads(raw)

    rows = []
    for item in data:
        desc = item.get("description", {})
        scenario = item.get("user_scenario", {})
        instructions = scenario.get("instructions", {})
        eval_criteria = item.get("evaluation_criteria", {})

        nl_assertions = "\n".join(eval_criteria.get("nl_assertions", []) or [])

        compliance_items = eval_criteria.get("compliance", []) or []
        compliance_descriptions = "\n".join(c.get("description", "") for c in compliance_items)

        reward_basis = eval_criteria.get("reward_basis", []) or []
        has_compliance = "yes" if "COMPLIANCE" in reward_basis else "no"

        rows.append({
            "id": item.get("id", ""),
            "ticket": item.get("ticket", ""),
            "description": desc.get("purpose", ""),
            "relevant_policies": desc.get("relevant_policies", ""),
            "notes": desc.get("notes", ""),
            "persona": scenario.get("persona", ""),
            "task_instructions": instructions.get("task_instructions", ""),
            "reason_for_call": instructions.get("reason_for_call", ""),
            "known_info": instructions.get("known_info", ""),
            "nl_assertions": nl_assertions,
            "compliance": compliance_descriptions,
            "Compliance": has_compliance,
        })

    fieldnames = [
        "id","description", "ticket", "relevant_policies", "notes", "persona",
        "task_instructions", "reason_for_call", "known_info",
        "nl_assertions", "compliance", "Compliance"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {len(rows)} rows to {csv_path}")


def main():
    json_path = os.path.join(SCRIPT_DIR, "../data/tau2/domains/telecom/tasks_small.json")
    csv_path = os.path.join(SCRIPT_DIR, "output/output_telecom.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    convert(json_path, csv_path)

main()