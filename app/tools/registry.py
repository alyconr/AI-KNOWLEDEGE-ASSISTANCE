
import json
from typing import Any

from pydantic import ValidationError

from app.tools.functions import (
    lookup_ai_concept,
    calculate_study_hours,
    create_study_task,
)

from app.tools.schemas import (
    ConceptArgs,
    HoursArgs,
    TaskArgs,
)


REGISTRY = {
    "lookup_ai_concept": (
        ConceptArgs,
        lookup_ai_concept,
        False,
        (
            "Consulta la definición de un concepto "
            "educativo de IA."
        ),
    ),

    "calculate_study_hours": (
        HoursArgs,
        calculate_study_hours,
        False,
        "Calcula las horas totales de estudio.",
    ),

    "create_study_task": (
        TaskArgs,
        create_study_task,
        True,
        (
            "Crea una tarea de estudio. "
            "Requiere confirmación explícita."
        ),
    ),
}


def tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": model.model_json_schema(),
            },
        }
        for name, (
            model,
            _func,
            _write,
            description,
        ) in REGISTRY.items()
    ]


def execute_tool(
    name: str,
    raw_arguments: str,
    confirm_actions: bool,
) -> dict[str, Any]:

    entry = REGISTRY.get(name)

    if entry is None:
        return {
            "ok": False,
            "error": "unknown_tool",
        }

    model, function, is_write, _description = entry

    if is_write and not confirm_actions:
        return {
            "ok": False,
            "error": "confirmation_required",
        }

    try:
        parsed = json.loads(raw_arguments)

        args = model.model_validate(parsed)

    except (
        json.JSONDecodeError,
        ValidationError,
        ValueError,
    ):
        return {
            "ok": False,
            "error": "invalid_arguments",
        }

    return {
        "ok": True,
        "data": function(args),
    }
