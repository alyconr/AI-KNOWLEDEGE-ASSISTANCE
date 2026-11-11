
import json

from app.tools.registry import (
    execute_tool,
    tool_definitions,
)


def test_tools_exposed():
    names = {
        tool["function"]["name"]
        for tool in tool_definitions()
    }

    assert names == {
        "lookup_ai_concept",
        "calculate_study_hours",
        "create_study_task",
    }


def test_calculation():
    result = execute_tool(
        "calculate_study_hours",
        json.dumps({
            "hours_per_day": 1.5,
            "days": 4,
        }),
        False,
    )

    assert (
        result["data"]["total_hours"]
        == 6
    )


def test_invalid_args():
    invalid_range = execute_tool(
        "calculate_study_hours",
        json.dumps({
            "hours_per_day": 50,
            "days": 2,
        }),
        False,
    )

    assert (
        invalid_range["error"]
        == "invalid_arguments"
    )

    invalid_json = execute_tool(
        "calculate_study_hours",
        "not-json",
        False,
    )

    assert (
        invalid_json["error"]
        == "invalid_arguments"
    )


def test_unknown_tool():
    result = execute_tool(
        "run_shell",
        "{}",
        True,
    )

    assert (
        result["error"]
        == "unknown_tool"
    )
