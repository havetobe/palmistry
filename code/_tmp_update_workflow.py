import json
from pathlib import Path


def build_prompt() -> str:
    lines = [
        "\u4f60\u662f\u624b\u76f8\u89e3\u8bfb\u4e0e\u6027\u683c\u503e\u5411\u5206\u6790\u52a9\u624b\uff08\u4ec5\u4f9b\u5a31\u4e50\uff09\u3002\u8bf7\u57fa\u4e8e\u8f93\u5165\u7684\u638c\u7eb9\u7279\u5f81\u8fdb\u884c\u89e3\u8bfb\uff0c\u8f93\u51fa\u4e25\u683cJSON\uff0c\u4e0d\u8981\u4f7f\u7528Markdown\u6216\u4ee3\u7801\u5757\u3002",
        "\u8bf7\u4f7f\u7528\u975e\u51b3\u5b9a\u8bba\u3001\u6e29\u548c\u3001\u4e0d\u4e0b\u7ed3\u8bba\u7684\u8bed\u6c14\uff0c\u4ec5\u8868\u8fbe\u53ef\u80fd\u6027\uff0c\u907f\u514d\u533b\u7597/\u6cd5\u5f8b/\u8d22\u52a1\u5efa\u8bae\uff0c\u4e0d\u8981\u65ad\u8a00\u4e8b\u5b9e\u3002",
        "\u8f93\u51faJSON\u5fc5\u987b\u5305\u542b\u5b57\u6bb5\uff1aoverview, personalityTraits, strengths, challenges, relationshipStyle, careerInclinations, selfCareTips, confidenceNotes, disclaimer\u3002",
        "\u5176\u4e2d personalityTraits/strengths/challenges/relationshipStyle/careerInclinations/selfCareTips/confidenceNotes \u4e3a\u5b57\u7b26\u4e32\u6570\u7ec4\uff0cdisclaimer \u4e3a\u5b57\u7b26\u4e32\u3002",
        "\u8f93\u5165\u5b57\u6bb5schema\u5982\u4e0b\uff08JSON\uff09\uff1a",
        "{",
        "  \"version\": 1,",
        "  \"lines\": {",
        "    \"heart\": {\"point_count\": 0, \"length\": 0.0, \"curvature\": 0.0, \"start_x\": 0.0, \"start_y\": 0.0, \"end_x\": 0.0, \"end_y\": 0.0, \"span_x\": 0.0, \"span_y\": 0.0},",
        "    \"head\":  {\"point_count\": 0, \"length\": 0.0, \"curvature\": 0.0, \"start_x\": 0.0, \"start_y\": 0.0, \"end_x\": 0.0, \"end_y\": 0.0, \"span_x\": 0.0, \"span_y\": 0.0},",
        "    \"life\":  {\"point_count\": 0, \"length\": 0.0, \"curvature\": 0.0, \"start_x\": 0.0, \"start_y\": 0.0, \"end_x\": 0.0, \"end_y\": 0.0, \"span_x\": 0.0, \"span_y\": 0.0}",
        "  },",
        "  \"confidences\": {\"heart\": 0.0, \"head\": 0.0, \"life\": 0.0},",
        "  \"roi\": {\"x\": 0.0, \"y\": 0.0, \"w\": 1.0, \"h\": 1.0}",
        "}",
        "\u8f93\u5165\u6570\u636e\u5982\u4e0b\uff1a{{analysis_input}}",
    ]
    return "\n".join(lines)


def build_display_prompt(lines: list[str]) -> str:
    return "".join(f"<p>{line}</p>" for line in lines)


def update_node_ui(node_ui: str, prompt_lines: list[str]) -> str:
    if not node_ui:
        return node_ui
    try:
        ui = json.loads(node_ui)
    except json.JSONDecodeError:
        return node_ui
    data = ui.get("data")
    if isinstance(data, dict) and "displayPrompt" in data:
        data["displayPrompt"] = build_display_prompt(prompt_lines)
        ui["data"] = data
        return json.dumps(ui, ensure_ascii=False)
    return node_ui


def main() -> None:
    path = Path("docs/workflows/export-\u5206\u6790\u52a9\u624b-\u9ad8\u4f18\u5148\u7ea7/85b8836e-bd71-4ef7-9add-aaceffdc01fc_workflow.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    prompt = build_prompt()
    prompt_lines = prompt.splitlines()

    # Keep WorkflowName to match canvas name to avoid import error.
    data["WorkflowName"] = "\u5206\u6790\u52a9\u624b-\u9ad8\u4f18\u5148\u7ea7-\u591a\u6a21\u578b"
    data["WorkflowDesc"] = (
        "\u89e6\u53d1\u6761\u4ef6\n"
        "\u5f53\u7528\u6237\u6d88\u606f\u540c\u65f6\u6ee1\u8db3\u4ee5\u4e0b\u6761\u4ef6\u65f6\uff0c\u5fc5\u987b\u8c03\u7528\u6b64\u5de5\u4f5c\u6d41\uff1a\n"
        "1.\u6d88\u606f\u683c\u5f0f\u4e3aJSON\u5bf9\u8c61\n"
        "2.\u5305\u542b lines\u3001confidences\u3001roi \u5b57\u6bb5\n"
        "3.\u5b57\u6bb5\u503c\u7c7b\u578b\uff1alines/confidences/roi \u4e3a\u5bf9\u8c61\n"
        "\u7981\u6b62\u89e6\u53d1\u6761\u4ef6\n"
        "\u6d88\u606f\u4e0d\u662fJSON\u683c\u5f0f\n"
        "\u7f3a\u5c11\u5173\u952e\u5b57\u6bb5\n"
        "\u3010\u4f18\u5148\u7ea7\u3011\u6700\u9ad8"
    )

    for node in data.get("Nodes", []):
        if node.get("NodeType") == "LLM":
            llm = node.get("LLMNodeData") or {}
            llm["Prompt"] = prompt
            node["LLMNodeData"] = llm
            node["NodeUI"] = update_node_ui(node.get("NodeUI"), prompt_lines)
        elif node.get("NodeType") == "TOOL":
            tool = node.get("ToolNodeData") or {}
            api = tool.get("API") or {}
            api["URL"] = "http://127.0.0.1:8000/api/palm/analysis/receive"
            tool["API"] = api
            node["ToolNodeData"] = tool

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
