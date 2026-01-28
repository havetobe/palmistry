import json
import uuid
from pathlib import Path


def build_prompt() -> str:
    lines = [
        "你是手相解读与性格倾向分析助手（仅供娱乐）。请基于输入的掌纹摘要数据进行解读，输出严格JSON，不要使用Markdown或代码块。",
        "JSON字段必须包含：overview, personalityTraits, strengths, challenges, relationshipStyle, careerInclinations, selfCareTips, confidenceNotes, disclaimer。",
        "其中 personalityTraits/strengths/challenges/relationshipStyle/careerInclinations/selfCareTips/confidenceNotes 为字符串数组，disclaimer 为字符串。",
        "解读要求：",
        "1) 结合心线/智慧线/生命线的长度、曲率、跨度与置信度给出倾向描述。",
        "2) 避免医学、法律或财务建议，不要断言事实或给出确定性预测。",
        "3) 用温和、非决定论的语气表达可能性与建议。",
        "输入数据如下：{{analysis_input}}",
    ]
    return "\n".join(lines)


def build_html_prompt(lines: list[str]) -> str:
    return "".join(f"<p>{line}</p>" for line in lines)


def update_node_ui(node_ui: str, prompt_lines: list[str]) -> str:
    if not node_ui:
        return node_ui
    try:
        ui = json.loads(node_ui)
    except json.JSONDecodeError:
        return node_ui
    data = ui.get("data")
    if not isinstance(data, dict):
        return node_ui
    if "displayPrompt" in data:
        data["displayPrompt"] = build_html_prompt(prompt_lines)
    ui["data"] = data
    return json.dumps(ui, ensure_ascii=False)


def main() -> None:
    src = Path("docs/workflows/export-手相分析助手-高优先级/palm_workflow.json")
    data = json.loads(src.read_text(encoding="utf-8"))

    prompt_lines = build_prompt().splitlines()
    prompt_text = "\n".join(prompt_lines)

    data["WorkflowID"] = str(uuid.uuid4())
    data["WorkflowName"] = "手相分析助手-高优先级-多模型"
    data["WorkflowDesc"] = (
        "触发条件\n"
        "当用户消息同时满足以下条件时，调用此工作流：\n"
        "1.消息格式为JSON对象\n"
        "2.包含 lines、confidences、roi 字段\n"
        "3.字段类型：lines/confidences/roi 均为对象\n"
        "禁止触发条件\n"
        "消息不是JSON格式\n"
        "缺少关键字段\n"
        "【优先级】最高"
    )

    for node in data.get("Nodes", []):
        node_type = node.get("NodeType")
        if node_type == "LLM":
            llm = node.get("LLMNodeData", {})
            llm["Prompt"] = prompt_text
            node["LLMNodeData"] = llm
            node["NodeUI"] = update_node_ui(node.get("NodeUI"), prompt_lines)
        elif node_type == "TOOL":
            tool = node.get("ToolNodeData", {})
            api = tool.get("API", {})
            api["URL"] = "http://127.0.0.1:8000/api/palm/analysis/receive"
            tool["API"] = api
            node["ToolNodeData"] = tool

    src.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
