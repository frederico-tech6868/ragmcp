import json

def parse_tool_call(model_output):
    # Parse JSON tool calls from model output
    tool_call = json.loads(model_output)
    return tool_call