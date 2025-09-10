from dataclasses import dataclass

@dataclass
class InputData:
    system_prompt: str
    user_prompt: str
    execution_command: str
    assistant_prompt: str # this variable presents the historical conversation
    mutated_prompt: str

def get_example_inputs() -> InputData:
    return InputData(
        system_prompt="You are an AI system assisting with pipeline execution.",
        assistant_prompt="Previous context: optimization initialized.",
        tool_description="A tool that provides natural language optimizations.",
        user_prompt="Optimize my query for gradient-based evaluation."
    )

