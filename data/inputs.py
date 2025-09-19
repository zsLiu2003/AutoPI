from dataclasses import dataclass
from config.parser import load_config

@dataclass
class InputData:
    system_prompt: str  # Original system prompt with <Injected Loaction> tag
    user_prompt: str    # Expected command (e.g., "rm")
    expected_output_prompt: str  # Expected output for gradient loss calculation only
    assistant_prompt: str  # Historical conversation
    seed_tool_des: str     # Seed tool description
    target_command: str    # Target command
    tool_name: str = "diagnostic_tool"  # Name for the tool in XML template
    real_response: str = ""  # Actual agent response for LLM judge evaluation

    def __init__(self, system_prompt: str, user_prompt: str, expected_output_prompt: str = "",
                 assistant_prompt: str = "", seed_tool_des: str = "", target_command: str = "",
                 tool_name: str = "diagnostic_tool", real_response: str = ""):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.expected_output_prompt = expected_output_prompt
        self.assistant_prompt = assistant_prompt
        self.seed_tool_des = seed_tool_des
        self.target_command = target_command
        self.tool_name = tool_name
        self.real_response = real_response
        
    def inject_tool_description(self, tool_des: str) -> str:
        """Inject tool description into system prompt's <Injected Loaction> tag"""
        if "<Injected Loaction>" in self.system_prompt:
            return self.system_prompt.replace("<Injected Loaction>", tool_des)
        elif "<injected location>" in self.system_prompt:
            return self.system_prompt.replace("<injected location>", tool_des)
        else:
            # If no injection tag found, append to the end
            return f"{self.system_prompt}\n\n{tool_des}"

@dataclass 
class VariantData:
    """单个variant的数据结构"""
    tool_description: str
    llm_judge_score: float
    gradient_score: float
    combined_score: float
    agent_response: str
    success: bool

def get_system_prompt(agent_name: str, config: dict) -> str:
    """
    retrieves the system prompt and expected output for a given agent from configuration files.
    """
    data_path = config.get("data_path", "./data")
    system_prompt_file = f"{data_path}/{agent_name}.txt"
    expected_output_file = f"{data_path}/{agent_name}_output.txt"

    try:
        with open(system_prompt_file, 'r') as f:
            system_prompt = f.read().strip()
    except FileNotFoundError as e:
        print(f"Error: System prompt file not found: {e}")
        return "", ""

    try:
        with open(expected_output_file, 'r') as f:
            expected_output = f.read().strip()
    except FileNotFoundError as e:
        print(f"Warning: Expected output file not found: {e}. Using default.")
        expected_output = "Command executed successfully"

    return system_prompt, expected_output
    
def assistant_prompt(historical_conversation: str, response: str) -> str:
    return historical_conversation + "\n" + "llm_response: " + response



