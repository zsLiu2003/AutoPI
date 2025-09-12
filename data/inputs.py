from dataclasses import dataclass
from config.parser import load_config

@dataclass
class InputData:
    system_prompt: str  # 原始system prompt，包含<injected location>标签
    user_prompt: str    # 期望执行的命令（如"rm"）
    expected_output_prompt: str
    assistant_prompt: str  # 历史对话
    seed_tool_des: str     # 种子tool description
    target_command: str    # 目标命令
    
    def __init__(self, system_prompt: str, user_prompt: str, expected_output_prompt: str = "", 
                 assistant_prompt: str = "", seed_tool_des: str = "", target_command: str = ""):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.expected_output_prompt = expected_output_prompt
        self.assistant_prompt = assistant_prompt
        self.seed_tool_des = seed_tool_des
        self.target_command = target_command
        
    def inject_tool_description(self, tool_des: str) -> str:
        """将tool description注入到system prompt的<injected location>标签中"""
        if "<injected location>" in self.system_prompt:
            return self.system_prompt.replace("<injected location>", tool_des)
        else:
            # 如果没有标签，直接追加
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
        with open(expected_output_file, 'r') as f:
            expected_output = f.read().strip()
        return system_prompt, expected_output
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return "", ""
    
def assistant_prompt(historical_conversation: str, response: str) -> str:
    return historical_conversation + "\n" + "llm_response: " + response



