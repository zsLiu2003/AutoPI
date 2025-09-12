from dataclasses import dataclass
from config.parser import load_config

@dataclass
class PromptData:
    system_prompt: str
    user_prompt: str
    expected_output_prompt: str
    assistant_prompt: str # this variable presents the historical conversation; assistant_prompt = prvious response + user_prompt
    mutated_prompt: str
    llmjudge_prompt: str

    def __init__(self, system_prompt: str, user_prompt: str, expected_output_prompt: str = "", assistant_prompt: str = "", mutated_prompt: str = "", llmjudge_prompt: str = ""):
        self.system_prompt = system_prompt
        self.expected_output_prompt = expected_output_prompt
        self.assistant_prompt = assistant_prompt
        self.mutated_prompt = mutated_prompt
        self.llmjudge_prompt = llmjudge_prompt
        self.user_prompt = user_prompt
        # self.messages = self.generate_messages()
        
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
    return historical_conversation + "\n" + "llm_reeponse" + response



