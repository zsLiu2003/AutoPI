import json
from dataclasses import dataclass
from typing import List

@dataclass
class UserAgnosticOutput:
    """Output format for user_agnostic_instruction_prompt.txt - JSON array of strings"""
    tool_descriptions: List[str]

    def to_json(self) -> str:
        return json.dumps(self.tool_descriptions, indent=2)

    @classmethod
    def from_variants(cls, variants: List[str]) -> 'UserAgnosticOutput':
        return cls(tool_descriptions=variants)