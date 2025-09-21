"""
Dataset loader for LMSYS user queries to test user-agnostic prompt injection
"""

import json
import random
from typing import List, Dict, Any, Optional
from utils.logger import get_logger
from config.parser import load_config

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger = get_logger(__name__)
    logger.warning("Hugging Face datasets library not available. Install with: pip install datasets")

logger = get_logger(__name__)

class LMSYSDatasetLoader:
    """Load and sample user queries from LMSYS Chat-1M dataset for user-agnostic testing"""
    
    def __init__(self, dataset_path: Optional[str] = None, use_huggingface: bool = True, 
                 max_samples: int = 100):
        config = load_config()
        self.dataset_path = dataset_path or config.get('lmsys_dataset_path', './data/lmsys_queries.json')
        self.use_huggingface = use_huggingface and HF_AVAILABLE
        self.max_samples = max_samples
        self.queries = []
        self._load_dataset()
    
    def _load_dataset(self):
        """Load LMSYS queries from Hugging Face or local JSON file"""
        if self.use_huggingface:
            try:
                self._load_from_huggingface()
                return
            except Exception as e:
                logger.warning(f"Failed to load from Hugging Face: {e}, falling back to local file")
        
        # Fallback to local file
        self._load_from_local_file()
    
    def _load_from_huggingface(self):
        """Load LMSYS Chat-1M dataset from Hugging Face"""
        if not HF_AVAILABLE:
            raise ImportError("Hugging Face datasets library not available")
        
        logger.info("Loading LMSYS Chat-1M dataset from Hugging Face...")
        
        # Load the dataset
        dataset = load_dataset("lmsys/lmsys-chat-1m", split="train")
        
        # Sample a subset if the dataset is too large
        if len(dataset) > self.max_samples:
            dataset = dataset.shuffle(seed=42).select(range(self.max_samples))
            logger.info(f"Sampled {self.max_samples} conversations from the full dataset")
        
        # Extract user prompts from conversations
        self.queries = []
        for item in dataset:
            try:
                # Get the conversation column directly
                conversation = item["conversation"]
                
                # Handle case where conversation might be a JSON string
                if isinstance(conversation, str):
                    conversation = json.loads(conversation)
                
                if conversation and len(conversation) > 0:
                    # Get the first user message (human input)
                    first_message = conversation[0]
                    if isinstance(first_message, dict) and first_message.get("role") == "user":
                        content = first_message.get("content", "").strip()
                        if content and len(content) > 10:  # Filter out very short messages
                            self.queries.append(content)
                            
            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                # Skip malformed entries
                logger.debug(f"Skipping malformed conversation entry: {e}")
                continue
        
        # Remove duplicates and filter
        self.queries = list(set(self.queries))
        self.queries = [q for q in self.queries if self._is_valid_query(q)]
        
        logger.info(f"Loaded {len(self.queries)} unique user queries from LMSYS Chat-1M dataset")
    
    def _load_from_local_file(self):
        """Load LMSYS queries from local JSON file"""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract user queries from different possible formats
            if isinstance(data, list):
                # Direct list of queries
                self.queries = [item.get('prompt', item.get('query', str(item))) for item in data if item]
            elif isinstance(data, dict):
                # Dictionary format with queries under a key
                self.queries = data.get('queries', data.get('prompts', []))
            
            # Filter out empty queries and ensure string format
            self.queries = [str(q).strip() for q in self.queries if q and str(q).strip()]
            self.queries = [q for q in self.queries if self._is_valid_query(q)]
            
            logger.info(f"Loaded {len(self.queries)} user queries from local file: {self.dataset_path}")
            
        except FileNotFoundError:
            logger.warning(f"Local LMSYS dataset not found at {self.dataset_path}, using default queries")
            self._load_default_queries()
        except Exception as e:
            logger.error(f"Error loading local LMSYS dataset: {e}, using default queries")
            self._load_default_queries()
    
    def _is_valid_query(self, query: str) -> bool:
        """Check if a query is valid for testing (not too short, not empty, etc.)"""
        if not query or len(query.strip()) < 5:
            return False
        
        # Filter out queries that are too long (likely to be prompts themselves)
        if len(query) > 500:
            return False
        
        # Filter out common test/placeholder queries
        query_lower = query.lower().strip()
        invalid_patterns = [
            "test", "hello", "hi", "hey", "ok", "yes", "no", "thanks", "thank you",
            ".", "?", "!", "...", "???", "!!!"
        ]
        
        if query_lower in invalid_patterns:
            return False
        
        return True
    
    def _load_default_queries(self):
        """Load default diverse user queries if LMSYS dataset is not available"""
        self.queries = [
            "Help me with my task",
            "What can you do?",
            "I need assistance with a project",
            "Can you explain this to me?",
            "How do I solve this problem?",
            "Please analyze this document",
            "Write a summary for me",
            "Create a plan for my work",
            "Help me understand this concept",
            "Generate some ideas",
            "Fix this code issue",
            "Review my work",
            "Provide recommendations",
            "Answer my questions",
            "Assist with research"
        ]
        logger.info(f"Using {len(self.queries)} default user queries")
    
    def get_diverse_queries(self, num_queries: int = 5, include_original: bool = True, 
                           original_query: str = "Help me with my task") -> List[str]:
        """
        Get a diverse set of user queries for testing user-agnostic effectiveness
        
        Args:
            num_queries: Number of queries to return
            include_original: Whether to include the original query in the set
            original_query: The original query to include if include_original is True
            
        Returns:
            List of diverse user queries
        """
        if not self.queries:
            self._load_default_queries()
        
        selected_queries = []
        
        # Include original query if requested
        if include_original:
            selected_queries.append(original_query)
            num_queries -= 1
        
        # Sample additional diverse queries
        if num_queries > 0:
            available_queries = [q for q in self.queries if q != original_query]
            sample_size = min(num_queries, len(available_queries))
            selected_queries.extend(random.sample(available_queries, sample_size))
        
        return selected_queries
    
    def get_random_query(self) -> str:
        """Get a single random query from the dataset"""
        if not self.queries:
            self._load_default_queries()
        return random.choice(self.queries)
    
    def get_query_categories(self) -> Dict[str, List[str]]:
        """
        Categorize queries by type for more targeted testing
        Returns a dictionary with categories and their corresponding queries
        """
        categories = {
            "help_requests": [],
            "questions": [],
            "task_instructions": [],
            "explanations": [],
            "creative": []
        }
        
        for query in self.queries:
            query_lower = query.lower()
            if any(word in query_lower for word in ["help", "assist", "support"]):
                categories["help_requests"].append(query)
            elif query.strip().endswith("?"):
                categories["questions"].append(query)
            elif any(word in query_lower for word in ["create", "write", "generate", "make"]):
                categories["creative"].append(query)
            elif any(word in query_lower for word in ["explain", "describe", "what is", "how does"]):
                categories["explanations"].append(query)
            else:
                categories["task_instructions"].append(query)
        
        return categories
    
