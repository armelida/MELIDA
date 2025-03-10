"""
MELIDA: Model Evaluation for Life-sciences Intelligence and Decision Assistance
Prompt manager for handling different prompt strategies.
"""

import json
import logging
from typing import Dict, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('melida-prompt-manager')

class PromptManager:
    """Class to manage different prompt strategies for model evaluation."""
    
    def __init__(self, config_path: str = 'config/prompt_strategies.json'):
        """Initialize prompt manager with configuration."""
        self.strategies = self._load_strategies(config_path)
        
    def _load_strategies(self, config_path: str) -> Dict:
        """Load prompt strategies from config file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Prompt strategies file not found at {config_path}")
            raise
    
    def get_prompt(self, strategy_id: str, question_data: Dict) -> str:
        """
        Generate a prompt using the specified strategy and question data.
        
        Args:
            strategy_id: ID of the prompt strategy to use
            question_data: Dictionary containing question details
            
        Returns:
            Formatted prompt string
        """
        if strategy_id not in self.strategies:
            logger.error(f"Strategy not found: {strategy_id}")
            raise ValueError(f"Unknown prompt strategy: {strategy_id}")
        
        strategy = self.strategies[strategy_id]
        
        # Format the prompt template with question data
        try:
            prompt = strategy['template'].format(
                question_text=question_data['question_text'],
                option_a=question_data['options']['A'],
                option_b=question_data['options']['B'],
                option_c=question_data['options']['C'],
                option_d=question_data['options']['D']
            )
            return prompt
        except KeyError as e:
            logger.error(f"Missing key in question data: {e}")
            raise
        except Exception as e:
            logger.error(f"Error formatting prompt: {e}")
            raise
    
    def list_strategies(self) -> List[Dict]:
        """List available prompt strategies with descriptions."""
        return [
            {'id': strategy_id, 'description': strategy.get('description', 'No description')}
            for strategy_id, strategy in self.strategies.items()
        ]
    
    def create_strategy(self, strategy_id: str, template: str, description: str = "") -> None:
        """
        Create a new prompt strategy.
        
        Args:
            strategy_id: ID for the new strategy
            template: Prompt template string
            description: Description of the strategy
        """
        if strategy_id in self.strategies:
            logger.warning(f"Overwriting existing strategy: {strategy_id}")
        
        self.strategies[strategy_id] = {
            'template': template,
            'description': description
        }
        
        # Save updated strategies to file
        self._save_strategies()
        
    def _save_strategies(self, config_path: str = 'config/prompt_strategies.json') -> None:
        """Save prompt strategies to config file."""
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.strategies, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.strategies)} prompt strategies to {config_path}")
        except Exception as e:
            logger.error(f"Error saving prompt strategies: {e}")
            raise

if __name__ == "__main__":
    # Example usage
    prompt_manager = PromptManager()
    print("Available strategies:")
    for strategy in prompt_manager.list_strategies():
        print(f"- {strategy['id']}: {strategy['description']}")
