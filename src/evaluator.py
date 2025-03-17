import os
import json
import time
import datetime
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import openai
import anthropic
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('melida-evaluator')


class ModelEvaluator:
    """Main evaluator class for testing AI models on medical exams."""

    def __init__(self, config_path: str = 'config/api_config.json'):
        """Initialize evaluator with configuration."""
        self.config = self._load_config(config_path)
        self.setup_clients()
        self.results = []
        self.prompt_strategies = self._load_prompt_strategies()

    def _load_config(self, config_path: str) -> Dict:
        """Load API configuration."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found at {config_path}")
            raise

    def _load_prompt_strategies(self) -> Dict:
        """Load prompt strategies from config."""
        try:
            with open('config/prompt_strategies.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error("Prompt strategies file not found")
            raise

    def setup_clients(self):
        """Set up API clients based on configuration."""
        if 'openai' in self.config:
            openai.api_key = self.config['openai']['api_key']
            # Create a client instance for the new OpenAI SDK
            self.openai_client = openai.OpenAI(api_key=self.config['openai']['api_key'])

        if 'anthropic' in self.config:
            self.anthropic_client = anthropic.Anthropic(
                api_key=self.config['anthropic']['api_key']
            )

    def load_questions(self, file_path: str) -> List[Dict]:
        """Load test questions from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Questions file not found at {file_path}")
            raise

    def load_answer_key(self, file_path: str) -> Dict:
        """Load answer key from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Answer key file not found at {file_path}")
            raise

    def evaluate_question(self, question: Dict, prompt_strategy: str, model: str) -> Dict:
        """Evaluate a single question using specified prompt strategy and model."""
        # Start timing the evaluation
        start_time = time.time()

        # Get the prompt strategy dictionary and template string
        prompt_strategy_dict = self.prompt_strategies[prompt_strategy]
        prompt_template = prompt_strategy_dict["template"]

        # Format the prompt with the question details
        prompt = prompt_template.format(
            question_text=question['question_text'],
            option_a=question['options']['A'],
            option_b=question['options']['B'],
            option_c=question['options']['C'],
            option_d=question['options']['D']
        )

        # Call the appropriate model API.
        # Now also support models that include "o3-mini" in their name.
        if 'openai' in model.lower() or 'gpt' in model.lower() or 'o3-mini' in model.lower():
            response = self._call_openai(prompt, model)
        elif 'claude' in model.lower():
            response = self._call_anthropic(prompt, model)
        else:
            raise ValueError(f"Unsupported model: {model}")

        end_time = time.time()

        # Process the response to extract the answer (A, B, C, D, or NO)
        model_answer = self._extract_answer(response)

        # Return result details
        return {
            'question_id': question['id'],
            'prompt_strategy': prompt_strategy,
            'model': model,
            'model_answer': model_answer,
            'raw_response': response,
            'response_time': end_time - start_time,
            'tokens_used': self._count_tokens(prompt, response, model),
            'timestamp': datetime.datetime.now().isoformat()
        }

    def _call_openai(self, prompt: str, model: str) -> str:
        """Call OpenAI API with prompt using new API client."""
        try:
            # For models like "o3-mini", use max_completion_tokens instead of max_tokens.
            if "o3-mini" in model.lower():
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_completion_tokens=10  # Use max_completion_tokens for o3-mini
                )
            else:
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=10
                )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return "ERROR"

    def _call_anthropic(self, prompt: str, model: str) -> str:
        """Call Anthropic API with prompt."""
        try:
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=10,  # We only need a short response (A/B/C/D/NO)
                temperature=0,
                system="You are taking a medical examination. Answer only with the letter of the correct option (A, B, C, D) or 'NO' if you prefer not to answer.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {e}")
            return "ERROR"

    def _extract_answer(self, response: str) -> str:
        """Extract the answer (A, B, C, D, or NO) from the model response."""
        # Clean and normalize the response
        clean_response = response.upper().strip()

        # Look for valid answers
        valid_answers = ['A', 'B', 'C', 'D', 'NO']

        # First check if response is exactly one of the valid answers
        if clean_response in valid_answers:
            return clean_response

        # If not, try to extract the first occurrence of a valid answer
        for answer in valid_answers:
            if answer in clean_response:
                return answer

        # If no valid answer found, return "INVALID"
        return "INVALID"

    def _count_tokens(self, prompt: str, response: str, model: str) -> int:
        """Estimate token count for the request and response."""
        # This is a simple approximation; for production, use model-specific tokenizers
        words = len(prompt.split()) + len(response.split())
        return int(words * 1.3)  # Rough approximation

    def calculate_score(self, model_answer: str, correct_answer: str) -> int:
        """Calculate score based on model's answer and correct answer."""
        if model_answer in ["INVALID", "ERROR"]:
            return 0  # No score for invalid responses

        if model_answer == "NO":
            return 0  # No score for skipped questions

        if model_answer == correct_answer:
            return 3  # Correct answer
        else:
            return -1  # Incorrect answer

    def run_evaluation(self,
                       questions_file: str,
                       answer_key_file: str,
                       prompt_strategy: str,
                       model: str,
                       output_dir: str = 'data/results',
                       sample_size: Optional[int] = None) -> str:
        """
        Run evaluation on a set of questions using specified prompt strategy and model.

        Args:
            questions_file: Path to the questions file
            answer_key_file: Path to the answer key file
            prompt_strategy: Name of the prompt strategy to use
            model: Name of the model to use
            output_dir: Directory to save results
            sample_size: Optional number of questions to evaluate (for testing)

        Returns:
            Path to the results file
        """
        questions = self.load_questions(questions_file)
        answer_key = self.load_answer_key(answer_key_file)

        # Use a sample of questions if specified
        if sample_size and sample_size < len(questions):
            questions = questions[:sample_size]

        results = []
        total_score = 0
        correct_count = 0
        incorrect_count = 0
        skipped_count = 0
        invalid_count = 0

        # Extract test ID from questions file name
        test_id = Path(questions_file).stem

        logger.info(f"Starting evaluation with {len(questions)} questions using {prompt_strategy} on {model}")

        # Process each question
        for question in tqdm(questions, desc="Evaluating questions"):
            # Evaluate the question
            result = self.evaluate_question(question, prompt_strategy, model)

            # Get correct answer from answer key
            correct_answer = answer_key.get(question['id'], "UNKNOWN")

            # Calculate and add score
            score = self.calculate_score(result['model_answer'], correct_answer)
            result['correct_answer'] = correct_answer
            result['score'] = score

            # Generate detailed result ID
            result['result_id'] = f"{question['id']}-Resp{result['model_answer']}-Correct{correct_answer}-Score{score:+d}"

            # Update statistics
            total_score += score
            if score == 3:
                correct_count += 1
            elif score == -1:
                incorrect_count += 1
            elif result['model_answer'] == "NO":
                skipped_count += 1
            else:
                invalid_count += 1

            # Add to results list
            results.append(result)

            # Add a small delay to avoid API rate limits
            time.sleep(0.5)

        # Create summary statistics
        summary = {
            'test_id': test_id,
            'prompt_strategy': prompt_strategy,
            'model': model,
            'total_questions': len(questions),
            'total_score': total_score,
            'correct_count': correct_count,
            'incorrect_count': incorrect_count,
            'skipped_count': skipped_count,
            'invalid_count': invalid_count,
            'accuracy': correct_count / len(questions) if len(questions) > 0 else 0,
            'timestamp': datetime.datetime.now().isoformat()
        }

        # Prepare the full results object
        evaluation_results = {
            'summary': summary,
            'results': results
        }

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate output filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_filename = f"EVAL-{test_id}-{prompt_strategy}-{model.replace('/', '-')}-{timestamp}.json"
        output_path = os.path.join(output_dir, output_filename)

        # Save results to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Evaluation completed. Results saved to {output_path}")
        logger.info(f"Summary: Score={total_score}, Correct={correct_count}, Incorrect={incorrect_count}, Skipped={skipped_count}")

        # Also save a CSV version for easy import into Tableau
        results_df = pd.DataFrame(results)
        csv_path = output_path.replace('.json', '.csv')
        results_df.to_csv(csv_path, index=False)

        return output_path


if __name__ == 
