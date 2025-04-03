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
import re

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
            self.openai_client = openai

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
        start_time = time.time()
        prompt_strategy_dict = self.prompt_strategies[prompt_strategy]
        prompt_template = prompt_strategy_dict["template"]
        prompt = prompt_template.format(
            question_text=question.get('question_text', "Not available"),
            option_a=question['options'].get('A', ""),
            option_b=question['options'].get('B', ""),
            option_c=question['options'].get('C', ""),
            option_d=question['options'].get('D', "")
        )
        # Updated conditional to support Together-based models
        if 'openai' in model.lower() or 'gpt' in model.lower() or 'o3-mini' in model.lower():
            response = self._call_openai(prompt, model)
        elif 'claude' in model.lower():
            response = self._call_anthropic(prompt, model)
        elif ('together' in model.lower() or 'deepseek' in model.lower() or
          'meta-llama' in model.lower() or 'qwen' in model.lower() or
          'mistralai' in model.lower()):
            response = self._call_together(prompt, model)
        elif 'google' in model.lower():
            response = self._call_google(prompt, model)
        else:
            raise ValueError(f"Unsupported model: {model}")
        end_time = time.time()
        model_answer = self._extract_answer(response)
        return {
            'question_id': question['id'],
            'question_text': question.get('question_text', "Not available"),
            'prompt_strategy': prompt_strategy,
            'model': model,
            'prompt': prompt,
            'full_model_output': response,
            'model_answer': model_answer,
            'raw_response': response,
            'response_time': end_time - start_time,
            'tokens_used': self._count_tokens(prompt, response, model),
            'timestamp': datetime.datetime.now().isoformat()
        }

    def _call_openai(self, prompt: str, model: str) -> str:
        """Call OpenAI API with prompt using the new API client."""
        try:
            if "o3-mini" in model.lower():
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=1024
                )
            else:
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024
                )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return "ERROR"

    def _call_anthropic(self, prompt: str, model: str) -> str:
        """Call Anthropic API with prompt."""
        try:
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=1024,
                temperature=0,
                system="You are taking a medical examination. Answer only with the letter of the correct option (A, B, C, D) or 'NO' if you prefer not to answer.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {e}")
            return "ERROR"

    def _call_together(self, prompt: str, model: str) -> str:
        """Call the Together API with the given prompt."""
        import requests  # Ensure requests is imported
        # Retrieve Together API key from environment (secrets) or config
        api_key = os.environ.get("TOGETHER_API_KEY") or self.config.get("together", {}).get("api_key")
        if not api_key:
            logger.error("Together API key not set in secrets or config.")
            return "ERROR"
        try:
            endpoint = "https://api.together.xyz/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            messages = [
                {
                    "role": "system",
                    "content": ("You are a precise question-answering system. "
                                "Respond with only one letter: A, B, C, or D. No extra text is allowed.")
                },
                {"role": "user", "content": prompt}
            ]
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": 1024,
                "temperature": 0.0,
                "top_p": 0.7,
                "top_k": 50,
                "repetition_penalty": 1,
                "stop": ["<|eot_id|>", "<|eom_id|>"],
                "stream": False
            }
            response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                raise ValueError(f"API Error: {data['error']}")
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"]
            else:
                return "ERROR"
        except Exception as e:
            logger.error(f"Error calling Together API for model {model}: {e}")
            return "ERROR"

    def _call_google(self, prompt: str, model: str) -> str:
        """Placeholder for Google API call (not implemented)."""
        return "ERROR"

    def _extract_answer(self, response: str) -> str:
        """Extract the answer (A, B, C, D, or N) from the model response using regex."""
        clean_response = response.upper().strip()
        match = re.search(r'\b([ABCDN])\b', clean_response)
        if match:
            return match.group(1)
        return "INVALID"

    def _count_tokens(self, prompt: str, response: str, model: str) -> int:
        """Estimate token count for the request and response."""
        words = len(prompt.split()) + len(response.split())
        return int(words * 1.3)

    def calculate_score(self, model_answer: str, correct_answer: str) -> int:
        """Calculate score based on model's answer and correct answer."""
        if model_answer in ["INVALID", "ERROR"]:
            return 0
        if model_answer == "N":
            return 0
        if model_answer == correct_answer:
            return 3
        else:
            return -1

    def run_evaluation(self,
                       questions_file: str,
                       answer_key_file: str,
                       prompt_strategy: str,
                       model: str,
                       output_dir: str = 'data/results',
                       sample_size: Optional[int] = None) -> str:
        """
        Run evaluation on a set of questions using specified prompt strategy and model.
        """
        questions = self.load_questions(questions_file)
        answer_key = self.load_answer_key(answer_key_file)
        if sample_size and sample_size < len(questions):
            questions = questions[:sample_size]
        results = []
        total_score = 0
        correct_count = 0
        incorrect_count = 0
        skipped_count = 0
        invalid_count = 0
        test_id = Path(questions_file).stem
        logger.info(f"Starting evaluation with {len(questions)} questions using {prompt_strategy} on {model}")
        for question in tqdm(questions, desc="Evaluating questions"):
            result = self.evaluate_question(question, prompt_strategy, model)
            correct_answer = answer_key.get(question['id'], "UNKNOWN")
            score = self.calculate_score(result['model_answer'], correct_answer)
            result['correct_answer'] = correct_answer
            result['score'] = score
            result['result_id'] = f"{question['id']}-Resp{result['model_answer']}-Correct{correct_answer}-Score{score:+d}"
            total_score += score
            if score == 3:
                correct_count += 1
            elif score == -1:
                incorrect_count += 1
            elif result['model_answer'] == "NO":
                skipped_count += 1
            else:
                invalid_count += 1
            results.append(result)
            time.sleep(0.5)
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
        evaluation_results = {
            'summary': summary,
            'results': results
        }
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_filename = f"EVAL-{test_id}-{prompt_strategy}-{model.replace('/', '-')}-{timestamp}.json"
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Evaluation completed. Results saved to {output_path}")
        logger.info(f"Summary: Score={total_score}, Correct={correct_count}, Incorrect={incorrect_count}, Skipped={skipped_count}")
        results_df = pd.DataFrame(results)
        csv_path = output_path.replace('.json', '.csv')
        results_df.to_csv(csv_path, index=False)
        return output_path


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    print("Evaluator initialized successfully")
