"""
MELIDA: Model Evaluation for Life-sciences Intelligence and Decision Assistance
Utility functions for the project.
"""

import json
import logging
import pandas as pd
from typing import Dict, List

def format_result_id(question_id: str, model_answer: str, correct_answer: str, score: int) -> str:
    """Format a detailed result ID."""
    sign = "P" if score > 0 else "M" if score < 0 else "Z"
    return f"{question_id}-Resp{model_answer}-Correct{correct_answer}-Score{sign}{abs(score)}"

def results_to_dataframe(results_file: str) -> pd.DataFrame:
    """Load evaluation results into a pandas DataFrame for analysis."""
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert results list to DataFrame
    return pd.DataFrame(data['results'])

def compare_evaluations(eval_files: List[str]) -> pd.DataFrame:
    """Compare results from multiple evaluations."""
    summaries = []
    
    for file in eval_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        summaries.append(data['summary'])
    
    return pd.DataFrame(summaries)
