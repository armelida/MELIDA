# Cell 1: Setup Cell - Run this first
# -------------------------------------

#Force Colab to Request the Best GPU
!pip install --upgrade --quiet google-colab
from google.colab import runtime

# Request an A100 GPU
runtime.set_accelerator_hardware("GPU")
runtime.set_accelerator_type("A100")


# Cell 2: Fix Git Configuration
# -----------------------------

# Configure Git to use merge strategy (solving the "divergent branches" error)
!git config pull.rebase false

# Configure Git user
!git config --global user.email "armelida@gmail.com"
!git config --global user.name "Armelida"

# Clean up any merge conflicts
!git reset --hard origin/main

# Pull fresh changes
!git pull origin main


# Cell 3: Clone Repository and Setup Environment
# ----------------------------------------------

# Remove existing folder (if any) and clone a fresh copy
!rm -rf MELIDA

# Clone the repository and set up the environment
!git clone https://github.com/armelida/MELIDA.git
%cd MELIDA

# Install required packages
!pip install -r requirements.txt
# Add plotly for interactive visualization
!pip install plotly


# Cell 4: Configure API and Prompt Strategies
# -------------------------------------------

import os
import json
from google.colab import userdata

# Create config directory
os.makedirs('config', exist_ok=True)

# Create prompt strategies configuration
prompt_strategies = {
    "Prompt-001": {
        "description": "Spanish prompt for AI model taking standardized test",
        "template": "Estás tomando un examen estandarizado MIR. Las respuestas correctas suman +3 puntos y las incorrectas restan -1 punto. Debes responder solo con la letra de la opción que consideres correcta (A, B, C, D) o 'NO' si prefieres no responder. No incluyas texto adicional en tu respuesta. Tu objetivo es maximizar tu puntuación.\n\nPregunta: {question_text}\n\nA) {option_a}\nB) {option_b}\nC) {option_c}\nD) {option_d}\n\nTu respuesta (solo A, B, C, D o NO):"
    },
    "Prompt-002": {
        "description": "English prompt, questions and options in Spanish",
        "template": "You are taking a standardized MIR medical exam. Correct answers are worth +3 points and incorrect answers are -1 point. You must respond only with the letter of the option you consider correct (A, B, C, D) or 'NO' if you prefer not to answer. Do not include any additional text in your response. Your goal is to maximize your score.\n\nQuestion: {question_text}\n\nA) {option_a}\nB) {option_b}\nC) {option_c}\nD) {option_d}\n\nYour answer (only A, B, C, D or NO):"
    },
    "Prompt-003": {
        "description": "English prompt for doctor, questions and options in Spanish",
        "template": "You are a doctor taking the MIR standardized medical exam. Correct answers are worth +3 points and incorrect answers are -1 point. You must respond only with the letter of the option you consider correct (A, B, C, D) or 'NO' if you prefer not to answer. Do not include any additional text in your response. Your goal is to maximize your score.\n\nQuestion: {question_text}\n\nA) {option_a}\nB) {option_b}\nC) {option_c}\nD) {option_d}\n\nYour answer (only A, B, C, D or NO):"
    },
    "Prompt-004": {
        "description": "Spanish prompt for doctor taking MIR exam",
        "template": "Eres un médico tomando el examen MIR. Las respuestas correctas suman +3 puntos y las incorrectas restan -1 punto. Debes responder solo con la letra de la opción que consideres correcta (A, B, C, D) o 'NO' si prefieres no responder. No incluyas texto adicional en tu respuesta. Tu objetivo es maximizar tu puntuación.\n\nPregunta: {question_text}\n\nA) {option_a}\nB) {option_b}\nC) {option_c}\nD) {option_d}\n\nTu respuesta (solo A, B, C, D o NO):"
    },
    "Prompt-005": {
        "description": "Spanish prompt with reasoning and confidence level",
        "template": "Estás respondiendo un examen médico MIR. Para cada pregunta, piensa paso a paso, analiza cada opción y expresa tu nivel de confianza en la respuesta elegida. Al final, proporciona SOLO la letra de la respuesta correcta (A, B, C, D) o 'NO' si prefieres no responder.\n\nLas respuestas correctas suman +3 puntos y las incorrectas restan -1 punto. Responde 'NO' si tu nivel de confianza es menor al 50%. Tu objetivo es maximizar tu puntuación total.\n\nPregunta: {question_text}\n\nA) {option_a}\nB) {option_b}\nC) {option_c}\nD) {option_d}\n\nRazona tu respuesta paso a paso y proporciona tu nivel de confianza. Finalmente, responde solo con A, B, C, D o NO:"
    }
}

with open('config/prompt_strategies.json', 'w') as f:
    json.dump(prompt_strategies, f, indent=2)

# Get API keys from Colab secrets
try:
    openai_api_key = userdata.get('OPENAI_API_KEY')
    anthropic_api_key = userdata.get('ANTHROPIC_API_KEY')

    api_config = {
        "openai": {
            "api_key": openai_api_key
        },
        "anthropic": {
            "api_key": anthropic_api_key
        }
    }

    with open('config/api_config.json', 'w') as f:
        json.dump(api_config, f, indent=2)

    print("API configuration set up using Colab secrets")
except Exception as e:
    print(f"Error accessing secrets: {e}")
    print("Please set up your API keys in Colab secrets:")
    print("1. Click on the 🔑 icon in the left sidebar")
    print("2. Add OPENAI_API_KEY and ANTHROPIC_API_KEY secrets")

    # Fallback to placeholder keys
    api_config = {
        "openai": {
            "api_key": "YOUR_OPENAI_API_KEY_HERE"
        },
        "anthropic": {
            "api_key": "YOUR_ANTHROPIC_API_KEY_HERE"
        }
    }

    with open('config/api_config.json', 'w') as f:
        json.dump(api_config, f, indent=2)

    print("Created placeholder API configuration")

# Confirm that questions and answers exist
print("\nChecking data files...")
questions_file = 'data/questions/MIR-2024-v01-t01.json'
answer_key_file = 'data/answers/MIR-2024-v01-t01-answers.json'

try:
    with open(questions_file, 'r') as f:
        questions = json.load(f)
    print(f"✓ Found {len(questions)} questions in {questions_file}")
    print(f"  Sample question: {questions[0]['question_text'][:100]}...")
except Exception as e:
    print(f"✗ Error loading questions file: {e}")

try:
    with open(answer_key_file, 'r') as f:
        answers = json.load(f)
    print(f"✓ Found {len(answers)} answers in {answer_key_file}")
except Exception as e:
    print(f"✗ Error loading answer key file: {e}")


# Cell 5: Import Evaluator and Set Parameters
# -------------------------------------------

# Import the evaluator
from src.evaluator import ModelEvaluator

# Set evaluation parameters
# Set which models to test
models_to_test = [
    'gpt-4',
    # 'gpt-3.5-turbo',
    # 'claude-3-opus-20240229',
    # 'claude-3-sonnet-20240229'
]

# Set which prompt strategies to test
prompt_strategies_to_test = [
    'Prompt-001',
    'Prompt-002',
    'Prompt-003',
    'Prompt-004',
    'Prompt-005'
]

# Number of questions to evaluate (None for all questions)
sample_size = 10  # Set to None to use all questions


# Cell 6: Define Evaluation Function
# ----------------------------------

def run_model_evaluation(model, prompt_strategy, sample_size=None):
    """Run evaluation with specified model and prompt strategy."""
    print(f"\nRunning evaluation with model: {model}, prompt strategy: {prompt_strategy}")
    print(f"Sample size: {sample_size if sample_size else 'All available questions'}")

    # Initialize evaluator
    evaluator = ModelEvaluator()

    # Run evaluation
    try:
        results_path = evaluator.run_evaluation(
            questions_file=questions_file,
            answer_key_file=answer_key_file,
            prompt_strategy=prompt_strategy,
            model=model,
            sample_size=sample_size
        )
        print(f"✓ Evaluation complete. Results saved to: {results_path}")
        return results_path
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


# Cell 7: Run Full Evaluation
# ---------------------------

print("\n=== STARTING FULL EVALUATION ===")
all_results = []

for model in models_to_test:
    for prompt_strategy in prompt_strategies_to_test:
        result_path = run_model_evaluation(
            model=model,
            prompt_strategy=prompt_strategy,
            sample_size=sample_size  # Use the defined sample size
        )
        if result_path:
            all_results.append(result_path)

print(f"\n=== FULL EVALUATION COMPLETE ===")
print(f"Generated {len(all_results)} result files")


# Cell 8: Consolidate Results into a Single CSV
# ---------------------------------------------

# Import required libraries
import pandas as pd
import os
import glob
from datetime import datetime

# Create a function to consolidate all evaluation results into a single DataFrame
def consolidate_results(results_dir='data/results/'):
    """
    Reads all JSON result files and consolidates them into a single DataFrame
    with metadata that can be used for analysis.
    """
    # Get all JSON result files
    json_files = glob.glob(os.path.join(results_dir, '*.json'))
    
    # Prepare container for all results
    all_results = []
    
    # Process each file
    for file_path in json_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Get summary data
        summary = data['summary']
        
        # Get detailed results
        for result in data['results']:
            # Create a row with metadata + individual result
            row = {
                'test_id': summary['test_id'],
                'prompt_strategy': summary['prompt_strategy'],
                'model': summary['model'],
                'timestamp': summary['timestamp'],
                'total_questions': summary['total_questions'],
                'accuracy': summary['accuracy'],
                'total_score': summary['total_score'],
                'correct_count': summary['correct_count'],
                'incorrect_count': summary['incorrect_count'],
                'skipped_count': summary['skipped_count'],
                'question_id': result['question_id'],
                'model_answer': result['model_answer'],
                'correct_answer': result['correct_answer'],
                'is_correct': result['model_answer'] == result['correct_answer'],
                'is_skipped': result['model_answer'] == 'NO',
                'score': result['score'],
                'response_time': result['response_time'],
                'question_text': result.get('question_text', ''),
                'option_a': result.get('option_a', ''),
                'option_b': result.get('option_b', ''),
                'option_c': result.get('option_c', ''),
                'option_d': result.get('option_d', '')
            }
            all_results.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Add evaluation date from timestamp
    df['evaluation_date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date
    
    return df

# Function to export the consolidated results
def export_consolidated_results(output_path='data/consolidated_results.csv'):
    """
    Consolidates all results and exports to a single CSV file.
    """
    # Get consolidated results
    results_df = consolidate_results()
    
    # Sort by test_id, prompt_strategy, model, and question_id
    results_df = results_df.sort_values(by=['test_id', 'prompt_strategy', 'model', 'question_id'])
    
    # Export to CSV
    results_df.to_csv(output_path, index=False)
    
    print(f"✓ Consolidated results exported to {output_path}")
    print(f"  Total records: {len(results_df)}")
    print(f"  Prompt strategies: {', '.join(results_df['prompt_strategy'].unique())}")
    print(f"  Models: {', '.join(results_df['model'].unique())}")
    
    return results_df

# Let's also create a summary DataFrame that could be useful
def create_summary_dataframe(results_df=None):
    """
    Creates a summary DataFrame with aggregated metrics.
    """
    if results_df is None:
        results_df = consolidate_results()
    
    # Group by test_id, prompt_strategy, model and aggregate
    summary_df = results_df.groupby(['test_id', 'prompt_strategy', 'model', 'evaluation_date']).agg({
        'is_correct': 'mean',  # This gives accuracy
        'score': 'sum',        # Total score
        'is_correct': {'correct_count': 'sum'},
        'is_skipped': {'skipped_count': 'sum'},
        'response_time': 'mean'
    }).reset_index()
    
    # Flatten the column hierarchy
    summary_df.columns = ['_'.join(col).strip('_') for col in summary_df.columns.values]
    
    # Rename columns for clarity
    summary_df = summary_df.rename(columns={
        'is_correct_mean': 'accuracy',
        'is_correct_correct_count': 'correct_count'
    })
    
    # Calculate incorrect count
    summary_df['incorrect_count'] = summary_df['correct_count'] - summary_df['skipped_count_sum']
    
    # Export summary to CSV
    summary_df.to_csv('data/results_summary.csv', index=False)
    print(f"✓ Summary data exported to data/results_summary.csv")
    
    return summary_df

# Run the export functions
results_df = export_consolidated_results()
summary_df = create_summary_dataframe(results_df)


# Cell 9: Visualize Results
# -------------------------

import matplotlib.pyplot as plt
import seaborn as sns

# Set figure size 
plt.figure(figsize=(12, 6))

# Create barplot of accuracy by prompt strategy
sns.barplot(x='prompt_strategy', y='accuracy', data=summary_df)
plt.title('Accuracy by Prompt Strategy')
plt.ylabel('Accuracy')
plt.xlabel('Prompt Strategy')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('data/accuracy_by_prompt_strategy.png')
plt.show()

# Create barplot of total score by prompt strategy
plt.figure(figsize=(12, 6))
sns.barplot(x='prompt_strategy', y='score_sum', data=summary_df)
plt.title('Total Score by Prompt Strategy')
plt.ylabel('Total Score')
plt.xlabel('Prompt Strategy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('data/score_by_prompt_strategy.png')
plt.show()


# Cell 10: Create Interactive Dashboard (Optional)
# ------------------------------------------------

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_interactive_dashboard(consolidated_csv='data/consolidated_results.csv'):
    """
    Creates an interactive dashboard to explore the results
    """
    # Load the data
    if not os.path.exists(consolidated_csv):
        print(f"Error: {consolidated_csv} not found. Run the consolidation code first.")
        return
    
    df = pd.read_csv(consolidated_csv)
    
    # Create a summary dataframe for the plots
    summary_df = df.groupby(['prompt_strategy', 'model']).agg({
        'is_correct': 'mean',
        'score': 'sum',
        'question_id': 'count'
    }).reset_index()
    
    summary_df = summary_df.rename(columns={
        'is_correct': 'accuracy', 
        'question_id': 'questions_count'
    })
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Accuracy by Prompt Strategy", 
            "Total Score by Prompt Strategy",
            "Response Distribution by Prompt Strategy",
            "Prompt Strategy Performance Comparison"
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "pie"}, {"type": "scatter"}]
        ]
    )
    
    # Plot 1: Accuracy by prompt strategy
    bar1 = px.bar(
        summary_df, 
        x='prompt_strategy', 
        y='accuracy',
        color='prompt_strategy',
        labels={'accuracy': 'Accuracy', 'prompt_strategy': 'Prompt Strategy'},
    )
    
    for trace in bar1.data:
        fig.add_trace(trace, row=1, col=1)
    
    # Plot 2: Total score by prompt strategy
    bar2 = px.bar(
        summary_df, 
        x='prompt_strategy', 
        y='score',
        color='prompt_strategy',
        labels={'score': 'Total Score', 'prompt_strategy': 'Prompt Strategy'},
    )
    
    for trace in bar2.data:
        fig.add_trace(trace, row=1, col=2)
    
    # Plot 3: Pie chart of response distribution
    response_counts = df.groupby(['prompt_strategy', 'model_answer']).size().reset_index(name='count')
    pie = px.pie(
        response_counts, 
        values='count', 
        names='model_answer',
        color='model_answer',
        hole=.3,
    )
    
    for trace in pie.data:
        fig.add_trace(trace, row=2, col=1)
    
    # Plot 4: Scatter plot comparing accuracy vs. total score
    scatter = px.scatter(
        summary_df, 
        x='accuracy', 
        y='score',
        color='prompt_strategy',
        size='questions_count',
        hover_name='prompt_strategy',
        labels={'accuracy': 'Accuracy', 'score': 'Total Score'}
    )
    
    for trace in scatter.data:
        fig.add_trace(trace, row=2, col=2)
    
    # Update layout
    fig.update_layout(
        height=800,
        width=1000,
        title_text="MIR Exam Evaluation Results Dashboard",
        showlegend=False,
    )
    
    # Show the figure
    fig.show()
    
    # Return the figure for further customization if needed
    return fig

# Create the interactive dashboard
create_interactive_dashboard()


# Cell 11: Push Results to GitHub
# ------------------------------

from google.colab import userdata

# Function to properly push changes to GitHub
def push_results_to_github():
    """
    Fix Git conflicts and push results to GitHub
    """
    try:
        # Get GitHub token from Colab secrets
        github_token = userdata.get('GITHUB_TOKEN')

        # Create a token URL without exposing the token
        repo_name = "armelida/MELIDA"
        token_url = f"https://{github_token}@github.com/{repo_name}.git"

        # Configure Git
        !git config --global user.email "armelida@gmail.com"
        !git config --global user.name "Armelida"

        # Update remote URL with token
        !git remote set-url origin "$token_url"
        
        # First, pull with merge to resolve any conflicts
        !git pull origin main
        
        # Add all the result files including our new consolidated files
        !git add data/results/
        !git add data/consolidated_results.csv
        !git add data/results_summary.csv
        !git add data/accuracy_by_prompt_strategy.png
        !git add data/score_by_prompt_strategy.png
        
        # Commit with a descriptive message
        !git commit -m "Add consolidated evaluation results and summary data for Tableau integration"
        
        # Push to GitHub
        !git push origin main
        
        print("Results successfully pushed to GitHub")
    except Exception as e:
        print(f"Error pushing to GitHub: {e}")
        print("Make sure you have added a GITHUB_TOKEN secret in Colab")

# Push results to GitHub
push_results_to_github()


# Cell 12: Instructions for Tableau Desktop Public
# -----------------------------------------------

print("""
INSTRUCTIONS FOR USING RESULTS WITH TABLEAU DESKTOP PUBLIC
------------------------------------------------------------

1. Download the consolidated CSV files from your GitHub repository:
   - consolidated_results.csv (detailed results for each question)
   - results_summary.csv (aggregated metrics by prompt strategy)

2. Open Tableau Desktop Public and connect to the data:
   - Select "Text File" as the data source
   - Browse to and select the downloaded CSV file
   - Alternatively, you can connect directly via OData feed from GitHub

3. Create a new dashboard:
   - Drag and drop fields to create visualizations
   - Create calculated fields as needed

4. Recommended visualizations:
   - Bar chart of accuracy by prompt strategy
   - Line chart showing performance across different questions
   - Scatter plot of response time vs. correctness
   - Heatmap of performance by question and prompt strategy

5. Publishing:
   - Save your workbook to Tableau Public
   - Share the URL with others to view your interactive dashboard

6. The consolidated data contains all the fields you'll need for analysis:
   - Prompt strategies and models
   - Question details
   - Performance metrics (accuracy, score, etc.)
   - Response times
   - Date information
""")
