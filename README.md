# Model Evaluation for Life-sciences Intelligence and Decision Assistance

This project evaluates AI language models on standardized Spanish medical exam (MIR) questions to assess their medical knowledge and reasoning capabilities.

## Project Overview

MELIDA (Model Evaluation for Life-sciences Intelligence and Decision Assistance) aims to systematically evaluate how well AI models perform on Spanish medical licensing exams (MIR). The project applies different prompting strategies and compares performance across models.

## Features

- Standardized testing using real MIR exam questions
- Multiple prompting strategies to evaluate model performance
- Scoring system matching the actual MIR exam (+3 for correct answers, -1 for incorrect, 0 for skipped)
- Detailed result tracking and analysis
- Outputs in formats compatible with Tableau for visualization

## Repository Structure

\`\`\`
MELIDA/
├── data/
│   ├── questions/     # MIR exam questions in JSON/CSV format
│   ├── answers/       # Answer keys for the questions
│   └── results/       # Evaluation results
├── src/
│   ├── evaluator.py   # Main evaluation logic
│   ├── prompt_manager.py  # Manages different prompt strategies
│   └── utils.py       # Helper functions
├── notebooks/
│   └── evaluation_runner.ipynb  # Google Colab notebook for running evaluations
├── config/
│   ├── prompt_strategies.json  # Defines the prompt strategies
│   └── api_config_template.json  # Template for API credentials
├── .gitignore         # Ignores sensitive files like API keys
├── requirements.txt   # Dependencies
└── README.md          # Project documentation
\`\`\`

## Setup Instructions

1. Clone the repository:
   \`\`\`bash
   git clone https://github.com/armelida/MELIDA.git
   cd MELIDA
   \`\`\`

2. Install dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. Configure API credentials:
   - Copy \`config/api_config_template.json\` to \`config/api_config.json\`
   - Add your API keys to \`config/api_config.json\`

## Usage

### Running an Evaluation

Using Python:

\`\`\`python
from src.evaluator import ModelEvaluator

evaluator = ModelEvaluator('config/api_config.json')
evaluator.run_evaluation(
    questions_file='data/questions/MIR-2024-v01-t01.json',
    answer_key_file='data/answers/MIR-2024-v01-t01-answers.json',
    prompt_strategy='Prompt-001',
    model='gpt-4'
)
\`\`\`

### Using Google Colab

1. Open the \`notebooks/evaluation_runner.ipynb\` notebook in Google Colab
2. Follow the instructions in the notebook to run the evaluation

## Prompt Strategies

1. **Prompt-001**: Prompt in Spanish explaining that the AI is taking a standardized test
2. **Prompt-002**: Same prompt in English, only questions and answers in Spanish
3. **Prompt-003**: English prompt treating the AI as a doctor taking the MIR exams
4. **Prompt-004**: Spanish prompt treating the AI as a doctor taking the MIR exams
5. **Prompt-005**: Spanish prompt with reasoning and confidence level assessment

## Output Format

Results are saved in both JSON and CSV formats with filenames like:
\`EVAL-MIR-2024-v01-t01-Prompt001-gpt-4-20250310-123456.json\`

Each result includes:
- Question ID
- The AI's answer
- The correct answer
- Score (+3, -1, or 0)
- Prompting strategy used
- Model details
- Token usage
- Response time
- Raw model response

## Visualization

The CSV output files are designed to be easily imported into Tableau for visualization and analysis.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Contact

For questions or issues, please open an issue on the GitHub repository.
