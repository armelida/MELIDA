from src.evaluator import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator()

# Test loading questions and answers
questions = evaluator.load_questions('data/questions/MIR-2024-v01-t01.json')
answers = evaluator.load_answer_key('data/answers/MIR-2024-v01-t01-answers.json')

print(f"Loaded {len(questions)} questions and {len(answers)} answers")
print("Sample question:", questions[0]['question_text'])
print("Answer to first question:", answers[questions[0]['id']])
