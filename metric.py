from rouge_score import rouge_scorer

def compute_rouge(reference, generated):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return scores

# Example usage
reference_text = "The quick brown fox jumps over the lazy dog."
generated_text = "The fast brown fox leaps over the lazy dog."

scores = compute_rouge(reference_text, generated_text)
print(scores)
