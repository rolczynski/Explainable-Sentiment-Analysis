# first line: 78
@memory.cache(ignore=['nlp'])
# The pattern recognizer name is used to distinguish function calls (caching).
def _evaluate(nlp: Pipeline, domain: str, is_test: bool, name: str) -> np.ndarray:
    partial_results = []
    dataset = absa.load_examples('semeval', domain, is_test)
    batches = absa.utils.batches(dataset, batch_size=32)
    for batch in batches:
        predictions = nlp.transform(batch)
        predictions = list(predictions)  # Keep in memory to append at the end.

        new_batch = [mask_tokens(nlp, prediction, k=1) for prediction in predictions]
        new_predictions = nlp.transform(new_batch)

        y_ref = [e.sentiment.value for e in predictions]
        y_new = [e.sentiment.value for e in new_predictions]
        partial_results.extend(zip(y_ref, y_new))
    # It's not a generator because we cache function results.
    return np.array(partial_results)
