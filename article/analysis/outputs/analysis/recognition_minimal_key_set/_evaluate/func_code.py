# first line: 38
@memory.cache(ignore=['nlp'])
# The pattern recognizer name is used to distinguish function calls (caching).
def _evaluate(nlp: Pipeline, domain: str, name: str, max_k: int) -> np.ndarray:
    partial_results = []
    dataset = absa.load_examples('semeval', domain, test=True)
    batches = absa.utils.batches(dataset, batch_size=32)
    for batch_i, batch in enumerate(batches):
        predictions = nlp.transform(batch)
        predictions = list(predictions)  # Keep in memory.

        i = np.arange(len(predictions)) + batch_i * 32
        s = [e.sentiment.value for e in predictions]
        k = np.zeros_like(i)
        partial_results.extend(zip(i, s, s, k))

        masked_examples = mask_examples(nlp, predictions, max_k)
        masked_batches = absa.utils.batches(masked_examples, batch_size=32)
        for masked_batch in masked_batches:
            i, s, k, masked_batch_examples = zip(*masked_batch)
            i = np.array(i) + batch_i * 32
            masked_predictions = nlp.transform(masked_batch_examples)
            new_s = [e.sentiment.value for e in masked_predictions]
            partial_results.extend(zip(i, s, new_s, k))
    return np.array(partial_results)
