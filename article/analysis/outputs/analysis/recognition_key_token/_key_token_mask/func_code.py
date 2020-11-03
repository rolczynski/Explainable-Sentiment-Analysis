# first line: 56
@memory.cache(ignore=['nlp'])
def _key_token_mask(nlp: Pipeline, domain: str, is_test: bool) -> np.ndarray:
    partial_results = []
    examples = mask_examples(nlp, domain, is_test)
    batches = absa.utils.batches(examples, batch_size=32)
    for batch in batches:
        indices, mask_index, batch_examples = zip(*batch)
        predictions = nlp.transform(batch_examples)
        y_hat = [e.sentiment.value for e in predictions]
        partial_results.extend(zip(indices, mask_index, y_hat))
    return np.array(partial_results)
