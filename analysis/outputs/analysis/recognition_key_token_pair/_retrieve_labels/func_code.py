# first line: 44
@memory.cache(ignore=['nlp'])
def _retrieve_labels(
        nlp: Pipeline,
        domain: str,
        part_parts: Tuple[int, int]
) -> np.ndarray:
    partial_results = []
    examples = mask_examples(nlp, domain, part_parts)
    batches = absa.utils.batches(examples, batch_size=32)
    for batch in batches:
        indices, *masked_tokens_ij, batch_examples = zip(*batch)
        predictions = nlp.transform(batch_examples)
        y_hat = [e.sentiment.value for e in predictions]
        partial_results.extend(zip(indices, *masked_tokens_ij, y_hat))
    return np.array(partial_results)
