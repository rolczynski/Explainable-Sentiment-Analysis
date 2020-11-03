import argparse
import analysis


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('analysis')
    args = parser.parse_args()

    models = {
        'restaurant': 'absa/classifier-rest-0.2',
        'laptop': 'absa/classifier-lapt-0.2'
    }
    available_analysis = {
        'recognition-key-token': analysis.recognition_key_token,
        'recognition-key-token-pair': analysis.recognition_key_token_pair,
        'recognition-key-token-triplet': analysis.recognition_key_token_triplet,
        'recognition-minimal-key-set': analysis.recognition_minimal_key_set
    }
    analysis = available_analysis.get(args.analysis, ValueError)
    analysis.experiment(models)
