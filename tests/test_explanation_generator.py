from tests import patch_external_libs


def test_generate_explanation_basic():
    with patch_external_libs():
        from explanation_generator import generate_explanation

        analysis = {
            'matrix_name': 'M',
            'model_name': 'model',
            'response_count': 10,
            'sentiment_scores': [{}] * 10,
            'sentiment_correlations': {
                'main_diag_vs_vader_pos': 0.5,
                'anti_diag_vs_vader_neg': -0.2,
            },
            'word_frequencies': {'hello': 5, 'world': 3},
            'metacommentary_analysis': [
                {'constraint_difficulty': ['a'], 'emotional_detection': [], 'priority_elements': [], 'deprioritized_elements': []}
            ] * 10,
        }

        text = generate_explanation(analysis)
        assert 'Experiment Explanation' in text
        assert 'Most frequent words' in text
