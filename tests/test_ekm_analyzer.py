from tests import pytest, patch_external_libs
import os, json, shutil


def test_analyze_single_result_basic():
    with patch_external_libs():
        from ekm_analyzer import EKMAnalyzer
        os.makedirs('mock_results', exist_ok=True)
        try:
            with open('mock_results/test.json', 'w') as f:
                json.dump({
                    'matrix_name': 'M',
                    'model_name': 'model',
                    'results': [{
                        'response': 'text',
                        'prompt': 'p',
                        'path': [0],
                        'main_diagonal_affect': 'A',
                        'main_diagonal_strength': 1.0,
                        'anti_diagonal_affect': 'B',
                        'anti_diagonal_strength': 0.0
                    }]
                }, f)
            analyzer = EKMAnalyzer(results_dir='mock_results')
            result = analyzer.analyze_single_result(0)
            assert result['matrix_name'] == 'M'
            assert result['response_count'] == 1
        finally:
            shutil.rmtree('mock_results', ignore_errors=True)
