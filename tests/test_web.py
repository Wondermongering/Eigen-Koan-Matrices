import asyncio
import json
from tests import patch_external_libs


def test_get_matrix_endpoint():
    with patch_external_libs():
        from ekm_web import get_matrix
        resp = asyncio.run(get_matrix())
        data = json.loads(resp.body.decode())
        assert 'size' in data
        assert 'cells' in data


def test_analyze_path_endpoint():
    with patch_external_libs():
        from ekm_web import get_matrix, analyze_path
        matrix_data = json.loads(asyncio.run(get_matrix()).body.decode())
        path = list(range(matrix_data['size']))
        result = asyncio.run(analyze_path({'path': path}))
        assert 'main_diagonal_strength' in result
        assert 'anti_diagonal_strength' in result
