import asyncio
import json
from tests import patch_external_libs


def test_get_designer_matrix():
    with patch_external_libs():
        from ekm_designer import get_matrix
        resp = asyncio.run(get_matrix())
        data = json.loads(resp.body.decode())
        assert 'size' in data
        assert 'cells' in data


def test_swap_tasks_endpoint():
    with patch_external_libs():
        from ekm_designer import get_matrix, swap_tasks
        initial = json.loads(asyncio.run(get_matrix()).body.decode())
        if initial['size'] < 2:
            return
        asyncio.run(swap_tasks({'row1': 0, 'row2': 1}))
        updated = json.loads(asyncio.run(get_matrix()).body.decode())
        assert updated['task_rows'][0] == initial['task_rows'][1]


