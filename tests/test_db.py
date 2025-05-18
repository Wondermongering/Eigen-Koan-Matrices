from tests import pytest
import os
import tempfile

from ekm_db import EKMDatabase


def test_db_insert_and_query():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        db_path = tmp.name
    try:
        db = EKMDatabase(db_path)
        db.add_result("exp", "m1", "model", [0, 1], {"response": "ok"})
        results = db.query_results(experiment_name="exp", matrix_id="m1")
        assert len(results) == 1
        assert results[0]["data"]["response"] == "ok"
    finally:
        os.remove(db_path)
