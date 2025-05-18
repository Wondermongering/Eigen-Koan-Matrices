from tests import patch_external_libs
import os
import tempfile


def test_repository_add_and_get():
    with patch_external_libs():
        from eigen_koan_matrix import create_random_ekm
        from ekm_repository import EKMRepository
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            db_path = tmp.name
        try:
            repo = EKMRepository(db_path)
            matrix = create_random_ekm(2)
            version = repo.add_matrix(matrix, author="tester", description="init")
            assert version == 1
            fetched = repo.get_matrix(matrix.id)
            assert fetched.id == matrix.id
            assert fetched.size == matrix.size
        finally:
            os.remove(db_path)


def test_repository_versioning():
    with patch_external_libs():
        from eigen_koan_matrix import create_random_ekm
        from ekm_repository import EKMRepository
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            db_path = tmp.name
        try:
            repo = EKMRepository(db_path)
            matrix = create_random_ekm(2)
            v1 = repo.add_matrix(matrix, author="tester", description="v1")
            matrix.name = "Updated"
            v2 = repo.add_matrix(matrix, author="tester", description="v2")
            assert v1 == 1
            assert v2 == 2
            latest = repo.get_matrix(matrix.id)
            assert latest.name == "Updated"
            assert repo.get_matrix(matrix.id, version=1).name != latest.name
        finally:
            os.remove(db_path)
