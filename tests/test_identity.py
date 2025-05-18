from tests import patch_external_libs


def test_create_hierarchical_identity():
    with patch_external_libs():
        from hierarchical_identity_tests import create_hierarchical_identity_test
        from recursive_ekm import RecursiveEKM

        rekm = create_hierarchical_identity_test()
        assert isinstance(rekm, RecursiveEKM)
        assert rekm.root_matrix.size == 4
        assert (1, 2) in rekm.sub_matrices
        assert rekm.sub_matrices[(1, 2)].size == 3
