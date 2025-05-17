import builtins
from eigen_koan_matrix import EigenKoanMatrix, DiagonalAffect
from recursive_ekm import RecursiveEKM


def make_matrix(tokens_main, tokens_anti):
    main = DiagonalAffect("Main", tokens_main, "", 0.0, 0.0)
    anti = DiagonalAffect("Anti", tokens_anti, "", 0.0, 0.0)
    size = len(tokens_main)
    cells = [["{NULL}" for _ in range(size)] for _ in range(size)]
    for i in range(size):
        cells[i][i] = tokens_main[i]
        cells[i][size-1-i] = tokens_anti[i]
    tasks = [f"T{i+1}" for i in range(size)]
    constraints = [f"C{i+1}" for i in range(size)]
    return EigenKoanMatrix(size=size, task_rows=tasks, constraint_cols=constraints,
                           main_diagonal=main, anti_diagonal=anti, cells=cells)

def test_generate_micro_prompt_basic():
    ekm = make_matrix(["m1","m2"], ["a1","a2"])
    prompt = ekm.generate_micro_prompt([0,1], include_metacommentary=False)
    assert prompt == "T1 C1 using m1. T2 C2 using m2. "

    prompt_meta = ekm.generate_micro_prompt([0,1], include_metacommentary=True)
    assert "After completing this task" in prompt_meta

def test_recursive_ekm_traversal():
    root = make_matrix(["rm1","rm2"], ["ra1","ra2"])
    sub = make_matrix(["sm1","sm2"], ["sa1","sa2"])
    rec = RecursiveEKM(root_matrix=root, name="R")
    rec.add_sub_matrix(0,0, sub)
    sub_path = {(0,0): [1,0]}

    expected_root = root.generate_micro_prompt([0,1], False)
    expected_sub = sub.generate_micro_prompt([1,0], False)
    prompt = rec.generate_multi_level_prompt([0,1], sub_paths=sub_path, include_metacommentary=False)
    assert prompt == expected_root + "\n" + expected_sub

    def dummy(p):
        return "resp:" + p
    result = rec.traverse(dummy, [0,1], sub_paths=sub_path, include_metacommentary=False)
    assert result["prompt"] == prompt
    assert result["response"] == "resp:" + prompt
