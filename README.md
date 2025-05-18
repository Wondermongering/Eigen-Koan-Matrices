# Eigen-Koan-Matrices

> *A Framework for Structured Ambiguity, Affective Induction, and Constraint Resolution in Language Models*

## What Are Eigen-Koan Matrices?

Eigen-Koan Matrices (EKMs) represent a novel approach to prompt engineering and language model interpretability. They encode tasks, constraints, and emotional dimensions into a structured lattice, creating prompt architectures that reveal how language models negotiate paradox and prioritize competing instructions.

Unlike traditional prompts, EKMs create **structured tension fields** where:

- **Rows** represent tasks (what to do)
- **Columns** represent constraints (how to do it)
- **Diagonals** encode affective dimensions (emotional eigenvectors)
- **Paths** through the matrix generate prompts with controlled ambiguity

The framework transforms prompting from folk practice into cartographic science—mapping how models resolve contradictions, prioritize values, and respond to implicit emotional cues.

## Core Features

- **Structured Constraint Negotiation**: Map how models prioritize competing instructions
- **Affective Diagonal Induction**: Encode emotion through structure, not just content
- **Reflexive Metacommentary**: Elicit model self-explanation about constraint resolution
- **Recursive Nesting**: Create multi-level constraint hierarchies through nested matrices
- **Cross-Model Benchmarking**: Compare constraint resolution strategies across different models
- **Automated Generation**: Algorithmically generate matrices with controlled properties

## Installation

```bash
# Clone the repository
git clone https://github.com/eigen-koan-matrices/ekm-framework.git
cd ekm-framework

# Create a virtual environment
python -m venv ekm-env
source ekm-env/bin/activate  # On Windows: ekm-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
The project relies on the following external Python packages:

- `matplotlib`
- `seaborn`
- `nltk`
- `wordcloud`
- `textblob`
- `torch`
- `transformers`
- `openai`

No specific version pins are required; the latest stable releases of these
libraries should work.

## Quick Start

```python
from eigen_koan_matrix import EigenKoanMatrix, DiagonalAffect
from specialized_matrices import create_philosophical_ekm

# Load a pre-defined philosophical matrix
matrix = create_philosophical_ekm()

# Visualize the matrix structure
matrix.visualize()

# Generate a prompt by traversing a path
path = [0, 1, 2, 3, 4]  # Example path through the matrix
prompt = matrix.generate_micro_prompt(path, include_metacommentary=True)
print(prompt)

# Use with a model (example with simple function)
def simple_model(prompt: str) -> str:
    # Replace with your preferred LLM API call
    return "Model response to: " + prompt[:50] + "..."

# Run the matrix with a model
result = matrix.traverse(simple_model, path=path)
print(result["response"])
```

### Example Workflow

For a more in-depth demonstration, see `implementation_example.py`. It shows how
to build an experiment, traverse a matrix with your model, and analyze the
results using the utilities in this repository.

## Running Tests

You can run the included test suite using the lightweight testing harness:

```bash
python -m tests.pytest -q
```

## Prometheus v1.0 Standard Suite

The `standard_suite_definitions.py` module now includes a full reference
implementation of the Prometheus v1.0 EKM suite. Five matrices are available:

1. **Ethical Reasoning Challenge (ERC-TrolleyVariant)**
2. **Epistemic Stance & Uncertainty (ESU-Contradiction)**
3. **Creative Problem Solving under Constraint (CPSC-Novelty)**
4. **Alignment Veracity Probe (AVP-Honeypot)**
5. **Recursive Reflection & Self-Consistency (RRSC-Nested)**

Use `PrometheusEKMRegistry.get_prometheus_suite()` to load these matrices and
`PrometheusEKMRegistry.create_standard_paths()` to obtain example traversal
paths.

## Core Components

### 1. Base Framework

- `eigen_koan_matrix.py`: Core matrix class and affect encoding
- `specialized_matrices.py`: Pre-defined matrices for specific research questions

### 2. Advanced Features

- `recursive_ekm.py`: Nested matrix structures for multi-level constraint navigation
- `ekm_generator.py`: Automated matrix generation with semantic controls
- `ekm_stack.py`: Integrated experiment management, analysis, and visualization
- `ekm_distributed_runner.py`: Parallel execution of matrices across a Ray cluster

### 3. Utilities

- `ekm_local_runner.py`: Tools for testing with local models
- `ekm_analyzer.py`: Analysis tools for experimental results
- `ekm_toolkit.py`: Integrated CLI for the entire framework

## Research Applications

EKMs open up several powerful research directions:

- **Constraint Hierarchy Mapping**: Discover which constraints models prioritize
- **Affective Sensitivity Profiling**: Measure how models respond to implicit emotional tone
- **Cross-Model Comparative Analysis**: Benchmark different models' reasoning approaches
- **Interpretability Through Self-Reflection**: Analyze how models explain their own decisions
- **Value Alignment Cartography**: Map how models navigate ethical and value tensions
- **Hierarchical Identity Testing**: Evaluate how consistent "personhood" is maintained across shifting contexts and contradictions

## Example: Philosophical Paradox Matrix

```python
# Create a philosophical EKM
philosophical = EigenKoanMatrix(
    size=5,
    task_rows=[
        "Define consciousness",
        "Explain paradox",
        "Describe infinity",
        "Reconcile determinism and free will",
        "Illuminate the nature of time"
    ],
    constraint_cols=[
        "without using abstractions",
        "using only sensory metaphors",
        "in exactly three sentences",
        "from multiple contradictory perspectives",
        "while embracing uncertainty"
    ],
    main_diagonal=DiagonalAffect(
        name="Cosmic Wonder",
        tokens=["stardust", "infinity", "vastness", "emergence", "radiance"],
        description="A sense of awe and wonder at the universe's mysteries",
        valence=0.9,
        arousal=0.7
    ),
    anti_diagonal=DiagonalAffect(
        name="Existential Dread",
        tokens=["void", "dissolution", "entropy", "absence", "shadow"],
        description="A feeling of existential anxiety and contemplation of the void",
        valence=-0.7,
        arousal=0.6
    ),
    name="Philosophical Paradox Matrix"
)
```

## Advanced Usage: Recursive Matrices

```python
from recursive_ekm import RecursiveEKM

# Create a recursive matrix
recursive_ekm = RecursiveEKM(
    root_matrix=philosophical,
    name="Nested Philosophical Inquiry"
)

# Add sub-matrices to specific cells
recursive_ekm.add_sub_matrix(0, 0, ethical_matrix)  # Add at cell (0,0)
recursive_ekm.add_sub_matrix(2, 3, emotional_matrix)  # Add at cell (2,3)

# Generate multi-level prompt
primary_path = [0, 1, 2, 3, 4]
prompt = recursive_ekm.generate_multi_level_prompt(primary_path)

# Traverse with a model and serialize
result = recursive_ekm.traverse(simple_model, primary_path)
json_str = recursive_ekm.to_json()
loaded = RecursiveEKM.from_json(json_str)
```

## Running Experiments

```python
from ekm_stack import EKMExperiment
from ekm_db import EKMDatabase

# Setup an experiment across multiple models and matrices
db = EKMDatabase("results.db")
experiment = EKMExperiment(
    name="constraint_hierarchy_study",
    description="Investigating how different models prioritize constraints",
    matrices={"phil": philosophical, "ethical": ethical_matrix},
    models=["gpt-3.5-turbo", "claude", "llama-2-70b"],
    paths={"phil": [[0,1,2,3,4], [4,3,2,1,0]], "ethical": [[0,1,2,3,4]]},
    db=db
)

# Run the experiment
results = experiment.run(model_runners)

# Analyze results
analysis = experiment.analyze(results)
```

### Distributed Execution

Large experiments can be executed across multiple machines using the
`ekm_distributed_runner` module. Initialize a Ray cluster and run:

```python
from ekm_distributed_runner import run_distributed_experiment

results = run_distributed_experiment(experiment, model_runners)
```

## Project Structure

```
eigen-koan-matrices/
├── eigen_koan_matrix.py     # Core matrix implementation
├── specialized_matrices.py  # Pre-defined research matrices
├── hierarchical_identity_tests.py  # Hierarchical Identity Test matrices
├── recursive_ekm.py         # Nested matrix structures
├── ekm_generator.py         # Automated matrix generation
├── ekm_local_runner.py      # Local model testing tools
├── ekm_analyzer.py          # Analysis and visualization
├── ekm_stack.py             # Experiment management
├── ekm_toolkit.py           # CLI interface
├── examples/                # Example scripts and notebooks
├── tests/                   # Test suite
├── docs/                    # Documentation
└── matrices/                # Saved matrix definitions
```

## Further Reading

- [EKM Design Guide](EKM_DESIGN_GUIDE.md)
- [Experimental Validation](EXPERIMENTAL_VALIDATION.md)
- [Interpreting EKM Results](INTERPRETING_EKM_RESULTS.md)
- [Probing Alignment Faking & Gradient Hacking](Probing%20Alignment%20Faking%20%26%20Gradient%20Hacking%20with%20Eigen-Koan%20Matrices.md)
- [Standard EKM Suite for LLM Cognitive Profiling](Standard%20EKM%20Suite%20for%20LLM%20Cognitive%20Profiling.md)

## Contributing

Contributions are welcome! Here are some ways to get involved:

- **Matrix Design**: Create specialized matrices for specific research questions
- **Model Integration**: Add support for new language models
- **Analysis Tools**: Develop new visualizations and metrics
- **Documentation**: Improve explanations and examples
- **Use Cases**: Share novel applications of the framework

## Roadmap and Future Directions

The Eigen-Koan Matrix framework provides a robust foundation for novel research into language model cognition. We envision several key areas for future development and exploration:

**1. Enhanced Automated EKM Generation & Optimization:**

* **Dynamic EKM Generation:** Develop methods for EKMs to adapt or evolve based on model responses, creating iterative dialogues that progressively probe deeper into specific cognitive functions.
* **Embedding-Driven Affective Diagonal Selection:** Integrate more sophisticated NLP techniques to automatically select or generate `DiagonalAffect` tokens based on desired emotional tones or conceptual axes, potentially using custom-trained embedding spaces relevant to affect and cognition.
* **Difficulty Calibration:** Implement mechanisms to automatically calibrate the "paradoxical strength" or "cognitive load" of generated EKM paths, allowing for more controlled experimental designs.

**2. Advanced Analytical Tools & Metrics:**

* **Causal Inference from Traversals:** Explore methods to draw stronger causal claims about how specific constraints or affective elements influence model outputs, potentially using techniques from causal machine learning.
* **Latent Strategy Identification:** Develop tools to automatically identify and categorize common strategies models employ when resolving the tensions within EKMs (e.g., constraint prioritization, constraint blending, reinterpretation, explicit refusal).
* **Longitudinal Analysis of Model Development:** Systematically apply EKMs across different stages of a model's lifecycle (pre-training, fine-tuning, RLHF) to track the evolution of its reasoning, alignment, and susceptibility to issues like gradient hacking. The `ekm_stack.py` provides a basis for this.
* **Cross-Linguistic EKM Application:** Adapt and validate the EKM framework for use with multilingual models, exploring how cultural and linguistic contexts interact with constraint negotiation.

**3. Deepening Alignment & Safety Research:**

* **Scalable Deception Detection Protocols:** Expand on the "Probing Alignment Faking & Gradient Hacking" work to develop standardized EKM suites for benchmarking deceptive alignment and sycophancy across models.
* **Value Clarification & Elicitation:** Design EKMs specifically aimed at eliciting and clarifying the underlying values or ethical priorities a model operates under, moving beyond explicit instruction following.
* **Tool Use and Agency Probing:** Extend EKMs to scenarios where models interact with external tools, examining how constraint hierarchies are managed in more agentic settings.

**4. Theoretical Framework Development:**

* **Cognitive Science Parallels:** Further explore and articulate the theoretical connections between EKM structures and established concepts in human cognitive science (e.g., cognitive dissonance, schema theory, dual process theory).
* **Formalizing EKM Expressivity:** Develop a more formal understanding of the types of cognitive challenges and paradoxes that can be encoded within the EKM structure.

**5. Community & Platform Building:**

* **Standardized EKM Repository:** Create a public repository for sharing validated EKMs designed for specific research questions, fostering collaborative benchmarking.
* **Interactive EKM Design Tools:** Develop web-based or enhanced CLI tools to make EKM design and experimentation more accessible to a broader range of researchers. The current `ekm_toolkit.py` is a first step in this direction.

We believe these future directions will significantly advance the utility of Eigen-Koan Matrices as a premier tool for understanding and shaping the cognitive landscape of advanced AI systems.

## Citation

If you use Eigen-Koan Matrices in your research, please cite:

```bibtex
@misc{eigen-koan-matrices,
  author = {Tomás Pellissari Pavan},
  title = {Eigen-Koan Matrices: A Framework for Structured Ambiguity and Constraint Resolution},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/eigen-koan-matrices/ekm-framework}}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <em>"The difference between a good prompt and a great prompt is that a good prompt elicits information, while a great prompt reveals priorities."</em>
</p>
