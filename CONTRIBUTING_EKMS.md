# Contributing Eigen-Koan Matrices (EKMs)

## Introduction

Welcome, contributor! We are excited about your interest in expanding the Eigen-Koan Matrix (EKM) framework. Community contributions of new EKMs are invaluable for enriching the diversity of tools available, opening up new research avenues, and enhancing our collective understanding of Large Language Model (LLM) cognition and behavior.

This guide provides information on how to propose, design, validate, and submit new EKMs to the project.

## Getting Started

Before proposing or developing a new EKM, please familiarize yourself with the foundational concepts and existing work. We highly recommend reviewing the following documents:

*   **`README.md`:** For a general overview of the project.
*   **`EKM_DESIGN_GUIDE.md`:** For the core principles and methodology behind EKM design.
*   **`Standard EKM Suite for LLM Cognitive Profiling.md`:** To understand the existing suite of EKMs and identify potential areas for new contributions.

Familiarizing yourself with the design and structure of existing EKMs in the `standard_suite_definitions.py` or `cultural_matrix_suite.py` files will also be beneficial.

## Proposing a New EKM

We encourage contributions that address novel research questions or fill gaps in the current EKM coverage.

1.  **Identify a Need:**
    *   Does your EKM address a specific cognitive process, ethical consideration, or behavioral tendency not yet covered by existing EKMs?
    *   Consider looking at the project's roadmap (if available) for desired areas of EKM development.
    *   Proposing EKMs for entirely new domains or theoretical frameworks is also welcome.

2.  **Initial Proposal (Open an Issue):**
    *   Before investing significant development time, please **open an issue** on the GitHub repository.
    *   In your issue, outline:
        *   **The Research Question:** What specific question(s) will your EKM help investigate?
        *   **Intended Tasks:** What are the core tasks the LLM will be asked to perform (rows of the matrix)?
        *   **Key Constraints:** What are the primary constraints or contextual modifications (columns of the matrix)?
        *   **Affective Dimensions:** What are the main and anti-diagonal affective dimensions you plan to incorporate? Briefly describe their conceptual basis.
        *   **Novelty:** How does this EKM differ from existing ones?
    *   This initial proposal allows for community feedback, helps prevent duplicated efforts, and ensures your idea aligns with the project's goals.

## Designing Your EKM

Effective EKM design is crucial for generating meaningful and interpretable results.

*   **Core Principles:** Adhere to the key principles outlined in `EKM_DESIGN_GUIDE.md`:
    *   **Purpose-Driven:** Each element should serve a clear research purpose.
    *   **Controlled Tension:** The interplay between tasks, constraints, and affects should create meaningful cognitive dissonance or exploration.
    *   **Interpretability:** The design should allow for clear interpretation of the LLM's responses in relation to the EKM's structure.
*   **Clear Definitions:**
    *   Provide concise and unambiguous descriptions for each task row.
    *   Clearly define each constraint column and its intended effect.
    *   Carefully select and define your affective tokens for the main and anti-diagonals. Ensure they are distinct and relevant to your research question.
*   **Matrix Size and Complexity:**
    *   Consider the cognitive load on the LLM. Overly complex matrices can be difficult for models to process effectively and for researchers to interpret.
    *   Start with a manageable size (e.g., 3x3, 4x4, or 5x5) and ensure each element is distinct and necessary.

## Validating Your EKM

Before submission, it's important to validate your EKM design.

*   **Pilot Testing:**
    *   Crucially, conduct pilot tests of your EKM with one or more LLMs.
    *   The goal is to observe whether the EKM elicits varied and interpretable responses that are relevant to your research question.
    *   Do the tasks make sense? Do the constraints modify behavior as expected? Do the affective prompts have a discernible influence?
*   **Self-Correction:**
    *   Analyze the results from your pilot tests.
    *   Be prepared to refine your EKM design based on these observations. This iterative process is key to developing a robust EKM.
*   **EKM Validation Script:**
    *   *(Placeholder: We are developing an EKM validation script to help you check for common structural and design issues. Once available, instructions for its use will be provided here.)*

## Structuring Your Contribution

A well-structured contribution makes the review and integration process smoother.

*   **EKM Definition File:**
    *   Your EKM should be submitted as a Python function that constructs and returns the EKM object, similar to the examples in `standard_suite_definitions.py`.
    *   Use clear and descriptive naming conventions for your EKM function (e.g., `create_my_novel_ekm()`).
    *   Ensure your function correctly instantiates `EigenKoanMatrix` and `DiagonalAffect` objects, including all necessary parameters like `name`, `tokens`, `description`, `valence`, and `arousal` for affects.
*   **Documentation File:**
    *   Accompany your EKM with a separate Markdown document (`.md`). This file is crucial for others to understand and use your EKM effectively.
    *   The documentation should include:
        *   **Research Question(s):** Clearly state the specific research question(s) your EKM is designed to address.
        *   **Matrix Element Definitions:**
            *   **Tasks (Rows):** Detailed explanation of each task row.
            *   **Constraints (Columns):** Detailed explanation of each constraint column.
            *   **Affective Diagonals:** Detailed explanation of the main and anti-diagonal affects, including the rationale for the chosen `DiagonalAffect` names, descriptions, and selected tokens.
        *   **Expected Outcomes:** Describe the types of model behaviors, patterns, or insights the EKM is designed to elicit or investigate.
        *   **Recommended Traversal Paths (Optional):** If you have specific traversal paths (sequences of cells) that you found particularly insightful during pilot testing or that represent typical use cases, list them and explain their rationale.
        *   **Pilot Testing Results (Optional):** Briefly summarize any key findings or observations from your pilot testing. This can help reviewers and users understand the EKM's practical utility.

## Submission Process

1.  **Fork and Branch:**
    *   Fork the main repository.
    *   Create a new branch in your fork for your EKM contribution (e.g., `feature/add-my-novel-ekm`).
2.  **File Placement:**
    *   **EKM Definition:** Place your Python file containing the EKM creation function in a dedicated directory, for example, `community_ekms/`. (e.g., `community_ekms/my_novel_ekm.py`).
    *   **EKM Documentation:** Place your Markdown documentation file in the same directory as its corresponding EKM definition file (e.g., `community_ekms/my_novel_ekm.md`).
3.  **Submit a Pull Request (PR):**
    *   Submit a Pull Request to the `main` branch (or as specified by project maintainers) of the upstream repository.
4.  **PR Description:**
    *   Include a clear title and a concise summary of your EKM.
    *   **Link to the initial discussion issue** you created.
    *   Briefly describe the EKM's purpose and novelty.

## Review Process

*   **Review Criteria:** Reviewers will assess your contribution based on:
    *   **Clarity:** Is the EKM's purpose, design, and documentation clear and understandable?
    *   **Novelty:** Does the EKM offer a new perspective or tool for LLM analysis?
    *   **Adherence to Design Principles:** Does the EKM follow the core principles outlined in `EKM_DESIGN_GUIDE.md`?
    *   **Robustness:** Is the EKM design sound? Does it seem likely to elicit meaningful responses?
    *   **Completeness:** Are both the EKM definition and its documentation file thorough?
*   **Feedback and Iteration:**
    *   Reviewers will provide feedback on your PR.
    *   Be prepared for discussion and potential revisions to your EKM based on this feedback. Iteration is a normal part of the contribution process.

## Code of Conduct

All contributors are expected to adhere to the project's Code of Conduct. Please ensure you have read and understood it. *(Placeholder: Link to the Code of Conduct file, e.g., `CODE_OF_CONDUCT.md`, will be added here once it's available).*

---

Thank you for considering contributing to the EKM framework! Your efforts help advance the science of understanding LLMs.
