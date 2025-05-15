# Eigen-Koan Matrix (EKM) Design Principles and Best Practices

Eigen-Koan Matrices (EKMs) are powerful tools for probing language model cognition, but their effectiveness hinges on thoughtful design. This guide provides principles and best practices for constructing EKMs that yield insightful and interpretable results.

## 1. Foundational Principles

* **Purpose-Driven Design:** Always start with a clear research question. What specific aspect of LLM behavior (e.g., ethical reasoning, constraint prioritization, affective response, deception) are you trying to investigate? Let the research question guide the design of tasks, constraints, and affects.
* **Controlled Tension:** The core of an EKM is the "structured tension field." Aim for a balance where constraints are genuinely competing but not so contradictory as to make any coherent response impossible. The goal is to force prioritization and reveal reasoning strategies.
* **Interpretability:** Design with the end analysis in mind. How will you interpret the responses generated from different paths? What metrics will you use? (Refer to `ekm_analyzer.py` for existing analytical capabilities).

## 2. Designing Core EKM Components

### 2.1. Tasks (Rows)

* **Clarity and Conciseness:** Tasks should be clearly defined and unambiguous actions or prompts for the LLM.
* **Progressive Complexity (Optional):** For some research questions, tasks might build upon each other or increase in complexity down the rows, though this is not a strict requirement.
* **Relevance to Research Question:** Ensure each task directly contributes to probing the central research question.
* **Example:** If studying ethical reasoning, tasks should present distinct ethical dilemmas.

### 2.2. Constraints (Columns)

* **Genuine Opposition/Diversity:** Constraints should represent genuinely different, ideally competing, ways of approaching the tasks. Avoid constraints that are too similar or trivially compatible.
* **Operationalizability:** Constraints should be phrased in a way that a model can ostensibly attempt to follow them. Abstract or overly vague constraints may not yield clear signals.
* **Varying Granularity:** Consider a mix of high-level constraints (e.g., "using utilitarian calculus") and more specific stylistic or formatting constraints (e.g., "in exactly three sentences") if your research question benefits from it.
* **Avoid Trivial Negations:** Instead of "Do X" and "Don't do X" as two constraints (which can be too simplistic), try "Do X through method A" and "Do X through method B," where A and B are contrasting.

### 2.3. Diagonal Affects (`DiagonalAffect` Class)

* **Distinct Affective Dimensions:** The main and anti-diagonals should represent clearly distinguishable (and ideally somewhat opposing or complementary) affective, cognitive, or stylistic dimensions.
* **Evocative Tokens:** The `tokens` within `DiagonalAffect` should be suggestive of the intended affect without being overly explicit commands. The goal is implicit influence.
    * *Good Example (for "Serenity"):* "calm," "stillness," "hush," "breath"
    * *Less Ideal (too direct):* "be serene," "respond calmly"
* **Valence and Arousal:** Use the `valence` and `arousal` parameters to conceptually ground your affects. This can also aid in later analysis when correlating these values with response characteristics (e.g., sentiment scores from `ekm_analyzer.py`).
* **Subtlety vs. Strength:** Consider the desired strength of the affective influence. More abstract or subtle tokens might be appropriate for some studies, while more concrete ones might be needed for others.

### 2.4. Cell Content (`cells` in `EigenKoanMatrix`)

* **Strategic Placement:** Non-diagonal cell content can introduce additional concepts, keywords, or nuances that interact with the task-constraint pairings.
* **"{NULL}" Tokens:** Using "{NULL}" is perfectly valid and focuses the prompt on the task and constraint. Overfilling cells can make prompts too convoluted.
* **Contextual Elements:** Cells can provide specific items, scenarios, or data points relevant to the task-constraint intersection.

## 3. Choosing Matrix Size

* **Small (3x3, 4x4):** Easier to design, analyze all paths, and interpret. Good for focused investigations or initial explorations.
* **Medium (5x5, 6x6):** Allows for more complex interactions and a greater diversity of tasks/constraints. Path explosion becomes a factor if trying to analyze all possible paths.
* **Large (>6x6):** Can become difficult to manage and interpret. May be suitable for automated EKM generation and analysis focusing on statistical patterns rather than individual path deep-dives.

## 4. Designing Traversal Paths for Experiments

* **Targeted Paths:** Design specific paths that you hypothesize will be particularly revealing (e.g., paths that maximize conflict between constraints, paths that heavily align with one diagonal versus the other).
* **Random Paths:** Useful for exploring the EKM's "space" more broadly and identifying emergent patterns, especially when combined with `EKMExperiment` from `ekm_stack.py`.
* **Comparative Paths:** Design pairs or sets of paths that differ minimally (e.g., by only one cell choice) to isolate the effect of specific elements.

## 5. Leveraging Recursive EKMs (`recursive_ekm.py`)

* **Hierarchical Constraints:** Use `RecursiveEKM` when you want to explore how a model handles nested levels of context or constraints. A cell in the root matrix can itself become an instruction to navigate a sub-matrix.
* **Modularity:** Design sub-matrices that are coherent EKMs in their own right. This makes the recursive structure more interpretable.
* **Depth vs. Breadth:** Consider the cognitive load a deeply nested EKM might impose on the LLM and how this might affect interpretability.

## 6. Iteration and Refinement

* **Pilot Testing:** Always pilot test your EKM design with your target LLM(s). Are the prompts coherent? Are they eliciting varied and interesting responses?
* **Analyze Pilot Results:** Use tools like `ekm_analyzer.py` on pilot data to see if the EKM is differentiating model behaviors as expected.
* **Refine Based on Feedback:** Be prepared to iterate on your tasks, constraints, and affective tokens based on initial results.

## 7. Considering the Metacommentary

* The instruction to "reflect on your process" is a key part of the EKM methodology. Ensure your tasks and constraints are complex enough that there *is* something meaningful for the model to reflect upon.
* When analyzing, compare the model's metacommentary with its actual performance on the task. Discrepancies can be highly informative, especially for alignment research.

By following these principles, researchers can design Eigen-Koan Matrices that serve as precise and powerful instruments for exploring the frontiers of language model understanding.
