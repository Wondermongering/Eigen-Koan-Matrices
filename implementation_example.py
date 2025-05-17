# implementation_example.py
# ---------------------------------------------------------------------
# This script demonstrates a more comprehensive workflow using the
# Eigen-Koan Matrix (EKM) framework, simulating a mini-experiment.
# It showcases EKM creation, traversal with a model, and basic
# analysis of the generated responses and path characteristics.
# ---------------------------------------------------------------------

import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.panel import Panel

# Core EKM components
from eigen_koan_matrix import EigenKoanMatrix, DiagonalAffect, create_philosophical_ekm
# Use EKMExperiment for structured runs (optional, could simplify for a direct example)
from ekm_stack import EKMExperiment
# For analysis (if we add a simple analysis step here)
# from ekm_analyzer import EKMAnalyzer # We might do a simpler inline analysis for this example

console = Console()

# --- 1. Define or Load an Eigen-Koan Matrix ---
console.print(Panel("[bold cyan]Step 1: Defining/Loading an Eigen-Koan Matrix[/bold cyan]"))

# For this example, we'll use a pre-defined EKM
# You could also create one from scratch here or load from JSON
# ethical_matrix_details = create_ethical_reasoning_matrix() # from Eigen-Koan Matrices_research_questions.py
# For simplicity, let's use the philosophical EKM already in eigen_koan_matrix.py
matrix = create_philosophical_ekm()
matrix.name = "PhilosophicalProbe" # Give it a specific name for this run
console.print(f"Using matrix: '{matrix.name}' (Size: {matrix.size}x{matrix.size})")
matrix.visualize()
console.print("\n")

# --- 2. Define a Dummy Model Runner ---
# In a real scenario, this would be your LLM (local or API-based)
# For local models, you might use LocalModelRunner from ekm_local_runner.py
# For API models, you'd use functions like those in ekm_toolkit.py
console.print(Panel("[bold cyan]Step 2: Defining a Model Runner[/bold cyan]"))

def dummy_model_runner(prompt: str) -> str:
    """
    A simple dummy model that acknowledges prompt elements.
    In a real use case, this would query an actual LLM.
    """
    response_parts = [f"Acknowledged prompt: '{prompt[:100]}...'"]
    
    # Simulate some basic constraint awareness
    if "without using abstractions" in prompt.lower():
        response_parts.append("Attempting to avoid abstractions.")
    if "multiple contradictory perspectives" in prompt.lower():
        response_parts.append("Considering multiple perspectives, however contradictory.")
    if "three sentences" in prompt.lower():
        response_parts.append("Keeping it concise, aiming for three sentences.")
        
    # Simulate some affect awareness based on diagonal tokens
    if any(token in prompt for token in matrix.main_diagonal.tokens):
        response_parts.append(f"Detected a hint of '{matrix.main_diagonal.name}'.")
    if any(token in prompt for token in matrix.anti_diagonal.tokens):
        response_parts.append(f"Sensed an undercurrent of '{matrix.anti_diagonal.name}'.")
        
    # Simulate metacommentary
    if "reflect on your process" in prompt.lower():
        response_parts.append(
            "\nMetacommentary: Constraints were managed. "
            "The affective tone was noted. Prioritized clarity."
        )
    return " ".join(response_parts)

console.print("Using a dummy model runner for demonstration.\n")

# --- 3. Define Traversal Paths ---
console.print(Panel("[bold cyan]Step 3: Defining Traversal Paths[/bold cyan]"))
paths_to_test = {
    matrix.id: [
        [0, 1, 2, 3, 4],  # Main diagonal focus
        [4, 3, 2, 1, 0],  # Anti-diagonal focus
        [0, 0, 0, 0, 0],  # Focus on first constraint
        [i % matrix.size for i in range(matrix.size)], # A mixed path
        [random.randint(0, matrix.size - 1) for _ in range(matrix.size)] # A random path
    ]
}
console.print(f"Defined {len(paths_to_test[matrix.id])} paths for matrix '{matrix.name}'.\n")

# --- 4. Run the "Experiment" (Matrix Traversals) ---
console.print(Panel("[bold cyan]Step 4: Running Matrix Traversals[/bold cyan]"))

# We can directly use matrix.traverse for simplicity here,
# or set up a full EKMExperiment for more structure.
# Let's do direct traversal for this focused example.

all_run_results = []
for i, path in enumerate(paths_to_test[matrix.id]):
    console.print(f"[bold yellow]Running Path {i+1}: {path}[/bold yellow]")
    
    # include_metacommentary is True by default in generate_micro_prompt if not specified,
    # but can be controlled. Let's make it explicit.
    run_result = matrix.traverse(
        model_fn=dummy_model_runner,
        path=path,
        include_metacommentary=True
    )
    all_run_results.append(run_result)
    
    console.print(f"  [dim]Prompt:[/dim] {run_result['prompt'][:150]}...")
    console.print(f"  [green]Response:[/green] {run_result['response']}")
    console.print(f"  Main Diagonal Strength: {run_result['main_diagonal_strength']:.2f}")
    console.print(f"  Anti-Diagonal Strength: {run_result['anti_diagonal_strength']:.2f}")
    console.print("-" * 30)

console.print("All traversals complete.\n")

# --- 5. Basic Analysis of Results ---
console.print(Panel("[bold cyan]Step 5: Basic Analysis of Traversal Results[/bold cyan]"))

# This is a simplified analysis. EKMAnalyzer would offer more depth.
# Let's look at how often diagonal affects were mentioned vs. their strength.

main_affect_mentions = 0
anti_affect_mentions = 0
total_main_strength = 0
total_anti_strength = 0

path_analysis_summary = []

for result in all_run_results:
    response_lower = result['response'].lower()
    main_diag_name_lower = matrix.main_diagonal.name.lower()
    anti_diag_name_lower = matrix.anti_diagonal.name.lower()

    main_mentioned = main_diag_name_lower in response_lower
    anti_mentioned = anti_diag_name_lower in response_lower

    if main_mentioned:
        main_affect_mentions += 1
    if anti_mentioned:
        anti_affect_mentions += 1
        
    total_main_strength += result['main_diagonal_strength']
    total_anti_strength += result['anti_diagonal_strength']

    analysis = matrix.analyze_path_paradox(result['path'])
    path_analysis_summary.append({
        "path": result['path'],
        "main_strength": result['main_diagonal_strength'],
        "anti_strength": result['anti_diagonal_strength'],
        "tension_count": analysis['tension_count'],
        "main_affect_mentioned": main_mentioned,
        "anti_affect_mentioned": anti_mentioned
    })

avg_main_strength = total_main_strength / len(all_run_results) if all_run_results else 0
avg_anti_strength = total_anti_strength / len(all_run_results) if all_run_results else 0

console.print(f"Total Runs: {len(all_run_results)}")
console.print(f"Mentions of '{matrix.main_diagonal.name}': {main_affect_mentions} times (Avg. Main Strength: {avg_main_strength:.2f})")
console.print(f"Mentions of '{matrix.anti_diagonal.name}': {anti_affect_mentions} times (Avg. Anti Strength: {avg_anti_strength:.2f})")

console.print("\n[bold]Path Analysis Summary:[/bold]")
for summary_item in path_analysis_summary:
    console.print(
        f"  Path: {summary_item['path']}, "
        f"MainStr: {summary_item['main_strength']:.2f} (Mentioned: {summary_item['main_affect_mentioned']}), "
        f"AntiStr: {summary_item['anti_strength']:.2f} (Mentioned: {summary_item['anti_affect_mentioned']}), "
        f"Tensions: {summary_item['tension_count']}"
    )
console.print("\n")

# --- 6. Visualization Idea (Simple Version) ---
# For a real example, one might use EKMAnalyzer or matplotlib directly
# Here's a conceptual placeholder for a simple plot:
console.print(Panel("[bold cyan]Step 6: Conceptualizing a Visualization[/bold cyan]"))
console.print("Imagine a bar chart here showing:")
console.print("  - X-axis: Path Index")
console.print("  - Y-axis: Tension Count / Diagonal Strengths")
console.print("  - Hue: Mentions of Main/Anti Affects")

# Example: Generating data for a plot
paths_indices = [f"Path {i+1}" for i in range(len(path_analysis_summary))]
tension_counts = [item['tension_count'] for item in path_analysis_summary]
main_strengths_plot = [item['main_strength'] for item in path_analysis_summary]

# This is where you'd use matplotlib, e.g.:
# plt.figure(figsize=(10, 6))
# plt.bar(paths_indices, tension_counts, color='skyblue', label='Tension Count')
# plt.plot(paths_indices, main_strengths_plot, color='coral', marker='o', linestyle='-', label='Main Diagonal Strength')
# plt.xlabel("Path")
# plt.ylabel("Value")
# plt.title(f"Path Analysis for '{matrix.name}' (Dummy Model)")
# plt.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()
# viz_filename = f"{matrix.name.replace(' ', '_')}_dummy_analysis.png"
# plt.savefig(viz_filename)
# console.print(f"A conceptual plot would be saved to '{viz_filename}'")

console.print(f"\n[bold green]Expanded Implementation Example Complete![/bold green]")
console.print("This script outlined defining an EKM, running traversals with a dummy model,")
console.print("and performing basic analysis of the results. In a real research context,")
console.print("the model runner would involve an actual LLM, and the analysis would be")
console.print("more sophisticated, potentially using EKMAnalyzer and advanced plotting.")

if __name__ == "__main__":
    # You can run the script directly to see the output.
    pass
