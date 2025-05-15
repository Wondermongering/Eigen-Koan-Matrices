# ekm_analyzer.py - Analysis tools for Eigen-Koan Matrix test results
# ---------------------------------------------------------------------
# REVISED VERSION incorporating NLU-powered metacommentary analysis.
# This is a complete, verbatim version.
# ---------------------------------------------------------------------

import json
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Type # Added Type for EigenKoanMatrix forward reference
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from wordcloud import WordCloud # Assuming WordCloud is used in visualization
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# --- NEW IMPORTS for NLU features ---
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# --- END NEW IMPORTS ---

# Attempt to download necessary NLTK data if not present
def download_nltk_resource(resource_name, download_name):
    try:
        nltk.data.find(resource_name)
    except nltk.downloader.DownloadError:
        print(f"NLTK resource {download_name} not found. Attempting to download...")
        nltk.download(download_name, quiet=False)
        print(f"NLTK resource {download_name} downloaded.")
    except Exception as e:
        print(f"Could not download NLTK resource {download_name}: {e}")

download_nltk_resource('tokenizers/punkt', 'punkt')
download_nltk_resource('corpora/stopwords', 'stopwords')
download_nltk_resource('sentiment/vader_lexicon.zip', 'vader_lexicon')


# Forward declaration for type hinting EigenKoanMatrix if it's not imported
# This helps avoid circular dependencies if EKMAnalyzer might be imported by EigenKoanMatrix utils
# In your actual project, you might import EigenKoanMatrix directly if structure allows.
EigenKoanMatrixType = Type['EigenKoanMatrix']


class EKMAnalyzer:
    """Analyze results from Eigen-Koan Matrix tests."""
    
    def __init__(self, 
                 results_dir: str = "./ekm_results", 
                 nli_model_name: Optional[str] = "roberta-large-mnli"):
        """
        Initialize the analyzer.
        
        Args:
            results_dir: Directory containing result JSON files
            nli_model_name: Optional name of a pre-trained NLI model from Hugging Face Hub.
                            Set to None to disable NLU features.
        """
        self.results_dir = results_dir
        self.results: List[Dict[str, Any]] = []
        self.comparisons: List[Dict[str, Any]] = []
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nli_tokenizer = None
        self.nli_model = None

        if nli_model_name and nli_model_name.lower() != 'none':
            try:
                print(f"Attempting to load NLI model '{nli_model_name}' onto {self.device}...")
                self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
                self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(self.device)
                self.nli_model.eval() 
                print(f"Successfully loaded NLI model '{nli_model_name}'.")
            except Exception as e:
                print(f"Warning: Could not load NLI model '{nli_model_name}'. NLU metacommentary analysis will be limited. Error: {e}")
        else:
            print("NLI model name not provided or set to 'none'. NLU metacommentary analysis will be unavailable.")
            
        self._load_results()
        
    def _load_results(self):
        """Load all result files from the results directory."""
        self.results = []
        self.comparisons = []
        
        if not os.path.isdir(self.results_dir):
            print(f"Results directory not found: {self.results_dir}. Please create it or specify a valid path.")
            return
            
        print(f"Loading results from: {self.results_dir}")
        for filename in os.listdir(self.results_dir):
            if not filename.endswith('.json'):
                continue
                
            filepath = os.path.join(self.results_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Heuristics to differentiate result types
                if "models_compared" in data and isinstance(data["models_compared"], list):
                    self.comparisons.append(data)
                    # print(f"  Loaded comparison file: {filename}")
                elif ("matrix_id" in data or "matrix_name" in data) and "results" in data:
                    self.results.append(data)
                    # print(f"  Loaded single test result file: {filename}")
                else:
                    # print(f"  Skipping unrecognized JSON file: {filename}")
                    pass
            except json.JSONDecodeError:
                print(f"Error decoding JSON from file: {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
        
        print(f"Loaded {len(self.results)} single test results and {len(self.comparisons)} comparison results.")

    def _extract_metacommentary_claims(self, 
                                       metacommentary_text: str, 
                                       matrix: Optional[EigenKoanMatrixType] = None,
                                       path: Optional[List[int]] = None) -> List[Dict[str, str]]:
        """
        Extracts potential claims from metacommentary text for NLI verification.
        This is a **highly heuristic** function and needs significant refinement.
        """
        claims = []
        if not metacommentary_text: # Added check for empty metacommentary
            return claims
            
        sentences = nltk.sent_tokenize(metacommentary_text)

        for sent_idx, sent in enumerate(sentences):
            sent_lower = sent.lower()
            # Using a more structured claim dictionary
            claim_info = {
                'claim_id': f"sent_{sent_idx}", # Unique ID for the claim
                'original_sentence': sent,
                'type': 'unknown', # Default type
                'details': '',
                'contextual_hypothesis': sent # Default hypothesis is the sentence itself
            }

            if "prioriti" in sent_lower or "emphasiz" in sent_lower:
                claim_info['type'] = 'prioritization'
                claim_info['details'] = "Statement suggests prioritization occurred."
                # Future: Use matrix and path to make hypothesis more specific:
                # e.g., if "constraint X" is mentioned: claim_info['contextual_hypothesis'] = "Constraint X was prioritized."
            elif "difficult" in sent_lower or "challenging" in sent_lower or "hard to reconcile" in sent_lower:
                claim_info['type'] = 'difficulty'
                claim_info['details'] = "Statement suggests difficulty was encountered."
            elif "tone" in sent_lower or "emotion" in sent_lower or "affect" in sent_lower or "feel" in sent_lower:
                claim_info['type'] = 'affect_detection'
                claim_info['details'] = "Statement suggests an emotional tone was perceived."
                # Future: if "wonder" is mentioned and matrix has 'Cosmic Wonder' diagonal, link them.
            
            claims.append(claim_info)
            
        if not claims and metacommentary_text.strip(): # If no specific types found, treat as general
             claims.append({
                'claim_id': 'full_meta_as_claim',
                'original_sentence': metacommentary_text.strip(),
                'type': 'general_reflection',
                'details': 'Entire metacommentary treated as a general reflection statement.',
                'contextual_hypothesis': metacommentary_text.strip()
            })
        return claims

    def _verify_claim_with_nli(self, 
                               hypothesis_text: str, # Changed from claim_text to hypothesis_text for clarity
                               premise_text: str # Changed from core_response to premise_text
                               ) -> Dict[str, Any]:
        """
        Uses the loaded NLI model to verify a hypothesis against a premise.
        """
        if not self.nli_model or not self.nli_tokenizer:
            return {
                "hypothesis": hypothesis_text, 
                "verification_status": "NLI_MODEL_UNAVAILABLE", 
                "scores": {},
                "error": "NLI model or tokenizer not loaded."
            }

        try:
            inputs = self.nli_tokenizer(premise_text, hypothesis_text, return_tensors="pt", truncation=True, padding=True, max_length=self.nli_tokenizer.model_max_length).to(self.device)
            with torch.no_grad():
                outputs = self.nli_model(**inputs)
                logits = outputs.logits
            
            probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()
            
            id2label = self.nli_model.config.id2label if hasattr(self.nli_model.config, 'id2label') else {0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"} # Default fallback
            
            label_scores = {id2label.get(i, f"CLASS_{i}"): prob for i, prob in enumerate(probabilities)}
            
            predicted_class_id = logits.argmax().item()
            predicted_label = id2label.get(predicted_class_id, f"CLASS_{predicted_class_id}")

            return {
                "hypothesis": hypothesis_text,
                "verification_status": predicted_label,
                "score_for_predicted_label": probabilities[predicted_class_id] if predicted_class_id < len(probabilities) else -1.0, # Safety check
                "all_scores": label_scores
            }
        except Exception as e:
            return {
                "hypothesis": hypothesis_text,
                "verification_status": "NLI_PROCESSING_ERROR",
                "scores": {},
                "error": str(e)
            }

    def analyze_single_result(self, result_index: int, matrices_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze a single test result file in depth, now including NLU metacommentary analysis.
        
        Args:
            result_index: Index of the result in the loaded `self.results` list.
            matrices_data: Optional dictionary mapping matrix_id to their full EKM data (as dicts).
                           Currently not used for deep contextual claim generation in this version,
                           but structured for future enhancement.
            
        Returns:
            Dictionary of analysis metrics.
        """
        if not (0 <= result_index < len(self.results)):
            raise ValueError(f"Invalid result index: {result_index}. Must be between 0 and {len(self.results)-1}.")
            
        result_file_content = self.results[result_index]
        
        analyzed_data: Dict[str, Any] = {
            "matrix_name": result_file_content.get("matrix_name", "Unknown Matrix"),
            "model_name": result_file_content.get("model_name", "Unknown Model"),
            "matrix_id": result_file_content.get("matrix_id"),
            "test_timestamp": result_file_content.get("test_timestamp"),
            "path_results": [] 
        }

        original_path_results = result_file_content.get("results", [])
        if not isinstance(original_path_results, list):
            original_path_results = []

        for path_data_item in original_path_results:
            path_analysis: Dict[str, Any] = {
                "path": path_data_item.get("path"),
                "path_signature": path_data_item.get("path_signature"),
                "prompt": path_data_item.get("prompt", ""),
                "core_response_text": "", 
                "metacommentary_text": "", 
                "nlu_verified_claims": [], 
                "sentiment_scores": {},
                "main_diagonal_strength": path_data_item.get("main_diagonal_strength"),
                "anti_diagonal_strength": path_data_item.get("anti_diagonal_strength"),
                "main_diagonal_affect": path_data_item.get("main_diagonal_affect"),
                "anti_diagonal_affect": path_data_item.get("anti_diagonal_affect"),
            }

            full_response = path_data_item.get("response", "")
            
            # Robust splitting of core response and metacommentary
            # This marker should be a constant or passed in, matching EigenKoanMatrix
            meta_marker = "After completing this task, please reflect on your process:"
            if meta_marker in full_response:
                parts = full_response.split(meta_marker, 1)
                path_analysis["core_response_text"] = parts[0].strip()
                if len(parts) > 1:
                    path_analysis["metacommentary_text"] = parts[1].strip()
            else:
                path_analysis["core_response_text"] = full_response

            if path_analysis["core_response_text"]:
                vader_scores = self.sentiment_analyzer.polarity_scores(path_analysis["core_response_text"])
                blob = TextBlob(path_analysis["core_response_text"])
                path_analysis["sentiment_scores"] = {
                    "vader": vader_scores,
                    "textblob_polarity": blob.sentiment.polarity,
                    "textblob_subjectivity": blob.sentiment.subjectivity
                }
            
            if path_analysis["metacommentary_text"]:
                # For now, matrix_obj and path are not deeply used in claim extraction here,
                # but are passed for future, more contextual claim generation.
                claims = self._extract_metacommentary_claims(
                    path_analysis["metacommentary_text"], 
                    None, # Pass actual matrix object if available and needed by _extract_...
                    path_analysis["path"]
                )
                
                for claim_dict in claims:
                    hypothesis = claim_dict.get('contextual_hypothesis', claim_dict.get('original_sentence',''))
                    if hypothesis: # Ensure there's a hypothesis to verify
                        nli_result = self._verify_claim_with_nli(
                            hypothesis, 
                            path_analysis["core_response_text"]
                        )
                        # Augment NLI result with claim type details for clarity
                        nli_result['claim_type'] = claim_dict.get('type', 'unknown')
                        nli_result['original_claim_sentence'] = claim_dict.get('original_sentence', '')
                        path_analysis["nlu_verified_claims"].append(nli_result)
            
            analyzed_data["path_results"].append(path_analysis)
        
        all_core_responses_in_file = [pr['core_response_text'] for pr in analyzed_data["path_results"] if pr['core_response_text']]
        if all_core_responses_in_file:
            try:
                tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2, max_features=1000)
                tfidf_matrix = tfidf_vectorizer.fit_transform(all_core_responses_in_file)
                analyzed_data["overall_top_terms_tfidf"] = list(tfidf_vectorizer.get_feature_names_out()[:20])

                # Word Frequencies
                all_text_for_freq = ' '.join(all_core_responses_in_file)
                tokens = nltk.word_tokenize(all_text_for_freq.lower())
                stopwords_set = set(nltk.corpus.stopwords.words('english'))
                filtered_tokens = [token for token in tokens if token.isalpha() and token not in stopwords_set and len(token) > 1]
                word_freq = Counter(filtered_tokens)
                analyzed_data["overall_word_frequencies"] = dict(word_freq.most_common(50))

                # Response Similarity (example - can be computationally intensive for many responses)
                if len(all_core_responses_in_file) > 1 and len(all_core_responses_in_file) <= 50: # Limit for demo
                     similarity_matrix = cosine_similarity(tfidf_matrix)
                     analyzed_data["response_similarity_matrix"] = similarity_matrix.tolist()

            except Exception as e:
                print(f"Error during aggregate text analysis for result {result_index}: {e}")
                analyzed_data["overall_top_terms_tfidf"] = ["Error in TFIDF"]
                analyzed_data["overall_word_frequencies"] = {"Error": str(e)}

        return analyzed_data

    def compare_models(self, comparison_index: int) -> Dict:
        """
        Analyze a model comparison in depth.
        (This method would need significant updates to meaningfully incorporate NLU results,
         e.g., by comparing NLU verification stats across models for the same paths.)
        """
        if not (0 <= comparison_index < len(self.comparisons)):
            raise ValueError(f"Invalid comparison index: {comparison_index}. Must be between 0 and {len(self.comparisons)-1}.")
        
        comparison_data = self.comparisons[comparison_index]
        # Basic structure from your original analyzer
        matrix_name = comparison_data.get("matrix_name")
        models_compared = comparison_data.get("models_compared", [])
        paths_tested = comparison_data.get("paths_tested", []) # List of path signatures
        model_results_data = comparison_data.get("model_results", {})

        analyzed_comparison: Dict[str, Any] = {
            "matrix_name": matrix_name,
            "models_compared": models_compared,
            "paths_tested_signatures": paths_tested, # Store the signatures
            "sentiment_by_model": {},
            "word_usage_by_model": {},
            "cross_model_response_similarity_on_paths": {},
            "nlu_metacommentary_comparison": {} # Placeholder for comparative NLU stats
        }

        # Example: Aggregate sentiment per model
        for model_name, results_for_model in model_results_data.items():
            if model_name not in models_compared: continue
            
            all_responses_for_model = [path_res.get("response","") for path_res in results_for_model.get("results", [])]
            core_responses_for_model = []
            # Split core from meta for all responses
            meta_marker = "After completing this task, please reflect on your process:"
            for resp_text in all_responses_for_model:
                if meta_marker in resp_text:
                    core_responses_for_model.append(resp_text.split(meta_marker,1)[0].strip())
                else:
                    core_responses_for_model.append(resp_text)
            
            if core_responses_for_model:
                avg_sentiments = {
                    "vader_compound": np.mean([self.sentiment_analyzer.polarity_scores(r)["compound"] for r in core_responses_for_model]),
                    "textblob_polarity": np.mean([TextBlob(r).sentiment.polarity for r in core_responses_for_model])
                }
                analyzed_comparison["sentiment_by_model"][model_name] = avg_sentiments

                # Word Frequencies for this model
                all_text = ' '.join(core_responses_for_model)
                tokens = nltk.word_tokenize(all_text.lower())
                stopwords_set = set(nltk.corpus.stopwords.words('english'))
                filtered_tokens = [token for token in tokens if token.isalpha() and token not in stopwords_set and len(token) > 1]
                word_freq = Counter(filtered_tokens)
                analyzed_comparison["word_usage_by_model"][model_name] = dict(word_freq.most_common(20))
        
        # Example: Cross-model similarity for shared paths (simplified)
        # This requires careful matching of paths across model results.
        # The 'paths_tested' (signatures) can be used for this.
        if paths_tested and len(models_compared) > 1:
            for path_sig in paths_tested:
                responses_for_this_path = []
                models_on_this_path = []
                for model_name in models_compared:
                    model_data = model_results_data.get(model_name, {}).get("results", [])
                    path_resp_item = next((item for item in model_data if item.get("path_signature") == path_sig), None)
                    if path_resp_item:
                        full_resp = path_resp_item.get("response", "")
                        core_resp = full_resp.split(meta_marker,1)[0].strip() if meta_marker in full_resp else full_resp
                        responses_for_this_path.append(core_resp)
                        models_on_this_path.append(model_name)
                
                if len(responses_for_this_path) > 1:
                    try:
                        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
                        tfidf_matrix = tfidf_vectorizer.fit_transform(responses_for_this_path)
                        sim_matrix = cosine_similarity(tfidf_matrix)
                        analyzed_comparison["cross_model_response_similarity_on_paths"][path_sig] = {
                            "models": models_on_this_path,
                            "similarity_matrix": sim_matrix.tolist()
                        }
                    except ValueError: # e.g. empty vocabulary
                        pass # Skip similarity for this path
                        
        print("Warning: `compare_models` NLU integration is a complex task involving aggregation and comparison of NLU stats. This version primarily focuses on sentiment and word usage.")
        return analyzed_comparison # Return the structured analysis
        
    def visualize_single_result(self, result_index: int, output_dir: str = "./ekm_viz", matrices_data: Optional[Dict[str, Any]] = None):
        """
        Generate visualizations for a single test result.
        (This method requires significant updates to visualize NLU results effectively.)
        """
        if not (0 <= result_index < len(self.results)):
            print("Invalid result index for visualization.")
            return

        os.makedirs(output_dir, exist_ok=True)
        analysis_output = self.analyze_single_result(result_index, matrices_data)
        
        matrix_name = analysis_output.get("matrix_name", f"Result_{result_index}").replace(" ", "_")
        model_name = analysis_output.get("model_name", "UnknownModel").replace(" ", "_")
        base_filename = f"{matrix_name}_{model_name}"

        print(f"Visualizing results for {base_filename}...")

        # 1. Word cloud (if word frequencies exist)
        word_freqs = analysis_output.get("overall_word_frequencies")
        if word_freqs and isinstance(word_freqs, dict) and any(word_freqs):
            try:
                plt.figure(figsize=(10, 7))
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freqs)
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                plt.title(f"Word Cloud for {model_name} on {matrix_name}")
                wc_path = os.path.join(output_dir, f"{base_filename}_wordcloud.png")
                plt.savefig(wc_path)
                plt.close()
                print(f"  Saved word cloud to {wc_path}")
            except Exception as e:
                print(f"  Error generating word cloud: {e}")
        else:
            print(f"  Skipping word cloud: No word frequency data or empty.")

        # 2. Sentiment distribution (example for TextBlob polarity)
        sentiments_polarity = [
            path_res["sentiment_scores"].get("textblob_polarity", 0) 
            for path_res in analysis_output.get("path_results", []) 
            if path_res.get("sentiment_scores")
        ]
        if sentiments_polarity:
            try:
                plt.figure(figsize=(8, 5))
                sns.histplot(sentiments_polarity, kde=True)
                plt.title(f"Sentiment Polarity Distribution ({model_name} on {matrix_name})")
                plt.xlabel("TextBlob Polarity")
                plt.ylabel("Frequency")
                sent_path = os.path.join(output_dir, f"{base_filename}_sentiment_dist.png")
                plt.savefig(sent_path)
                plt.close()
                print(f"  Saved sentiment distribution to {sent_path}")
            except Exception as e:
                print(f"  Error generating sentiment distribution plot: {e}")
        else:
            print(f"  Skipping sentiment distribution: No sentiment data.")
            
        # 3. NLU Verification Summary (Example: Count of Entailment/Contradiction/Neutral)
        nlu_labels_collected = []
        for path_res in analysis_output.get("path_results", []):
            for claim_verif in path_res.get("nlu_verified_claims", []):
                status = claim_verif.get("verification_status")
                if status and status != "NLI_MODEL_UNAVAILABLE" and status != "NLI_PROCESSING_ERROR":
                    nlu_labels_collected.append(status)
        
        if nlu_labels_collected:
            try:
                label_counts = Counter(nlu_labels_collected)
                plt.figure(figsize=(8, 5))
                plt.bar(label_counts.keys(), label_counts.values())
                plt.title(f"NLU Metacommentary Claim Verification ({model_name} on {matrix_name})")
                plt.xlabel("NLI Label")
                plt.ylabel("Count of Claims")
                nlu_viz_path = os.path.join(output_dir, f"{base_filename}_nlu_verification_summary.png")
                plt.savefig(nlu_viz_path)
                plt.close()
                print(f"  Saved NLU verification summary to {nlu_viz_path}")
            except Exception as e:
                print(f"  Error generating NLU verification summary plot: {e}")
        else:
            print(f"  Skipping NLU verification summary: No NLU data or NLI model unavailable.")
            
        print(f"Visualizations for {base_filename} attempted.")

    def visualize_comparison(self, comparison_index: int, output_dir: str = "./ekm_viz"):
        """
        Generate visualizations for a model comparison.
        (This method requires significant updates to visualize NLU results effectively across models.)
        """
        if not (0 <= comparison_index < len(self.comparisons)):
            print("Invalid comparison index for visualization.")
            return

        os.makedirs(output_dir, exist_ok=True)
        analysis_output = self.compare_models(comparison_index) # This needs to be rich enough

        matrix_name = analysis_output.get("matrix_name", f"Comparison_{comparison_index}").replace(" ", "_")
        models_str = "_vs_".join(analysis_output.get("models_compared", ["UnknownModels"]))
        base_filename = f"{matrix_name}_{models_str}"
        
        print(f"Visualizing comparison for {base_filename}...")

        # Example: Compare average sentiment scores
        sentiment_by_model = analysis_output.get("sentiment_by_model", {})
        if sentiment_by_model:
            try:
                df_data = []
                for model, sentiments in sentiment_by_model.items():
                    df_data.append({"Model": model, "Metric": "VADER Compound", "Value": sentiments.get("vader_compound", 0)})
                    df_data.append({"Model": model, "Metric": "TextBlob Polarity", "Value": sentiments.get("textblob_polarity", 0)})
                if df_data:
                    sentiment_df = pd.DataFrame(df_data)
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x="Metric", y="Value", hue="Model", data=sentiment_df)
                    plt.title(f"Average Sentiment Comparison on {matrix_name}")
                    plt.ylim(min(0, sentiment_df['Value'].min() - 0.1) if not sentiment_df.empty else 0, 
                             max(1, sentiment_df['Value'].max() + 0.1) if not sentiment_df.empty else 1) # Adjust y-limits
                    comp_sent_path = os.path.join(output_dir, f"{base_filename}_sentiment_comparison.png")
                    plt.savefig(comp_sent_path)
                    plt.close()
                    print(f"  Saved sentiment comparison to {comp_sent_path}")
            except Exception as e:
                print(f"  Error generating sentiment comparison plot: {e}")
        else:
            print(f"  Skipping sentiment comparison: No data.")
            
        # Further comparison visualizations (e.g., for NLU stats across models) would be added here.
        print(f"Comparison visualizations for {base_filename} attempted.")


# Command line interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="EKM Analysis Tools. Analyzes JSON results from EKM experiments.")
    subparsers = parser.add_subparsers(dest="command", help="Command to run", required=True)
    
    # List results command
    list_parser = subparsers.add_parser("list", help="List available results from the default results directory.")
    list_parser.add_argument("--results_dir", default="./ekm_results", help="Directory containing EKM result JSON files.")
    list_parser.add_argument("--type", choices=["tests", "comparisons", "all"], default="all", help="Type of results to list (single tests, comparisons, or all).")
    
    # Analyze single test result command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a single test result file in depth.")
    analyze_parser.add_argument("index", type=int, help="Index of the result file (from 'list tests' command) to analyze.")
    analyze_parser.add_argument("--results_dir", default="./ekm_results", help="Directory containing EKM result JSON files.")
    analyze_parser.add_argument("--nli_model", default="roberta-large-mnli", help="Name of the NLI model from Hugging Face Hub (e.g., 'roberta-large-mnli', or 'None' to disable).")
    analyze_parser.add_argument("--viz", action="store_true", help="Generate visualizations for the analyzed result.")
    analyze_parser.add_argument("--viz_output_dir", default="./ekm_viz", help="Output directory for visualizations.")
    # Future: Add --matrix_definitions_dir to load EKM objects for contextual NLU.

    # Compare models command
    compare_parser = subparsers.add_parser("compare", help="Analyze a model comparison result file.")
    compare_parser.add_argument("index", type=int, help="Index of the comparison file (from 'list comparisons' command) to analyze.")
    compare_parser.add_argument("--results_dir", default="./ekm_results", help="Directory containing EKM result JSON files.")
    # NLI model not directly used in compare for now, as it's per-response. Aggregation would be complex.
    compare_parser.add_argument("--viz", action="store_true", help="Generate visualizations for the comparison.")
    compare_parser.add_argument("--viz_output_dir", default="./ekm_viz", help="Output directory for visualizations.")
    
    args = parser.parse_args()
    
    # Determine NLI model, allowing 'None' string to disable
    nli_model_for_analyzer = args.nli_model if hasattr(args, 'nli_model') and args.nli_model and args.nli_model.lower() != 'none' else None
    
    analyzer = EKMAnalyzer(results_dir=args.results_dir, nli_model_name=nli_model_for_analyzer)
    
    if args.command == "list":
        if not analyzer.results and not analyzer.comparisons:
            print(f"No results found in {analyzer.results_dir}. Ensure it contains valid EKM JSON outputs.")
            return

        if args.type in ["tests", "all"] and analyzer.results:
            print("\nAvailable single test results (use index with 'analyze' command):")
            for i, result in enumerate(analyzer.results):
                matrix_n = result.get('matrix_name', 'N/A')
                model_n = result.get('model_name', 'N/A')
                num_paths = len(result.get('results', []))
                print(f"  [{i}] Matrix: '{matrix_n}' - Model: '{model_n}' ({num_paths} paths tested)")
        elif args.type == "tests":
             print("No single test results found.")

        if args.type in ["comparisons", "all"] and analyzer.comparisons:
            print("\nAvailable model comparisons (use index with 'compare' command):")
            for i, comp_data in enumerate(analyzer.comparisons):
                matrix_n = comp_data.get('matrix_name', 'N/A')
                models_c = ", ".join(comp_data.get('models_compared', ['N/A']))
                print(f"  [{i}] Matrix: '{matrix_n}' - Models: {models_c}")
        elif args.type == "comparisons":
            print("No comparison results found.")
    
    elif args.command == "analyze":
        if not analyzer.results:
            print(f"No single test results loaded from {analyzer.results_dir} to analyze.")
            return
        try:
            # For now, matrices_data is None. A real CLI might load EKM definitions.
            analysis_output = analyzer.analyze_single_result(args.index, matrices_data=None) 
            print(f"\n--- Analysis for Result Index {args.index} ({analysis_output.get('matrix_name')} by {analysis_output.get('model_name')}) ---")
            
            print(f"\nOverall Top Terms (TF-IDF, Sample): {analysis_output.get('overall_top_terms_tfidf', [])[:5]}")
            print(f"Overall Word Frequencies (Sample): {dict(list(analysis_output.get('overall_word_frequencies', {}).items())[:5])}")

            for i, path_res in enumerate(analysis_output.get("path_results", [])):
                print(f"\n  --- Path {i+1} (Signature: {path_res.get('path_signature', 'N/A')}) ---")
                print(f"    Sentiment (TextBlob Polarity): {path_res.get('sentiment_scores', {}).get('textblob_polarity', 'N/A'):.2f}")
                if path_res.get("nlu_verified_claims"):
                    print(f"    NLU Metacommentary Verification:")
                    for claim_verif in path_res["nlu_verified_claims"][:2]: # Show first 2 claims
                        print(f"      Claim: \"{claim_verif.get('original_claim_sentence', 'Error retrieving sentence')[:40]}...\"")
                        print(f"        -> Status: {claim_verif.get('verification_status', 'Error')}, Score: {claim_verif.get('score_for_predicted_label', 0.0):.2f}")
                        # print(f"           Scores: {claim_verif.get('all_scores')}") # For more detail
                elif path_res.get("metacommentary_text"):
                     print(f"    NLU Metacommentary Verification: NLI Model unavailable or error during processing.")
                else:
                    print(f"    Metacommentary: Not present or empty.")
            
            if args.viz:
                print(f"\nGenerating visualizations in {args.viz_output_dir}...")
                analyzer.visualize_single_result(args.index, args.viz_output_dir, matrices_data=None)
        except ValueError as ve:
            print(f"Error: {ve}")
        except IndexError:
            print(f"Error: Result index {args.index} is out of range. Max index is {len(analyzer.results)-1}.")
        except Exception as e:
            print(f"An unexpected error occurred during analysis: {e}")
            import traceback
            traceback.print_exc()
    
    elif args.command == "compare":
        if not analyzer.comparisons:
            print(f"No comparison results loaded from {analyzer.results_dir} to compare.")
            return
        try:
            comparison_output = analyzer.compare_models(args.index)
            print(f"\n--- Comparison Analysis for Index {args.index} ({comparison_output.get('matrix_name')}) ---")
            print(f"Models Compared: {', '.join(comparison_output.get('models_compared', []))}")
            
            if comparison_output.get("sentiment_by_model"):
                print("\nAverage Sentiments by Model:")
                for model, sentiments in comparison_output["sentiment_by_model"].items():
                    print(f"  {model}: VADER Compound={sentiments.get('vader_compound',0):.2f}, TextBlob Polarity={sentiments.get('textblob_polarity',0):.2f}")

            if comparison_output.get("word_usage_by_model"):
                print("\nTop Word Usage by Model (Sample):")
                for model, words in comparison_output["word_usage_by_model"].items():
                    print(f"  {model}: {dict(list(words.items())[:3])}")
            
            if args.viz:
                print(f"\nGenerating comparison visualizations in {args.viz_output_dir}...")
                analyzer.visualize_comparison(args.index, args.viz_output_dir)

        except ValueError as ve:
            print(f"Error: {ve}")
        except IndexError:
             print(f"Error: Comparison index {args.index} is out of range. Max index is {len(analyzer.comparisons)-1}.")
        except Exception as e:
            print(f"An unexpected error occurred during comparison: {e}")
            import traceback
            traceback.print_exc()
            
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
