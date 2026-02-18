#!/usr/bin/env python
# coding: utf-8

# AIMO3 Consensus Voting Submission - Optimized for 2x T4 GPUs
# Competition: AI Mathematical Olympiad - Progress Prize 3
# Strategy: Generate 3 solutions with consensus voting, model parallel on 2x T4

import ast
import json
import os
import re
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import kaggle_evaluation.aimo_3_inference_server
import pandas as pd
import polars as pl
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Suppress warnings
warnings.filterwarnings("ignore")

print("✓ Imports successful")


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class Config:
    """Configuration for consensus voting system."""

    # Model configuration - DeepSeek-Math-7B-Instruct from Kaggle dataset
    # Using local Kaggle dataset: markwijkhuizen/deepseek-math-7b-instruct
    MODEL_NAME = "/kaggle/input/deepseek-math-7b-instruct/"
    MAX_NEW_TOKENS = 4096
    BASE_TEMPERATURE = 0.1
    TOP_P = 0.95

    # 2x T4 GPU Configuration
    USE_FP16 = True  # Use fp16 instead of 4-bit for speed
    DEVICE_MAP = "auto"  # Automatically split across GPUs

    # Consensus configuration
    NUM_SOLUTIONS = 3  # Number of solutions for voting
    TEMPERATURES = [0.1, 0.3, 0.5]  # Diversity through temperature

    # Constraints
    MIN_ANSWER = 0
    MAX_ANSWER = 99999
    TIMEOUT_PER_PROBLEM = 300  # 5 minutes max


print(f"✓ Configuration:")
print(f"  Model: {Config.MODEL_NAME}")
print(f"  Consensus solutions: {Config.NUM_SOLUTIONS}")
print(f"  Temperatures: {Config.TEMPERATURES}")


# =============================================================================
# REFERENCE PROBLEMS (for context)
# =============================================================================

REFERENCE_PROBLEMS = """
Here are examples of AIMO3 problems and their solutions:

Example 1 (Geometry):
Problem: Let $ABC$ be an acute-angled triangle with integer side lengths and $AB<AC$. Points $D$ and $E$ lie on segments $BC$ and $AC$, respectively, such that $AD=AE=AB$. Line $DE$ intersects $AB$ at $X$. Circles $BXD$ and $CED$ intersect for the second time at $Y \\neq D$. Suppose that $Y$ lies on line $AD$. There is a unique such triangle with minimal perimeter. This triangle has side lengths $a=BC$, $b=CA$, and $c=AB$. Find the remainder when $abc$ is divided by $10^{5}$.
Answer: 336

Example 2 (Number Theory):
Problem: Define a function $f \\colon \\mathbb{Z}_{\\geq 1} \\to \\mathbb{Z}_{\\geq 1}$ by $f(n) = \\sum_{i = 1}^n \\sum_{j = 1}^n j^{1024} \\left\\lfloor\\frac1j + \\frac{n-i}{n}\\right\\rfloor$. Let $M=2 \\cdot 3 \\cdot 5 \\cdot 7 \\cdot 11 \\cdot 13$ and let $N = f{(M^{15})} - f{(M^{15}-1)}$. Let $k$ be the largest non-negative integer such that $2^k$ divides $N$. What is the remainder when $2^k$ is divided by $5^7$?
Answer: 32951

Example 3 (Algebra):
Problem: Alice and Bob are each holding some integer number of sweets. Alice says to Bob: ``If we each added the number of sweets we're holding to our (positive integer) age, my answer would be double yours. If we took the product, then my answer would be four times yours.'' Bob replies: ``Why don't you give me five of your sweets because then both our sum and product would be equal.'' What is the product of Alice and Bob's ages?
Answer: 50

Example 4 (Combinatorics):
Problem: A $500 \\times 500$ square is divided into $k$ rectangles, each having integer side lengths. Given that no two of these rectangles have the same perimeter, the largest possible value of $k$ is $\\mathcal{K}$. What is the remainder when $k$ is divided by $10^{5}$?
Answer: 520
"""

print("✓ Reference problems loaded")


# =============================================================================
# MULTI-GPU MODEL MANAGER
# =============================================================================


class MultiGPUModelManager:
    """
    Manages model loading across 2x T4 GPUs.
    Uses automatic device mapping for optimal memory distribution.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.num_gpus = torch.cuda.device_count()

    def _find_model_path(self):
        """Search for the model files in Kaggle input directories."""
        import glob

        # First check the configured path
        if os.path.exists(Config.MODEL_NAME):
            files = os.listdir(Config.MODEL_NAME)
            if any(f.endswith((".bin", ".safetensors", "config.json")) for f in files):
                return Config.MODEL_NAME

        # Search recursively in /kaggle/input/
        print("Searching for model files in /kaggle/input/...")
        for root, dirs, files in os.walk("/kaggle/input/"):
            # Look for config.json which indicates a model directory
            if "config.json" in files:
                print(f"Found model directory: {root}")
                # Verify it has model weights
                if any(f.endswith((".bin", ".safetensors")) for f in files):
                    return root

        # If not found, list all directories for debugging
        print("All directories in /kaggle/input/:")
        for root, dirs, files in os.walk("/kaggle/input/"):
            level = root.replace("/kaggle/input/", "").count(os.sep)
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 2 * (level + 1)
            for file in files[:5]:  # Show first 5 files
                print(f"{subindent}{file}")

        return None

    def load(self):
        """Load model across available GPUs."""
        if self.model is not None:
            return self

        print(f"Loading model from configured path: {Config.MODEL_NAME}")
        print(f"Available GPUs: {self.num_gpus}")

        # Find the actual model path
        model_path = self._find_model_path()

        if model_path is None:
            raise FileNotFoundError(f"Could not find model files in /kaggle/input/")

        print(f"✓ Found model at: {model_path}")

        # List files in the directory
        files = os.listdir(model_path)
        print(f"Files in model directory: {files[:10]}...")  # Show first 10 files

        # Set memory limits per GPU (T4 has ~16GB each)
        max_memory = {i: "15GiB" for i in range(self.num_gpus)}

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, local_files_only=True
            )
            print("✓ Tokenizer loaded")

            # Load model with automatic device mapping
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if Config.USE_FP16 else torch.float32,
                device_map=Config.DEVICE_MAP,
                max_memory=max_memory,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            print("✓ Model loaded")
        except Exception as e:
            print(f"ERROR loading model: {e}")
            print(f"Model path content:")
            for f in os.listdir(model_path):
                print(f"  - {f}")
            raise

        # Check device allocation
        if hasattr(self.model, "hf_device_map"):
            print(f"✓ Model sharded across devices: {self.model.hf_device_map}")
        else:
            self.device = next(self.model.parameters()).device
            print(f"✓ Model loaded on device: {self.device}")

        print(f"✓ Model ready")
        return self

    def generate(self, prompt: str, temperature: float = None) -> str:
        """Generate response with specified temperature."""
        if self.model is None:
            self.load()

        temp = temperature if temperature is not None else Config.BASE_TEMPERATURE

        # Format as chat
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt")

        # Move to appropriate device (device_map handles this automatically)
        if hasattr(self.model, "hf_device_map"):
            # Model is split across GPUs, no need to move
            pass
        else:
            inputs = inputs.to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=Config.MAX_NEW_TOKENS,
                temperature=temp,
                top_p=Config.TOP_P,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1,
            )

        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        return response.strip()


# Global model manager (lazy loading)
model_manager = MultiGPUModelManager()
print("✓ Multi-GPU model manager initialized")


# =============================================================================
# AGENT CLASSES
# =============================================================================

# Domain-specific solving instructions
DOMAIN_INSTRUCTIONS = {
    "algebra": """For this algebra problem:
- Identify key variables and equations
- Use substitution, elimination, or factorization
- Check for special cases and edge conditions
- Verify your solution satisfies the original equation""",
    "geometry": """For this geometry problem:
- Use pure synthetic reasoning (theorems, not coordinates)
- Apply circle theorems, triangle properties, angle chasing
- Look for cyclic quadrilaterals, power of a point
- Draw auxiliary lines when helpful""",
    "combinatorics": """For this combinatorics problem:
- Identify if counting, probability, or existence
- Consider permutations, combinations, inclusion-exclusion
- Look for symmetries and bijections
- Check small cases first to identify patterns""",
    "number_theory": """For this number theory problem:
- Consider divisibility, prime factorization, modular arithmetic
- Apply Fermat's Little Theorem, Euler's theorem, CRT
- Look for patterns in residues modulo small primes
- Use Euclidean algorithm for gcd/lcm""",
    "unknown": "Use general mathematical reasoning and careful step-by-step analysis.",
}


class BaseAgent:
    """Base class for all agents."""

    def __init__(self, name: str):
        self.name = name

    def run(self, *args, **kwargs):
        raise NotImplementedError


class AnalyzerAgent(BaseAgent):
    """Analyzes the problem to understand its domain and requirements."""

    def __init__(self):
        super().__init__("Analyzer")

    def run(self, problem: str) -> Dict:
        """Analyze the problem and return structured information."""

        prompt = f"""You are a mathematical problem analyzer for competition mathematics (IMO/AIME level).

{REFERENCE_PROBLEMS}

Now analyze this problem:
Problem: {problem}

You MUST respond with a valid Python dictionary in this EXACT format:
{{
    "domain": "algebra",
    "problem_type": "computation",
    "difficulty_estimate": "hard",
    "key_concepts": ["concept1", "concept2"],
    "suggested_approach": "brief description"
}}

IMPORTANT:
- Use DOUBLE QUOTES for all strings
- Domain must be exactly: algebra, geometry, combinatorics, number_theory, or unknown
- Provide 1-3 key concepts as a list
- Keep suggested_approach brief (under 100 characters)

Respond ONLY with the dictionary, no other text."""

        try:
            response = model_manager.generate(prompt, temperature=0.1)

            # Extract JSON from response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                analysis = ast.literal_eval(json_match.group())
                return analysis
        except Exception as e:
            print(f"Analyzer error: {e}")

        # Fallback analysis
        return {
            "domain": "unknown",
            "problem_type": "computation",
            "difficulty_estimate": "medium",
            "key_concepts": [],
            "suggested_approach": "General mathematical reasoning",
        }


class SolverAgent(BaseAgent):
    """Generates solutions with chain-of-thought reasoning."""

    def __init__(self):
        super().__init__("Solver")

    def create_prompt(self, problem: str, analysis: Dict, temperature_idx: int) -> str:
        """Create solving prompt with appropriate instructions."""

        domain = analysis.get("domain", "unknown")
        approach = analysis.get("suggested_approach", "")
        domain_instruction = DOMAIN_INSTRUCTIONS.get(
            domain, DOMAIN_INSTRUCTIONS["unknown"]
        )

        prompt = f"""You are an expert mathematical problem solver specializing in competition mathematics (IMO/AIME level).

{REFERENCE_PROBLEMS}

Now solve this problem:
Problem: {problem}

Analysis: This is a {domain} problem. {approach}

{domain_instruction}

Important Instructions:
1. Think step-by-step using chain-of-reasoning
2. Show all your work clearly
3. The answer MUST be a non-negative integer between 0 and 99999
4. If the problem asks for a remainder, compute it correctly

Work through the solution step by step.

After solving, add a VERIFICATION section:
- Double-check your calculations
- Verify the answer satisfies all problem conditions
- Confirm the answer is in the valid range (0-99999)

State your final answer clearly as: FINAL ANSWER: [number]"""

        return prompt

    def extract_answer(self, response: str) -> Optional[int]:
        """Extract the integer answer from model response."""

        # Multiple patterns for answer extraction
        patterns = [
            r"FINAL ANSWER:\s*(\d+)",
            r"final answer is:?\s*(\d+)",
            r"answer is:?\s*(\d+)",
            r"the answer is:?\s*(\d+)",
            r"\\boxed\{(\d+)\}",
            r"\*\*(\d+)\*\*",
            r"\b(\d{1,5})\b(?!\s*\.\d)",  # Standalone numbers
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                # Get the last match (usually the final answer)
                for match in reversed(matches):
                    try:
                        num = int(match)
                        if Config.MIN_ANSWER <= num <= Config.MAX_ANSWER:
                            return num
                    except ValueError:
                        continue

        return None

    def run(self, problem: str, analysis: Dict, temp_idx: int) -> Dict:
        """Generate a solution with specified temperature."""

        temperature = Config.TEMPERATURES[temp_idx % len(Config.TEMPERATURES)]
        prompt = self.create_prompt(problem, analysis, temp_idx)

        try:
            response = model_manager.generate(prompt, temperature=temperature)
            answer = self.extract_answer(response)

            return {
                "success": answer is not None,
                "answer": answer,
                "raw_response": response,
                "temperature": temperature,
                "temp_idx": temp_idx,
            }

        except Exception as e:
            return {
                "success": False,
                "answer": None,
                "error": str(e),
                "temperature": temperature,
                "temp_idx": temp_idx,
            }


class ConsensusVotingAgent(BaseAgent):
    """
    Implements consensus voting across multiple solutions.
    Generates N solutions and selects the majority answer.
    """

    def __init__(self):
        super().__init__("ConsensusVoting")
        self.analyzer = AnalyzerAgent()
        self.solver = SolverAgent()

    def run(self, problem: str) -> int:
        """
        Solve problem using consensus voting.
        Returns the majority answer or best available.
        """
        print(f"\n{'=' * 60}")
        print("CONSENSUS VOTING SOLVER")
        print(f"{'=' * 60}")

        # Step 1: Analyze problem
        print("Step 1: Analyzing problem...")
        analysis = self.analyzer.run(problem)
        domain = analysis.get("domain", "unknown")
        print(f"  Domain: {domain}")
        print(f"  Approach: {analysis.get('suggested_approach', 'N/A')}")

        # Step 2: Generate multiple solutions
        print(f"\nStep 2: Generating {Config.NUM_SOLUTIONS} solutions...")
        solutions = []

        for i in range(Config.NUM_SOLUTIONS):
            print(
                f"  Solution {i + 1}/{Config.NUM_SOLUTIONS} (temp={Config.TEMPERATURES[i]})..."
            )

            result = self.solver.run(problem, analysis, i)
            solutions.append(result)

            if result["success"]:
                print(f"    ✓ Answer: {result['answer']}")
            else:
                print(f"    ✗ Failed: {result.get('error', 'Unknown')}")

        # Step 3: Consensus voting
        print(f"\nStep 3: Consensus voting...")

        # Collect all valid answers
        valid_answers = [
            s["answer"] for s in solutions if s["success"] and s["answer"] is not None
        ]

        if not valid_answers:
            print("  No valid answers! Returning 0")
            return 0

        # Count votes
        answer_counts = Counter(valid_answers)
        print(f"  Vote distribution: {dict(answer_counts)}")

        # Get majority
        majority_answer, vote_count = answer_counts.most_common(1)[0]

        # Check if we have a clear majority or tie
        if vote_count > len(valid_answers) / 2:
            print(
                f"  ✓ Clear majority: {majority_answer} ({vote_count}/{len(valid_answers)} votes)"
            )
        else:
            # Tie or no clear majority - pick most common
            print(
                f"  ~ Plurality winner: {majority_answer} ({vote_count}/{len(valid_answers)} votes)"
            )

        print(f"\nFinal answer: {majority_answer}")
        print(f"{'=' * 60}")

        return majority_answer


print("✓ All agent classes defined")


# =============================================================================
# MAIN SOLVER INTERFACE
# =============================================================================


class AIMO3Solver:
    """Main solver with consensus voting."""

    def __init__(self):
        self.consensus_agent = ConsensusVotingAgent()

    def solve(self, problem: str) -> int:
        """Solve a single problem."""
        try:
            answer = self.consensus_agent.run(problem)
            # Ensure valid range
            return max(Config.MIN_ANSWER, min(Config.MAX_ANSWER, int(answer)))
        except Exception as e:
            print(f"Critical error: {e}")
            return 0


# Global solver (lazy loaded)
_solver = None


def get_solver():
    """Get or create solver instance."""
    global _solver
    if _solver is None:
        _solver = AIMO3Solver()
    return _solver


print("✓ Main solver interface ready")


# =============================================================================
# KAGGLE INFERENCE INTERFACE
# =============================================================================


def predict(id_: pl.Series, problem: pl.Series) -> pl.DataFrame:
    """
    Kaggle inference API function.

    Args:
        id_: Polars Series containing problem ID
        problem: Polars Series containing problem text

    Returns:
        Polars DataFrame with 'id' and 'answer' columns
    """
    # Unpack values
    problem_id = id_.item(0)
    problem_text = problem.item(0)

    print(f"\n{'=' * 60}")
    print(f"Processing: {problem_id}")
    print(f"{'=' * 60}")
    print(f"Problem: {problem_text[:100]}...")

    # Get solver and solve
    solver = get_solver()
    answer = solver.solve(problem_text)

    print(f"✓ Answer: {answer}")

    # Return as DataFrame
    return pl.DataFrame({"id": problem_id, "answer": answer})


print("✓ Predict function defined")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

print("\n" + "=" * 60)
print("AIMO3 Consensus Voting Solver")
print("Optimized for 2x T4 GPUs")
print("=" * 60)

# Initialize inference server
inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(
    predict
)

# Check if running in competition mode
if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    print("\n🚀 Starting inference server in PRODUCTION mode...")
    inference_server.serve()
else:
    print("\n🧪 Running in LOCAL TEST mode...")

    # Try to run on test data
    test_path = "/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv"

    if os.path.exists(test_path):
        print(f"Loading test data from: {test_path}")
        inference_server.run_local_gateway((test_path,))
    else:
        print("Test file not found. Running sample problem...")

        # Test with a simple problem
        sample_problem = (
            "What is the sum of all positive integers $n$ such that $n^2 - 3n + 2 = 0$?"
        )

        test_id = pl.Series(["test001"])
        test_problem = pl.Series([sample_problem])

        result = predict(test_id, test_problem)
        print(f"\nResult:")
        print(result)
