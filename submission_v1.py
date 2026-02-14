#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Install required packages
get_ipython().system("pip install bitsandbytes -q")
print("✓ Dependencies installed")


# # AIMO3 Multi-Agent Submission Notebook - Version 1
#
# This notebook implements a minimal viable multi-agent system for solving AIMO3 competition problems.
#
# **Architecture:**
# 1. **Analyzer Agent** - Classifies problem domain and approach
# 2. **Direct Solver Agent** - Uses pure mathematical reasoning with chain-of-thought
# 3. **Verifier Agent** - Validates answer correctness and constraints
# 4. **Aggregator Agent** - Manages fixed retry logic and selects final answer
#
# **Model:** Qwen2.5-Math-7B-Instruct (4-bit quantized)

# In[ ]:


# =============================================================================
# IMPORTS AND CONFIGURATION
# =============================================================================

import os
import re
import ast
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import polars as pl
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Kaggle evaluation imports
import kaggle_evaluation.aimo_3_inference_server

print("✓ Imports successful")


# In[ ]:


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class Config:
    """Configuration for the multi-agent system."""

    # Model configuration
    MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"
    MAX_NEW_TOKENS = 4096
    TEMPERATURE = 0.1
    TOP_P = 0.95

    # Quantization for 16GB GPU
    USE_4BIT = True
    BNB_4BIT_COMPUTE_DTYPE = torch.float16

    # Agent configuration
    MAX_ATTEMPTS = 3
    TIMEOUT_PER_PROBLEM = 300  # 5 minutes max per problem

    # Answer constraints
    MIN_ANSWER = 0
    MAX_ANSWER = 99999


print(f"✓ Configuration set")
print(f"  Model: {Config.MODEL_NAME}")
print(f"  Max attempts: {Config.MAX_ATTEMPTS}")
print(f"  Answer range: {Config.MIN_ANSWER}-{Config.MAX_ANSWER}")


# In[ ]:


# =============================================================================
# REFERENCE PROBLEMS (for context understanding)
# =============================================================================

REFERENCE_PROBLEMS = """
Here are examples of AIMO3 problems and their solutions:

Example 1 (Geometry):
Problem: Let $ABC$ be an acute-angled triangle with integer side lengths and $AB<AC$. Points $D$ and $E$ lie on segments $BC$ and $AC$, respectively, such that $AD=AE=AB$. Line $DE$ intersects $AB$ at $X$. Circles $BXD$ and $CED$ intersect for the second time at $Y \neq D$. Suppose that $Y$ lies on line $AD$. There is a unique such triangle with minimal perimeter. This triangle has side lengths $a=BC$, $b=CA$, and $c=AB$. Find the remainder when $abc$ is divided by $10^{5}$.
Answer: 336

Example 2 (Number Theory):
Problem: Define a function $f \colon \mathbb{Z}_{\geq 1} \to \mathbb{Z}_{\geq 1}$ by $f(n) = \sum_{i = 1}^n \sum_{j = 1}^n j^{1024} \left\lfloor\frac1j + \frac{n-i}{n}\right\rfloor$. Let $M=2 \cdot 3 \cdot 5 \cdot 7 \cdot 11 \cdot 13$ and let $N = f{(M^{15})} - f{(M^{15}-1)}$. Let $k$ be the largest non-negative integer such that $2^k$ divides $N$. What is the remainder when $2^k$ is divided by $5^7$?
Answer: 32951

Example 3 (Algebra):
Problem: Alice and Bob are each holding some integer number of sweets. Alice says to Bob: ``If we each added the number of sweets we're holding to our (positive integer) age, my answer would be double yours. If we took the product, then my answer would be four times yours.'' Bob replies: ``Why don't you give me five of your sweets because then both our sum and product would be equal.'' What is the product of Alice and Bob's ages?
Answer: 50

Example 4 (Combinatorics):
Problem: A $500 \times 500$ square is divided into $k$ rectangles, each having integer side lengths. Given that no two of these rectangles have the same perimeter, the largest possible value of $k$ is $\mathcal{K}$. What is the remainder when $k$ is divided by $10^{5}$?
Answer: 520
"""

print("✓ Reference problems loaded")


# In[ ]:


# =============================================================================
# MODEL MANAGER
# =============================================================================


class ModelManager:
    """Manages the LLM loading and inference."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None

    def load(self):
        """Load the model with 4-bit quantization for 16GB GPU."""
        if self.model is not None:
            return self

        print("Loading Qwen2.5-Math-7B-Instruct...")

        # Configure quantization
        if Config.USE_4BIT:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=Config.BNB_4BIT_COMPUTE_DTYPE,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            bnb_config = None

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            Config.MODEL_NAME, trust_remote_code=True
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        self.device = next(self.model.parameters()).device
        print(f"✓ Model loaded on device: {self.device}")

        return self

    def generate(self, prompt: str, temperature: float = None) -> str:
        """Generate response from the model."""
        if self.model is None:
            self.load()

        temp = temperature if temperature is not None else Config.TEMPERATURE

        # Format as chat
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=Config.MAX_NEW_TOKENS,
                temperature=temp,
                top_p=Config.TOP_P,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        return response.strip()


# Global model manager
model_manager = ModelManager()
print("✓ Model manager initialized")


# In[ ]:


# =============================================================================
# AGENT CLASSES
# =============================================================================

# Domain-specific solving instructions
DOMAIN_INSTRUCTIONS = {
    "algebra": "\nFor this algebra problem:\n- Identify the key variables and equations\n- Use substitution, elimination, or factorization as appropriate\n- Check for special cases and edge conditions\n- Verify your solution satisfies the original equation",
    "geometry": "\nFor this geometry problem, use pure synthetic reasoning:\n- Work with geometric properties, theorems, and relationships\n- Apply circle theorems, triangle properties, angle chasing, similar triangles\n- Look for cyclic quadrilaterals, power of a point, harmonic divisions\n- Use coordinate geometry only if synthetic approach is not clear\n- Draw auxiliary lines or constructions when helpful",
    "combinatorics": "\nFor this combinatorics problem:\n- Identify whether this is counting, probability, or existence\n- Consider: permutations, combinations, inclusion-exclusion, generating functions\n- Look for symmetries and bijections to simplify counting\n- Check small cases first to identify patterns\n- Verify your counting doesn't overcount or undercount",
    "number_theory": "\nFor this number theory problem:\n- Consider divisibility, prime factorization, modular arithmetic\n- Apply Fermat's Little Theorem, Euler's theorem, Chinese Remainder Theorem\n- Look for patterns in residues modulo small primes\n- Use the Euclidean algorithm for gcd/lcm problems\n- Check if Chinese Remainder Theorem or lifting the exponent (LTE) applies",
    "unknown": "",
}


class BaseAgent:
    """Base class for all agents."""

    def __init__(self, name: str):
        self.name = name

    def run(self, *args, **kwargs):
        raise NotImplementedError


class AnalyzerAgent(BaseAgent):
    """
    Analyzes the problem to understand its domain and requirements.
    """

    def __init__(self):
        super().__init__("Analyzer")

    def run(self, problem: str) -> Dict:
        """Analyze the problem and return structured information."""

        prompt = f"""You are a mathematical problem analyzer for competition mathematics (IMO/AIME level). Your task is to analyze the problem and classify it.{REFERENCE_PROBLEMS}Now analyze this problem:Problem: {problem}You MUST respond with a valid Python dictionary in this EXACT format (use double quotes for strings, True/False for booleans, None for null):{{    "domain": "algebra",    "problem_type": "computation",    "difficulty_estimate": "hard",    "key_concepts": ["modular arithmetic", "prime factorization"],    "requires_modular_arithmetic": False,    "modulus": None,    "suggested_approach": "Use Chinese Remainder Theorem and analyze the pattern"}}IMPORTANT: - Use DOUBLE QUOTES for all strings- Use True or False (not true/false)- Use None (not null)- Domain must be exactly: algebra, geometry, combinatorics, number_theory, or unknown- Provide 1-3 key concepts as a list- Keep suggested_approach brief (under 100 characters)Respond ONLY with the dictionary, no other text."""

        try:
            response = model_manager.generate(prompt)
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
            "requires_modular_arithmetic": False,
            "modulus": None,
            "suggested_approach": "General mathematical reasoning",
        }


class DirectSolverAgent(BaseAgent):
    """
    Solves problems using pure mathematical reasoning.
    Uses chain-of-thought and synthetic reasoning for geometry.
    """

    def __init__(self):
        super().__init__("DirectSolver")

    def run(
        self, problem: str, analysis: Dict, attempt: int = 1, temperature: float = None
    ) -> Dict:
        """Solve the problem and return answer with confidence."""

        # Use provided temperature or default based on attempt
        if temperature is None:
            temperature = 0.1 if attempt == 1 else (0.3 if attempt == 2 else 0.5)

        domain = analysis.get("domain", "unknown")
        approach = analysis.get("suggested_approach", "")

        # Special instructions for geometry
        # Get domain-specific instructions
        domain_instruction = DOMAIN_INSTRUCTIONS.get(domain, "")

        # Special geometry handling within the domain instruction
        if domain == "geometry":
            domain_instruction = DOMAIN_INSTRUCTIONS["geometry"]

        prompt = f"""You are an expert mathematical problem solver specializing in competition mathematics (IMO/AIME level).{REFERENCE_PROBLEMS}Now solve this problem:Problem: {problem}Analysis: This is a {domain} problem. {approach}{domain_instruction}Important Instructions:1. Think step-by-step using chain-of-reasoning2. Show all your work clearly3. For geometry, use synthetic methods (theorems, not coordinates)4. The answer MUST be a non-negative integer between 0 and 999995. If the problem asks for a remainder, compute it correctlyWork through the solution step by step.After solving, add a VERIFICATION section:- Double-check your calculations- Verify the answer satisfies all problem conditions- Confirm the answer is in the valid range (0-99999)- If you find any errors, correct themState your final answer clearly as: FINAL ANSWER: [number]"""

        try:
            response = model_manager.generate(prompt, temperature=temperature)

            # Extract answer
            answer = self._extract_answer(response)

            return {
                "success": answer is not None,
                "answer": answer,
                "raw_response": response,
                "attempt": attempt,
                "method": "direct_reasoning",
            }

        except Exception as e:
            return {
                "success": False,
                "answer": None,
                "error": str(e),
                "attempt": attempt,
                "method": "direct_reasoning",
            }

    def _extract_answer(self, response: str) -> Optional[int]:
        """Extract the integer answer from model response."""
        # Look for explicit final answer marker
        patterns = [
            r"FINAL ANSWER:\s*(\d+)",
            r"final answer is:?\s*(\d+)",
            r"answer is:?\s*(\d+)",
            r"the answer is:?\s*(\d+)",
            r"\\boxed\{(\d+)\}",
            r"\*\*(\d+)\*\*",
            r"(?<![\d.])\b(\d{1,5})\b(?!\s*\.\d)",  # Standalone numbers
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

        # Fallback: find all numbers and pick the most likely one
        all_numbers = re.findall(r"\b(\d{1,5})\b", response)
        if all_numbers:
            # Return the last number that's in valid range
            for num_str in reversed(all_numbers):
                try:
                    num = int(num_str)
                    if Config.MIN_ANSWER <= num <= Config.MAX_ANSWER:
                        return num
                except ValueError:
                    continue

        return None


class VerifierAgent(BaseAgent):
    """
    Verifies if an answer is valid and makes sense given the problem.
    """

    def __init__(self):
        super().__init__("Verifier")

    def run(self, problem: str, answer: int, analysis: Dict) -> Dict:
        """Verify the answer and return validation result."""

        if answer is None:
            return {"valid": False, "score": 0.0, "reason": "No answer provided"}

        # Basic range check
        if not (Config.MIN_ANSWER <= answer <= Config.MAX_ANSWER):
            return {"valid": False, "score": 0.0, "reason": "Answer out of valid range"}

        # Check if answer is integer
        if not isinstance(answer, int):
            return {"valid": False, "score": 0.0, "reason": "Answer not an integer"}

        # Domain-specific sanity checks
        score = 1.0
        checks = []

        # Check for remainder problems
        if "remainder" in problem.lower():
            modulus = analysis.get("modulus")
            if modulus and answer >= modulus:
                score *= 0.5
                checks.append("Answer may exceed expected modulus")

        # Check for counting problems (usually positive)
        if analysis.get("problem_type") == "counting" and answer == 0:
            score *= 0.8
            checks.append("Counting problem with zero answer")

        return {"valid": True, "score": score, "checks": checks, "answer": answer}


class AggregatorAgent(BaseAgent):
    """
    Aggregates results from multiple solver attempts and selects the best answer.
    Uses consensus voting and self-verification for improved accuracy.
    """

    def __init__(self):
        super().__init__("Aggregator")
        self.analyzer = AnalyzerAgent()
        self.solver = DirectSolverAgent()
        self.verifier = VerifierAgent()

    def run(self, problem: str) -> int:
        """
        Solve the problem using consensus voting and verification.
        Generates multiple solutions and picks the most reliable answer.
        """
        print(f"{'=' * 60}")
        print(f"Solving problem with consensus voting...")
        print(f"{'=' * 60}")

        # Step 1: Analyze the problem
        print("Step 1: Analyzing problem...")
        analysis = self.analyzer.run(problem)
        print(f"  Domain: {analysis.get('domain')}")
        print(f"  Type: {analysis.get('problem_type')}")
        print(f"  Approach: {analysis.get('suggested_approach', 'N/A')}")

        # Step 2: Generate multiple solutions for consensus
        print(f"Step 2: Generating multiple solutions for consensus...")
        NUM_SOLUTIONS = 5
        all_answers = []
        all_results = []

        for i in range(NUM_SOLUTIONS):
            print(f"  Solution {i + 1}/{NUM_SOLUTIONS}:")

            # Use different temperature for diversity
            temperature = 0.1 + (i * 0.15)  # 0.1, 0.25, 0.4, 0.55, 0.7

            result = self.solver.run(
                problem, analysis, attempt=i + 1, temperature=temperature
            )
            all_results.append(result)

            if result["success"]:
                answer = result["answer"]
                all_answers.append(answer)
                print(f"    -> Answer: {answer}")
            else:
                print(f"    -> Failed: {result.get('error', 'Unknown error')}")

        if not all_answers:
            print("  No valid answers generated, returning 0")
            return 0

        # Step 3: Consensus voting
        print(f"Step 3: Consensus voting...")
        from collections import Counter

        answer_counts = Counter(all_answers)
        print(f"  Answer distribution: {dict(answer_counts)}")

        # Get the most common answer
        most_common = answer_counts.most_common()
        consensus_answer = most_common[0][0]
        consensus_count = most_common[0][1]

        print(
            f"  Consensus answer: {consensus_answer} (appears {consensus_count}/{len(all_answers)} times)"
        )

        # Return consensus answer directly (no self-verification)
        print(f"Final answer: {consensus_answer}")
        print(f"{'=' * 60}")
        return consensus_answer


print("✓ All agents defined")


# In[ ]:


# =============================================================================
# MAIN SOLVER INTERFACE
# =============================================================================


class AIMO3Solver:
    """Main solver class that orchestrates the multi-agent system."""

    def __init__(self):
        self.aggregator = AggregatorAgent()

    def solve(self, problem: str) -> int:
        """Solve a single problem and return the answer."""
        try:
            answer = self.aggregator.run(problem)
            return answer
        except Exception as e:
            print(f"Critical error solving problem: {e}")
            return 0  # Safe fallback


# Global solver instance (lazy loaded)
_solver = None


def get_solver():
    """Get or create the solver instance."""
    global _solver
    if _solver is None:
        _solver = AIMO3Solver()
    return _solver


print("✓ Main solver interface ready")


# In[ ]:


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

    print(f"\nProcessing problem ID: {problem_id}")
    print(f"Problem text: {problem_text[:100]}...")

    # Get solver and solve
    solver = get_solver()
    answer = solver.solve(problem_text)

    # Ensure answer is valid
    answer = max(Config.MIN_ANSWER, min(Config.MAX_ANSWER, int(answer)))

    print(f"Final answer for {problem_id}: {answer}")

    # Return as DataFrame
    return pl.DataFrame({"id": problem_id, "answer": answer})


print("✓ Predict function defined")


# In[ ]:


# =============================================================================
# MAIN EXECUTION
# =============================================================================

print("AIMO3 Multi-Agent Solver initialized")
print("Model will be loaded on first prediction")

# Set up inference server
inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(
    predict
)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    # Production mode - serve indefinitely
    print("Starting inference server in production mode...")
    inference_server.serve()
else:
    # Local testing mode
    print("Running in local test mode...")
    test_path = "/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv"

    # Check if we have a local test file
    if os.path.exists(test_path):
        inference_server.run_local_gateway((test_path,))
    else:
        print(f"Test file not found at {test_path}")
        print("Testing with sample problem...")

        # Test with sample problem
        sample_problem = "What is $1+1$?"
        test_id = pl.Series(["test001"])
        test_problem = pl.Series([sample_problem])

        result = predict(test_id, test_problem)
        print(f"\nTest result:")
        print(result)


# In[ ]:


# =============================================================================
# TEST ON REFERENCE PROBLEMS
# =============================================================================


def test_reference_problems():
    """Test the solver on all 10 reference problems."""
    import pandas as pd
    import time

    # Load reference problems
    df = pd.read_csv(
        "/kaggle/input/ai-mathematical-olympiad-progress-prize-3/reference.csv"
    )

    print("=" * 80)
    print("TESTING ON REFERENCE PROBLEMS")
    print("=" * 80)
    print(f"Total problems: {len(df)}")
    print()

    results = []
    solver = get_solver()

    for idx, row in df.iterrows():
        problem_id = row["id"]
        problem_text = row["problem"]
        expected_answer = int(row["answer"])

        print(f"\n{'=' * 80}")
        print(f"Problem {idx + 1}/10 [ID: {problem_id}]")
        print(f"Expected answer: {expected_answer}")
        print(f"{'=' * 80}")

        start_time = time.time()

        try:
            predicted_answer = solver.solve(problem_text)
            elapsed = time.time() - start_time

            is_correct = predicted_answer == expected_answer

            result = {
                "id": problem_id,
                "expected": expected_answer,
                "predicted": predicted_answer,
                "correct": is_correct,
                "time": elapsed,
            }
            results.append(result)

            status = "✓ CORRECT" if is_correct else "✗ WRONG"
            print(f"\n{status}")
            print(f"  Predicted: {predicted_answer}")
            print(f"  Expected:  {expected_answer}")
            print(f"  Time: {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n✗ ERROR: {e}")
            results.append(
                {
                    "id": problem_id,
                    "expected": expected_answer,
                    "predicted": None,
                    "correct": False,
                    "time": elapsed,
                    "error": str(e),
                }
            )

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    correct_count = sum(1 for r in results if r["correct"])
    total_time = sum(r["time"] for r in results)

    print(f"\nCorrect: {correct_count}/10 ({100 * correct_count / 10:.1f}%)")
    print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"\nPer-problem results:")

    for r in results:
        status = "✓" if r["correct"] else "✗"
        print(
            f"  {status} {r['id']}: pred={r['predicted']}, exp={r['expected']}, time={r['time']:.1f}s"
        )

    return results


# Run the test
results = test_reference_problems()
