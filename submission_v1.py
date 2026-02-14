"""
AIMO3 Multi-Agent Submission Notebook - Version 1
Minimal viable implementation with 4 core agents
"""

import os
import re
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import polars as pl
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Kaggle evaluation imports
import sys

sys.path.append("/kaggle/input/ai-mathematical-olympiad-progress-prize-3")
import kaggle_evaluation.aimo_3_inference_server


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class Config:
    """Configuration for the multi-agent system."""

    # Model configuration
    MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
    MAX_NEW_TOKENS = 2048
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


# =============================================================================
# REFERENCE PROBLEMS (for context understanding)
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
        print(f"Model loaded on device: {self.device}")

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


# =============================================================================
# AGENT CLASSES
# =============================================================================


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

        prompt = f"""You are a mathematical problem analyzer. Analyze the following problem and classify it.

{REFERENCE_PROBLEMS}

Now analyze this problem:

Problem: {problem}

Provide your analysis in this exact JSON format:
{{
    "domain": "algebra|geometry|combinatorics|number_theory",
    "problem_type": "computation|proof|optimization|counting",
    "difficulty_estimate": "easy|medium|hard",
    "key_concepts": ["concept1", "concept2"],
    "requires_modular_arithmetic": true|false,
    "modulus": null|number,
    "suggested_approach": "brief description of solution strategy"
}}

Respond ONLY with the JSON, no other text."""

        try:
            response = model_manager.generate(prompt)
            # Extract JSON from response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
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

    def run(self, problem: str, analysis: Dict, attempt: int = 1) -> Dict:
        """Solve the problem and return answer with confidence."""

        # Adjust temperature for retry attempts
        temperature = 0.1 if attempt == 1 else (0.3 if attempt == 2 else 0.5)

        domain = analysis.get("domain", "unknown")
        approach = analysis.get("suggested_approach", "")

        # Special instructions for geometry
        geometry_instruction = ""
        if domain == "geometry":
            geometry_instruction = """
For this geometry problem, use pure synthetic reasoning:
- Work with geometric properties, theorems, and relationships
- Use coordinate geometry only if synthetic approach fails
- Apply circle theorems, triangle properties, angle chasing
- Look for similar triangles, cyclic quadrilaterals, power of a point"""

        prompt = f"""You are an expert mathematical problem solver specializing in competition mathematics (IMO/AIME level).

{REFERENCE_PROBLEMS}

Now solve this problem:

Problem: {problem}

Analysis: This is a {domain} problem. {approach}{geometry_instruction}

Important Instructions:
1. Think step-by-step using chain-of-reasoning
2. Show all your work clearly
3. For geometry, use synthetic methods (theorems, not coordinates)
4. The answer MUST be a non-negative integer between 0 and 99999
5. If the problem asks for a remainder, compute it correctly
6. State your final answer clearly as: FINAL ANSWER: [number]

Work through the solution:
"""

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
    Aggregates results from solver attempts and selects the best answer.
    Uses fixed retry logic.
    """

    def __init__(self):
        super().__init__("Aggregator")
        self.analyzer = AnalyzerAgent()
        self.solver = DirectSolverAgent()
        self.verifier = VerifierAgent()

    def run(self, problem: str) -> int:
        """
        Solve the problem with fixed attempts and return the best answer.
        """
        print(f"\n{'=' * 60}")
        print(f"Solving problem...")
        print(f"{'=' * 60}")

        # Step 1: Analyze the problem
        print("Step 1: Analyzing problem...")
        analysis = self.analyzer.run(problem)
        print(f"  Domain: {analysis.get('domain')}")
        print(f"  Type: {analysis.get('problem_type')}")
        print(f"  Approach: {analysis.get('suggested_approach', 'N/A')}")

        # Step 2: Attempt to solve (fixed attempts)
        best_answer = 0
        best_score = -1
        all_results = []

        for attempt in range(1, Config.MAX_ATTEMPTS + 1):
            print(
                f"\nStep 2.{attempt}: Solving (attempt {attempt}/{Config.MAX_ATTEMPTS})..."
            )

            # Try direct solving
            result = self.solver.run(problem, analysis, attempt)
            all_results.append(result)

            if result["success"]:
                answer = result["answer"]
                print(f"  Raw answer: {answer}")

                # Verify the answer
                verification = self.verifier.run(problem, answer, analysis)
                print(f"  Verification: {verification}")

                if verification["valid"]:
                    score = verification["score"]

                    # Prefer this answer if it has higher score
                    if score > best_score:
                        best_score = score
                        best_answer = answer
                        print(f"  -> New best answer: {answer} (score: {score})")

                    # If we have a perfect score, we can stop
                    if score >= 1.0:
                        print(f"  -> Perfect score achieved, stopping.")
                        break
                else:
                    print(f"  -> Answer failed verification: {verification['reason']}")
            else:
                print(f"  -> Solver failed: {result.get('error', 'Unknown error')}")

        # Step 3: Return best answer
        print(f"\nStep 3: Final answer selected: {best_answer}")
        print(f"{'=' * 60}\n")

        return best_answer


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


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Initialize model on first call (lazy loading happens in predict)
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
            print(f"\nTest result:\n{result}")
