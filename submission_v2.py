#!/usr/bin/env python
# coding: utf-8

"""
AIMO3 Improved Submission - Optimized for 2x T4 GPUs
Strategy: Robust answer extraction + Self-verification + Adaptive consensus
"""

import ast
import json
import os
import re
import time
import warnings
import glob
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import kaggle_evaluation.aimo_3_inference_server
import pandas as pd
import polars as pl
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Suppress warnings
warnings.filterwarnings("ignore")

print("✓ Imports successful")


@dataclass
class Config:
    """Configuration for improved consensus voting system."""

    # Model configuration - Try multiple paths
    MODEL_PATHS = [
        "/kaggle/input/qwen2-5-32b-instruct-quant/",
        "/kaggle/input/qwen2-5-32b-instruct-awq/",
        "/kaggle/input/deepseek-math-7b-instruct/",
        "/kaggle/input/deepseek-math-7b/",
    ]
    MAX_NEW_TOKENS = 4096
    BASE_TEMPERATURE = 0.1
    TOP_P = 0.95

    # GPU Configuration
    USE_FP16 = True
    DEVICE_MAP = "auto"

    # Consensus configuration - Adaptive
    MAX_SOLUTIONS = 5  # Maximum solutions to generate
    MIN_SOLUTIONS = 2  # Minimum for consensus
    TEMPERATURES = [0.1, 0.2, 0.3, 0.4, 0.5]

    # Verification
    VERIFY_ANSWERS = True
    VERIFICATION_TEMPERATURE = 0.1

    # Constraints
    MIN_ANSWER = 0
    MAX_ANSWER = 99999
    TIMEOUT_PER_PROBLEM = 300


print(f"✓ Configuration loaded")


# Domain-specific instructions
DOMAIN_INSTRUCTIONS = {
    "algebra": """Solve this algebra problem step-by-step:
- Identify variables and set up equations
- Use substitution or elimination methods
- Check special cases and edge conditions
- Verify the final answer numerically""",
    "geometry": """Solve this geometry problem using synthetic methods:
- Apply relevant theorems (circle, triangle, angle properties)
- Use coordinate geometry only if necessary
- Verify geometric constraints
- Double-check calculations""",
    "combinatorics": """Solve this combinatorics problem:
- Consider counting principles, permutations, combinations
- Use inclusion-exclusion if needed
- Check small cases for patterns
- Verify the counting logic""",
    "number_theory": """Solve this number theory problem:
- Use modular arithmetic, prime factorization
- Apply relevant theorems (Fermat's, Euler's, CRT)
- Check divisibility conditions
- Verify modular calculations""",
    "unknown": "Solve this problem using careful mathematical reasoning and step-by-step analysis.",
}


REFERENCE_PROBLEMS = """
Here are examples of AIMO3 problems and their solutions:

Example 1 (Geometry):
Problem: Let $ABC$ be an acute-angled triangle with integer side lengths and $AB<AC$. Points $D$ and $E$ lie on segments $BC$ and $AC$, respectively, such that $AD=AE=AB$. Line $DE$ intersects $AB$ at $X$. Circles $BXD$ and $CED$ intersect for the second time at $Y \\neq D$. Suppose that $Y$ lies on line $AD$. There is a unique such triangle with minimal perimeter. This triangle has side lengths $a=BC$, $b=CA$, and $c=AB$. Find the remainder when $abc$ is divided by $10^{5}$.
Solution: After geometric analysis with the given constraints, we find the triangle has sides 13, 14, 15, giving $abc = 2730$. The remainder when divided by $10^5$ is 2730.
Answer: 2730

Example 2 (Number Theory):
Problem: Define a function $f \\colon \\mathbb{Z}_{\\geq 1} \\to \\mathbb{Z}_{\\geq 1}$ by $f(n) = \\sum_{i = 1}^n \\sum_{j = 1}^n j^{1024} \\left\\lfloor\\frac1j + \\frac{n-i}{n}\\right\\rfloor$. Let $M=2 \\cdot 3 \\cdot 5 \\cdot 7 \\cdot 11 \\cdot 13$ and let $N = f{(M^{15})} - f{(M^{15}-1)}$. Let $k$ be the largest non-negative integer such that $2^k$ divides $N$. What is the remainder when $2^k$ is divided by $5^7$?
Solution: Analyzing the floor function and the structure of $f(n)$, we determine the power of 2 dividing $N$ and compute $2^k \\mod 5^7 = 32951$.
Answer: 32951

Example 3 (Algebra):
Problem: Alice and Bob are each holding some integer number of sweets. Alice says to Bob: ``If we each added the number of sweets we're holding to our (positive integer) age, my answer would be double yours. If we took the product, then my answer would be four times yours.'' Bob replies: ``Why don't you give me five of your sweets because then both our sum and product would be equal.'' What is the product of Alice and Bob's ages?
Solution: Setting up equations from the statements: Let $A$ = Alice's sweets, $B$ = Bob's sweets, $a$ = Alice's age, $b$ = Bob's age. From the conditions: $A+a = 2(B+b)$, $AB = 4ab$, and after transfer: $A-5+a = B+5+b$, $(A-5)a = (B+5)b$. Solving gives $a=10$, $b=5$, so the product is 50.
Answer: 50

Example 4 (Combinatorics):
Problem: A $500 \\times 500$ square is divided into $k$ rectangles, each having integer side lengths. Given that no two of these rectangles have the same perimeter, the largest possible value of $k$ is $\\mathcal{K}$. What is the remainder when $k$ is divided by $10^{5}$?
Solution: For a rectangle with integer sides $a, b$ where $1 \\leq a \\leq 500$ and $1 \\leq b \\leq 500$, the perimeter is $2(a+b)$. The number of distinct perimeters equals the number of distinct values of $a+b$ where $2 \\leq a+b \\leq 1000$, which is 999. So $k = 999$ and $999 \\mod 10^5 = 999$.
Answer: 999
"""


class RobustAnswerExtractor:
    """Robust answer extraction with multiple strategies."""

    @staticmethod
    def extract_answer(response: str) -> Optional[int]:
        """
        Extract answer using multiple strategies, from most to least specific.
        Returns the best candidate in valid range [0, 99999].
        """
        if not response:
            return None

        # Strategy 1: Look for explicit answer markers (case-insensitive)
        explicit_patterns = [
            r"FINAL ANSWER[:\s]+(\d+)",
            r"The answer is[:\s]+(\d+)",
            r"answer[:\s]+(\d+)",
            r"result[:\s]+(\d+)",
            r"Therefore,\s+\$?(\d+)\$?",
            r"Thus,\s+\$?(\d+)\$?",
            r"\\boxed\{(\d+)\}",
            r"\\mathbf\{(\d+)\}",
            r"\\textbf\{(\d+)\}",
            r"\*\*(\d+)\*\*",  # Markdown bold
            r"`(\d+)`",  # Inline code
        ]

        candidates = []

        for pattern in explicit_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
            for match in matches:
                try:
                    num = int(match)
                    if Config.MIN_ANSWER <= num <= Config.MAX_SWER:
                        candidates.append((num, "explicit", len(candidates)))
                except ValueError:
                    continue

        # Strategy 2: Find the last standalone number in the response
        # This is a fallback when no explicit markers are found
        last_number_pattern = r"(?<!\d)(\d{1,5})(?!\d)"
        all_numbers = re.findall(last_number_pattern, response)

        if all_numbers:
            # Get the last few numbers (likely the answer)
            for num_str in reversed(all_numbers[-5:]):  # Check last 5 numbers
                try:
                    num = int(num_str)
                    if Config.MIN_ANSWER <= num <= Config.MAX_ANSWER:
                        candidates.append((num, "last_number", len(candidates)))
                except ValueError:
                    continue

        # Strategy 3: Look for numbers after common math operators
        math_patterns = [
            r"=\s*(\d+)",
            r"\\equiv\s*(\d+)",
            r"\\equiv\s*\\boxed\{(\d+)\}",
        ]

        for pattern in math_patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                try:
                    num = int(match)
                    if Config.MIN_ANSWER <= num <= Config.MAX_ANSWER:
                        candidates.append((num, "math", len(candidates)))
                except ValueError:
                    continue

        if not candidates:
            return None

        # Prioritize candidates: explicit markers > math context > last number
        priority_order = {"explicit": 0, "math": 1, "last_number": 2}
        candidates.sort(key=lambda x: (priority_order.get(x[1], 3), x[2]))

        # Return the highest priority candidate
        return candidates[0][0]

    @staticmethod
    def extract_all_answers(response: str) -> List[int]:
        """Extract all valid answers from response for consensus analysis."""
        numbers = []
        pattern = r"(?<!\d)(\d{1,5})(?!\d)"
        matches = re.findall(pattern, response)

        for match in set(matches):  # Use set to avoid duplicates
            try:
                num = int(match)
                if Config.MIN_ANSWER <= num <= Config.MAX_ANSWER:
                    numbers.append(num)
            except ValueError:
                continue

        return numbers


class MultiGPUModelManager:
    """Manages model loading with fallback paths."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_path = None
        self.num_gpus = torch.cuda.device_count()

    def _find_model_path(self) -> str:
        """Search for the model files in Kaggle input directories."""
        # First check configured paths
        for path in Config.MODEL_PATHS:
            if os.path.exists(path):
                files = os.listdir(path)
                if any(
                    f.endswith((".bin", ".safetensors", "config.json")) for f in files
                ):
                    print(f"✓ Found model at: {path}")
                    return path

        # Search recursively in /kaggle/input/
        print("Searching for model files in /kaggle/input/...")
        for root, dirs, files in os.walk("/kaggle/input/"):
            if "config.json" in files:
                print(f"Found model directory: {root}")
                if any(f.endswith((".bin", ".safetensors")) for f in files):
                    return root

        raise FileNotFoundError(
            f"Could not find model files. Checked: {Config.MODEL_PATHS}"
        )

    def load(self):
        """Load model with fallback paths."""
        if self.model is not None:
            return self

        print("Loading model...")
        print(f"Available GPUs: {self.num_gpus}")

        try:
            self.model_path = self._find_model_path()
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            raise

        max_memory = (
            {i: "15GiB" for i in range(self.num_gpus)} if self.num_gpus > 0 else None
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True, local_files_only=True
            )
            print("✓ Tokenizer loaded")

            load_kwargs = {
                "torch_dtype": torch.float16 if Config.USE_FP16 else torch.float32,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "local_files_only": True,
            }

            if self.num_gpus > 0:
                load_kwargs["device_map"] = Config.DEVICE_MAP
                load_kwargs["max_memory"] = max_memory

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, **load_kwargs
            )
            print("✓ Model loaded")

        except Exception as e:
            print(f"ERROR loading model: {e}")
            raise

        if hasattr(self.model, "hf_device_map"):
            print(f"✓ Model sharded across: {self.model.hf_device_map}")
        else:
            self.device = next(self.model.parameters()).device
            print(f"✓ Model on device: {self.device}")

        return self

    def generate(
        self, prompt: str, temperature: float = None, max_new_tokens: int = None
    ) -> str:
        """Generate response with error handling."""
        if self.model is None:
            self.load()

        temp = temperature if temperature is not None else Config.BASE_TEMPERATURE
        max_tokens = (
            max_new_tokens if max_new_tokens is not None else Config.MAX_NEW_TOKENS
        )

        try:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.tokenizer(text, return_tensors="pt")

            if hasattr(self.model, "hf_device_map"):
                inputs = inputs.to("cuda:0")
            elif self.device:
                inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temp,
                    top_p=Config.TOP_P,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

            return response.strip()

        except Exception as e:
            print(f"Generation error: {e}")
            return ""


# Global model manager (lazy loading)
model_manager = MultiGPUModelManager()
print("✓ Multi-GPU model manager initialized")


class AnalyzerAgent:
    """Analyzes problem to understand domain and requirements."""

    def __init__(self):
        self.name = "Analyzer"

    def run(self, problem: str) -> Dict:
        """Analyze problem using keyword-based detection (faster than LLM)."""
        problem_lower = problem.lower()

        domain = "unknown"
        if any(
            kw in problem_lower
            for kw in [
                "triangle",
                "circle",
                "angle",
                "line",
                "point",
                "segment",
                "perpendicular",
                "parallel",
                "geometry",
            ]
        ):
            domain = "geometry"
        elif any(
            kw in problem_lower
            for kw in [
                "prime",
                "divisible",
                "modular",
                "remainder",
                "gcd",
                "lcm",
                "factor",
                "mod",
            ]
        ):
            domain = "number_theory"
        elif any(
            kw in problem_lower
            for kw in [
                "count",
                "arrange",
                "choose",
                "select",
                "combinatorics",
                "permutation",
                "combination",
            ]
        ):
            domain = "combinatorics"
        elif any(
            kw in problem_lower
            for kw in [
                "equation",
                "solve",
                "find x",
                "variable",
                "polynomial",
                "function",
                "algebra",
            ]
        ):
            domain = "algebra"

        return {
            "domain": domain,
            "problem_type": "computation",
            "difficulty_estimate": "hard",
            "key_concepts": [],
            "suggested_approach": f"Use {domain} problem-solving techniques",
        }


class SolverAgent:
    """Generates solutions with robust answer extraction."""

    def __init__(self):
        self.name = "Solver"
        self.extractor = RobustAnswerExtractor()

    def create_prompt(self, problem: str, analysis: Dict) -> str:
        """Create solving prompt."""
        domain = analysis.get("domain", "unknown")
        domain_instruction = DOMAIN_INSTRUCTIONS.get(
            domain, DOMAIN_INSTRUCTIONS["unknown"]
        )

        prompt = f"""You are an expert mathematical problem solver for competition mathematics (IMO/AIME level).

{REFERENCE_PROBLEMS}

Now solve this problem:
Problem: {problem}

This is a {domain} problem.

{domain_instruction}

IMPORTANT INSTRUCTIONS:
1. Work through the solution step-by-step
2. Show all calculations clearly
3. The answer MUST be a non-negative integer between 0 and 99999
4. After your solution, state: FINAL ANSWER: [your number]

Example: "...therefore the answer is 42. FINAL ANSWER: 42"

Solve the problem now:"""

        return prompt

    def run(self, problem: str, analysis: Dict, temperature: float) -> Dict:
        """Generate solution and extract answer."""
        prompt = self.create_prompt(problem, analysis)

        try:
            response = model_manager.generate(prompt, temperature=temperature)
            answer = self.extractor.extract_answer(response)

            # Get all candidates for analysis
            all_answers = self.extractor.extract_all_answers(response)

            return {
                "success": answer is not None,
                "answer": answer,
                "raw_response": response,
                "temperature": temperature,
                "all_candidates": all_answers,
            }
        except Exception as e:
            return {
                "success": False,
                "answer": None,
                "error": str(e),
                "temperature": temperature,
            }


class VerificationAgent:
    """Verifies if an answer is correct for the given problem."""

    def __init__(self):
        self.name = "Verifier"

    def verify(self, problem: str, answer: int) -> float:
        """Verify the answer. Returns confidence score 0.0-1.0."""
        if not Config.VERIFY_ANSWERS:
            return 1.0

        prompt = f"""You are verifying a solution to a mathematics competition problem.

Problem: {problem}

Proposed Answer: {answer}

Verify if this answer is correct:
1. Does the answer satisfy all conditions in the problem?
2. Is the answer a non-negative integer between 0 and 99999?
3. Are there any obvious errors or contradictions?

Respond with EXACTLY one of:
- "CORRECT: [brief reason]" if the answer is definitely correct
- "LIKELY_CORRECT: [brief reason]" if the answer seems correct but you're not 100% certain
- "UNCERTAIN: [brief reason]" if you cannot verify
- "INCORRECT: [brief reason]" if the answer is definitely wrong

Your verification:"""

        try:
            response = model_manager.generate(
                prompt, temperature=Config.VERIFICATION_TEMPERATURE, max_new_tokens=256
            )

            response_upper = response.upper()

            if (
                "CORRECT:" in response_upper
                and "LIKELY" not in response_upper
                and "UN" not in response_upper
            ):
                return 1.0
            elif "LIKELY_CORRECT" in response_upper:
                return 0.8
            elif "UNCERTAIN" in response_upper:
                return 0.5
            elif "INCORRECT" in response_upper:
                return 0.0
            else:
                return 0.5

        except Exception as e:
            print(f"Verification error: {e}")
            return 0.5


class AdaptiveConsensusAgent:
    """Implements adaptive consensus with verification."""

    def __init__(self):
        self.name = "AdaptiveConsensus"
        self.analyzer = AnalyzerAgent()
        self.solver = SolverAgent()
        self.verifier = VerificationAgent()

    def run(self, problem: str) -> int:
        """Solve problem using adaptive consensus."""
        print(f"\n{'=' * 60}")
        print("ADAPTIVE CONSENSUS SOLVER")
        print(f"{'=' * 60}")

        # Step 1: Analyze
        print("Step 1: Analyzing problem...")
        analysis = self.analyzer.run(problem)
        print(f"  Domain: {analysis['domain']}")

        # Step 2: Generate solutions adaptively
        solutions = []
        verified_answers = {}

        for i, temp in enumerate(Config.TEMPERATURES[: Config.MAX_SOLUTIONS]):
            print(f"\n  Solution {i + 1}/{Config.MAX_SOLUTIONS} (temp={temp})...")

            result = self.solver.run(problem, analysis, temp)
            solutions.append(result)

            if result["success"] and result["answer"] is not None:
                print(f"    ✓ Candidate: {result['answer']}")

                # Verify if not already verified
                answer = result["answer"]
                if answer not in verified_answers:
                    print(f"    Verifying...")
                    verification_score = self.verifier.verify(problem, answer)
                    verified_answers[answer] = verification_score
                    print(f"    Verification score: {verification_score:.2f}")

                # Early exit if we have a high-confidence verified answer
                if verified_answers[answer] >= 0.9:
                    print(f"    ✓✓ High confidence answer found!")
                    # Check if this answer appears multiple times
                    answer_count = sum(
                        1 for s in solutions if s.get("answer") == answer
                    )
                    if answer_count >= 2:
                        print(
                            f"\n✓ Early termination: {answer} (verified and consistent)"
                        )
                        return answer
            else:
                print(f"    ✗ Failed: {result.get('error', 'No answer found')}")
                # Try fallback extraction
                if result.get("raw_response"):
                    all_nums = RobustAnswerExtractor.extract_all_answers(
                        result["raw_response"]
                    )
                    if all_nums:
                        print(f"    Fallback: Found numbers {all_nums}")

        # Step 3: Select best answer
        print(f"\nStep 2: Selecting best answer...")

        # Collect all valid answers with their verification scores
        valid_solutions = [
            s for s in solutions if s["success"] and s["answer"] is not None
        ]

        if not valid_solutions:
            print("  No valid answers found! Using fallback extraction...")
            # Try to extract any number from any response
            for s in solutions:
                if s.get("raw_response"):
                    all_nums = RobustAnswerExtractor.extract_all_answers(
                        s["raw_response"]
                    )
                    if all_nums:
                        print(
                            f"  Fallback: Using last number from response: {all_nums[-1]}"
                        )
                        return all_nums[-1]
            print("  Complete fallback failure. Returning 0.")
            return 0

        # Count occurrences and calculate weighted scores
        answer_votes = Counter(s["answer"] for s in valid_solutions)

        # Calculate weighted score for each answer
        best_answer = None
        best_score = -1

        print(f"  Vote distribution: {dict(answer_votes)}")

        for answer, votes in answer_votes.items():
            vote_ratio = votes / len(valid_solutions)
            verification_score = verified_answers.get(answer, 0.5)
            total_score = vote_ratio * verification_score

            print(
                f"    Answer {answer}: {votes} votes, verification={verification_score:.2f}, score={total_score:.2f}"
            )

            if total_score > best_score:
                best_score = total_score
                best_answer = answer

        if best_answer is None:
            best_answer = valid_solutions[0]["answer"]

        print(f"\n✓ Final answer: {best_answer} (score: {best_score:.2f})")
        print(f"{'=' * 60}")

        return best_answer


class AIMO3Solver:
    """Main solver interface."""

    def __init__(self):
        self.consensus_agent = AdaptiveConsensusAgent()

    def solve(self, problem: str) -> int:
        """Solve a single problem."""
        try:
            answer = self.consensus_agent.run(problem)
            if answer is None:
                return 0
            return max(Config.MIN_ANSWER, min(Config.MAX_ANSWER, int(answer)))
        except Exception as e:
            print(f"Critical error: {e}")
            import traceback

            traceback.print_exc()
            return 0


# Global solver (lazy loaded)
_solver = None


def get_solver():
    """Get or create solver instance."""
    global _solver
    if _solver is None:
        _solver = AIMO3Solver()
    return _solver


print("✓ All classes initialized")


# Kaggle inference interface
def predict(id_: pl.Series, problem: pl.Series) -> pl.DataFrame:
    """
    Kaggle inference API function.

    Args:
        id_: Polars Series containing problem ID
        problem: Polars Series containing problem text

    Returns:
        Polars DataFrame with 'id' and 'answer' columns
    """
    problem_id = id_.item(0)
    problem_text = problem.item(0)

    print(f"\n{'=' * 60}")
    print(f"Processing: {problem_id}")
    print(f"{'=' * 60}")
    print(f"Problem: {problem_text[:100]}...")

    solver = get_solver()
    answer = solver.solve(problem_text)

    print(f"✓ Answer: {answer}")

    return pl.DataFrame({"id": problem_id, "answer": answer})


print("✓ Predict function ready")


# Main execution
print("\n" + "=" * 60)
print("AIMO3 Improved Consensus Solver")
print("Features: Robust extraction + Verification + Adaptive consensus")
print("=" * 60)

inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(
    predict
)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    print("\n🚀 Starting inference server in PRODUCTION mode...")
    inference_server.serve()
else:
    print("\n🧪 Running in LOCAL TEST mode...")

    test_path = "/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv"

    if os.path.exists(test_path):
        print(f"Loading test data from: {test_path}")
        inference_server.run_local_gateway((test_path,))
    else:
        print("Test file not found. Running sample problem...")

        sample_problem = (
            "What is the sum of all positive integers $n$ such that $n^2 - 3n + 2 = 0$?"
        )

        test_id = pl.Series(["test001"])
        test_problem = pl.Series([sample_problem])

        result = predict(test_id, test_problem)
        print(f"\nResult:")
        print(result)
