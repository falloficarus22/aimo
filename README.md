# AIMO3 Multi-Agent Submission - Version 1

**Competition:** AI Mathematical Olympiad - Progress Prize 3  
**Submission:** Multi-Agent System with Qwen2.5-Math-7B  
**Date:** February 2026

## Overview

This submission implements a minimal viable multi-agent system for solving olympiad-level mathematics problems. The system uses a 4-agent architecture with zero-shot prompting and synthetic reasoning for geometry problems.

## Architecture

```
Problem Input
      ↓
┌─────────────────┐
│ Analyzer Agent  │ → Classifies domain (algebra/geometry/combinatorics/number theory)
└─────────────────┘
      ↓
┌──────────────────┐
│ Direct Solver    │ → Chain-of-thought reasoning (3 attempts with varying temperature)
│ Agent            │
└──────────────────┘
      ↓
┌─────────────────┐
│ Verifier Agent  │ → Validates answer range and sanity checks
└─────────────────┘
      ↓
┌──────────────────┐
│ Aggregator Agent │ → Selects best answer from attempts
└──────────────────┘
      ↓
  Final Answer (0-99999)
```

## Files

- `submission_v1.ipynb` - Main Kaggle submission notebook
- `submission_v1.py` - Python script version (for reference)
- `README.md` - This documentation

## Model Details

**Primary Model:** Qwen2.5-Math-7B-Instruct
- **Parameters:** 7 billion
- **Quantization:** 4-bit (NF4) with double quantization
- **Memory Usage:** ~7-8 GB (fits comfortably in 16GB GPU)
- **Strengths:** #1 open-source on math benchmarks (MATH, GSM8K)

**Alternative Models:**
- DeepSeek-Math-7B-Instruct (similar performance)
- Llama-3.1-8B-Instruct (less math-specialized)

## Agent Specifications

### 1. Analyzer Agent
**Purpose:** Problem classification and approach suggestion

**Output:**
```json
{
    "domain": "algebra|geometry|combinatorics|number_theory",
    "problem_type": "computation|proof|optimization|counting",
    "difficulty_estimate": "easy|medium|hard",
    "key_concepts": ["concept1", "concept2"],
    "requires_modular_arithmetic": true|false,
    "modulus": null|number,
    "suggested_approach": "brief description"
}
```

**Strategy:** Uses reference problems as context for classification

### 2. Direct Solver Agent
**Purpose:** Core problem solving using mathematical reasoning

**Key Features:**
- Zero-shot prompting with reference examples
- Chain-of-thought reasoning
- Temperature variation for retries (0.1 → 0.3 → 0.5)
- Special synthetic reasoning instructions for geometry
- Multiple answer extraction patterns

**Prompt Strategy:**
```
You are an expert mathematical problem solver specializing in 
competition mathematics (IMO/AIME level).

[Reference Problems Context]

Problem: {problem}
Analysis: {domain} problem. {approach}

[Geometry-specific instructions if applicable]

1. Think step-by-step using chain-of-reasoning
2. Show all your work clearly
3. The answer MUST be a non-negative integer between 0 and 99999
4. State your final answer clearly as: FINAL ANSWER: [number]
```

### 3. Verifier Agent
**Purpose:** Answer validation and sanity checking

**Validation Checks:**
- Range validation (0-99999)
- Integer type check
- Remainder problem sanity (answer < modulus)
- Counting problem sanity (usually positive)

**Scoring:** 0.0-1.0 confidence score

### 4. Aggregator Agent
**Purpose:** Orchestrate solving process and select final answer

**Workflow:**
1. Analyze problem
2. Attempt solving (up to 3 times with different temperatures)
3. Verify each answer
4. Select answer with highest verification score
5. Return best answer or 0 (safe fallback)

## Configuration

```python
class Config:
    MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
    MAX_NEW_TOKENS = 2048
    TEMPERATURE = 0.1
    TOP_P = 0.95
    
    USE_4BIT = True  # Quantization for 16GB GPU
    MAX_ATTEMPTS = 3  # Fixed retry attempts
    
    MIN_ANSWER = 0
    MAX_ANSWER = 99999
```

## Geometry Handling

Uses **pure synthetic reasoning** as requested:
- Geometric properties and theorems
- Circle theorems, triangle properties
- Angle chasing
- Similar triangles
- Cyclic quadrilaterals
- Power of a point

Coordinate geometry is only used as fallback if synthetic approach fails.

## Reference Problems

The system includes 4 reference problems in the prompts:
1. Geometry (triangle configuration) → Answer: 336
2. Number Theory (divisibility) → Answer: 32951
3. Algebra (word problem) → Answer: 50
4. Combinatorics (rectangle division) → Answer: 520

These provide context for the model to understand problem difficulty and format.

## Answer Extraction

Multiple patterns are used to extract answers:
1. `FINAL ANSWER: (\d+)` - Explicit marker
2. `\boxed{(\d+)}` - LaTeX box notation
3. `**(\d+)**` - Bold formatting
4. Standalone numbers (last occurrence)

Fallback to last valid number in response if no pattern matches.

## Performance Considerations

**Time Budget:**
- 50 problems in 5 hours = 6 minutes/problem average
- Breakdown:
  - Analysis: ~30s
  - Solving (3 attempts): ~3-4 minutes
  - Verification: ~30s
  - Overhead: ~1 minute

**Memory Management:**
- Model loaded once (lazy initialization)
- 4-bit quantization reduces VRAM usage
- Gradient checkpointing available if needed

**Error Handling:**
- Model timeout → Fallback to simpler prompt
- Parse errors → Retry with same parameters
- All failures → Return 0 (valid per competition rules)

## Usage

### On Kaggle:
1. Upload `submission_v1.ipynb` to Kaggle Notebooks
2. Attach competition dataset
3. Select GPU accelerator (T4 or better)
4. Run all cells
5. Submit to competition

### Local Testing:
```python
# The notebook will auto-detect local mode and test with sample problem
# Or manually test:
from submission_v1 import predict
import polars as pl

result = predict(
    pl.Series(["test001"]),
    pl.Series(["What is $1+1$?"])
)
print(result)
```

## Known Limitations (Version 1)

1. **No Code Execution:** All reasoning is done by LLM without Python/SymPy verification
2. **No Multi-Agent Voting:** Only uses Direct Solver (no Code Generator or parallel solving)
3. **Limited Verification:** Basic sanity checks only, no mathematical proof verification
4. **Fixed Attempts:** Always does 3 attempts, no early stopping on high confidence
5. **No Fine-tuning:** Uses base model without competition-specific fine-tuning

## Future Improvements (Roadmap)

### Version 2:
- [ ] Add Code Generator Agent with SymPy execution
- [ ] Implement parallel agent voting
- [ ] Add problem-specific verification (e.g., modular arithmetic checking)
- [ ] Dynamic temperature adjustment based on confidence

### Version 3:
- [ ] Fine-tune on reference problems + additional math datasets
- [ ] Add geometric diagram parsing (if problems include diagrams)
- [ ] Implement self-consistency with majority voting
- [ ] Add caching for similar problems

### Version 4:
- [ ] Multi-model ensemble (Qwen + DeepSeek + Llama)
- [ ] Advanced retry strategies (problem-dependent)
- [ ] Meta-learning for approach selection
- [ ] Automated hyperparameter tuning

## Competition Strategy

**Expected Performance:**
- With 7B model: ~20-40% accuracy on AIMO3 problems (estimated)
- Strengths: Algebra, Number Theory
- Weaknesses: Complex geometry proofs, multi-step combinatorics

**Submission Plan:**
1. Version 1: Baseline with 4-agent system (this submission)
2. Version 2: Add code execution and parallel solving
3. Version 3: Fine-tuned model with improved verification
4. Version 4: Ensemble approach with multiple models

## Technical Requirements

**Hardware:**
- GPU: 16GB VRAM minimum (T4, P100, V100, A100)
- CPU: Standard Kaggle notebook CPU sufficient
- RAM: 16GB+ recommended

**Dependencies:**
```
torch>=2.0.0
transformers>=4.35.0
bitsandbytes>=0.41.0
accelerate>=0.24.0
polars>=0.19.0
pandas>=2.0.0
```

**Kaggle Environment:**
- Internet: Disabled (per competition rules)
- Accelerator: GPU T4x2 or better
- Runtime: 5 hours GPU maximum

## Debugging

If you encounter issues:

1. **Model loading fails:**
   - Check GPU memory availability
   - Verify model name is correct
   - Ensure internet was enabled during model download

2. **Answer extraction fails:**
   - Check model output format
   - Review answer extraction regex patterns
   - Add custom patterns for specific problem types

3. **Out of memory:**
    - Enable 4-bit quantization (already enabled)
    - Reduce MAX_NEW_TOKENS
    - Clear cache between problems

4. **Timeout errors:**
    - Reduce MAX_ATTEMPTS
    - Decrease MAX_NEW_TOKENS
    - Optimize prompt length

## Credits

- **Model:** Qwen2.5-Math by Alibaba Cloud
- **Competition:** AI Mathematical Olympiad by XTX Markets
- **Framework:** Hugging Face Transformers, PyTorch

## License

This submission follows the competition rules and uses open-source models permitted by AIMO3 guidelines.

---

**Note:** This is Version 1 - a minimal viable implementation. Future versions will build upon this foundation with additional agents and improvements.
