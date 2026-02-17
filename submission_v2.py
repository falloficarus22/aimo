import ast
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import kaggle_evaluation.aimo_3_inference_server
import pandas as pd
import polars as pl
import torch
from transformers import AutoModelForCasualLM, AutoTokenizer

print("Imports successful")


@dataclass
class Config:
    """Configuration for consensus voting system."""

    MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
    MAX_NEW_TOKENS = 4096
    TEMPERATURE = 0.1
    TOP_P = 0.95

    USE_FP16 = True
    DEVICE_MAP = "auto"

    NUM_SOLUTIONS = 3
    TEMPERATURES = [0.1, 0.3, 0.5]

    MIN_ANSWER = 0
    MAX_ANSWER = 99999
    TIMEOUT_PER_PROBLEM = 300


print(f"Configuration:")
print(f"Model: {Config.MODEL_NAME}")
print(f"Consensus solutions: {Config.NUM_SOLUTIONS}")
print(f"Temperature: {Config.TEMPERATURES}")

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

print("Reference problems loaded")


class MultiGPUModelManager:
    """
    Manages model loading across 2x T4 GPUs.
    Uses automatic device mapping for optimal memory distribution."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.num_gpus = torch.cuda.device_count()

    def load(self):
        """Load model across available GPUs."""
        if self.model is not None:
            return self

        print(f"Loading {Config.MODEL_NAME}...")
        print(f"Availabel GPUs: {self.num_gpus}")

        max_memory = {i: "15.GiB" for i in range(self.num_gpus)}

        self.tokenizer = AutoTokenizer.from_pretrained(
            Config.MODEL_NAME, trust_remote_code=True
        )

        self.model = AutoModelForCasualLM.from_pretrained(
            Config.MODEL_NAME,
            torch_dtype=torch.float16 if Config.USE_FP16 else torch.float32,
            device_map=Config.DEVICE_MAP,
            max_memory=max_memory,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        if hasattr(self.model, "hf_device_map"):
            print(f"Model sharded across device: {self.device}")
        else:
            self.device = next(self.model.parameters()).device()
            print(f"Model loaded on device: {self.device}")

        print(f"Model ready")
        return self
