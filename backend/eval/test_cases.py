"""
Test cases for the RAG evaluation suite.

Pre-populated with comprehensive test cases for the indexed papers in storage:
- 019ed9f9-52b8-40f6-ba89-5146192f001a: Euclid MCP.pdf
- 50b44e24-90d5-46a4-8bea-ad896c4a212d: NIPS-2017-attention-is-all-you-need-Paper.pdf
"""

PAPER_EUCLID = "019ed9f9-52b8-40f6-ba89-5146192f001a"
PAPER_TRANSFORMER = "50b44e24-90d5-46a4-8bea-ad896c4a212d"

# 1. QA Test Cases (22 cases)
QA_CASES = [
    {
        "id": "qa_001",
        "question": "What is the Transformer architecture based on and what does it dispense with?",
        "paper_ids": [PAPER_TRANSFORMER],
        "expected_intent": "qa",
        "ground_truth": (
            "The Transformer architecture is based entirely on attention mechanisms, "
            "dispensing with recurrence and convolutions entirely."
        ),
        "relevant_chunk_keywords": ["attention", "recurrence", "convolution"],
    },
    {
        "id": "qa_002",
        "question": "What optimizer and learning rate formula or schedule were used for training the Transformer?",
        "paper_ids": [PAPER_TRANSFORMER],
        "expected_intent": "qa",
        "ground_truth": (
            "The Adam optimizer was used with beta1=0.9, beta2=0.98, and epsilon=10^-9. "
            "The learning rate was varied over training according to a formula with warmup_steps=4000."
        ),
        "relevant_chunk_keywords": ["Adam", "beta1", "learning rate", "warmup"],
    },
    {
        "id": "qa_003",
        "question": "What is Euclid-MCP and what engine does it use for logical reasoning?",
        "paper_ids": [PAPER_EUCLID],
        "expected_intent": "qa",
        "ground_truth": (
            "Euclid-MCP is an open-source MCP server that provides deterministic logical reasoning "
            "via SWI-Prolog and introduces Euclid-IR for Horn-clause logic."
        ),
        "relevant_chunk_keywords": ["Euclid-MCP", "SWI-Prolog", "Prolog", "Horn-clause"],
    },
    {
        "id": "qa_004",
        "question": "How does Scaled Dot-Product Attention compute its outputs mathematically?",
        "paper_ids": [PAPER_TRANSFORMER],
        "expected_intent": "qa",
        "ground_truth": (
            "Scaled Dot-Product Attention computes the matrix of outputs as softmax((Q * K^T) / sqrt(d_k)) * V, "
            "where Q, K, and V are queries, keys, and values matrices."
        ),
        "relevant_chunk_keywords": ["softmax", "sqrt", "dot-product", "keys", "values"],
    },
    {
        "id": "qa_005",
        "question": "What is the purpose of scaling by the square root of d_k in attention?",
        "paper_ids": [PAPER_TRANSFORMER],
        "expected_intent": "qa",
        "ground_truth": (
            "Scaling by 1/sqrt(d_k) prevents the dot products from growing large in magnitude for large d_k, "
            "which would push the softmax function into regions with extremely small gradients."
        ),
        "relevant_chunk_keywords": ["scaling", "dot products", "gradients", "softmax"],
    },
    {
        "id": "qa_006",
        "question": "How is Multi-Head Attention structured and why is it beneficial compared to single attention?",
        "paper_ids": [PAPER_TRANSFORMER],
        "expected_intent": "qa",
        "ground_truth": (
            "Multi-Head Attention linearly projects queries, keys, and values h times with different learned projections, "
            "allowing the model to jointly attend to information from different representation subspaces at different positions."
        ),
        "relevant_chunk_keywords": ["Multi-Head Attention", "projections", "subspaces", "attend"],
    },
    {
        "id": "qa_007",
        "question": "What positional encoding formulation is used in the Transformer and why?",
        "paper_ids": [PAPER_TRANSFORMER],
        "expected_intent": "qa",
        "ground_truth": (
            "Sine and cosine functions of different frequencies are used for positional encoding so the model can inject "
            "token sequence order without recurrence, allowing it to easily learn to attend by relative positions."
        ),
        "relevant_chunk_keywords": ["positional encoding", "sine", "cosine", "frequencies"],
    },
    {
        "id": "qa_008",
        "question": "What is the dimensionality d_model and number of layers N in the base Transformer model?",
        "paper_ids": [PAPER_TRANSFORMER],
        "expected_intent": "qa",
        "ground_truth": (
            "The base Transformer model has N=6 identical layers for both encoder and decoder, "
            "with an embedding and hidden dimension of d_model=512."
        ),
        "relevant_chunk_keywords": ["d_model", "512", "layers", "N=6"],
    },
    {
        "id": "qa_009",
        "question": "What BLEU scores did the Transformer achieve on the WMT 2014 English-to-German and English-to-French tasks?",
        "paper_ids": [PAPER_TRANSFORMER],
        "expected_intent": "qa",
        "ground_truth": (
            "The Transformer (big) model achieved 28.4 BLEU on English-to-German and 41.8 BLEU on English-to-French, "
            "establishing new state-of-the-art results."
        ),
        "relevant_chunk_keywords": ["28.4", "41.8", "BLEU", "English-to-German"],
    },
    {
        "id": "qa_010",
        "question": "Why does the Transformer use masked multi-head attention in its decoder?",
        "paper_ids": [PAPER_TRANSFORMER],
        "expected_intent": "qa",
        "ground_truth": (
            "Masking prevents positions from attending to subsequent positions, ensuring that predictions for position i "
            "can depend only on known outputs at positions less than i to preserve autoregressive generation."
        ),
        "relevant_chunk_keywords": ["masking", "subsequent", "prevent", "auto-regressive"],
    },
    {
        "id": "qa_011",
        "question": "What regularization techniques were used during training the Transformer?",
        "paper_ids": [PAPER_TRANSFORMER],
        "expected_intent": "qa",
        "ground_truth": (
            "Residual dropout (rate P_drop = 0.1) applied to sub-layer outputs and embedding sums, "
            "and label smoothing with epsilon_ls = 0.1 were employed."
        ),
        "relevant_chunk_keywords": ["dropout", "label smoothing", "regularization"],
    },
    {
        "id": "qa_012",
        "question": "What is Euclid-IR and how is it used in Euclid-MCP?",
        "paper_ids": [PAPER_EUCLID],
        "expected_intent": "qa",
        "ground_truth": (
            "Euclid-IR is an engine-agnostic intermediate representation for Horn-clause logic that is human-readable, "
            "easy for LLMs to generate, and straightforward to compile into Prolog or alternative backends."
        ),
        "relevant_chunk_keywords": ["Euclid-IR", "Horn-clause", "intermediate representation", "Prolog"],
    },
    {
        "id": "qa_013",
        "question": "What is the translate-run-inspect-repair loop in Euclid-MCP?",
        "paper_ids": [PAPER_EUCLID],
        "expected_intent": "qa",
        "ground_truth": (
            "It is a tool interaction loop allowing LLM clients to translate natural language rules into Euclid-IR, "
            "run inference on the Prolog backend, inspect proof traces and logs, and repair logical errors iteratively."
        ),
        "relevant_chunk_keywords": ["translate", "run", "inspect", "repair", "loop"],
    },
    {
        "id": "qa_014",
        "question": "In what scenario do LLMs hallucinate compared to Euclid-MCP in the compliance evaluation?",
        "paper_ids": [PAPER_EUCLID],
        "expected_intent": "qa",
        "ground_truth": (
            "While LLMs alone are sufficient on small knowledge bases, they hallucinate systematically on larger, "
            "multi-step compliance problems, whereas Euclid-MCP delivers exact, deterministic answers."
        ),
        "relevant_chunk_keywords": ["hallucinate", "compliance", "knowledge bases", "systematic"],
    },
    {
        "id": "qa_015",
        "question": "Why does the Euclid-MCP paper argue semantic RAG is fundamentally unsuited for rule enforcement?",
        "paper_ids": [PAPER_EUCLID],
        "expected_intent": "qa",
        "ground_truth": (
            "Semantic RAG retrieves chunks based on semantic similarity rather than formal logical validity, "
            "making it incapable of guaranteeing strict deterministic rule enforcement and invariant checking."
        ),
        "relevant_chunk_keywords": ["semantic RAG", "rule enforcement", "unsuited", "logical"],
    },
    {
        "id": "qa_016",
        "question": "What tools or interface does the Euclid-MCP server expose to LLM clients?",
        "paper_ids": [PAPER_EUCLID],
        "expected_intent": "qa",
        "ground_truth": (
            "Euclid-MCP exposes a compact tool interface supporting translation, execution of logic queries, "
            "inspection of derivations, and error repair over the Model Context Protocol."
        ),
        "relevant_chunk_keywords": ["tools", "interface", "server", "Model Context Protocol"],
    },
    {
        "id": "qa_017",
        "question": "How does Euclid-MCP provide deterministic explainability or proof traces?",
        "paper_ids": [PAPER_EUCLID],
        "expected_intent": "qa",
        "ground_truth": (
            "By delegating inference to SWI-Prolog, Euclid-MCP produces formal derivation logs and proof trees "
            "that show step-by-step clause resolution for every deduction."
        ),
        "relevant_chunk_keywords": ["proof trace", "derivation", "SWI-Prolog", "resolution"],
    },
    {
        "id": "qa_018",
        "question": "What hardware and training time were required for the base Transformer model?",
        "paper_ids": [PAPER_TRANSFORMER],
        "expected_intent": "qa",
        "ground_truth": (
            "The base Transformer model was trained on 8 NVIDIA P100 GPUs for 100,000 steps (about 12 hours)."
        ),
        "relevant_chunk_keywords": ["8 NVIDIA", "P100", "12 hours", "100,000 steps"],
    },
    {
        "id": "qa_019",
        "question": "What tokenization approaches were used for English-German and English-French in Attention Is All You Need?",
        "paper_ids": [PAPER_TRANSFORMER],
        "expected_intent": "qa",
        "ground_truth": (
            "English-German used byte-pair encoding (BPE) with a shared vocabulary of about 37,000 tokens; "
            "English-French used a 32,000 word-piece vocabulary."
        ),
        "relevant_chunk_keywords": ["byte-pair encoding", "BPE", "word-piece", "vocabulary"],
    },
    {
        "id": "qa_020",
        "question": "What IT security and compliance use case is evaluated in the Euclid-MCP paper?",
        "paper_ids": [PAPER_EUCLID],
        "expected_intent": "qa",
        "ground_truth": (
            "A realistic IT security and access control compliance scenario where organizational policies "
            "and user privileges are verified deterministically against safety rules."
        ),
        "relevant_chunk_keywords": ["compliance", "security", "access control", "rules"],
    },
    {
        "id": "qa_021",
        "question": "How does the Transformer feed-forward network compute its activations?",
        "paper_ids": [PAPER_TRANSFORMER],
        "expected_intent": "qa",
        "ground_truth": (
            "The feed-forward network consists of two linear transformations with a ReLU activation in between: "
            "FFN(x) = max(0, xW1 + b1)W2 + b2, with inner dimension d_ff = 2048."
        ),
        "relevant_chunk_keywords": ["feed-forward", "network", "layer", "linear"],
    },
    {
        "id": "qa_022",
        "question": "What is the main conclusion of the Euclid-MCP paper regarding hybrid cognitive architectures?",
        "paper_ids": [PAPER_EUCLID],
        "expected_intent": "qa",
        "ground_truth": (
            "Euclid-MCP shows that coupling neural LLMs with deterministic symbolic solvers creates a robust "
            "substrate for reliable reasoning in safety-critical and compliance-sensitive agentic systems."
        ),
        "relevant_chunk_keywords": ["hybrid", "neuro-symbolic", "substrate", "reasoning"],
    },
]

# 2. Comparison Test Cases (20 cases)
COMPARISON_CASES = [
    {
        "id": "cmp_001",
        "question": "Compare the core methodologies and goals of Euclid MCP and the Attention Is All You Need paper.",
        "paper_ids": [PAPER_EUCLID, PAPER_TRANSFORMER],
        "expected_intent": "comparison",
        "ground_truth": (
            "Attention Is All You Need proposes the Transformer model relying solely on attention mechanisms "
            "for sequence modeling and translation, eliminating recurrence. In contrast, Euclid-MCP addresses "
            "the limitations of neural LLMs in multi-step logical reasoning by coupling them with a symbolic "
            "SWI-Prolog engine via the Model Context Protocol."
        ),
    },
    {
        "id": "cmp_002",
        "question": "Contrast how each paper approaches the problem of reliable reasoning in artificial intelligence.",
        "paper_ids": [PAPER_EUCLID, PAPER_TRANSFORMER],
        "expected_intent": "comparison",
        "ground_truth": (
            "The Transformer paper aims for statistical pattern recognition and representation learning via massive "
            "data and self-attention, while Euclid-MCP argues statistical neural models cannot guarantee multi-step logical "
            "correctness alone and therefore pairs neural generation with deterministic symbolic deduction."
        ),
    },
    {
        "id": "cmp_003",
        "question": "Compare the evaluation benchmarks and metrics used in Attention Is All You Need versus Euclid MCP.",
        "paper_ids": [PAPER_EUCLID, PAPER_TRANSFORMER],
        "expected_intent": "comparison",
        "ground_truth": (
            "Attention Is All You Need evaluates on machine translation tasks (WMT 2014) measuring BLEU scores and parsing "
            "F1 scores. Euclid-MCP evaluates on an IT security and compliance rule enforcement domain measuring logical correctness, "
            "hallucination rates, latency, and proof trace completeness."
        ),
    },
    {
        "id": "cmp_004",
        "question": "Compare how each paper handles multi-step sequential dependencies.",
        "paper_ids": [PAPER_EUCLID, PAPER_TRANSFORMER],
        "expected_intent": "comparison",
        "ground_truth": (
            "The Transformer handles sequential dependencies by connecting all pairs of positions in constant time via self-attention, "
            "whereas Euclid-MCP handles logical dependencies via recursive Horn-clause resolution and Prolog unification."
        ),
    },
    {
        "id": "cmp_005",
        "question": "Contrast the computational paradigms of the Transformer with Euclid-MCP.",
        "paper_ids": [PAPER_EUCLID, PAPER_TRANSFORMER],
        "expected_intent": "comparison",
        "ground_truth": (
            "The Transformer is purely connectionist and differentiable, computing floating-point matrix multiplications on GPUs, "
            "while Euclid-MCP is neuro-symbolic, bridging neural generation with discrete, exact symbolic deduction."
        ),
    },
    {
        "id": "cmp_006",
        "question": "Compare the scalability characteristics described in both papers.",
        "paper_ids": [PAPER_EUCLID, PAPER_TRANSFORMER],
        "expected_intent": "comparison",
        "ground_truth": (
            "The Transformer scales efficiently with sequence parallelization and training compute, while Euclid-MCP studies scaling "
            "in knowledge base size, showing that LLMs fail systematically as rule bases grow while Prolog maintains exactness."
        ),
    },
    {
        "id": "cmp_007",
        "question": "Contrast the error modes of standard Transformers with Euclid-MCP's verification framework.",
        "paper_ids": [PAPER_EUCLID, PAPER_TRANSFORMER],
        "expected_intent": "comparison",
        "ground_truth": (
            "Transformers suffer from hallucinations and attention dispersion on complex logic, whereas Euclid-MCP provides a "
            "closed-world verifiable engine where deductions must strictly follow from declared axioms."
        ),
    },
    {
        "id": "cmp_008",
        "question": "Compare how explainability and interpretability are treated in Attention Is All You Need versus Euclid MCP.",
        "paper_ids": [PAPER_EUCLID, PAPER_TRANSFORMER],
        "expected_intent": "comparison",
        "ground_truth": (
            "Attention Is All You Need visualizes attention heads as soft weights between tokens, whereas Euclid-MCP produces "
            "rigorous, human-verifiable derivation proofs and deduction chains."
        ),
    },
    {
        "id": "cmp_009",
        "question": "Contrast the software infrastructure stack required to run a Transformer versus Euclid-MCP.",
        "paper_ids": [PAPER_EUCLID, PAPER_TRANSFORMER],
        "expected_intent": "comparison",
        "ground_truth": (
            "Transformers require deep learning frameworks (TensorFlow/PyTorch) and high-memory GPUs for tensor operations, "
            "while Euclid-MCP runs on CPU as a lightweight Model Context Protocol server backed by SWI-Prolog."
        ),
    },
    {
        "id": "cmp_010",
        "question": "Compare the training and data requirements of the Transformer against Euclid-MCP's rule engine.",
        "paper_ids": [PAPER_EUCLID, PAPER_TRANSFORMER],
        "expected_intent": "comparison",
        "ground_truth": (
            "The Transformer requires millions of sentence pairs and days of GPU training, while Euclid-MCP relies on zero-shot "
            "rule compilation into Horn clauses without retraining model weights."
        ),
    },
    {
        "id": "cmp_011",
        "question": "Contrast the role of natural language processing in both systems.",
        "paper_ids": [PAPER_EUCLID, PAPER_TRANSFORMER],
        "expected_intent": "comparison",
        "ground_truth": (
            "In Attention Is All You Need, NLP is the primary task (sequence-to-sequence machine translation). In Euclid-MCP, "
            "NLP serves as the translation front-end converting human language policies into structured logic."
        ),
    },
    {
        "id": "cmp_012",
        "question": "Compare how both papers validate their claims using empirical experiments.",
        "paper_ids": [PAPER_EUCLID, PAPER_TRANSFORMER],
        "expected_intent": "comparison",
        "ground_truth": (
            "The Transformer paper uses standard machine translation test sets and English parsing ablations; Euclid-MCP constructs "
            "realistic synthetic and organizational IT compliance test benches measuring accuracy and latency under varied rule sizes."
        ),
    },
    {
        "id": "cmp_013",
        "question": "Contrast how each system handles rule violations and edge cases.",
        "paper_ids": [PAPER_EUCLID, PAPER_TRANSFORMER],
        "expected_intent": "comparison",
        "ground_truth": (
            "Transformers may generate plausible-sounding but invalid text when encountering edge cases, whereas Euclid-MCP detects "
            "contradictions and unmet predicates deterministically during Prolog resolution."
        ),
    },
    {
        "id": "cmp_014",
        "question": "Compare the latency and computational cost profiles of both systems.",
        "paper_ids": [PAPER_EUCLID, PAPER_TRANSFORMER],
        "expected_intent": "comparison",
        "ground_truth": (
            "Transformer autoregressive decoding scales with generated token length and GPU FLOPs, while Euclid-MCP delegates reasoning "
            "to fast compiled Prolog queries, yielding lower latency and concise symbolic responses."
        ),
    },
    {
        "id": "cmp_015",
        "question": "Contrast the open-source software artifacts released by each paper.",
        "paper_ids": [PAPER_EUCLID, PAPER_TRANSFORMER],
        "expected_intent": "comparison",
        "ground_truth": (
            "The Transformer authors released Tensor2Tensor and model checkpoints; Euclid-MCP provides an open-source MCP server "
            "package supporting the Anthropic Model Context Protocol specification."
        ),
    },
    {
        "id": "cmp_016",
        "question": "Compare the reliance on attention matrices versus Horn-clause unification.",
        "paper_ids": [PAPER_EUCLID, PAPER_TRANSFORMER],
        "expected_intent": "comparison",
        "ground_truth": (
            "Attention matrices model soft probabilistic associations across tokens, whereas Horn-clause unification performs "
            "strict boolean variable substitutions based on first-order predicate logic."
        ),
    },
    {
        "id": "cmp_017",
        "question": "Contrast how context length limitations impact Transformer self-attention versus Euclid-MCP knowledge bases.",
        "paper_ids": [PAPER_EUCLID, PAPER_TRANSFORMER],
        "expected_intent": "comparison",
        "ground_truth": (
            "Transformers suffer quadratic computational complexity O(n^2) with context length, while Euclid-MCP externalizes knowledge "
            "into Prolog clauses, avoiding in-context memory saturation."
        ),
    },
    {
        "id": "cmp_018",
        "question": "Compare the primary target application domains envisioned by each paper.",
        "paper_ids": [PAPER_EUCLID, PAPER_TRANSFORMER],
        "expected_intent": "comparison",
        "ground_truth": (
            "The Transformer targets sequence transduction tasks like language translation and parsing, while Euclid-MCP targets "
            "safety-critical enterprise compliance, legal verification, and deterministic tool-augmented AI agents."
        ),
    },
    {
        "id": "cmp_019",
        "question": "Contrast the future work proposed in Attention Is All You Need versus Euclid-MCP.",
        "paper_ids": [PAPER_EUCLID, PAPER_TRANSFORMER],
        "expected_intent": "comparison",
        "ground_truth": (
            "Attention Is All You Need outlines extending attention to images, audio, and restricted local attention. Euclid-MCP proposes "
            "supporting richer logic formalisms, alternative constraint solvers, and automated repair mechanisms."
        ),
    },
    {
        "id": "cmp_020",
        "question": "Compare how both papers contribute to the evolution towards autonomous AI agents.",
        "paper_ids": [PAPER_EUCLID, PAPER_TRANSFORMER],
        "expected_intent": "comparison",
        "ground_truth": (
            "The Transformer provides the foundational neural architecture powering modern foundation LLMs, while Euclid-MCP provides "
            "the external symbolic reasoning interface essential for making those LLMs safe, deterministic, and verifiable."
        ),
    },
]

# 3. Summary Test Cases (20 cases)
SUMMARY_CASES = [
    {
        "id": "sum_001",
        "question": "Summarize the Attention Is All You Need paper.",
        "paper_ids": [PAPER_TRANSFORMER],
        "expected_intent": "summary",
        "expected_topics": ["Transformer", "Self-Attention", "Multi-Head Attention", "Translation", "Encoder-Decoder"],
    },
    {
        "id": "sum_002",
        "question": "Summarize the Euclid MCP paper.",
        "paper_ids": [PAPER_EUCLID],
        "expected_intent": "summary",
        "expected_topics": ["Euclid-MCP", "SWI-Prolog", "Euclid-IR", "Horn-clause", "Logical Reasoning"],
    },
    {
        "id": "sum_003",
        "question": "Provide an executive summary of the Transformer architecture and its advantages.",
        "paper_ids": [PAPER_TRANSFORMER],
        "expected_intent": "summary",
        "expected_topics": ["Transformer", "Parallelization", "Self-Attention", "Recurrence Elimination", "Efficiency"],
    },
    {
        "id": "sum_004",
        "question": "Summarize the motivation and key features of Euclid-MCP.",
        "paper_ids": [PAPER_EUCLID],
        "expected_intent": "summary",
        "expected_topics": ["Neuro-Symbolic", "Hallucination", "SWI-Prolog", "Deterministic Inference", "Tool Interface"],
    },
    {
        "id": "sum_005",
        "question": "Summarize the experimental results and translation performance of the Transformer model.",
        "paper_ids": [PAPER_TRANSFORMER],
        "expected_intent": "summary",
        "expected_topics": ["WMT 2014", "BLEU", "English-to-German", "English-to-French", "Training Cost"],
    },
    {
        "id": "sum_006",
        "question": "Summarize the IT security and compliance case study in the Euclid-MCP paper.",
        "paper_ids": [PAPER_EUCLID],
        "expected_intent": "summary",
        "expected_topics": ["Access Control", "Security Rules", "Evaluation Benchmarks", "Accuracy", "Scalability"],
    },
    {
        "id": "sum_007",
        "question": "Summarize the Multi-Head Attention mechanism described in Attention Is All You Need.",
        "paper_ids": [PAPER_TRANSFORMER],
        "expected_intent": "summary",
        "expected_topics": ["Multi-Head Attention", "Scaled Dot-Product", "Linear Projections", "Subspaces", "d_k"],
    },
    {
        "id": "sum_008",
        "question": "Summarize the design and role of Euclid-IR in the Euclid-MCP architecture.",
        "paper_ids": [PAPER_EUCLID],
        "expected_intent": "summary",
        "expected_topics": ["Euclid-IR", "Horn-Clause", "Intermediate Representation", "Compiler", "Prolog Backend"],
    },
    {
        "id": "sum_009",
        "question": "Summarize the training hyperparameters and optimizer settings in Attention Is All You Need.",
        "paper_ids": [PAPER_TRANSFORMER],
        "expected_intent": "summary",
        "expected_topics": ["Adam Optimizer", "Learning Rate Schedule", "Warmup", "Dropout", "Label Smoothing"],
    },
    {
        "id": "sum_010",
        "question": "Summarize the translate-run-inspect-repair workflow of Euclid-MCP.",
        "paper_ids": [PAPER_EUCLID],
        "expected_intent": "summary",
        "expected_topics": ["Translate", "Run", "Inspect", "Repair", "Loop", "Proof Trace"],
    },
    {
        "id": "sum_011",
        "question": "Summarize how positional encoding works in the Transformer architecture.",
        "paper_ids": [PAPER_TRANSFORMER],
        "expected_intent": "summary",
        "expected_topics": ["Positional Encoding", "Sine and Cosine", "Frequencies", "Relative Position", "Wavelengths"],
    },
    {
        "id": "sum_012",
        "question": "Summarize why the Euclid-MCP paper argues semantic RAG is insufficient for compliance rules.",
        "paper_ids": [PAPER_EUCLID],
        "expected_intent": "summary",
        "expected_topics": ["Semantic RAG", "Embedding Similarity", "Rule Enforcement", "Non-Determinism", "Hallucination"],
    },
    {
        "id": "sum_013",
        "question": "Summarize the encoder and decoder stack construction in Attention Is All You Need.",
        "paper_ids": [PAPER_TRANSFORMER],
        "expected_intent": "summary",
        "expected_topics": ["Encoder Stack", "Decoder Stack", "Layer Normalization", "Residual Connections", "Masking"],
    },
    {
        "id": "sum_014",
        "question": "Summarize the latency and output compactness results of Euclid-MCP versus standalone LLMs.",
        "paper_ids": [PAPER_EUCLID],
        "expected_intent": "summary",
        "expected_topics": ["Latency", "Compact Output", "Token Overhead", "Evaluation", "Prolog Speed"],
    },
    {
        "id": "sum_015",
        "question": "Summarize the model variations and ablation experiments presented in the Transformer paper.",
        "paper_ids": [PAPER_TRANSFORMER],
        "expected_intent": "summary",
        "expected_topics": ["Ablation", "Number of Heads", "Key Size", "Feed-Forward Dimensions", "Table 3"],
    },
    {
        "id": "sum_016",
        "question": "Summarize the derivation inspection and proof trace capabilities of Euclid-MCP.",
        "paper_ids": [PAPER_EUCLID],
        "expected_intent": "summary",
        "expected_topics": ["Proof Traces", "Derivation Trees", "Auditability", "Debugging", "Explainability"],
    },
    {
        "id": "sum_017",
        "question": "Summarize the English constituency parsing experiments in the Transformer paper.",
        "paper_ids": [PAPER_TRANSFORMER],
        "expected_intent": "summary",
        "expected_topics": ["Constituency Parsing", "Penn Treebank", "WSJ", "Generalization", "F1 Score"],
    },
    {
        "id": "sum_018",
        "question": "Summarize the neuro-symbolic philosophy presented in the Euclid-MCP paper.",
        "paper_ids": [PAPER_EUCLID],
        "expected_intent": "summary",
        "expected_topics": ["Neuro-Symbolic", "System 1 vs System 2", "Deterministic Reasoning", "Shared Substrate"],
    },
    {
        "id": "sum_019",
        "question": "Summarize the main contributions and future directions of the Transformer paper.",
        "paper_ids": [PAPER_TRANSFORMER],
        "expected_intent": "summary",
        "expected_topics": ["Contributions", "Attention Mechanisms", "Future Work", "Non-Recurrent", "Translation"],
    },
    {
        "id": "sum_020",
        "question": "Summarize the conclusions and implications of Euclid-MCP for autonomous agents.",
        "paper_ids": [PAPER_EUCLID],
        "expected_intent": "summary",
        "expected_topics": ["Agentic AI", "MCP Protocol", "Enterprise Compliance", "Conclusions", "SWI-Prolog"],
    },
]

# 4. Planner-only Test Cases (25 cases)
PLANNER_CASES = [
    {
        "id": "plan_001",
        "question": "How does multi-head attention work in the Transformer paper?",
        "paper_ids": [PAPER_TRANSFORMER],
        "expected_intent": "qa",
        "expected_target_paper_ids": [PAPER_TRANSFORMER],
    },
    {
        "id": "plan_002",
        "question": "Summarize the Euclid MCP paper.",
        "paper_ids": [PAPER_EUCLID],
        "expected_intent": "summary",
        "expected_target_paper_ids": [PAPER_EUCLID],
    },
    {
        "id": "plan_003",
        "question": "Compare Euclid MCP and Attention Is All You Need.",
        "paper_ids": [PAPER_EUCLID, PAPER_TRANSFORMER],
        "expected_intent": "comparison",
        "expected_target_paper_ids": [PAPER_EUCLID, PAPER_TRANSFORMER],
    },
    {
        "id": "plan_004",
        "question": "What is the capital of France?",
        "paper_ids": [PAPER_TRANSFORMER],
        "expected_intent": "unanswerable",
        "expected_target_paper_ids": [],
    },
    {
        "id": "plan_005",
        "question": "What is the formula for scaled dot-product attention in Attention Is All You Need?",
        "paper_ids": [PAPER_TRANSFORMER],
        "expected_intent": "qa",
        "expected_target_paper_ids": [PAPER_TRANSFORMER],
    },
    {
        "id": "plan_006",
        "question": "What is Euclid-IR and how does it interface with SWI-Prolog?",
        "paper_ids": [PAPER_EUCLID],
        "expected_intent": "qa",
        "expected_target_paper_ids": [PAPER_EUCLID],
    },
    {
        "id": "plan_007",
        "question": "Give me a comprehensive summary of Attention Is All You Need.",
        "paper_ids": [PAPER_TRANSFORMER],
        "expected_intent": "summary",
        "expected_target_paper_ids": [PAPER_TRANSFORMER],
    },
    {
        "id": "plan_008",
        "question": "Provide a detailed summary of the Euclid MCP research paper.",
        "paper_ids": [PAPER_EUCLID],
        "expected_intent": "summary",
        "expected_target_paper_ids": [PAPER_EUCLID],
    },
    {
        "id": "plan_009",
        "question": "How do the training paradigms differ between the Transformer and Euclid MCP?",
        "paper_ids": [PAPER_EUCLID, PAPER_TRANSFORMER],
        "expected_intent": "comparison",
        "expected_target_paper_ids": [PAPER_EUCLID, PAPER_TRANSFORMER],
    },
    {
        "id": "plan_010",
        "question": "Who won the 2022 FIFA World Cup in Qatar?",
        "paper_ids": [PAPER_TRANSFORMER],
        "expected_intent": "unanswerable",
        "expected_target_paper_ids": [],
    },
    {
        "id": "plan_011",
        "question": "What optimizer settings were used during training in Attention Is All You Need?",
        "paper_ids": [PAPER_TRANSFORMER],
        "expected_intent": "qa",
        "expected_target_paper_ids": [PAPER_TRANSFORMER],
    },
    {
        "id": "plan_012",
        "question": "How does the translate-run-inspect-repair loop operate in Euclid MCP?",
        "paper_ids": [PAPER_EUCLID],
        "expected_intent": "qa",
        "expected_target_paper_ids": [PAPER_EUCLID],
    },
    {
        "id": "plan_013",
        "question": "Compare the reasoning reliability of standard neural models versus Euclid-MCP and Transformers.",
        "paper_ids": [PAPER_EUCLID, PAPER_TRANSFORMER],
        "expected_intent": "comparison",
        "expected_target_paper_ids": [PAPER_EUCLID, PAPER_TRANSFORMER],
    },
    {
        "id": "plan_014",
        "question": "Summarize the experimental machine translation results from the Transformer paper.",
        "paper_ids": [PAPER_TRANSFORMER],
        "expected_intent": "summary",
        "expected_target_paper_ids": [PAPER_TRANSFORMER],
    },
    {
        "id": "plan_015",
        "question": "Summarize the IT security compliance case study in the Euclid MCP paper.",
        "paper_ids": [PAPER_EUCLID],
        "expected_intent": "summary",
        "expected_target_paper_ids": [PAPER_EUCLID],
    },
    {
        "id": "plan_016",
        "question": "What is the recipe for baking chocolate chip cookies?",
        "paper_ids": [PAPER_EUCLID],
        "expected_intent": "unanswerable",
        "expected_target_paper_ids": [],
    },
    {
        "id": "plan_017",
        "question": "What are the inner dimensions of the feed-forward network in the Transformer?",
        "paper_ids": [PAPER_TRANSFORMER],
        "expected_intent": "qa",
        "expected_target_paper_ids": [PAPER_TRANSFORMER],
    },
    {
        "id": "plan_018",
        "question": "Why is semantic RAG unsuited for compliance rule enforcement according to Euclid MCP?",
        "paper_ids": [PAPER_EUCLID],
        "expected_intent": "qa",
        "expected_target_paper_ids": [PAPER_EUCLID],
    },
    {
        "id": "plan_019",
        "question": "Contrast the symbolic approach of Euclid MCP with the deep neural approach of the Transformer.",
        "paper_ids": [PAPER_EUCLID, PAPER_TRANSFORMER],
        "expected_intent": "comparison",
        "expected_target_paper_ids": [PAPER_EUCLID, PAPER_TRANSFORMER],
    },
    {
        "id": "plan_020",
        "question": "Summarize the architectural components of the Transformer encoder and decoder.",
        "paper_ids": [PAPER_TRANSFORMER],
        "expected_intent": "qa",
        "expected_target_paper_ids": [PAPER_TRANSFORMER],
    },
    {
        "id": "plan_021",
        "question": "Summarize the open-source tools and proof trace features in Euclid MCP.",
        "paper_ids": [PAPER_EUCLID],
        "expected_intent": "summary",
        "expected_target_paper_ids": [PAPER_EUCLID],
    },
    {
        "id": "plan_022",
        "question": "How do I calculate the gravitational orbit of Jupiter?",
        "paper_ids": [PAPER_TRANSFORMER],
        "expected_intent": "unanswerable",
        "expected_target_paper_ids": [],
    },
    {
        "id": "plan_023",
        "question": "What BLEU score did the Transformer model achieve on English-to-German?",
        "paper_ids": [PAPER_TRANSFORMER],
        "expected_intent": "qa",
        "expected_target_paper_ids": [PAPER_TRANSFORMER],
    },
    {
        "id": "plan_024",
        "question": "What Horn-clause representation is introduced by the Euclid-MCP server?",
        "paper_ids": [PAPER_EUCLID],
        "expected_intent": "qa",
        "expected_target_paper_ids": [PAPER_EUCLID],
    },
    {
        "id": "plan_025",
        "question": "Compare the error modes and hallucination behavior of Transformers and Euclid MCP.",
        "paper_ids": [PAPER_EUCLID, PAPER_TRANSFORMER],
        "expected_intent": "comparison",
        "expected_target_paper_ids": [PAPER_EUCLID, PAPER_TRANSFORMER],
    },
]

