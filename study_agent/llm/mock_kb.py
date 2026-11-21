"""Tiny hand-written knowledge base used ONLY by the offline mock LLM.

It lets demos and tests produce realistic content without an API key. With a real
LLM configured (LLM_MODEL=openai:..., anthropic:..., ...), none of this is used.
"""
from __future__ import annotations

import re

SUBJECT_TOPICS: dict[str, list[str]] = {
    "machine learning": [
        "linear regression", "gradient descent", "overfitting", "logistic regression",
        "model evaluation", "decision trees", "neural networks basics", "clustering",
    ],
    "python": [
        "variables and types", "control flow", "functions", "data structures",
        "recursion", "file handling", "object-oriented programming", "error handling",
    ],
    "algorithms": [
        "big-o notation", "binary search", "recursion", "sorting algorithms",
        "hash tables", "graphs and bfs/dfs", "dynamic programming", "greedy algorithms",
    ],
    "statistics": [
        "descriptive statistics", "probability basics", "distributions", "sampling",
        "hypothesis testing", "confidence intervals", "correlation", "regression",
    ],
}
SUBJECT_ALIASES = {
    "ml": "machine learning", "machine-learning": "machine learning",
    "data structures": "algorithms", "dsa": "algorithms", "algorithm": "algorithms",
    "stats": "statistics", "python programming": "python",
}

TOPICS: dict[str, dict] = {
    "gradient descent": {
        "summary": "Gradient descent finds a minimum of a loss function by repeatedly stepping in the direction of the negative gradient.",
        "explanation": (
            "A model's loss tells you how wrong it is for a given set of parameters. The gradient is the vector of "
            "partial derivatives of the loss with respect to each parameter; it points in the direction of steepest "
            "increase. Gradient descent updates every parameter as  θ ← θ − η·∇L(θ), where η (the learning rate) "
            "controls the step size. Repeating this moves the parameters downhill until the loss stops improving. "
            "Variants differ in how much data is used per step: batch (all data), stochastic (one example) and "
            "mini-batch (a small subset, the usual choice in practice)."
        ),
        "analogy": "Walking down a foggy hill: you can't see the valley, but you can feel which way the ground slopes and take a step that way.",
        "example": "Fitting y = w·x to points (1,2),(2,4): with w=0 the MSE gradient is negative, so w increases; after a few steps with η=0.05, w approaches 2.",
        "key_points": [
            "The update rule is θ ← θ − η·∇L(θ).",
            "The learning rate trades off speed against stability.",
            "Too large a learning rate overshoots and can diverge; too small converges slowly.",
            "Mini-batch gradient descent balances noise and compute cost.",
        ],
        "common_mistakes": [
            "Adding the gradient instead of subtracting it.",
            "Not scaling features, which makes the loss surface elongated and convergence slow.",
        ],
        "check_question": "What happens to training if the learning rate is far too large?",
        "questions": [
            {"type": "mcq", "question": "In gradient descent, parameters are updated in which direction?",
             "options": ["Along the gradient", "Opposite to the gradient", "Perpendicular to the gradient", "In a random direction"],
             "correct_option": "B", "reference_answer": "Opposite to the gradient, because the gradient points uphill (steepest increase).", "concept": "update direction"},
            {"type": "mcq", "question": "What is the most likely effect of a learning rate that is far too large?",
             "options": ["Training converges faster and more accurately", "The loss oscillates or diverges", "The model underfits less", "Nothing; the learning rate only affects memory use"],
             "correct_option": "B", "reference_answer": "Steps overshoot the minimum, so the loss oscillates or explodes.", "concept": "learning rate"},
            {"type": "mcq", "question": "Mini-batch gradient descent computes each update using…",
             "options": ["The entire training set", "A single example", "A small random subset of examples", "Only the validation set"],
             "correct_option": "C", "reference_answer": "A small random subset — a compromise between batch and stochastic GD.", "concept": "gd variants"},
            {"type": "short", "question": "Why does feature scaling usually speed up gradient descent?",
             "reference_answer": "Features on very different scales create an elongated loss surface, so gradient steps zig-zag; scaling makes the surface more round so steps point more directly at the minimum and a single learning rate works for all parameters.",
             "keywords": ["scale|scaling", "loss surface shape|elongated|contour|surface|shape", "direct steps|zig|oscillat|direct", "one learning rate|learning rate"], "concept": "feature scaling"},
            {"type": "mcq", "question": "Which quantity does gradient descent try to minimise?",
             "options": ["The number of parameters", "The loss (cost) function", "The learning rate", "The size of the dataset"],
             "correct_option": "B", "reference_answer": "The loss function measuring prediction error.", "concept": "objective"},
        ],
    },
    "overfitting": {
        "summary": "Overfitting is when a model learns noise and quirks of the training data and therefore generalises poorly to new data.",
        "explanation": (
            "A model that is too flexible for the amount of data can memorise the training set — including random "
            "noise — instead of learning the underlying pattern. The tell-tale sign is a gap: low training error but "
            "much higher validation/test error. Remedies include more data, simpler models, regularisation "
            "(L1/L2, dropout), early stopping and cross-validation to choose hyper-parameters honestly."
        ),
        "analogy": "A student who memorises past exam answers word-for-word aces the practice paper but fails when the questions are reworded.",
        "example": "A degree-15 polynomial fitted to 20 noisy points passes through almost every point (train MSE≈0) but swings wildly between them (high test MSE); a degree-2 fit generalises better.",
        "key_points": [
            "Signature: training error much lower than validation error.",
            "Caused by high model capacity relative to data size/noise.",
            "Regularisation penalises complexity to reduce variance.",
            "Always judge a model on data it was not trained on.",
        ],
        "common_mistakes": [
            "Tuning hyper-parameters on the test set, which leaks information.",
            "Assuming more epochs always help.",
        ],
        "check_question": "Train accuracy 99%, validation accuracy 70% — what is happening and name one fix?",
        "questions": [
            {"type": "mcq", "question": "Which pattern most strongly suggests overfitting?",
             "options": ["High training error, high validation error", "Low training error, much higher validation error", "Training and validation error both low and close", "Validation error lower than training error"],
             "correct_option": "B", "reference_answer": "A large gap with low training error indicates the model memorised training data.", "concept": "diagnosing overfitting"},
            {"type": "mcq", "question": "Which technique does NOT directly reduce overfitting?",
             "options": ["L2 regularisation", "Early stopping", "Adding more model parameters", "Collecting more training data"],
             "correct_option": "C", "reference_answer": "Adding parameters increases capacity, which usually makes overfitting worse.", "concept": "remedies"},
            {"type": "mcq", "question": "Why should hyper-parameters not be tuned on the test set?",
             "options": ["It is computationally expensive", "The test score becomes an optimistic, biased estimate", "Test sets are always too small", "It causes underfitting"],
             "correct_option": "B", "reference_answer": "The test set then influences model choice, so it no longer measures generalisation honestly.", "concept": "data leakage"},
            {"type": "short", "question": "Explain in your own words how regularisation helps against overfitting.",
             "reference_answer": "Regularisation adds a penalty on model complexity (e.g. large weights) to the loss, which discourages fitting noise and pushes the model towards simpler functions that generalise better.",
             "keywords": ["penalty|penalt|penalis|penaliz", "complexity|complex|weight", "less noise fitting|noise|simpler|simple", "generalisation|generali"], "concept": "regularisation"},
        ],
    },
    "linear regression": {
        "summary": "Linear regression models a numeric target as a weighted sum of input features plus a bias, fitted by minimising squared error.",
        "explanation": (
            "The model predicts ŷ = w₁x₁ + … + wₙxₙ + b. Training chooses weights that minimise the mean squared error "
            "between predictions and true values, either analytically (normal equation) or with gradient descent. "
            "Each weight is the expected change in the target for a one-unit change in that feature, holding others "
            "fixed. Key assumptions: a roughly linear relationship, independent errors and constant error variance."
        ),
        "analogy": "Drawing the straight line through a scatter plot that keeps the total squared vertical distance to the points as small as possible.",
        "example": "Predicting house price from size: price ≈ 3,000·m² + 50,000 means each extra square metre adds about 3,000 to the predicted price.",
        "key_points": [
            "Prediction is a weighted sum of features plus a bias.",
            "Fitted by minimising mean squared error.",
            "Coefficients are interpretable as marginal effects.",
            "Sensitive to outliers and multicollinearity.",
        ],
        "common_mistakes": ["Interpreting correlation as causation.", "Ignoring outliers that dominate squared error."],
        "check_question": "What does a coefficient of −2 on feature x mean?",
        "questions": [
            {"type": "mcq", "question": "Ordinary least squares linear regression chooses weights that minimise…",
             "options": ["The sum of absolute errors", "The sum of squared errors", "The number of misclassified points", "The largest single error"],
             "correct_option": "B", "reference_answer": "OLS minimises the (mean) squared error.", "concept": "loss function"},
            {"type": "mcq", "question": "A coefficient of 3 on feature x means…",
             "options": ["x explains 3% of the variance", "A one-unit increase in x raises the prediction by 3, holding other features fixed", "x is three times more important than other features", "The model has 3 parameters"],
             "correct_option": "B", "reference_answer": "Coefficients are marginal effects holding other features fixed.", "concept": "interpretation"},
            {"type": "mcq", "question": "Why are outliers especially harmful to OLS?",
             "options": ["They are removed automatically", "Squared error gives large residuals disproportionate weight", "They make the bias term zero", "They only affect classification models"],
             "correct_option": "B", "reference_answer": "Squaring amplifies large residuals so outliers pull the line towards them.", "concept": "outliers"},
            {"type": "short", "question": "Name one assumption of linear regression and what happens if it is violated.",
             "reference_answer": "Linearity: if the true relationship is non-linear the model is biased and residuals show patterns. (Also: independent errors, constant variance, little multicollinearity.)",
             "keywords": ["a named assumption|linear|independ|variance|homosced|multicollinear", "consequence of violation|bias|residual|pattern|unreliable|wrong"], "concept": "assumptions"},
        ],
    },
    "recursion": {
        "summary": "Recursion solves a problem by having a function call itself on smaller instances until it reaches a base case.",
        "explanation": (
            "Every recursive function needs (1) a base case that returns without recursing and (2) a recursive case "
            "that moves the input closer to the base case. Each call gets its own stack frame; when the base case "
            "returns, results flow back up the call stack. Missing or unreachable base cases cause infinite recursion "
            "and a stack overflow."
        ),
        "analogy": "Russian nesting dolls: open one to find a smaller one, until you reach the solid doll in the middle.",
        "example": "factorial(n) = 1 if n == 0 else n * factorial(n-1); factorial(3) → 3·2·1·1 = 6.",
        "key_points": ["Always define a base case.", "Each call must shrink the problem.", "Calls use stack memory.", "Many recursive solutions can be rewritten iteratively."],
        "common_mistakes": ["Forgetting the base case.", "Recursive step that doesn't shrink the input."],
        "check_question": "What happens if a recursive function's base case is never reached?",
        "questions": [
            {"type": "mcq", "question": "What prevents a recursive function from running forever?",
             "options": ["A loop counter", "A base case", "A global variable", "Tail call syntax"],
             "correct_option": "B", "reference_answer": "The base case stops the recursion.", "concept": "base case"},
            {"type": "mcq", "question": "What error typically results from infinite recursion in Python?",
             "options": ["SyntaxError", "RecursionError (stack overflow)", "KeyError", "MemoryLeakWarning"],
             "correct_option": "B", "reference_answer": "Python raises RecursionError when the maximum recursion depth is exceeded.", "concept": "call stack"},
            {"type": "short", "question": "Why does every recursive call need to make the problem smaller?",
             "reference_answer": "So that the sequence of calls eventually reaches the base case; otherwise recursion never terminates and overflows the stack.",
             "keywords": ["base case", "termination|terminat|stop|end", "stack overflow|stack|infinite"], "concept": "progress toward base case"},
        ],
    },
}

TOPICS["binary search"] = {
    "summary": "Binary search finds a target in a sorted array by repeatedly halving the search interval.",
    "explanation": (
        "Keep two indices, low and high, bounding where the target could be. Look at the middle element: if it "
        "equals the target you're done; if it's smaller, the target can only be to the right, so set low = mid + 1; "
        "otherwise set high = mid − 1. Each comparison discards half of the remaining elements, so the search takes "
        "O(log n) steps. It only works because the array is sorted."
    ),
    "analogy": "Finding a word in a paper dictionary: open the middle, decide which half the word is in, repeat.",
    "example": "Searching 23 in [2, 5, 8, 12, 16, 23, 38, 56]: mid=12 → go right; mid=23 → found in 2 comparisons.",
    "key_points": ["Requires sorted input.", "Halves the search space each step.", "Runs in O(log n) time.",
                   "Watch the loop condition (low <= high) and the mid update to avoid infinite loops."],
    "common_mistakes": ["Using it on unsorted data.", "Off-by-one errors: setting low = mid instead of mid + 1."],
    "check_question": "Why does binary search need the array to be sorted?",
    "questions": [
        {"type": "mcq", "question": "What is the worst-case time complexity of binary search on n sorted elements?",
         "options": ["O(n)", "O(log n)", "O(n log n)", "O(1)"], "correct_option": "B",
         "reference_answer": "Each step halves the interval, so about log₂ n steps.", "concept": "complexity"},
        {"type": "mcq", "question": "Binary search can be applied directly to…",
         "options": ["Any array", "Only sorted arrays (or monotonic search spaces)", "Only linked lists", "Only arrays of even length"],
         "correct_option": "B", "reference_answer": "Sortedness is what lets one comparison discard half the elements.", "concept": "preconditions"},
        {"type": "mcq", "question": "If arr[mid] < target, the next step is…",
         "options": ["high = mid - 1", "low = mid + 1", "low = mid", "return -1"], "correct_option": "B",
         "reference_answer": "The target must be to the right of mid, so move low past mid.", "concept": "interval update"},
        {"type": "short", "question": "What can go wrong if you write `low = mid` instead of `low = mid + 1`?",
         "reference_answer": "The interval may stop shrinking (when low and mid are equal), causing an infinite loop.",
         "keywords": ["infinite loop|infinite|never ends|loop forever", "interval doesn't shrink|shrink|same|stuck"], "concept": "off-by-one"},
    ],
}

TOPIC_ALIASES = {
    "gd": "gradient descent", "sgd": "gradient descent", "stochastic gradient descent": "gradient descent",
    "over-fitting": "overfitting", "overfit": "overfitting", "regularisation": "overfitting", "regularization": "overfitting",
    "ols": "linear regression", "least squares": "linear regression", "recursive functions": "recursion",
    "feature scaling": "gradient descent", "learning rate": "gradient descent",
}


def lookup_topic(topic: str | None) -> tuple[str, dict] | tuple[None, None]:
    if not topic:
        return None, None
    t = topic.lower().strip()
    t = TOPIC_ALIASES.get(t, t)
    if t in TOPICS:
        return t, TOPICS[t]
    # Prefer the topic mentioned earliest in the string ("linear regression, gradient descent" -> linear regression)
    hits = [(t.find(name), name) for name in TOPICS if name in t]
    hits += [(t.find(alias), name) for alias, name in TOPIC_ALIASES.items() if re.search(rf"\b{re.escape(alias)}\b", t)]
    if hits:
        name = min(hits)[1]
        return name, TOPICS[name]
    for name in TOPICS:
        if t and t in name:
            return name, TOPICS[name]
    return None, None


def subject_topics(subject: str | None) -> list[str]:
    if not subject:
        return []
    s = subject.lower().strip()
    s = SUBJECT_ALIASES.get(s, s)
    for name, topics in SUBJECT_TOPICS.items():
        if name in s or s in name:
            return topics
    return []
