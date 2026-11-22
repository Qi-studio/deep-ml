import math

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    max_val = max(scores)
    scores = [math.exp(i - max_val) for i in scores]
    sum_val = sum(scores)
    result = [i / sum_val for i in scores]
    return result