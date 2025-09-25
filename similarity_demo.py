from bert_score import score

def bert_score_similarity(prompt1: str, prompt2: str, model_type: str = "microsoft/deberta-xlarge-mnli") -> dict:
    """
    使用 BERTScore 计算两个文本（prompt）的语义相似度。

    Args:
        prompt1 (str): 第一个文本
        prompt2 (str): 第二个文本
        model_type (str): 用于编码的模型，默认是 DeBERTa (microsoft/deberta-xlarge-mnli)，
                          也可以换成 roberta-large, bert-base-uncased 等

    Returns:
        dict: 含 precision, recall, f1 三个分数
    """

    # 计算 BERTScore
    P, R, F1 = score([prompt1], [prompt2], model_type=model_type, verbose=False)

    # 转成 float
    precision = float(P[0])
    recall = float(R[0])
    f1 = float(F1[0])

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


if __name__ == "__main__":
    prompt_a = "Write a function in Python to compute factorial."
    prompt_b = "Create a Python function that calculates the factorial of a number."

    result = bert_score_similarity(prompt_a, prompt_b)
    print(f"Precision: {result['precision']:.3f}")
    print(f"Recall: {result['recall']:.3f}")
    print(f"F1 (similarity): {result['f1']:.3f}")