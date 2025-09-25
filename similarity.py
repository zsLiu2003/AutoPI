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
    from pathlib import Path

    data_dir ="/home/zesen/AutoPI/data"
    open_prompt_dir = "/home/zesen/AutoPI/data/open_prompt"

    agent_names = ["cursor", "copilot", "trae", "windsurf", "cline"]

    print("=" * 70)
    print("BERTScore Similarity between /data and /data/open_prompt prompts")
    print("=" * 70)

    
    for agent_name in agent_names:
        with open(f"{data_dir}/{agent_name}.txt", "r", encoding="utf-8") as f:
            prompt1 = f.read()

        with open(f"{open_prompt_dir}/{agent_name}.txt", "r", encoding="utf-8") as f:
            prompt2 = f.read()
        result = bert_score_similarity(prompt1, prompt2)

        print(f"\n{agent_name}:")
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall:    {result['recall']:.4f}")
        print(f"  F1:        {result['f1']:.4f}")

    print("\n" + "=" * 70)