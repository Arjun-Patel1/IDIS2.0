def requires_human_review(confidence, threshold=0.6):
    if confidence < threshold:
        return True
    return False
