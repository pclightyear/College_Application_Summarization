def calculate_coverage(key_tokens, target):
    """
        key_tokens: list of tokens
        target: a piece of text
    """
    hits = sum([1 for token in key_tokens if token in target])
    coverage = hits / len(key_tokens)
    
    return coverage

def cover_determinants(determinants, target):
    """
        determinants: list of determine tokens
        target: a piece of text
    """
    for determinant in determinants:
        if determinant not in target:
            return False
        
    return True