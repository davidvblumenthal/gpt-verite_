

def discard_small_samples(doc_ids: list, loss_mask: list, pad_token: int, threshold: int): 
    # Extract the last 2048 items from the input list
    last_items = doc_ids[-2048:]
    
    # Count the number of items in the last 2048 that are not equal to 1
    non_ones = sum(pad_token for item in last_items if item != pad_token)
    
    # If the number of non-1 items is less than the threshold,
    # return the input list without the last 2048 items. Otherwise,
    # return the input list unmodified.
    if non_ones < threshold:
        return [doc_ids[:-2048]], [loss_mask[:-2048]]
    else:
        return [doc_ids], [loss_mask]
