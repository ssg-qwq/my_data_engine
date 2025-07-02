import heapq

def interleave_lists(*lists):
    """
    交错多个列表，使它们的元素按照时间顺序排列。
    [1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2], [3, 3, 3, 3]
    -> [1, 2, 3, 2, 1, 3, 2, 1, 2, 3, 2, 1, 2, 3, 1, 2]
    """
    total = sum(len(lst) for lst in lists)
    heap = []
    result = []

    if not total:
        return result

    for idx, lst in enumerate(lists):
        if lst:
            freq = total / len(lst)  
            heapq.heappush(
                heap, (0, idx, 0, freq)
            )  

    while heap:
        time, list_idx, item_idx, step = heapq.heappop(heap)
        result.append(lists[list_idx][item_idx])
        if item_idx + 1 < len(lists[list_idx]):
            next_time = time + step
            heapq.heappush(heap, (next_time, list_idx, item_idx + 1, step))

    return result