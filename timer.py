import time

total_time = {}

def timer(func):
    total_time[func] = 0
    def _inner(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        total_time[func] += end - start
        # print(f"{func.__name__} elapsed: {end - start}")
        return res
    return _inner


if __name__ == "main":
    for func in total_time:
        print(f"total time in {func.__name__}: {total_time[func]}")