import json
import statistics
import sys

def analyze_response_times(test_file):
    with open(test_file, 'r') as f:
        data = json.load(f)

    rt = data['raw_data']['response_times']
    timestamps = data['raw_data']['timestamps']

    # 找到最长的10个响应时间
    longest = sorted(enumerate(rt), key=lambda x: x[1], reverse=True)[:10]

    print('Top 10 longest response times:')
    for idx, t in longest:
        print(f'  #{idx+1}: {t:.2f}s (timestamp: {timestamps[idx]})')

    # 分析响应时间的分布
    print(f'\nResponse time distribution:')
    print(f'  Min: {min(rt):.2f}s')
    print(f'  Max: {max(rt):.2f}s')
    print(f'  Mean: {statistics.mean(rt):.2f}s')
    print(f'  Median: {statistics.median(rt):.2f}s')
    if len(rt) > 1:
        print(f'  Std dev: {statistics.stdev(rt):.2f}s')

    # 分析响应时间的聚类
    print(f'\nResponse time clusters (grouped by similar times):')
    ranges = [(0, 5), (5, 10), (10, 15), (15, 20)]
    for low, high in ranges:
        count = sum(1 for t in rt if low <= t < high)
        print(f'  {low}-{high}s: {count} requests ({count/len(rt)*100:.1f}%)')

    # 分析连续的响应时间模式
    print(f'\nAnalyzing response time patterns...')
    batch_patterns = []
    current_batch = []
    for i in range(len(rt)):
        if i == 0 or abs(rt[i] - rt[i-1]) < 0.5:
            current_batch.append(i)
        else:
            if len(current_batch) > 1:
                batch_patterns.append(current_batch)
            current_batch = [i]
    if len(current_batch) > 1:
        batch_patterns.append(current_batch)

    print(f'Found {len(batch_patterns)} batch patterns (consecutive requests with similar response times):')
    for pattern in batch_patterns[:10]:  # 只显示前10个
        start = pattern[0]
        end = pattern[-1]
        avg_time = statistics.mean([rt[i] for i in pattern])
        print(f'  Requests {start+1}-{end+1}: {len(pattern)} requests, avg time: {avg_time:.2f}s')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <test_result_file>')
        sys.exit(1)
    analyze_response_times(sys.argv[1])
