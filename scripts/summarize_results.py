"""Rebuild the spatially paired RIVER tables from released task outcomes."""
from collections import Counter, defaultdict
import csv
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / 'data'


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def summarize(study):
    assigned = read_csv(DATA / study / 'all_assigned.csv')
    rows = read_csv(DATA / study / 'per_task.csv')
    method = 'method' if study == 'initialization' else 'arm'
    retention = 'last10_correct' if study == 'initialization' else 'retained'
    index = 'case_index' if study == 'initialization' else 'index'
    arms = sorted({row[method] for row in assigned})
    expected = defaultdict(set)
    observed = defaultdict(set)
    key_fields = ('group', 'task', 'depth_swap', 'exit_mode', method)
    for ledger, by_group in ((assigned, expected), (rows, observed)):
        if len({row[index] for row in ledger}) != len(ledger):
            raise ValueError(f'Duplicate case index in {study}')
        for row in ledger:
            by_group[row['group']].add(tuple(row[field] for field in key_fields))
    groups = sorted(group for group in expected if expected[group] == observed[group])
    status = Counter(row['status'] for row in assigned)
    print(f'\n## {study.title()}\n')
    print(f'{len(groups)} complete spatial groups. {len(rows)} scored of {len(assigned)} assigned. '
        + ', '.join(f'{n} {label}' for label, n in sorted(status.items())) + '.\n')
    has_costs = 'detector_calls' in rows[0]
    columns = ['Arm', 'Tasks', 'Correct frames', 'Last-10 retention', 'Sustained wrong identity']
    if has_costs:
        columns.extend(['Detector calls', 'Module time (s)'])
    print('| ' + ' | '.join(columns) + ' |')
    print('| ' + ' | '.join(['---'] * len(columns)) + ' |')
    for arm in arms:
        selected = [row for row in rows if row['group'] in groups and row[method] == arm]
        count = len(selected)
        correct = sum(int(row['correct_frames']) for row in selected)
        kept = sum(int(row[retention]) for row in selected)
        wrong = sum(int(row['sustained_wrong']) for row in selected)
        values = [arm, count, f'{correct}/{45 * count}', f'{kept}/{count}', f'{wrong}/{count}']
        if has_costs:
            values.extend([sum(int(row['detector_calls']) for row in selected),
                f"{sum(float(row['module_wall_s']) for row in selected):.3f}"])
        print('| ' + ' | '.join(map(str, values)) + ' |')


if __name__ == '__main__':
    for name in ('mobile', 'initialization', 'rejection'):
        summarize(name)
