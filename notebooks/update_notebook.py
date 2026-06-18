import json

def update_notebook():
    with open('c:/Users/aymen/Desktop/sisyphus/notebooks/model.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    with open('c:/Users/aymen/Desktop/sisyphus/notebooks/model_code.py', 'r', encoding='utf-8') as f:
        new_source = f.read()

    # Split lines and append newlines to match notebook format
    source_lines = [line + '\n' for line in new_source.split('\n')]
    if source_lines and source_lines[-1] == '\n':
        source_lines = source_lines[:-1]

    # The first cell contains the benchmarking code
    nb['cells'][0]['source'] = source_lines

    # Remove the second cell which has the error (execution_count null, source empty)
    if len(nb['cells']) > 1 and not nb['cells'][1]['source']:
        nb['cells'].pop(1)

    with open('c:/Users/aymen/Desktop/sisyphus/notebooks/model.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        f.write('\n')

if __name__ == '__main__':
    update_notebook()
