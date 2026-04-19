import json

def dump_code(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for i, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') == 'code':
            source = "".join(cell.get('source', []))
            print(f'# ----- Cell {i} -----')
            print(source)
            print()

dump_code('f:/Projects/CSE-475/cse-475-assignment-02-dino out.ipynb')
