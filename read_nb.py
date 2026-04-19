import json

with open(r'f:\Projects\CSE-475\cse-475-assignment-01-01-tm.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open(r'f:\Projects\CSE-475\nb_output.txt', 'w', encoding='utf-8') as out:
    for i, cell in enumerate(nb['cells']):
        out.write(f"\n========== CELL {i} ({cell['cell_type']}) ==========\n")
        source = ''.join(cell['source'])
        out.write(source + "\n")
