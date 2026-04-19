import json
import textwrap

def analyze_notebook(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb.get('cells', [])
    print(f'Total cells: {len(cells)}\n')
    
    headers = []
    
    for i, cell in enumerate(cells):
        cell_type = cell.get('cell_type')
        print(f'--- Cell {i} ({cell_type}) ---')
        
        if cell_type == 'markdown':
            source = ''.join(cell.get('source', []))
            print(f'Markdown snippet: {textwrap.shorten(source, width=200)}')
            for line in source.splitlines():
                if line.startswith('#'):
                    headers.append(line.strip())
            
        elif cell_type == 'code':
            source = ''.join(cell.get('source', []))
            print(f'Source snippet: {textwrap.shorten(source, width=200, placeholder="...")}')
            outputs = cell.get('outputs', [])
            print(f'Outputs: {len(outputs)}')
            has_error = any(out.get('output_type') == 'error' for out in outputs)
            if has_error:
                print('  [Contains Error]')

    print('\n--- Headers ---')
    for h in headers:
        print(h)

analyze_notebook('f:/Projects/CSE-475/cse-475-assignment-02-dino-group-d.ipynb')
