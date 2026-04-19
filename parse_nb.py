import json

def analyze_notebook(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb.get('cells', [])
    print(f'Total cells: {len(cells)}\n')

    for i, cell in enumerate(cells):
        cell_type = cell.get('cell_type')
        print(f'--- Cell {i} ({cell_type}) ---')
        
        if cell_type == 'code':
            source = ''.join(cell.get('source', []))
            print(f'Source snippet: {source[:150]}...')
            
            outputs = cell.get('outputs', [])
            print(f'Outputs: {len(outputs)}')
            
            for j, gout in enumerate(outputs):
                out_type = gout.get('output_type')
                if out_type == 'stream':
                    print(f'  [Stream] {gout.get("name")}: {"".join(gout.get("text", []))[:200]}...')
                elif out_type == 'error':
                    print(f'  [Error] {gout.get("ename")}: {gout.get("evalue")}')
                    traceback_text = '\n'.join(gout.get('traceback', []))
                    print(f'  Traceback snippet: {traceback_text[:300]}...')
                elif out_type in ('display_data', 'execute_result'):
                    data = gout.get('data', {})
                    keys = list(data.keys())
                    print(f'  [{out_type}] Data keys: {keys}')
                    if 'text/plain' in data:
                        print(f'    text/plain snippet: {"".join(data["text/plain"])[:200]}...')

analyze_notebook('f:/Projects/CSE-475/cse-475-assignment-02-dino out.ipynb')
