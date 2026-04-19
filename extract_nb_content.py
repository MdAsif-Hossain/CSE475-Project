import json, sys

def extract_notebook(path, out_path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    with open(out_path, 'w', encoding='utf-8') as out:
        for i, cell in enumerate(nb['cells']):
            ct = cell['cell_type']
            source = ''.join(cell['source'])
            out.write(f"{'='*60}\n")
            out.write(f"CELL {i} [{ct}]\n")
            out.write(f"{'='*60}\n")
            out.write(source + "\n")
            
            # Print outputs (text only, skip images)
            outputs = cell.get('outputs', [])
            if outputs:
                out.write(f"--- OUTPUTS ({len(outputs)} items) ---\n")
                for j, o in enumerate(outputs):
                    otype = o.get('output_type', '')
                    if otype == 'stream':
                        text = ''.join(o.get('text', []))
                        # Limit long outputs
                        if len(text) > 2000:
                            out.write(text[:1000] + "\n...[TRUNCATED]...\n" + text[-500:] + "\n")
                        else:
                            out.write(text + "\n")
                    elif otype == 'execute_result' or otype == 'display_data':
                        data = o.get('data', {})
                        if 'text/plain' in data:
                            t = ''.join(data['text/plain'])
                            out.write(f"[{otype}]: {t[:500]}\n")
                        if 'image/png' in data:
                            out.write(f"[{otype}]: <IMAGE OUTPUT>\n")
                    elif otype == 'error':
                        out.write(f"[ERROR] {o.get('ename','')}: {o.get('evalue','')}\n")
                        tb = ''.join(o.get('traceback', []))[:500]
                        out.write(tb + "\n")
            out.write("\n")

if __name__ == '__main__':
    # Extract BYOL
    extract_notebook(
        'cse-475-assignment-02-byol output.ipynb',
        'byol_output_extracted.txt'
    )
    # Extract DINO
    extract_notebook(
        'cse-475-assignment-02-dino out.ipynb',
        'dino_output_extracted.txt'
    )
    print("Done extracting both notebooks!")
