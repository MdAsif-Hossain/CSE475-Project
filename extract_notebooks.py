import json

for nb_name in ['CSE_475_Assignment_02_BYOL.ipynb', 'CSE_475_Assignment_02_DINO.ipynb']:
    nb = json.load(open(nb_name, encoding='utf-8'))
    out_name = nb_name.replace('.ipynb', '_content.txt')
    with open(out_name, 'w', encoding='utf-8') as f:
        cells = nb['cells']
        f.write(f"=== {nb_name} === ({len(cells)} cells)\n\n")
        for i, c in enumerate(cells):
            src = ''.join(c['source'])
            cell_type = c['cell_type']
            f.write(f"{'='*80}\n")
            f.write(f"CELL {i} [{cell_type.upper()}]\n")
            f.write(f"{'='*80}\n")
            f.write(src)
            f.write("\n\n")
            # If code cell, check for outputs
            if cell_type == 'code' and c.get('outputs'):
                for out in c['outputs']:
                    if out.get('output_type') == 'stream':
                        text = ''.join(out.get('text', []))
                        if len(text) > 500:
                            text = text[:500] + '... [TRUNCATED]'
                        f.write(f"--- OUTPUT (stream) ---\n{text}\n")
                    elif out.get('output_type') in ('execute_result', 'display_data'):
                        data = out.get('data', {})
                        if 'text/plain' in data:
                            text = ''.join(data['text/plain'])
                            if len(text) > 300:
                                text = text[:300] + '... [TRUNCATED]'
                            f.write(f"--- OUTPUT (display) ---\n{text}\n")
                        if 'image/png' in data:
                            f.write(f"--- OUTPUT (image/png present) ---\n")
    print(f"Written {out_name}")
