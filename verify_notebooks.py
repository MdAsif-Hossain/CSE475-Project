import json
byol = json.load(open('CSE_475_Assignment_02_BYOL.ipynb', encoding='utf-8'))
dino = json.load(open('CSE_475_Assignment_02_DINO.ipynb', encoding='utf-8'))
print(f"BYOL: {len(byol['cells'])} cells")
print(f"DINO: {len(dino['cells'])} cells")

# Print structure
for name, nb in [("BYOL", byol), ("DINO", dino)]:
    print(f"\n{'='*60}")
    print(f"{name} Notebook Structure:")
    print(f"{'='*60}")
    for i, c in enumerate(nb['cells']):
        src = ''.join(c['source'])
        first_line = src.strip().split('\n')[0][:80] if src.strip() else "(empty)"
        print(f"  Cell {i:>2} [{c['cell_type'][:4].upper()}] {first_line}")
