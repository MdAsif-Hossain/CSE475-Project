import sys
try:
    import pypdf
except:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pypdf', '-q'])
    import pypdf

reader = pypdf.PdfReader(r'f:\Projects\CSE-475\Spring 2026-Assignment 2.pdf')
with open(r'f:\Projects\CSE-475\assignment2_text.txt', 'w', encoding='utf-8') as f:
    f.write('\n===PAGE===\n'.join([page.extract_text() for page in reader.pages]))
