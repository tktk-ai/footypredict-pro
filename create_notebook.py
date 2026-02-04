# Save this script as 'create_notebook.py' and run it
import json

notebook_content = '''<paste the JSON content here>'''

with open('ultimate_quantum_football_predictor_v5.ipynb', 'w') as f:
    f.write(notebook_content)

print("Notebook created successfully!")