import json
import io
import sys
import base64
import contextlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Global execution namespace
exec_globals = {
    'np': np,
    'plt': plt,
    'io': io,
    'base64': base64
}

cells = []

def add_md(content):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in content.strip().split("\n")]
    })

def add_code(code_str):
    # Capture stdout and matplotlib plots
    stdout_buf = io.StringIO()
    plt.close('all')
    
    outputs = []
    
    with contextlib.redirect_stdout(stdout_buf):
        try:
            exec(code_str, exec_globals)
        except Exception as e:
            print(f"Error executing code: {e}", file=sys.stderr)
            raise e
            
    # Check stdout
    stdout_str = stdout_buf.getvalue()
    if stdout_str:
        outputs.append({
            "output_type": "stream",
            "name": "stdout",
            "text": [line + "\n" for line in stdout_str.splitlines()]
        })
        
    # Check figures
    figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=100)
        img_buf.seek(0)
        img_b64 = base64.b64encode(img_buf.read()).decode('utf-8')
        outputs.append({
            "output_type": "display_data",
            "data": {
                "image/png": img_b64,
                "text/plain": ["<Figure size ...>"]
            },
            "metadata": {}
        })
    plt.close('all')
    
    cells.append({
        "cell_type": "code",
        "execution_count": len([c for c in cells if c["cell_type"] == "code"]) + 1,
        "metadata": {},
        "outputs": outputs,
        "source": [line + "\n" for line in code_str.strip().split("\n")]
    })

print("Helper ready")
