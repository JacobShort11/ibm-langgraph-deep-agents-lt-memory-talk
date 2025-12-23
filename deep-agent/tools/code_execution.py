"""
Code execution tool using Daytona sandbox.

Provides safe Python code execution with data science libraries.
"""

import os
from daytona_sdk import Daytona

daytona = Daytona()

# Get absolute paths to deep-agent/scratchpad directories
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEEP_AGENT_DIR = os.path.dirname(_TOOLS_DIR)
PLOTS_DIR = os.path.join(_DEEP_AGENT_DIR, "scratchpad", "plots")
DATA_DIR = os.path.join(_DEEP_AGENT_DIR, "scratchpad", "data")


def execute_python_code(code: str) -> str:
    """
    Execute Python code in a Daytona sandbox for data analysis and visualization.

    Available libraries: pandas, numpy, matplotlib, seaborn, scipy, sklearn

    File paths:
    - Input data: Files from scratchpad/data/ are uploaded to /home/daytona/data/
    - Output plots: Save to /home/daytona/outputs/ → downloaded to scratchpad/plots/

    Args:
        code: Python code to execute

    Returns:
        Execution output, generated file paths, and download locations
    """
    # Create a sandbox
    sandbox = daytona.create()

    try:
        # Upload any data files from local scratchpad/data to sandbox
        if os.path.exists(DATA_DIR):
            sandbox.process.code_run("import os; os.makedirs('/home/daytona/data', exist_ok=True)")
            for filename in os.listdir(DATA_DIR):
                local_path = os.path.join(DATA_DIR, filename)
                if os.path.isfile(local_path):
                    remote_path = f"/home/daytona/data/{filename}"
                    try:
                        sandbox.fs.upload_file(local_path, remote_path)
                    except Exception as e:
                        pass  # Silently skip upload errors

        # Setup code with common imports
        setup = """
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.makedirs('/home/daytona/outputs', exist_ok=True)
"""
        # Run setup + user code
        response = sandbox.process.code_run(setup + "\n" + code)

        output_parts = []

        # If the code does print("hello") → that goes into response.result
        if response.result:
            output_parts.append(f"Output:\n{response.result}")

        # Check for generated files and download them
        try:
            files = sandbox.fs.list_files("/home/daytona/outputs")
            if files:
                # Extract file names from FileInfo objects
                file_names = [f.name if hasattr(f, 'name') else str(f) for f in files]
                output_parts.append(f"Generated files: {', '.join(file_names)}")

                # Download files to local filesystem
                os.makedirs(PLOTS_DIR, exist_ok=True)
                downloaded = []

                for file_info in files:
                    file_name = file_info.name if hasattr(file_info, 'name') else str(file_info)
                    remote_path = f"/home/daytona/outputs/{file_name}"
                    local_path = os.path.join(PLOTS_DIR, file_name)

                    try:
                        # Download file from sandbox
                        sandbox.fs.download_file(remote_path, local_path)
                        downloaded.append(local_path)
                    except Exception as e:
                        output_parts.append(f"Warning: Could not download {file_name}: {e}")

                if downloaded:
                    output_parts.append(f"Plots saved to scratchpad/plots/: {', '.join(file_names)}")
        except Exception:
            pass

        return "\n\n".join(output_parts) if output_parts else "Code executed successfully"

    finally:
        daytona.delete(sandbox)
