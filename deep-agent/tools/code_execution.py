"""
Code execution tool using Daytona sandbox.

Provides safe Python code execution with data science libraries.
"""

import os
from daytona_sdk import Daytona

daytona = Daytona()


def execute_python_code(code: str) -> str:
    """
    Execute Python code in a Daytona sandbox for data analysis and visualization.

    Available libraries: pandas, numpy, matplotlib, seaborn, scipy, sklearn

    To save plots, use: plt.savefig('/home/daytona/outputs/chart.png')
    Files are automatically downloaded and stored in scratchpad/plots/

    Args:
        code: Python code to execute

    Returns:
        Execution output, generated file paths, and download locations
    """
    # Create a sandbox
    sandbox = daytona.create()

    try:
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
                os.makedirs("scratchpad/plots", exist_ok=True)
                downloaded = []

                for file_info in files:
                    file_name = file_info.name if hasattr(file_info, 'name') else str(file_info)
                    remote_path = f"/home/daytona/outputs/{file_name}"
                    local_path = os.path.join("scratchpad/plots", file_name)

                    try:
                        # Download file from sandbox
                        sandbox.fs.download_file(remote_path, local_path)
                        downloaded.append(local_path)
                    except Exception as e:
                        output_parts.append(f"Warning: Could not download {file_name}: {e}")

                if downloaded:
                    output_parts.append(f"\n📥 Plots saved locally:")
                    for path in downloaded:
                        output_parts.append(f"   • {path}")
                    output_parts.append(f"\n✅ Main agent can now access these files using Read tool")
        except Exception:
            pass

        return "\n\n".join(output_parts) if output_parts else "Code executed successfully"

    finally:
        daytona.delete(sandbox)
