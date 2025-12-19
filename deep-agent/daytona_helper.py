"""
Standalone Daytona code execution helper.

This module provides the execute_python_code function without requiring
the full agent setup (database, checkpointer, etc.).
"""

import os
from daytona_sdk import Daytona

# Initialize Daytona client
daytona = Daytona()


def execute_python_code(code: str, download_outputs: bool = True, output_dir: str = "outputs") -> str:
    """
    Execute Python code in a Daytona sandbox for data analysis and visualization.

    Available libraries: pandas, numpy, matplotlib, seaborn, scipy, sklearn

    To save plots, use: plt.savefig('/home/daytona/outputs/chart.png')
    Files are automatically downloaded to the local outputs directory.

    Args:
        code: Python code to execute
        download_outputs: If True, download generated files to local directory (default: True)
        output_dir: Local directory to save downloaded files (default: "outputs")

    Returns:
        Execution output and any generated file paths
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

        # Check for generated files and optionally download them
        try:
            # First verify the directory exists
            try:
                files = sandbox.fs.list_files("/home/daytona/outputs")
            except Exception as list_err:
                # Directory might not exist if code failed
                output_parts.append(f"\n⚠️ Could not access outputs directory: {list_err}")
                files = None

            if files is None:
                files = []

            if files:
                # Extract file names from FileInfo objects
                file_names = [f.name if hasattr(f, 'name') else str(f) for f in files]
                output_parts.append(f"Generated files: {', '.join(file_names)}")

                # Download files to local directory
                if download_outputs:
                    os.makedirs(output_dir, exist_ok=True)
                    downloaded = []

                    for file_info in files:
                        file_name = file_info.name if hasattr(file_info, 'name') else str(file_info)
                        remote_path = f"/home/daytona/outputs/{file_name}"
                        local_path = os.path.join(output_dir, file_name)

                        try:
                            # Download file from sandbox
                            sandbox.fs.download_file(remote_path, local_path)
                            downloaded.append(local_path)
                        except Exception as e:
                            output_parts.append(f"Warning: Could not download {file_name}: {e}")

                    if downloaded:
                        output_parts.append(f"\n📥 Downloaded to local directory:")
                        for path in downloaded:
                            output_parts.append(f"   • {path}")
        except Exception as e:
            output_parts.append(f"\n⚠️ Error downloading files: {e}")

        return "\n\n".join(output_parts) if output_parts else "Code executed successfully"

    finally:
        daytona.delete(sandbox)
