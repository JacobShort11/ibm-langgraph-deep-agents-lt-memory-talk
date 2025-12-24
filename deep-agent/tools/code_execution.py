"""
Code execution tool using Daytona sandbox.

Provides safe Python code execution with data science libraries. Plot outputs
are pulled from the sandbox and uploaded from the host to Cloudinary when
environment variables are configured.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from typing import Dict, Iterable, List, Tuple
from uuid import uuid4

import requests
from daytona_sdk import Daytona

daytona = Daytona()

# Get absolute paths
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEEP_AGENT_DIR = os.path.dirname(_TOOLS_DIR)


def _cloudinary_config() -> Tuple[Dict[str, str], List[str]]:
    """Read Cloudinary env config and return settings + warnings."""
    warnings: List[str] = []

    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
    api_key = os.getenv("CLOUDINARY_API_KEY")
    api_secret = os.getenv("CLOUDINARY_API_SECRET")
    upload_preset = os.getenv("CLOUDINARY_UPLOAD_PRESET")  # for unsigned uploads if preferred
    prefix = os.getenv("CLOUDINARY_PUBLIC_ID_PREFIX", "plots/")

    # Allow CLOUDINARY_URL to populate fields if present
    cloudinary_url = os.getenv("CLOUDINARY_URL")
    if cloudinary_url and not (cloud_name and api_key and (api_secret or upload_preset)):
        try:
            # cloudinary://<api_key>:<api_secret>@<cloud_name>
            _, rest = cloudinary_url.split("://", 1)
            creds, cloud = rest.split("@", 1)
            key, secret = creds.split(":", 1)
            cloud_name = cloud_name or cloud
            api_key = api_key or key
            api_secret = api_secret or secret
        except Exception:
            warnings.append("CLOUDINARY_URL is set but could not be parsed.")

    if not cloud_name:
        warnings.append("CLOUDINARY_CLOUD_NAME is not set; cannot upload plots.")
    if not api_key:
        warnings.append("CLOUDINARY_API_KEY is not set; cannot upload plots.")
    if not (api_secret or upload_preset):
        warnings.append("Set CLOUDINARY_API_SECRET (for signed uploads) or CLOUDINARY_UPLOAD_PRESET (for unsigned).")

    cfg = {
        "cloud_name": cloud_name,
        "api_key": api_key,
        "api_secret": api_secret,
        "upload_preset": upload_preset,
        "prefix": prefix,
    }
    if warnings:
        return {}, warnings
    return cfg, warnings


def _upload_cloudinary_host(file_paths: Iterable[str]) -> Tuple[List[str], List[str]]:
    """Upload plot files from host to Cloudinary (used when sandbox egress is blocked)."""
    cfg, warnings = _cloudinary_config()
    if not cfg:
        return [], warnings

    upload_url = f"https://api.cloudinary.com/v1_1/{cfg['cloud_name']}/image/upload"
    uploaded: List[str] = []

    for path in file_paths:
        base = os.path.basename(path)
        stem, _ = os.path.splitext(base)
        public_id = f"{cfg['prefix'].rstrip('/')}/{uuid4().hex}_{stem}".lstrip("/")
        timestamp = int(time.time())
        data: Dict[str, str] = {
            "api_key": cfg["api_key"],
            "timestamp": str(timestamp),
            "public_id": public_id,
        }
        if cfg["upload_preset"]:
            data["upload_preset"] = cfg["upload_preset"]
        elif cfg["api_secret"]:
            to_sign = f"public_id={public_id}&timestamp={timestamp}{cfg['api_secret']}"
            import hashlib

            data["signature"] = hashlib.sha1(to_sign.encode("utf-8")).hexdigest()
        else:
            warnings.append("Missing both CLOUDINARY_API_SECRET and CLOUDINARY_UPLOAD_PRESET; cannot upload.")
            continue

        try:
            with open(path, "rb") as f:
                resp = requests.post(upload_url, data=data, files={"file": f}, timeout=30)
            if resp.status_code >= 400:
                warnings.append(f"Host upload failed for {path}: {resp.status_code} {resp.text}")
                continue
            if resp.headers.get("Content-Type", "").startswith("application/json"):
                body = resp.json()
                url = body.get("secure_url") or body.get("url")
                if url:
                    uploaded.append(url)
                else:
                    warnings.append(f"Host upload returned no URL for {path}")
            else:
                warnings.append(f"Host upload returned non-JSON response for {path}")
        except Exception as exc:
            warnings.append(f"Host upload error for {path}: {exc}")

    return uploaded, warnings


def execute_python_code(code: str) -> str:
    """
    Execute Python code in a Daytona sandbox for data analysis and visualization.

    Available libraries: pandas, numpy, matplotlib, seaborn, scipy, sklearn

    File paths:
    - Output plots: Save to /home/daytona/outputs/ → downloaded then uploaded from host to Cloudinary (when configured)

    Args:
        code: Python code to execute

    Returns:
        Execution output, generated file paths, and public URLs when available
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

        # Check for generated files, download them, and upload from host
        try:
            files = sandbox.fs.list_files("/home/daytona/outputs")
            if files:
                file_names = [f.name if hasattr(f, "name") else str(f) for f in files]
                output_parts.append("Generated files:\n" + "\n".join(f"- {name}" for name in file_names))
                temp_dir = tempfile.mkdtemp(prefix="plots_")
                downloaded: List[str] = []
                for name in file_names:
                    remote_path = f"/home/daytona/outputs/{name}"
                    local_path = os.path.join(temp_dir, name)
                    try:
                        sandbox.fs.download_file(remote_path, local_path)
                        downloaded.append(local_path)
                    except Exception as exc:
                        output_parts.append(f"Plot upload warnings: host download failed for {name}: {exc}")

                if downloaded:
                    urls, warns = _upload_cloudinary_host(downloaded)
                    if urls:
                        output_parts.append("Plot URLs:\n" + "\n".join(f"- {url}" for url in urls))
                    if warns:
                        output_parts.append("Plot upload warnings:\n" + "\n".join(f"- {w}" for w in warns))
                shutil.rmtree(temp_dir, ignore_errors=True)
            else:
                output_parts.append("No plot files found to upload.")
        except Exception as exc:
            output_parts.append(f"Plot upload warnings: {exc}")

        return "\n\n".join(output_parts) if output_parts else "Code executed successfully"

    finally:
        daytona.delete(sandbox)
