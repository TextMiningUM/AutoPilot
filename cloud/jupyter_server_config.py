"""System-wide jupyter_server config (applies to every JupyterHub single-user
server). Restricts the kernel picker to only the AutoPilot project venv, so
nobody can accidentally pick the bare system Python 3.8 kernel and hit
"no matching distribution" pip errors.
"""
c = get_config()  # noqa: F821

c.KernelSpecManager.allowed_kernelspecs = {"autopilot-venv"}
