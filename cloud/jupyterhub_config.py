"""JupyterHub configuration for the shared AutoPilot GPU server.

- PAM authentication (existing Linux accounts).
- Every user's single-user server opens directly in the shared repo, so
  nobody needs to `cd` around -- the repo's shared "autopilot" group
  permissions (see cloud/setup_users.sh) make it writable by the whole team.
- Hub listens on localhost only; nginx (with the Let's Encrypt cert) is the
  public HTTPS endpoint reverse-proxying to it.
"""
c = get_config()  # noqa: F821

c.JupyterHub.ip = "127.0.0.1"
c.JupyterHub.port = 8000
c.JupyterHub.bind_url = "http://127.0.0.1:8000"

c.Authenticator.admin_users = {"ubuntu"}
c.Authenticator.allow_all = True
c.PAMAuthenticator.open_sessions = False

# Every spawned single-user server starts in the shared project directory
# regardless of the user's own $HOME.
c.Spawner.notebook_dir = "/home/ubuntu/AutoPilot"
c.Spawner.default_url = "/lab"

# The hub's proxy trusts nginx's X-Forwarded-* headers.
c.JupyterHub.trust_downstream_proxy = True
