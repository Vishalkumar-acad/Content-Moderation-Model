# Running this service on AWS (EC2)

The model is baked into the image at build time, so a container starts ready
to score text — nothing is downloaded on the first request.

## 1. One-time setup on the box

```bash
# Docker (Amazon Linux / Ubuntu)
sudo dnf install -y docker        # or: sudo apt-get install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
# The group change only applies to NEW logins. Either reconnect your SSH
# session, or run `newgrp docker` once in this shell. Until then, prefix
# docker commands with sudo — e.g. `sudo docker compose up -d --build`.
# If you skip the group entirely, `sudo docker ...` works every time.

# Compose plugin, if your distro does not ship it
sudo dnf install -y docker-compose-plugin   # or: sudo apt-get install -y docker-compose-v2
```

## 2. Build and start

```bash
git clone https://github.com/Vishalkumar-acad/Content-Moderation-Model.git
cd Content-Moderation-Model
docker compose up -d --build
```

The first build downloads the 67 MB model and verifies it against the same
sha256 the service checks at runtime, so the image cannot contain a wrong or
truncated model.

## 3. Check it

```bash
docker compose ps                 # moderation should be "healthy"
curl -s localhost:8000/           # model_loaded should be true
curl -s -X POST localhost:8000/moderate \
  -H 'Content-Type: application/json' \
  -d '{"text":"You are a wonderful person"}'
```

## 4. Open the web ports and point a name at the box

**Security group.** In the AWS console, open the instance's security group and
add inbound rules for HTTP (80) and HTTPS (443) from `0.0.0.0/0`. Without
them Caddy cannot reach Let's Encrypt and no certificate is issued.

**DNS.** In Cloudflare, add an `A` record for the name you want
(e.g. `moderation`) pointing at the box's public IPv4 address. Leave it
**DNS only** (grey cloud) for now — that lets Caddy get its certificate
directly. You can switch to the proxied (orange) setting later, with the
SSL/TLS mode set to **Full**.

The box's public IP:

```bash
curl -s ifconfig.me
```

## 5. Put it behind the reverse proxy

The service listens on **127.0.0.1:8000 only** — Caddy is the single way in.

```bash
# Install Caddy (official repository)
sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt-get update && sudo apt-get install -y caddy

# Add the site block (append — the file may already hold other sites)
sudo tee -a /etc/caddy/Caddyfile >/dev/null <<'EOF'

moderation.pixelabs.in {
    reverse_proxy localhost:8000
}
EOF

sudo systemctl reload caddy
```

Replace `moderation.pixelabs.in` with the name you created in step 4. Caddy
fetches and renews the certificate on its own, exactly like the other
hostnames on your Azure box. Give it a few seconds, then check:

```bash
curl -s https://moderation.pixelabs.in/
```

`"model_loaded":true` in the reply means the public path is live.

## 6. Point the tools site at it

The tools worker proxies `/api/moderate` to this service. Set the
`MODERATION_URL` environment variable in the Cloudflare dashboard (Worker →
Settings → Variables) to the new origin:

```
MODERATION_URL = https://moderation.pixelabs.in
```

Nothing else changes — the worker picks it up on the next deploy, and the
tool page keeps working the same way.

## Notes

- **Memory.** The service idles around 250–350 MB with the model loaded, so
  it fits a 2 GB box comfortably and is capped at 900 MB in
  `docker-compose.yml` so it can never take the host down.
- **CORS.** Browsers cannot call this service directly (the origin allow-list
  in `main.py` does not include the tools site) — and they do not need to,
  because the worker proxies the call server-side. If you ever want a browser
  to hit it directly, add that origin to the allow-list in `main.py`.
- **Updating.** `git pull && docker compose up -d --build`.
- **Logs.** `docker compose logs -f moderation`.
- **The model files are git-ignored** (`.gitignore`) — they live in the
  `model-v3` release, never in the repo.
