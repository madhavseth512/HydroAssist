# Deploying HydroAssist on Oracle Cloud Always Free

HydroAssist runs as two containers on one small VM:

- **app** — FastAPI, serving the API and the built React workspace
- **Caddy** — HTTPS, via a free Let's Encrypt certificate, plus basic auth

The whole setup costs nothing.

```
Browser ──HTTPS──▶ Caddy (TLS, basic auth) ──▶ app: FastAPI + React build ──▶ Gemini (free tier)
```

## Why this host

Checked on 11 September 2026:

| Option | Why not |
|---|---|
| Render / Koyeb free | 512 MB RAM and 0.1 CPU: out-of-memory risk, and fits would crawl |
| Google Cloud Run / Azure Container Apps | They bill the card on file once the free grant is used up |
| Hugging Face Spaces, Fly.io, Railway | No free option for an app like this |
| **Oracle Cloud Always Free** | **Ampere A1: 2 OCPU and 12 GB RAM.** Measured peak use is about 415 MB. |

Oracle's risks, stated plainly:
- **Allowance changes:** Oracle halved the Always Free allowance without notice in June 2026.
- **Idle reclaim:** Oracle may reclaim Always Free instances it considers idle. Check the current Always Free documentation.
- **Card required:** sign-up asks for a card for identity verification. It is not charged while the account stays on Always Free — **never click "Upgrade"**.

## What you need

- An Oracle Cloud account, set up in step 1.
- A free DuckDNS subdomain, set up in step 3.
- A **private** copy of this repository on GitHub, needed from step 4 (see the licensing warning at the end).
- A Gemini API key.

## 1. Create the VM

1. Sign up at <https://www.oracle.com/cloud/free/> and choose a home region close
   to your users. Always Free resources are only available in your home region.
2. Go to **Compute → Instances → Create instance**:
   - **Image:** Canonical Ubuntu 24.04.
   - **Shape:** Ampere, `VM.Standard.A1.Flex` with **2 OCPU and 12 GB memory**.
   - **Networking:** create a VCN with a public subnet, and assign a public IPv4 address.
   - **SSH keys:** upload your public key, or download the generated one.
3. Note the instance's **public IP address**.

> Ampere capacity is sometimes exhausted in popular regions. If creation fails
> with "Out of capacity", retry later or pick another availability domain.

## 2. Open ports 80 and 443 — in two places

Oracle blocks traffic at the network **and** on the VM itself. Both must be
opened, or Caddy cannot obtain a certificate.

**a. Network security list.** Open **Networking → Virtual cloud networks → your VCN → Security Lists → Default**. Add two ingress rules, both with source `0.0.0.0/0` and protocol TCP: one for destination port **80** and one for port **443**.

**b. The VM's firewall.** Oracle's Ubuntu images ship `iptables` rules that reject new inbound connections other than SSH:

```bash
ssh ubuntu@<public-ip>
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save
```

## 3. Point a domain at the VM

1. Sign in at <https://www.duckdns.org>.
2. Create a subdomain, for example `hydroassist-kgp`.
3. Set its IP to the VM's public IP.

Your site will be `https://hydroassist-kgp.duckdns.org`.

## 4. Install Docker and get the code

On the VM:

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu && newgrp docker
docker compose version            # confirms the compose plugin is present
```

The repository is private, so give the VM a read-only **deploy key**:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/hydroassist_deploy -N ""
cat ~/.ssh/hydroassist_deploy.pub
```

Add that public key to the GitHub repository under **Settings → Deploy keys**, leaving write access **off**. Then clone:

```bash
GIT_SSH_COMMAND="ssh -i ~/.ssh/hydroassist_deploy" \
  git clone git@github.com:<you>/HydroAssist.git ~/hydroassist
cd ~/hydroassist
```

## 5. Configure

```bash
cp .env.example .env
docker run --rm caddy:2 caddy hash-password --plaintext 'choose-a-password'
nano .env
```

Fill in `.env`:
- `GEMINI_API_KEY`
- `DOMAIN` — for example `hydroassist-kgp.duckdns.org`
- `BASIC_AUTH_USER`
- `BASIC_AUTH_HASH` — the hash printed above, **inside single quotes**, because it contains `$`

`.env` never leaves the VM. It is excluded from git and from the Docker build context.

## 6. Start

```bash
docker compose up -d --build
docker compose logs -f
```

The first build takes several minutes on the Ampere VM. It installs the Python dependencies, builds the React workspace, and downloads the ONNX embedding model into the image. Caddy obtains the certificate once the app is healthy.

## Verify

Run the smoke check from the VM. It checks health, a session, an upload, a Theis
fit against the golden value and the PDF report, and names the first step that
fails:

```bash
python3 scripts/smoke_deploy.py https://<your-domain> --user <user> --password '<password>'
```

Then check in a browser:

1. `docker compose ps` — both services are running, and `app` is `healthy`.
2. Open `https://<your-domain>`. The browser asks for the basic-auth credentials, then loads the workspace.
3. Choose `tests/theis_cooper_jacob_sample.csv` from your local copy. Leave Q = 500 m³/day and r = 50 m, and run **Theis (1935)**. Expect **T ≈ 215.29 m²/day** and S ≈ 5.05×10⁻⁵.
4. Click **Download report**. A PDF downloads.
5. Open **Consultant** and ask *"What are the assumptions of the Theis method?"* The answer should list **Sources consulted**. If the answer says it lacks information, the vector store did not deploy.

## Updating

```bash
cd ~/hydroassist
GIT_SSH_COMMAND="ssh -i ~/.ssh/hydroassist_deploy" git pull
docker compose up -d --build
```

Sessions live in memory. A redeploy ends open sessions, so anyone mid-analysis loses unsaved work; a saved project file reopens as normal.

## Operating notes

- **Gemini free tier.** About 1,500 requests a day, and one question can cost several model calls. The app therefore caps the consultant at 20 questions per visitor per 10 minutes and 500 per day in total. It says so plainly when a cap is reached, and the calculators are never limited. Tune the caps in `.env` (see `.env.example`). On the free tier, Google may use conversations to improve its products; the chat panel tells visitors this.
- **No consultant?** If the key is missing or the model is unreachable, the workspace still works — upload, fit, report, compare — and the chat explains why it cannot answer.
- **One configuration.** `configs/default.yaml` is the only config, and it is what every test runs against, so the image sets `HYDROASSIST_ENV=default`. Untested `production.yaml` and `development.yaml` variants were deleted: the first set the retrieval `similarity_threshold` to 0.75 against 0.2, which would have cut the sources the consultant finds and looked like a broken knowledge base. `tests/unit/test_config.py` fails if an untested variant reappears.
- **Memory.** One worker, with sessions capped at 200 and expired after two idle hours. The VM has ample headroom.

## Before making the site public

`knowledge source/` contains roughly thirty publisher PDFs (SPE, Elsevier, Wiley/AGU) tracked in git.

- **Kept out of the image.** `.dockerignore` excludes them, and the Dockerfile copies only the built index.
- **Still quoted.** The consultant's answers quote passages from them.
- **Hence private.** The repository stays private, and the site sits behind basic auth.

Removing the auth, or making the repository public, is a licensing decision, not a technical one, and should be settled first.
