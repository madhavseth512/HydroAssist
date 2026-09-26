# Deploying HydroAssist

Two hosts are supported, both free. Pick by whether you have a card.

| | **Render** | **Oracle Cloud Always Free** |
|---|---|---|
| Card to sign up | **No** | Yes, for identity verification |
| Memory | 512 MB | 12 GB |
| CPU | 0.1 | 2 Ampere cores |
| Sleeps when idle | After 15 min, ~1 min to wake | Never |
| HTTPS | Included | Caddy + Let's Encrypt |
| Password protection | The app's own (`src/api/auth.py`) | Caddy's `basic_auth` |
| Setup | Connect the repo | VM, firewall in two places, DuckDNS |
| Upkeep | None | You patch the machine |

Both run the same image, so moving later costs an afternoon rather than a
rewrite.

**Does HydroAssist fit in 512 MB?** Measured, with the container capped at
Render's limit and no swap: 301 MB after a full analysis (upload, Theis fit,
PDF), and 401 MB after three consultant questions, never OOM-killed. About
110 MB spare — enough for one person, and the case to watch is two people asking
questions at the same moment.

Rejected, and why (checked 11 and 23 September 2026):

| Option | Why not |
|---|---|
| Railway | No free tier since 2023. $5/month minimum, $1 even on the "Free" plan. |
| Hugging Face Spaces | 16 GB free, but the Docker SDK now requires a paid PRO plan. |
| Google Cloud Run / Azure Container Apps | Bill the card on file once the free grant runs out. |
| Fly.io | No free allowance, and a card is required. |
| Vercel / Netlify | Serverless size limits; this image is 1.8 GB. |
| PythonAnywhere free | No Docker, and outbound network is allow-listed, so Gemini is unreachable. |

---

# Path A — Render (no card)

1. **Create a Render account** at <https://render.com>. Sign in with GitHub so it
   can see your private repository. No card is asked for.
2. **Choose a password** for the site and hash it, locally:
   ```bash
   python scripts/make_auth_hash.py 'the password you choose'
   ```
   The corpus quotes paywalled papers, so this is not optional until the
   licensing question is settled. Without it the app logs a loud warning and
   serves anyone with the URL.
3. **New → Blueprint**, pick this repository. Render reads `render.yaml` and
   creates the service.
4. **Set the three secrets** it asks for, which are deliberately not in git:
   - `GEMINI_API_KEY`
   - `HYDROASSIST_AUTH_USER` — whatever you like, e.g. `kgp`
   - `HYDROASSIST_AUTH_PASSWORD_HASH` — the `pbkdf2_sha256$...` line from step 2
5. **Wait for the first build.** It compiles the React workspace and installs
   the Python stack, so expect roughly ten minutes. Watch the log for
   `Password protection is ON`.
6. **Check it**, from your own machine:
   ```bash
   python scripts/smoke_deploy.py https://hydroassist.onrender.com \
       --user kgp --password 'the password you chose'
   ```
   Eight checks, ending with a PDF and `T = 215.29 m^2/day`. Without
   `--user`/`--password` it should fail at the first step with a 401, which is
   itself proof the lock works.

   One of the eight reads a photographed field sheet. It is there because
   Tesseract is the only dependency that is not a Python wheel — the Dockerfile
   installs it with `apt-get` — and a missing apt layer stays invisible until a
   user photographs a sheet and is told their image is the problem. A 503 on that
   check means the image was built without it.

**What to expect day to day.** The first visit after a quiet spell takes a
minute or so: Render wakes the container, then the knowledge base loads. After
that it is quick. Projects are saved to the visitor's own machine, so nothing is
lost when the service sleeps — but a conversation in progress is, because
sessions live in memory.

**Your DuckDNS subdomain is not needed here.** Render provides
`hydroassist.onrender.com` with HTTPS already. Keep the subdomain for Oracle, or
point it at Render later as a custom domain.

---

# Path B — Oracle Cloud Always Free

HydroAssist runs as two containers on one small VM:

- **app** — FastAPI, serving the API and the built React workspace
- **Caddy** — HTTPS, via a free Let's Encrypt certificate, plus basic auth

```
Browser ──HTTPS──▶ Caddy (TLS, basic auth) ──▶ app: FastAPI + React build ──▶ Gemini (free tier)
```

Oracle's risks, stated plainly:
- **Allowance changes:** Oracle halved the Always Free allowance without notice in June 2026.
- **Idle reclaim:** Oracle may reclaim Always Free instances it considers idle.
- **Card required:** sign-up asks for a card for identity verification. A debit
  card is usually accepted. It is not charged while the account stays on Always
  Free — **never click "Upgrade"**.

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

**Your DuckDNS token is not one of these.** Nothing in HydroAssist reads it; it
is the password to your subdomain, and it has one job here — putting the current
IP back into DNS if Oracle ever gives the instance a different one. An instance
that is stopped and started can come back on a new address, and the site then
resolves to nothing until DNS is corrected.

Either fix it by hand at duckdns.org when it happens, or have the VM do it:

```bash
# Keeps the subdomain pointed at this machine. Token stays on the VM.
( crontab -l 2>/dev/null
  echo "*/5 * * * * curl -fsS 'https://www.duckdns.org/update?domains=<subdomain>&token=<token>&ip=' >/dev/null" \
) | crontab -
```

Leaving `ip=` empty tells DuckDNS to use the address the request came from, so
the VM always registers itself.

## 6. Start

```bash
docker compose up -d --build
docker compose logs -f
```

The first build takes several minutes on the Ampere VM. It installs the Python dependencies, builds the React workspace, and downloads the ONNX embedding model into the image. Caddy obtains the certificate once the app is healthy.

## Verify

Run the smoke check from the VM. It checks health, a session, an upload, a Theis
fit against the golden value, a photographed field sheet and the PDF report, and
names the first step that fails:

```bash
python3 scripts/smoke_deploy.py https://<your-domain> --user <user> --password '<password>'
```

Then check in a browser:

1. `docker compose ps` — both services are running, and `app` is `healthy`.
2. Open `https://<your-domain>`. The browser asks for the basic-auth credentials, then loads the workspace.
3. Choose `tests/theis_cooper_jacob_sample.csv` from your local copy. Leave Q = 500 m³/day and r = 50 m, and run **Theis (1935)**. Expect **T ≈ 215.29 m²/day** and S ≈ 5.05×10⁻⁵.
4. Click **Download report**. A PDF downloads.
5. Open **Consultant** and ask *"What are the assumptions of the Theis method?"* The answer should list **Sources consulted**. If the answer says it lacks information, the vector store did not deploy.
6. In Step 1, press **Photograph of a table** and choose
   `tests/fixtures/field_data/field_sheet.png`. A grid of 13 readings opens with
   one cell marked uncertain. If instead it reports that Tesseract is missing,
   the image was built without the `apt-get` layer; nothing else is affected.

## Updating

```bash
cd ~/hydroassist
GIT_SSH_COMMAND="ssh -i ~/.ssh/hydroassist_deploy" git pull
docker compose up -d --build
```

Sessions live in memory. A redeploy ends open sessions, so anyone mid-analysis loses unsaved work; a saved project file reopens as normal.

## When the build or the site misbehaves

| What you see | What it means |
|---|---|
| `ERROR: No matching distribution found for langchain>=0.3.0` — or any package, with `(from versions: none)` | Not a dependency problem. `from versions: none` means pip reached no index at all, so it is the network. It happens when Docker's networking is not up yet, or the VM briefly has no route out. Wait a minute and run `docker compose up -d --build` again; do not edit `requirements.txt`. Check with `docker run --rm python:3.11-slim pip index versions pypi`. |
| Caddy logs `could not get certificate` or keeps retrying | Ports 80 and 443 are not open in **both** places — the VCN security list *and* the VM's own `iptables` (step 2). Let's Encrypt must reach the VM on 80. Also confirm the DuckDNS record points at this machine: `dig +short <your-domain>`. |
| The site resolves nowhere after a stop/start | Oracle gave the instance a new public IP. Update it at duckdns.org, or install the cron above. |
| `docker compose up` is killed partway through the build | The Ampere VM ran out of memory building the frontend. Give Docker swap: `sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile`. |
| The consultant answers "not enough information" for everything | The vector store did not reach the image. Check `data/vectordb/` exists in the clone; it is committed, so a shallow or partial clone is the usual cause. |
| The workspace loads but every action fails | `.env` is missing or unreadable. `docker compose logs app` shows it; the API needs `GEMINI_API_KEY` at start-up. |

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
