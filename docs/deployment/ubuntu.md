# Ubuntu deployment and continued Codex development

This is the first-deployment checklist for the Gradio multi-agent UI. Deploy a
reviewed Git commit; do not copy the Windows working directory or its `.env` to
the server. Keep the server's `.env`, uploaded PDFs, and database volume outside
Git. Continue editing and testing in the local Codex checkout, then push a new
commit and update the server deliberately.

## 1. Confirm access and prerequisites

- Verify the server's SSH host-key fingerprint with its administrator before
  accepting it. Do not use `StrictHostKeyChecking=no` to bypass this check.
- Confirm that Ubuntu has Git, Docker with Compose, `uv`, and Python 3.11 or
  3.12. Check disk space before installing the `rag` extra and model weights.
- Use a private repository if the code should not be public. GitHub never
  needs the production `.env` or uploaded papers.

## 2. Install one fixed version

On the server, clone the GitHub repository into a directory owned by the
deployment user. If it is private, set up GitHub authentication without putting
tokens into a command or repository URL. From the repository root:

```bash
git clone https://github.com/subrosa1224/matAgent.git
cd matAgent
git rev-parse HEAD
uv sync --frozen --extra workflow --extra web-ui --extra literature --extra rag --extra analysis
```

Record the commit ID from `git rev-parse HEAD` for rollback and comparison with
the local Codex checkout. `rag` may download large embedding/reranking models
on first use; do not treat a successful dependency install as a completed
end-to-end test.

## 3. Configure secrets and the literature database

```bash
cp .env.example .env
chmod 600 .env
```

Edit `.env` locally on the server. Set at least `MP_API_KEY` for live Materials
Project queries. For the literature database, set a strong
`MATAGENT_POSTGRES_PASSWORD` and set `LITERATURE_DATABASE_URL` to
`postgresql://matagent:<same-password>@127.0.0.1:5432/matagent`. Percent-encode
special characters in the URL password. If using PDF ingestion, set
`LITERATURE_INGEST_ROOTS` to an existing, writable absolute server path. Set
`LLM_PROVIDER=intern`, `INTERN_BASE_URL`, `INTERN_MODEL`, and `INTERN_API_KEY`
only after validating the lab model endpoint; check `INTERN_THINKING_MODE`
compatibility there. Never paste these secrets into an issue or chat log.

```bash
docker compose --env-file .env -f deploy/literature/compose.yaml config --quiet
docker compose --env-file .env -f deploy/literature/compose.yaml up -d
docker compose --env-file .env -f deploy/literature/compose.yaml ps
uv run materials-screen literature migrate --yes
uv run materials-screen literature doctor
```

The database port is bound to server loopback only. The named Docker volume
contains persistent data: do not run `docker compose down -v` during normal
updates. Changing the password in `.env` after the database volume is first
initialized does not automatically change the existing database user's
password.

## 4. Smoke-test before making a persistent service

Run the UI on the server in a foreground shell:

```bash
uv run materials-screen master ui --port 8501
```

The UI binds to `127.0.0.1`, not the public interface. From your own computer,
after verifying the SSH host key and logging in successfully, open a separate
terminal and forward the port (replace the SSH endpoint if it changes):

```bash
ssh -N -L 8501:127.0.0.1:8501 -p 20322 ubuntu@221.239.50.147
```

Open `http://127.0.0.1:8501/` locally. If 8501 is occupied, the UI may choose
another port; read the server output and forward that exact port. Test one
question through database screening, literature search, and the user-confirmed
full-text stage. Record actual successes and partial failures separately. Do
not claim production readiness from CLI help or database health checks alone.

Only after the foreground smoke test passes, arrange a supervised service
(for example systemd) with the same working directory and environment. Keep
the UI bound to loopback behind SSH forwarding or an authenticated reverse
proxy; do not directly expose unauthenticated Gradio to the Internet.

## 5. Subsequent updates from Codex

In the local Codex checkout, review `git status` and tests, commit the intended
changes, and push to GitHub. On the server, record the currently deployed
commit, fetch, inspect the new commit, update to that reviewed commit, and run
`uv sync --frozen` with the same extras. Restart the service, repeat the smoke
test, and roll back to the recorded commit if necessary. Do not use the server
as the primary place for code edits, and do not overwrite its `.env`, uploads,
or database volume when updating code.
