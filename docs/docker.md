# Running in Docker

`elbi serve` is the usual way to run Elbi, and it needs nothing but the CLI. Docker is
for when you want it running continuously on a home server, a workstation you leave
on, or a VM you keep around, without a terminal open.

## Start it

```bash
curl -O https://raw.githubusercontent.com/Intelligible/elbi/main/docker-compose.yml
curl -o .env https://raw.githubusercontent.com/Intelligible/elbi/main/.env.example
docker compose up -d
```

Then open <http://localhost:7700>. No source tree and no build: the compose file pulls
the published image.

Set `LLM_API_KEY` in `.env` before starting, or add a model in Settings on first run.
Either works; the app is usable without a key until you send a message.

## What persists

Everything the app writes goes to a named volume mounted at `/data`: the database, the
warehouse, notebook state, and the derivation cache. `docker compose down` stops the app
and keeps that volume. `docker compose down -v` deletes it.

To back it up, copy the volume out while the app is stopped:

```bash
docker compose down
docker run --rm -v elbi_elbi_data:/data -v "$PWD:/backup" \
  alpine tar czf /backup/elbi-backup.tar.gz -C /data .
```

## Configuration worth knowing

`APP_SECRET_KEY` encrypts stored connection secrets at rest. Without it, adding a
connection that needs a password is refused rather than stored in the clear, so set it
before you connect anything:

```bash
openssl rand -hex 32
```

`DB_URI` is SQLite inside the volume by default. Point it at Postgres
(`postgres://user:pass@host:5432/elbi`) if you outgrow that; the schema is created on
first start either way.

`STORAGE_URI` is where synced warehouse tables land. A path inside the volume by
default, or `s3://`, `gs://`, `az://` for object storage.

`APP_PORT` changes the published port. The compose file binds it to `127.0.0.1`
deliberately: Elbi has no sign-in, so publishing it on every interface would put an
open app on your network. If you need it reachable from another machine, put it behind
something that authenticates, or use an SSH tunnel:

```bash
ssh -N -L 7700:127.0.0.1:7700 you@your-server
```

## Upgrading

Pin `APP_VERSION` to a release tag for anything you intend to keep, because `latest`
moves with `main`. To upgrade, change the tag and recreate:

```bash
docker compose pull && docker compose up -d
```

The schema migrates forward on start. Take a backup first if the data matters.

`elbi update` works inside the container too. The image sets `ELBI_INSTALL=docker`, so
it prints the compose command rather than a `pip` one that would not survive a restart.
[Upgrading](upgrading.md) covers the other install methods.
