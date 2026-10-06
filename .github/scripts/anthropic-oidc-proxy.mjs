// Forwards Claude API requests to api.anthropic.com with a Workload Identity
// Federation token it keeps fresh. Open Code Review reads its token once at
// startup, and a federated token expires after about ten minutes, so a long
// review would otherwise start failing partway through.
import http from 'node:http';
import https from 'node:https';
import { pipeline } from 'node:stream';

const UPSTREAM_HOST = 'api.anthropic.com';
const AUDIENCE = 'https://api.anthropic.com';
// Same advisory window the Anthropic SDKs use before expiry.
const REFRESH_MARGIN_MS = 120_000;
// A cached token this close to expiry is not worth sending when a refresh fails.
const MIN_REMAINING_MS = 10_000;
const TOKEN_FETCH_TIMEOUT_MS = 30_000;

function requireEnv(name) {
  const value = process.env[name];
  if (!value) throw new Error(`${name} is not set`);
  return value;
}

const config = {
  port: Number(requireEnv('ANTHROPIC_PROXY_PORT')),
  idTokenUrl: requireEnv('ACTIONS_ID_TOKEN_REQUEST_URL'),
  idTokenRequestToken: requireEnv('ACTIONS_ID_TOKEN_REQUEST_TOKEN'),
  federationRuleId: requireEnv('ANTHROPIC_FEDERATION_RULE_ID'),
  organizationId: requireEnv('ANTHROPIC_ORGANIZATION_ID'),
  serviceAccountId: requireEnv('ANTHROPIC_SERVICE_ACCOUNT_ID'),
};

function describe(err) {
  return err.cause ? `${err.message}: ${err.cause.message}` : err.message;
}

function log(message) {
  process.stderr.write(`[anthropic-oidc-proxy] ${new Date().toISOString()} ${message}\n`);
}

async function fetchGithubIdToken() {
  const url = `${config.idTokenUrl}&audience=${encodeURIComponent(AUDIENCE)}`;
  const res = await fetch(url, {
    headers: { Authorization: `Bearer ${config.idTokenRequestToken}` },
    signal: AbortSignal.timeout(TOKEN_FETCH_TIMEOUT_MS),
  });
  if (!res.ok) throw new Error(`GitHub OIDC token request failed: ${res.status} ${await res.text()}`);
  const { value } = await res.json();
  if (typeof value !== 'string' || !value) throw new Error('GitHub OIDC response has no token value');
  return value;
}

// Each exchange presents a newly minted GitHub token, because Anthropic
// rejects a reused jti.
async function exchange() {
  const res = await fetch(`https://${UPSTREAM_HOST}/v1/oauth/token`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      grant_type: 'urn:ietf:params:oauth:grant-type:jwt-bearer',
      assertion: await fetchGithubIdToken(),
      federation_rule_id: config.federationRuleId,
      organization_id: config.organizationId,
      service_account_id: config.serviceAccountId,
    }),
    signal: AbortSignal.timeout(TOKEN_FETCH_TIMEOUT_MS),
  });
  const body = await res.text();
  if (!res.ok) throw new Error(`Anthropic token exchange failed: ${res.status} ${body}`);
  const { access_token: token, expires_in: expiresIn } = JSON.parse(body);
  if (typeof token !== 'string' || !token || !Number.isFinite(expiresIn) || expiresIn <= 0) {
    throw new Error('Anthropic token exchange response has no access_token or expires_in');
  }
  log(`minted a token that expires in ${expiresIn}s`);
  return { token, expiresAt: Date.now() + expiresIn * 1000 };
}

let cached = null;
let pending = null;

function getToken() {
  const remaining = cached ? cached.expiresAt - Date.now() : 0;
  if (remaining > REFRESH_MARGIN_MS) return Promise.resolve(cached.token);

  pending ??= exchange()
    .then((fresh) => {
      cached = fresh;
      return fresh.token;
    })
    .finally(() => {
      pending = null;
    });
  if (remaining <= MIN_REMAINING_MS) return pending;

  // The current token is still good for a while, so a failed refresh should
  // not fail the request. The next request past the margin tries again.
  const stillValid = cached.token;
  return pending.catch((err) => {
    log(`refresh failed, using the cached token: ${describe(err)}`);
    return stillValid;
  });
}

function fail(res, err) {
  log(describe(err));
  if (res.headersSent) {
    res.destroy();
    return;
  }
  res.writeHead(502, { 'content-type': 'application/json' });
  res.end(JSON.stringify({ type: 'error', error: { type: 'api_error', message: describe(err) } }));
}

const server = http.createServer(async (req, res) => {
  if (req.url === '/healthz') {
    res.writeHead(200).end('ok');
    return;
  }

  let token;
  try {
    token = await getToken();
  } catch (err) {
    fail(res, err);
    return;
  }

  const headers = { ...req.headers, host: UPSTREAM_HOST, authorization: `Bearer ${token}` };
  delete headers['x-api-key'];

  const upstream = https.request({ host: UPSTREAM_HOST, method: req.method, path: req.url, headers }, (upstreamRes) => {
    res.writeHead(upstreamRes.statusCode ?? 502, upstreamRes.headers);
    pipeline(upstreamRes, res, (err) => {
      if (err) log(`response stream failed: ${describe(err)}`);
    });
  });
  upstream.on('error', (err) => fail(res, err));
  // Stop generation upstream (and its billing) when the client goes away first.
  res.on('close', () => {
    if (!res.writableFinished) upstream.destroy();
  });
  pipeline(req, upstream, (err) => {
    if (err) fail(res, err);
  });
});

try {
  await getToken();
} catch (err) {
  log(describe(err));
  process.exit(1);
}
server.listen(config.port, '127.0.0.1', () => log(`listening on 127.0.0.1:${config.port}`));
