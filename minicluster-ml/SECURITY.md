# Security Policy

MiniCluster ML is built for trusted university lab networks. Recommended
hardening for anything beyond a closed lab:

1. Set `MINICLUSTER_CLUSTER_SECRET` so only your PCs can join as workers.
2. Set `MINICLUSTER_API_TOKEN` so only authorised users can submit jobs.
3. Put the dashboard behind TLS (see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)).
4. Firewall ports 8000/9000 to the lab subnet.

By design, workers never execute code received over the network: only a
fixed set of numeric task kinds is accepted and NumPy payloads are loaded
with `allow_pickle=False`.

## Reporting a vulnerability

Please open a private security advisory on GitHub ("Security" tab → "Report a
vulnerability") rather than a public issue. We aim to respond within 7 days.
