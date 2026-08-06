# Exo Astrid store adapter

This optional extension links Exo directly to `astrid-storage`. It does not
start an Astrid daemon or cross a capsule/CLI boundary. Exo remains responsible
for model identities and MLX checkpoint serialization; Astrid supplies verified
content, durable KV metadata, convergence, recovery, and compaction.

Install it explicitly while the integration is experimental:

```bash
uv pip install ./integrations/astrid_store
```

The default Exo build does not compile or import Astrid.

Configure Exo with both variables:

```bash
EXO_ASTRID_STORE="$HOME/.local/share/exo/astrid"
EXO_ASTRID_RUNTIME_PROFILE="exact-model-and-runtime-identity"
```

The adapter deliberately exposes only verified named content and durable KV.
Exo owns MLX checkpoint serialization and semantic compatibility. The initial
integration uses KV as a replaceable lookup projection; a future shared
computation index can replace that projection without changing Exo's
`KVPrefixPersistence` protocol.

Publication happens after inference on a background worker. A crash can leave
unreferenced content or incomplete metadata, but lookup advertises only a
complete checkpoint. Storage failure is always a cache miss, never inference
failure.
