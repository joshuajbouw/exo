# Exo computation-store adapter

This optional extension implements Exo's generic computation-store seam. Its
current backend links directly to `astrid-storage`; the public Python API and
Exo integration do not carry that provider name. It does not start a daemon or
cross a capsule/CLI boundary. Exo remains responsible for model identities and
MLX checkpoint serialization; the backend supplies verified content, durable
KV metadata, convergence, and recovery.

Install it explicitly while the integration is experimental:

```bash
uv pip install ./integrations/computation_store
```

The default Exo build does not compile or import a computation-store backend.

Configure Exo with both variables:

```bash
EXO_COMPUTATION_STORE="$HOME/.local/share/exo/computation"
EXO_COMPUTATION_RUNTIME_PROFILE="exact-model-and-runtime-identity"
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

The adapter also maintains an evictable contiguous projection beside the
authoritative store. Admission computes an algorithm-tagged representation
digest during the existing input pass. A later process hashes the projection
against that digest before MLX opens it; a missing or modified projection is
reconstructed from verified content. Projection corruption can therefore lose
warmth, never computation correctness.

MLX checkpoints use exact logical-tensor deltas when a verified shorter
frontier is available. The delta pack stores only newly computed tensor tails;
every inherited range must byte-compare against the base or publication falls
back to a complete checkpoint. After a successor publishes, its reconstructible
ancestor projection is evicted, retaining one warm projection per lineage leaf.
Cold recovery verifies the base and every delta pack, rebuilds an ordinary MLX
safetensors checkpoint, and checks its complete digest before inference.
