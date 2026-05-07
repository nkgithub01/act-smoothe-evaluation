## Scripts

These scripts launch the ACT Docker image automatically when run from the host.
No separate `scripts/docker.sh` step is needed.

## Generate backend

```bash
./scripts/generate.sh QKV.py
```

## Build backend

```bash
./scripts/build_backend.sh QKV_improved
```

## Compile a kernel

```bash
./scripts/compile_hlo.sh QKV_improved attention.hlo
```

Backend names may be replaced with any generated target name.
