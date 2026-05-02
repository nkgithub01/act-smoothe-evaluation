## Run docker environment:

```bash
docker.sh --compile
```

### The following commands should be executed in the docker environment
## Next build backend

```bash
build_backend.sh QKV
```

## Finally, run the backend to compile a kernel

```bash
compile_hlo.sh QKV
```

'QKV' may be replaced with any backend name.