```bash
docker build -t <tag> --build-arg BUILDKIT_INLINE_CACHE=1 --build-arg CUDA_ARCH="7.0 7.5 8.6" .
```
