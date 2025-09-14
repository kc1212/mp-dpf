# mp-dpf

## Benchmark

### Linux

Make sure flamegraph is installed: `cargo install flamegraph`.

### macOS

Make sure samply is installed: `cargo install samply`.

```bash
samply record target/release/deps/my_benchmark-b988e4327dc7f9b6 'half niote bob encode' --bench --profile-time 60
```

You can find the executable name when running the benchmark normally.
