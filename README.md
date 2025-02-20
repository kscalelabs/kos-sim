# kos-sim

`kos-sim` is a pure-simulation backend for the [K-Scale Operating System (KOS)](https://github.com/kscalelabs/kos), using the same gRPC interface.

## Installation

```bash
pip install kos-sim
```

## Getting Started

First, start the `kos-sim` backend:

```bash
kos-sim kbot-v1
```

Then, in a separate terminal, run the example client:

```bash
python -m examples.kbot
```

You should see the simulated K-Bot move in response to the client commands.
