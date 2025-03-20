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

The simulator will automatically check for and update the latest robot assets from the [kscale-assets](https://github.com/kscalelabs/kscale-assets) repository.

Then, in a separate terminal, run the example client:

```bash
python -m examples.kbot
```

You should see the simulated K-Bot move in response to the client commands.

## Development Setup

1. Clone the repository
```bash
git clone --recursive https://github.com/kscalelabs/kos-sim.git
```

Or if you've already cloned the repository, initialize the submodule:

```bash
git submodule update --init --recursive
```

2. Make sure you're using Python 3.11 or greater

```bash
python --version  # Should show Python 3.11 or greater
```

3. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

4. Run the simulator

```bash
kos-sim kbot-v2
```


## Possible Bugs

If you find that your robot is jittering on the ground, try increasing `iterations` and `ls_iterations` in your mjcf options.

```xml
<option iterations="6" ls_iterations="6">
</option>
```

Also, to clip actuator ctrl values, be sure to send a `configure_actuator` KOS command with `max_torque` set.

```python
await kos.actuator.configure_actuator(
    actuator_id=actuator.actuator_id,
    max_torque=actuator.max_torque,
)
```