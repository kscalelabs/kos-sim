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
