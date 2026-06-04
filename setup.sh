#!/bin/bash
echo "Setting up Newton + mesh tools..."
pip install "newton[examples]"
pip install tetgen pyvista
echo "Setup complete. Verifying..."
python -m newton.examples basic_pendulum --viewer null
echo "If you saw the RTX GPU detected above, you're good to go."
