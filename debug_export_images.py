import plotly.graph_objects as go
import io
import sys

print("Checking environment for image generation support...")

try:
    import kaleido
    print(f"Kaleido version: {kaleido.__version__}")
except ImportError:
    print("ERROR: kaleido is NOT installed. Images cannot be generated.")

try:
    fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
    print("Attempting to generate image bytes...")
    img_bytes = fig.to_image(format="png")
    print(f"Success! Generated {len(img_bytes)} bytes.")
except Exception as e:
    print(f"ERROR generating image: {e}")
    import traceback
    traceback.print_exc()

print("Done.")
