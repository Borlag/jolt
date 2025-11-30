import sys
import traceback

with open("error_log.txt", "w", encoding="utf-8") as f:
    print("Checking imports...")
    try:
        import xlsxwriter
        f.write("xlsxwriter: OK\n")
    except ImportError as e:
        f.write(f"xlsxwriter: FAILED ({e})\n")

    try:
        import kaleido
        f.write("kaleido: OK\n")
    except ImportError as e:
        f.write(f"kaleido: FAILED ({e})\n")

    try:
        import plotly
        f.write("plotly: OK\n")
    except ImportError as e:
        f.write(f"plotly: FAILED ({e})\n")

    try:
        import streamlit
        f.write("streamlit: OK\n")
    except ImportError as e:
        f.write(f"streamlit: FAILED ({e})\n")

    try:
        import jolt.export
        f.write("jolt.export: OK\n")
    except Exception as e:
        f.write(f"jolt.export: FAILED\n")
        traceback.print_exc(file=f)

    try:
        import jolt.ui.export_ui
        f.write("jolt.ui.export_ui: OK\n")
    except Exception as e:
        f.write(f"jolt.ui.export_ui: FAILED\n")
        traceback.print_exc(file=f)
