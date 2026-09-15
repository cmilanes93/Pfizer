"""Execute and save the exploration notebook from the project root."""

from pathlib import Path
import argparse
import base64
from io import BytesIO
import os
import nbformat
from nbclient import NotebookClient


def execute_in_process(notebook):
    """Run ordinary Python cells where the host disallows kernel sockets."""
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    from matplotlib.figure import Figure

    shell = InteractiveShell.instance()

    def figure_png(figure):
        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=110)
        return buffer.getvalue()

    shell.display_formatter.formatters["image/png"].for_type(Figure, figure_png)
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        count += 1
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=True)
        result.raise_error()
        cell.execution_count = count
        cell.outputs = []
        for stream in ["stdout", "stderr"]:
            if getattr(captured, stream):
                cell.outputs.append(nbformat.v4.new_output(
                    "stream", name=stream, text=getattr(captured, stream)))
        for output in captured.outputs:
            data = dict(output.data)
            for mime, value in data.items():
                if isinstance(value, bytes):
                    data[mime] = base64.b64encode(value).decode("ascii")
            cell.outputs.append(nbformat.v4.new_output(
                "display_data", data=data, metadata=output.metadata))
    notebook.metadata["execution_method"] = "IPython in process, sequential Python cells"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-process", action="store_true",
                        help="Run Python cells without starting a socket-based kernel")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks" / "01_exploration.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    if args.in_process:
        os.chdir(root)
        execute_in_process(notebook)
    else:
        NotebookClient(notebook, timeout=180, kernel_name="python3",
                       resources={"metadata": {"path": str(root)}}).execute()
        notebook.metadata["execution_method"] = "Jupyter kernel, nbclient"
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in code)
    assert not any(output.output_type == "error" for cell in code for output in cell.outputs)
    print(f"Executed {len(code)} code cells: {path}")


if __name__ == "__main__":
    main()
