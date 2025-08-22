# /// script
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import io
    from ase.io import formats, read, write
    from ase.build import bulk

    from weas_widget.base_widget import BaseWidget
    from weas_widget.atoms_viewer import AtomsViewer
    from weas_widget.utils import ASEAdapter

    import tempfile
    from pathlib import Path

    import ast

    # List of supported ASE formats
    write_formats = sorted([fname for fname, ftype in formats.all_formats.items() if ftype.can_write])
    read_formats = sorted([fname for fname, ftype in formats.all_formats.items() if ftype.can_read])
    # I disabled the controls in the GUi, because the style is not loaded properly inside Marimo notebook
    guiConfig={"controls": {"enabled": False}}
    return (
        ASEAdapter,
        AtomsViewer,
        BaseWidget,
        Path,
        ast,
        bulk,
        formats,
        guiConfig,
        io,
        read,
        tempfile,
        write,
        write_formats,
    )


@app.cell
def _(mo):
    mo.md(
        """
    # ASE-based structure converter

    This is a quick test of how a marimo app could be deployed on GH pages to make ASE functionality available to a wider audience. It's not a robust or full-featured app - please do not use for anything important yet!

    It uses [ASE](https://wiki.fysik.dtu.dk/ase/) to read and write a variety of structure and trajectory file formats. See the [ASE documentation](https://wiki.fysik.dtu.dk/ase/ase/io/io.html#file-formats) for the full list of supported formats.
    """
    )
    return


@app.cell
def _(bulk, formats, io, write):
    # Example Atoms
    temp_atoms = bulk("Si")

    # --- Build a safe list of formats ---
    valid_write_formats = []
    for fname, ftype in formats.all_formats.items():
        if not ftype.can_write:
            continue
        try:
            # Try a dummy write to see if it supports file-like objects
            buf = io.StringIO()
            write(buf, temp_atoms, format=fname)
            valid_write_formats.append(fname)
        except Exception:
            # skip formats like gif that need a real filename
            pass
    return


@app.cell
def _(ASEAdapter, AtomsViewer, BaseWidget, guiConfig):
    def view_atoms(atoms, model_style=1, boundary = [[-0.1, 1.1], [-0.1, 1.1], [-0.1, 1.1]], show_bonded_atoms=True):
        v = AtomsViewer(BaseWidget(guiConfig=guiConfig))
        v.atoms = ASEAdapter.to_weas(atoms)
        v.model_style = model_style
        v.boundary = boundary
        v.show_bonded_atoms = show_bonded_atoms
        v.color_type = "VESTA"
        v.cell.settings['showAxes'] = True
        return v._widget

    return (view_atoms,)


@app.cell
def _(ast):
    def parse_kwargs_string(text: str):
        """
        Safely parse user input as a dict.
        Accepts Python-style dicts (single or double quotes, True/False)
        and JSON-style dicts.
        Returns {} if parsing fails.
        """
        text = text.strip()
        if not text:
            return {}
        try:
            # literal_eval is safe: no arbitrary code execution
            result = ast.literal_eval(text)
            if isinstance(result, dict):
                return result
            else:
                return {}
        except Exception:
            # fallback to empty dict if parsing fails
            return {}
    return (parse_kwargs_string,)


@app.cell
def _(mo, write_formats):
    # Create UI elements for file upload and format selection
    file_upload = mo.ui.file(label='Upload File', multiple=False)
    # input_format_dropdown = mo.ui.dropdown(options=all_formats, label='Input Format')
    output_format_dropdown = mo.ui.dropdown(options=write_formats, label='Output Format')


    # # Display UI elements in a vertical stack
    # mo.vstack([
    #     file_upload,
    #     input_kwargs_text,
    #     # input_format_dropdown,
    # ])

    in_form = (
        mo.md("""
        ## Read in structure
        ASE will try to guess the file format from the file extension/contents, but you can also specify it in the input kwargs (e.g., `'format': 'xyz'`).

        Input file:
        {file_upload}

        {input_kwargs_text}

        """)
        .batch(
            file_upload = mo.ui.file(label='Upload File', multiple=False),
            input_kwargs_text = mo.ui.text_area(label='Input Kwargs (as dict)', value="{'index': -1}"),
        ).form(submit_button_label="Load File")
    )




    return (in_form,)


@app.cell
def _():
    return


@app.cell
def _(in_form, mo, vis):
    mo.vstack([in_form, vis])
    return


@app.cell
def _(Path, read, tempfile):
    def process_new_file(structure_file, input_kwargs):
        """Process new structure/trajectory file"""
        suffix = f".{structure_file.name.split('.')[-1]}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(structure_file.contents)
            temp_path = Path(tmp.name)
        atoms = read(temp_path, **input_kwargs)
        temp_path.unlink()  # Delete the temporary file
        return atoms

    return (process_new_file,)


@app.cell
def _(in_form, parse_kwargs_string, process_new_file):
    atoms = None
    input_kwargs = parse_kwargs_string(in_form.value['input_kwargs_text']) if in_form.value else {}
    if in_form.value and in_form.value['file_upload']:
        atoms = process_new_file(in_form.value['file_upload'][0],input_kwargs=input_kwargs)
    atoms
    return (atoms,)


@app.cell
def _(mo):
    model_style = mo.ui.dropdown(options={'Ball': 0, 'Ball and Stick': 1, 'Polyhedral': 2, 'Stick': 3}, label='Model Style', value='Ball and Stick')
    show_bonded_atoms = mo.ui.checkbox(label='Show bonded atoms', value=True)
    # 

    return model_style, show_bonded_atoms


@app.cell
def _(atoms, mo, model_style, show_bonded_atoms, view_atoms):
    try:
        v = view_atoms(atoms, model_style=model_style.value, show_bonded_atoms=show_bonded_atoms.value) if atoms else mo.md("Upload a file to view the structure.")
    except Exception as e:
        v = mo.md(f"**Error displaying structure:** {e}")


    vis = mo.vstack([v, mo.hstack([model_style, show_bonded_atoms], justify='space-around', align='center')])
    return (vis,)


@app.cell
def _(io, write):
    def atoms_to_contents(atoms, fmt: str, **write_kwargs):
        """
        Try to write Atoms in text mode; fall back to binary if needed.
        Returns str (for text) or bytes (for binary).
        """

        def _handle_single_atom_error(error_msg: str, format_name: str) -> ValueError:
            """Handle the specific error when format can only store 1 Atoms object."""
            if "can only store 1 Atoms object" in error_msg:
                return ValueError(
                    f"Format '{format_name}' can only store 1 Atoms object. "
                    f"Try adding 'index': 0 to the write kwargs to select the first structure, "
                    f"or use a format that supports multiple structures (e.g., 'xyz', 'traj')."
                )
            return ValueError(f"Unable to write atoms in format '{format_name}': {error_msg}")

        def _try_write(buffer_type, buffer_class):
            """Attempt to write to the given buffer type."""
            try:
                buf = buffer_class()
                write(buf, atoms, format=fmt, **write_kwargs)
                return buf.getvalue()
            except (TypeError, UnicodeDecodeError):
                # These are buffer-type specific errors, let caller try other buffer type
                raise
            except ValueError as e:
                # Format-specific errors should be handled immediately
                raise _handle_single_atom_error(str(e), fmt) from e
            except Exception as e:
                # Any other unexpected error
                raise ValueError(f"Unexpected error writing {buffer_type} format '{fmt}': {str(e)}") from e

        # Try text mode first
        try:
            return _try_write("text", io.StringIO)
        except (TypeError, UnicodeDecodeError):
            # Fall back to binary mode for encoding issues
            pass
        except ValueError:
            # Format errors should not fall back to binary - re-raise immediately
            raise

        # Try binary mode
        return _try_write("binary", io.BytesIO)
    return (atoms_to_contents,)


@app.cell
def _(atoms, atoms_to_contents, mo, write_formats):
    fmt_dropdown = mo.ui.dropdown(options=write_formats, value="cif", label="Output format")

    def make_download(fmt_choice, file_prefix="converted_structure", write_kwargs={}):
        contents = atoms_to_contents(atoms, fmt_choice, **write_kwargs)
        return mo.download(contents, filename=f"{file_prefix}")


    output_kwargs_text = mo.ui.text_area(
        label="Output kwargs", 
        value="{}"
    )



    return fmt_dropdown, make_download, output_kwargs_text


@app.cell
def _(fmt_dropdown, mo):
    output_file_name = mo.ui.text(label='Output File Name', value=f"converted_file.{fmt_dropdown.value}")
    return (output_file_name,)


@app.cell
def _(
    atoms,
    fmt_dropdown,
    make_download,
    mo,
    output_file_name,
    output_kwargs_text,
    parse_kwargs_string,
):
    output_kwargs = parse_kwargs_string(output_kwargs_text.value)
    download_link = make_download(fmt_dropdown.value, file_prefix=output_file_name.value, write_kwargs=output_kwargs) if atoms else mo.md("Upload and load a file to enable download.")

    mo.vstack([
        mo.md("## Download Converted File"),
        fmt_dropdown, output_file_name, output_kwargs_text, download_link])
    return


if __name__ == "__main__":
    app.run()
