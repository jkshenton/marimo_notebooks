# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "castep-outputs==0.2.0",
#     "pandas",
#     "numpy",
#     "ase",
#     "altair<6.0.0",
#     "weas-widget==0.1.26",
# ]
# ///
"""
CASTEP Phonon Visualisation Dashboard

Visualise phonon modes from CASTEP .phonon files using weas-widget.

Usage:
  marimo run castep_phonon_viewer.py --sandbox
"""

import marimo

__generated_with = "0.18.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # CASTEP Phonon Mode Viewer

    Upload a CASTEP `.phonon` file to visualise phonon modes.

    - Select a **q-point** and **mode** to visualise
    - Adjust **amplitude** and **animation speed** for better visualisation
    """)
    return


@app.cell
def _():
    import castep_outputs as co
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from ase import Atoms
    import tempfile
    return Atoms, Path, co, np, pd, tempfile


@app.cell
def _():
    from weas_widget.base_widget import BaseWidget
    from weas_widget.atoms_viewer import AtomsViewer
    from weas_widget.utils import ASEAdapter
    return ASEAdapter, AtomsViewer, BaseWidget


@app.cell
def _(mo):
    file_upload = mo.ui.file(
        label="Upload CASTEP phonon file (.phonon)",
        filetypes=[".phonon"],
        multiple=False
    )
    return (file_upload,)


@app.cell
def _(file_upload, mo):
    mo.vstack([
        mo.md("## Upload File"),
        file_upload,
        mo.md(f"✅ Loaded: **{file_upload.value[0].name}**") if file_upload.value else mo.md("_Upload a `.phonon` file to begin_")
    ])
    return


@app.cell
def _(Atoms, Path, co, file_upload, np, tempfile):
    def parse_phonon_file(uploaded_file):
        """Parse uploaded .phonon file using castep_outputs.

        Returns dict with:
          - atoms: ASE Atoms object
          - qpts: list of q-point data (qpt, weight, eigenvalues, eigenvectors, ir_intensity)
          - n_ions: number of ions
          - n_branches: number of branches (3 * n_ions)
        """
        if uploaded_file is None:
            return None

        with tempfile.NamedTemporaryFile(suffix=".phonon", delete=False) as tmp:
            tmp.write(uploaded_file.contents)
            temp_path = Path(tmp.name)

        try:
            data = co.parse_single(temp_path)

            # Build ASE Atoms from parsed data
            cell = np.array(data['unit_cell'])
            coords = data['coords']
            scaled_positions = np.array([coords['u'], coords['v'], coords['w']]).T
            symbols = coords['spec']
            masses = coords['mass']

            atoms = Atoms(
                symbols=symbols,
                scaled_positions=scaled_positions,
                cell=cell,
                pbc=True
            )
            atoms.set_masses(masses)

            # Process q-points
            # CASTEP outputs q-points with optional direction vector for LO-TO splitting
            # at Gamma. We keep all q-points and store the direction if present.
            qpts = []
            for qpt_data in data['qpts']:
                qpt = qpt_data['qpt']
                weight = qpt_data['weight']
                direction = qpt_data.get('dir', None)  # LO-TO direction if present
                eigenvalues = qpt_data['eigenvalues']  # frequencies in cm-1
                ir_intensity = qpt_data.get('ir_intensity', [])

                # Eigenvectors: list of (n_ions * n_branches) tuples of (x, y, z) complex
                # Reshape to (n_branches, n_ions, 3) complex array
                raw_evecs = qpt_data['eigenvectors']
                n_ions = data['ions']
                n_branches = data['branches']

                # Each mode has n_ions eigenvector entries
                eigenvectors = []
                for mode_idx in range(n_branches):
                    mode_evecs = []
                    for ion_idx in range(n_ions):
                        evec_idx = mode_idx * n_ions + ion_idx
                        evec = raw_evecs[evec_idx]  # (x, y, z) complex tuple
                        mode_evecs.append([complex(evec[0]), complex(evec[1]), complex(evec[2])])
                    eigenvectors.append(mode_evecs)

                eigenvectors = np.array(eigenvectors)  # (n_branches, n_ions, 3) complex

                qpts.append({
                    'qpt': qpt,
                    'weight': weight,
                    'direction': direction,
                    'eigenvalues': eigenvalues,
                    'eigenvectors': eigenvectors,
                    'ir_intensity': ir_intensity
                })

            return {
                'atoms': atoms,
                'qpts': qpts,
                'n_ions': data['ions'],
                'n_branches': data['branches'],
                'filename': uploaded_file.name
            }
        finally:
            temp_path.unlink()

    phonon_data = parse_phonon_file(file_upload.value[0]) if file_upload.value else None
    return (phonon_data,)


@app.cell
def _(mo, phonon_data):
    # Q-point selector
    if phonon_data:
        def _format_qpt(i, q):
            qpt_str = f"({q['qpt'][0]:.3f}, {q['qpt'][1]:.3f}, {q['qpt'][2]:.3f})"
            if q['direction']:
                d = q['direction']
                return f"q{i+1}: {qpt_str} dir=[{d[0]:.2f},{d[1]:.2f},{d[2]:.2f}]"
            else:
                return f"q{i+1}: {qpt_str} [w={q['weight']:.4f}]"
        
        qpt_options = {
            _format_qpt(i, q): i 
            for i, q in enumerate(phonon_data['qpts'])
        }
        qpt_selector = mo.ui.dropdown(
            options=qpt_options,
            value=list(qpt_options.keys())[0],
            label="Q-point"
        )
    else:
        qpt_selector = None
    return (qpt_selector,)


@app.cell
def _(mo):
    # State to remember mode index across q-point changes
    get_mode_idx, set_mode_idx = mo.state(0)
    return get_mode_idx, set_mode_idx


@app.cell
def _(get_mode_idx, mo, phonon_data, qpt_selector, set_mode_idx):
    # Mode selector - depends on selected q-point
    if phonon_data and qpt_selector:
        _qpt_idx = qpt_selector.value
        _qpt_data = phonon_data['qpts'][_qpt_idx]
        eigenvalues = _qpt_data['eigenvalues']
        _ir_intensities = _qpt_data.get('ir_intensity', [])

        _mode_options = {}
        for i, _freq in enumerate(eigenvalues):
            _ir_str = f", IR: {_ir_intensities[i]:.2f}" if _ir_intensities and i < len(_ir_intensities) else ""
            _mode_options[f"Mode {i+1}: {_freq:.2f} cm⁻¹{_ir_str}"] = i

        # Clamp remembered mode index to available modes
        _remembered_idx = min(get_mode_idx(), len(eigenvalues) - 1)
        _mode_keys = list(_mode_options.keys())
        _initial_value = _mode_keys[_remembered_idx]

        mode_selector = mo.ui.dropdown(
            options=_mode_options,
            value=_initial_value,
            label="Phonon Mode",
            on_change=lambda val: set_mode_idx(val)
        )
    else:
        mode_selector = None
        eigenvalues = []
    return (mode_selector,)


@app.cell
def _(mo):
    # Visualization controls
    amplitude_slider = mo.ui.slider(
        start=0.2, stop=5.0, step=0.2, value=1.0,
        label="Amplitude"
    )
    arrow_scale = mo.ui.slider(
        start=0.0, stop=2.0, step=0.1, value=1.0,
        label="Arrow scale"
    )
    static_arrows = mo.ui.checkbox(
        value=False,
        label="Static (at max amplitude)"
    )
    n_frames = mo.ui.slider(
        start=10, stop=60, step=5, value=20,
        label="Animation frames"
    )
    repeat_x = mo.ui.number(value=2, start=1, stop=6, step=1, label="Repeat X")
    repeat_y = mo.ui.number(value=2, start=1, stop=6, step=1, label="Repeat Y")
    repeat_z = mo.ui.number(value=2, start=1, stop=6, step=1, label="Repeat Z")
    arrow_color = mo.ui.dropdown(
        options=["blue", "red", "green", "orange", "purple", "cyan"],
        value="blue",
        label="Arrow color"
    )
    return (
        amplitude_slider,
        arrow_color,
        arrow_scale,
        n_frames,
        repeat_x,
        repeat_y,
        repeat_z,
        static_arrows,
    )


@app.cell
def _(
    amplitude_slider,
    arrow_color,
    arrow_scale,
    mo,
    mode_selector,
    n_frames,
    phonon_data,
    qpt_selector,
    repeat_x,
    repeat_y,
    repeat_z,
    static_arrows,
):
    # Display controls
    if phonon_data:
        mo.output.append(mo.vstack([
            mo.md("## Phonon Mode Selection"),
            mo.hstack([qpt_selector, mode_selector], justify='start', gap=2),
            mo.md("## Visualisation Controls"),
            mo.hstack([amplitude_slider, arrow_scale, static_arrows], justify='start', gap=2),
            mo.hstack([repeat_x, repeat_y, repeat_z, arrow_color] + ([] if static_arrows.value else [n_frames]), justify='start', gap=2),
        ]))
    return


@app.cell
def _(
    ASEAdapter,
    AtomsViewer,
    BaseWidget,
    amplitude_slider,
    arrow_color,
    arrow_scale,
    mo,
    mode_selector,
    n_frames,
    np,
    phonon_data,
    qpt_selector,
    repeat_x,
    repeat_y,
    repeat_z,
    static_arrows,
):
    # Phonon visualisation
    if phonon_data and qpt_selector and mode_selector:
        _qpt_idx = qpt_selector.value
        _mode_idx = mode_selector.value

        _qpt_data = phonon_data['qpts'][_qpt_idx]
        _eigenvector = _qpt_data['eigenvectors'][_mode_idx]  # (n_ions, 3) complex
        _frequency = _qpt_data['eigenvalues'][_mode_idx]
        _qpt = _qpt_data['qpt']

        # Convert eigenvector to format expected by weas-widget
        # Shape should be (n_ions, 3, 2) where last dim is [real, imag]
        _evec_for_weas = np.stack([_eigenvector.real, _eigenvector.imag], axis=-1)

        # Build phonon settings
        # For static arrows: nframes=1 freezes at max amplitude
        # Using smaller radius for long thin arrows that overlay better with atoms
        _phonon_setting = {
            "eigenvectors": _evec_for_weas,
            "kpoint": list(_qpt),
            "amplitude": amplitude_slider.value,
            "factor": arrow_scale.value,
            "nframes": 1 if static_arrows.value else int(n_frames.value),
            "repeat": [int(repeat_x.value), int(repeat_y.value), int(repeat_z.value)],
            "color": arrow_color.value,
            "radius": 0.05,
        }

        # Create viewer using AtomsViewer with BaseWidget (marimo compatible)
        _v = AtomsViewer(BaseWidget(guiConfig={"controls": {"enabled": True}}))
        _v.atoms = ASEAdapter.to_weas(phonon_data['atoms'])
        _v.model_style = 1  # Ball and Stick
        _v.boundary = [[-0.1, 1.1], [-0.1, 1.1], [-0.1, 1.1]]
        _v.color_type = "VESTA"
        _v.phonon_setting = _phonon_setting

        mo.output.append(mo.vstack([
            mo.md(f"### Mode {_mode_idx + 1}: {_frequency:.2f} cm⁻¹"),
            mo.md(f"Q-point: ({_qpt[0]:.3f}, {_qpt[1]:.3f}, {_qpt[2]:.3f})"),
            _v._widget,
        ]))
    else:
        mo.output.append(mo.md("_Upload a phonon file to visualise modes_"))
    return


@app.cell
def _(mo, pd, phonon_data):
    # Frequency table
    if phonon_data:
        _rows = []
        for _qpt_idx, _qpt_data in enumerate(phonon_data['qpts']):
            _qpt = _qpt_data['qpt']
            _weight = _qpt_data['weight']
            _direction = _qpt_data['direction']
            _ir_data = _qpt_data['ir_intensity']
            
            # Format direction if present
            if _direction:
                _dir_str = f"[{_direction[0]:.2f}, {_direction[1]:.2f}, {_direction[2]:.2f}]"
            else:
                _dir_str = '-'
            
            for _mode_idx, _freq in enumerate(_qpt_data['eigenvalues']):
                _rows.append({
                    'Q-point': f"({_qpt[0]:.3f}, {_qpt[1]:.3f}, {_qpt[2]:.3f})",
                    'Direction': _dir_str,
                    'Weight': _weight,
                    'Mode': _mode_idx + 1,
                    'Frequency (cm⁻¹)': _freq,
                    'IR Intensity': _ir_data[_mode_idx] if _ir_data else 'N/A',
                })

        _freq_df = pd.DataFrame(_rows)

        mo.output.append(mo.vstack([
            mo.md("## Frequency Summary"),
            mo.ui.table(_freq_df)
        ]))
    return


if __name__ == "__main__":
    app.run()
