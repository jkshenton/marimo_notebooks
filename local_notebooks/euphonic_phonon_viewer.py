# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "euphonic>=1.3.0",
#     "seekpath",
#     "pandas",
#     "numpy",
#     "altair<6.0.0",
# ]
# ///
"""
Euphonic Phonon Dispersion and DOS Viewer

*Work in progress - please use the Github issues page to suggest improvements*

https://github.com/jkshenton/marimo_notebooks/issues


Visualise phonon dispersion bands and density of states from force constants
using the Euphonic library.

Supported file formats:
  - CASTEP: .castep_bin or .check files
  - Phonopy: phonopy.yaml (with force constants)

Features:
  - Phonon dispersion along high-symmetry paths (using seekpath)
  - Density of states on Monkhorst-Pack grids
  - Partial DOS by species
  - Neutron-weighted DOS (coherent/incoherent scattering cross-sections)
  - Adaptive broadening using mode gradients
  - Interactive combined dispersion + DOS view with linked zoom

Usage:
  marimo run euphonic_phonon_viewer.py --sandbox
"""

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Euphonic Phonon Viewer

    Upload force constants to visualise phonon properties:

    **Supported formats:**
    - CASTEP: `.castep_bin` or `.check` files
    - Phonopy: `phonopy.yaml` (with force constants)

    **Calculations:**
    - **Phonon Dispersion** along high-symmetry paths (using seekpath)
    - **Density of States** on a Monkhorst-Pack grid
    - **Neutron-weighted (P)DOS** with coherent/incoherent scattering cross-sections

    Configure parameters for both calculations using the controls below.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import altair as alt
    from pathlib import Path
    import tempfile
    import warnings
    # Suppress spglib deprecation warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='spglib')
    return Path, alt, np, pd, tempfile


@app.cell
def _():
    from euphonic import ForceConstants, ureg
    from euphonic.util import mp_grid, mode_gradients_to_widths
    import seekpath

    return ForceConstants, mode_gradients_to_widths, mp_grid, seekpath, ureg


@app.cell
def _(mo):
    file_upload = mo.ui.file(
        label="Upload force constants file",
        filetypes=[".castep_bin", ".check", ".yaml"],
        multiple=False
    )
    return (file_upload,)


@app.cell
def _(file_upload, mo):
    mo.vstack([
        mo.md("## Upload File"),
        file_upload,
        mo.md(f"✅ Loaded: **{file_upload.value[0].name}**") if file_upload.value else mo.md("_Upload a `.castep_bin` or `.check` file to begin_")
    ])
    return


@app.cell
def _(ForceConstants, Path, file_upload, mo, tempfile):
    def load_force_constants(uploaded_file):
        """Load force constants from uploaded file (CASTEP or Phonopy)."""
        if uploaded_file is None:
            return None, None

        # Determine suffix from filename
        suffix = Path(uploaded_file.name).suffix.lower()

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded_file.contents)
            temp_path = Path(tmp.name)

        try:
            if suffix == '.yaml':
                # Phonopy format
                fc = ForceConstants.from_phonopy(summary_name=str(temp_path))
            else:
                # CASTEP format (.castep_bin or .check)
                fc = ForceConstants.from_castep(str(temp_path))
            return fc, None
        except RuntimeError as e:
            # Handle missing force constants
            error_msg = str(e)
            if "Force constants matrix could not be found" in error_msg:
                return None, mo.callout(
                    mo.md(f"""
    **Force constants not found in file**

    The uploaded file `{uploaded_file.name}` does not contain a force constants matrix.

    **To fix this, ensure your CASTEP calculation used:**
    ```
    PHONON_WRITE_FORCE_CONSTANTS : true
    PHONON_FINE_METHOD : interpolate  # or supercell
    ```

                    """),
                    kind="warn"
                )
            else:
                return None, mo.callout(
                    mo.md(f"**Error loading file:** {error_msg}"),
                    kind="danger"
                )
        except Exception as e:
            return None, mo.callout(
                mo.md(f"**Unexpected error:** {type(e).__name__}: {e}"),
                kind="danger"
            )
        finally:
            temp_path.unlink()

    _result = load_force_constants(file_upload.value[0]) if file_upload.value else (None, None)
    fc = _result[0]
    fc_error = _result[1]
    return fc, fc_error


@app.cell
def _(fc, fc_error, get_fc_summary, mo):
    # Display crystal info or error
    if fc_error:
        mo.output.append(fc_error)
    elif fc:
        crystal = fc.crystal
        n_atoms = crystal.n_atoms
        species = list(set(crystal.atom_type))
        cell_params = crystal.cell_vectors.magnitude

        mo.output.append(mo.vstack([
            mo.md(f"""## Crystal Information
            - **Atoms:** {n_atoms}
            - **Species:** {', '.join(species)}
            - **Cell vectors shape:** {cell_params.shape}
            """),
            mo.md(get_fc_summary(fc))
        ]))
    return


@app.cell
def _(fc, mo):
    # Global energy unit control
    energy_unit = mo.ui.dropdown(
        options=["meV", "THz", "1/cm"],
        value="meV",
        label="Energy unit"
    )
    if fc:
        mo.output.append(mo.hstack([
            mo.md("**Settings:**"),
            energy_unit
        ], justify='start', gap=2))
    return (energy_unit,)


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Dispersion Settings
    """)
    return


@app.cell
def _(mo):
    # Dispersion controls
    disp_q_spacing = mo.ui.slider(
        start=0.01, stop=0.1, step=0.005, value=0.025,
        label="q-spacing (1/Å)"
    )
    disp_asr = mo.ui.dropdown(
        options={"None": None, "Reciprocal": "reciprocal", "Realspace": "realspace"},
        value="Reciprocal",
        label="Acoustic Sum Rule"
    )
    disp_reorder = mo.ui.checkbox(
        value=False,
        label="Reorder bands (follow modes)"
    )
    disp_e_min = mo.ui.number(
        value=0.0, start=-100, stop=1000, step=1,
        label="E min"
    )
    disp_e_max = mo.ui.number(
        value=0.0, start=0, stop=2000, step=10,
        label="E max (0=auto)"
    )
    return disp_asr, disp_e_max, disp_e_min, disp_q_spacing, disp_reorder


@app.cell
def _(disp_asr, disp_e_max, disp_e_min, disp_q_spacing, disp_reorder, fc, mo):
    if fc:
        mo.output.append(mo.vstack([
            mo.hstack([disp_q_spacing, disp_asr], justify='start', gap=2),
            mo.hstack([disp_e_min, disp_e_max, disp_reorder], justify='start', gap=2),
        ]))
    return


@app.cell
def _(disp_asr, disp_q_spacing, disp_reorder, fc, np, seekpath):
    def calculate_dispersion(fc, q_spacing, asr, reorder):
        """Calculate phonon dispersion along high-symmetry path."""
        if fc is None:
            return None

        # Get high-symmetry path from seekpath
        cell = fc.crystal.to_spglib_cell()
        path_data = seekpath.get_explicit_k_path(cell, reference_distance=q_spacing)

        qpts = np.array(path_data["explicit_kpoints_rel"])

        # Calculate phonon modes.
        # Note: insert_gamma=False (default) is correct here because seekpath's
        # get_explicit_k_path already places Gamma at all segment endpoints.
        # Using insert_gamma=True would add duplicate Gamma points and shift
        # the q-point indices, misaligning the high-symmetry point labels.
        phonons = fc.calculate_qpoint_phonon_modes(
            qpts,
            asr=asr,
            reduce_qpts=True,
        )

        if reorder:
            phonons.reorder_frequencies()

        # Extract path labels and positions
        labels = path_data["explicit_kpoints_labels"]
        label_positions = []
        label_names = []
        for i, label in enumerate(labels):
            if label:
                label_positions.append(i)
                # Replace GAMMA with Γ
                label_names.append("Γ" if label == "GAMMA" else label)

        return {
            'phonons': phonons,
            'qpts': qpts,
            'labels': label_names,
            'label_positions': label_positions,
            'path_data': path_data
        }

    dispersion_data = calculate_dispersion(fc, disp_q_spacing.value, disp_asr.value, disp_reorder.value)
    return (dispersion_data,)


@app.cell
def _(
    alt,
    disp_e_max,
    disp_e_min,
    disp_reorder,
    dispersion_data,
    energy_unit,
    mo,
    pd,
):
    def plot_dispersion(data, energy_unit, e_min, e_max, reorder):
        """Create Altair dispersion plot."""
        if data is None:
            return None

        phonons = data['phonons']
        labels = data['labels']
        label_positions = data['label_positions']

        # Get frequencies in chosen unit
        frequencies = phonons.frequencies.to(energy_unit).magnitude
        n_qpts, n_branches = frequencies.shape

        # Build dataframe for plotting
        rows = []
        for q_idx in range(n_qpts):
            for branch_idx in range(n_branches):
                rows.append({
                    'q_index': q_idx,
                    'branch': branch_idx,
                    'frequency': frequencies[q_idx, branch_idx]
                })

        df = pd.DataFrame(rows)

        # Determine y-axis range
        y_min = e_min if e_min != 0 else df['frequency'].min() - 5
        y_max = e_max if e_max != 0 else df['frequency'].max() * 1.05

        # Build JavaScript object literal for label lookup
        label_lookup_js = "{" + ", ".join(f"{pos}: '{lbl}'" for pos, lbl in zip(label_positions, labels)) + "}"

        # Create main dispersion plot with custom x-axis tick labels
        dispersion_chart = alt.Chart(df).mark_line(
            strokeWidth=1.5,
            opacity=0.8
        ).encode(
            x=alt.X('q_index:Q', 
                    title='',
                    scale=alt.Scale(domain=[min(label_positions), max(label_positions)]),
                    axis=alt.Axis(
                        values=label_positions,
                        labelExpr=f"{label_lookup_js}[datum.value] || ''",
                        labelFontSize=11,
                        labelFontWeight='bold',
                        labelOverlap=True,
                        labelFlush=False
                    )),
            y=alt.Y('frequency:Q', 
                    title=f'Energy ({energy_unit})',
                    scale=alt.Scale(domain=[y_min, y_max])),
            color=alt.Color('branch:N', legend=None),
            detail='branch:N'
        ).properties(
            width=600,
            height=400,
            title='Phonon Dispersion'
        )

        # Add vertical lines at high-symmetry points
        label_df = pd.DataFrame({
            'q_index': label_positions,
            'label': labels
        })

        vlines = alt.Chart(label_df).mark_rule(
            opacity=0.3
        ).encode(
            x='q_index:Q'
        )

        return alt.layer(dispersion_chart, vlines).properties(
            width=600,
            height=400
        ).interactive()

    if dispersion_data:
        _disp_chart = plot_dispersion(
            dispersion_data, 
            energy_unit.value,
            disp_e_min.value,
            disp_e_max.value,
            disp_reorder.value
        )
        mo.output.append(mo.vstack([
            mo.md("## Phonon Dispersion"),
            _disp_chart if _disp_chart else mo.md("_No dispersion data_")
        ]))
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Density of States Settings
    """)
    return


@app.cell
def _(mo):
    # DOS controls
    dos_grid_x = mo.ui.number(value=10, start=1, stop=50, step=1, label="Grid X")
    dos_grid_y = mo.ui.number(value=10, start=1, stop=50, step=1, label="Grid Y")
    dos_grid_z = mo.ui.number(value=10, start=1, stop=50, step=1, label="Grid Z")
    dos_ebins = mo.ui.slider(
        start=50, stop=500, step=10, value=200,
        label="Energy bins"
    )
    dos_asr = mo.ui.dropdown(
        options={"None": None, "Reciprocal": "reciprocal", "Realspace": "realspace"},
        value="Reciprocal",
        label="Acoustic Sum Rule"
    )
    return dos_asr, dos_ebins, dos_grid_x, dos_grid_y, dos_grid_z


@app.cell
def _(mo):
    # Broadening controls
    dos_adaptive = mo.ui.checkbox(
        value=False,
        label="Adaptive broadening"
    )
    dos_adaptive_method = mo.ui.dropdown(
        options=["reference", "fast"],
        value="fast",
        label="Adaptive method"
    )
    dos_energy_broadening = mo.ui.slider(
        start=0.0, stop=10.0, step=0.1, value=1.0,
        label="Energy broadening"
    )
    dos_shape = mo.ui.dropdown(
        options=["gauss", "lorentz"],
        value="gauss",
        label="Broadening shape"
    )
    return dos_adaptive, dos_adaptive_method, dos_energy_broadening, dos_shape


@app.cell
def _(mo):
    # PDOS and weighting controls
    dos_pdos = mo.ui.checkbox(
        value=False,
        label="Show PDOS by species"
    )
    dos_weighting = mo.ui.dropdown(
        options={
            "DOS": "dos",
            "Coherent neutron": "coherent",
            "Incoherent neutron": "incoherent",
            "Total neutron": "coherent-plus-incoherent"
        },
        value="DOS",
        label="Weighting"
    )
    return dos_pdos, dos_weighting


@app.cell
def _(
    dos_adaptive,
    dos_adaptive_method,
    dos_asr,
    dos_ebins,
    dos_energy_broadening,
    dos_grid_x,
    dos_grid_y,
    dos_grid_z,
    dos_pdos,
    dos_shape,
    dos_weighting,
    fc,
    mo,
):
    if fc:
        # Grey out fixed broadening controls when adaptive is selected
        if dos_adaptive.value:
            _broadening_controls = mo.hstack([
                dos_adaptive, 
                dos_adaptive_method,
                mo.md(f"~~Energy broadening: {dos_energy_broadening.value}~~").style({"opacity": "0.4"}),
                mo.md(f"~~Shape: {dos_shape.value}~~").style({"opacity": "0.4"})
            ], justify='start', gap=2)
        else:
            _broadening_controls = mo.hstack([
                dos_adaptive, 
                mo.md("_Adaptive method_").style({"opacity": "0.4"}),
                dos_energy_broadening, 
                dos_shape
            ], justify='start', gap=2)

        mo.output.append(mo.vstack([
            mo.md("### Grid & Energy"),
            mo.hstack([dos_grid_x, dos_grid_y, dos_grid_z, dos_ebins], justify='start', gap=2),
            mo.hstack([dos_asr], justify='start', gap=2),
            mo.md("### Broadening"),
            _broadening_controls,
            mo.md("### Output"),
            mo.hstack([dos_pdos, dos_weighting], justify='start', gap=2),
        ]))
    return


@app.cell
def _(
    dos_adaptive,
    dos_adaptive_method,
    dos_asr,
    dos_ebins,
    dos_energy_broadening,
    dos_grid_x,
    dos_grid_y,
    dos_grid_z,
    dos_pdos,
    dos_shape,
    dos_weighting,
    energy_unit,
    fc,
    mode_gradients_to_widths,
    mp_grid,
    np,
    ureg,
):
    def calculate_dos(fc, grid, energy_unit, n_bins, asr, adaptive, adaptive_method, 
                      broadening, shape, pdos, weighting):
        """Calculate DOS on Monkhorst-Pack grid.

        Args:
            fc: ForceConstants object
            grid: [nx, ny, nz] Monkhorst-Pack grid dimensions
            energy_unit: Unit for energy ('meV', 'THz', '1/cm')
            n_bins: Number of energy bins
            asr: Acoustic sum rule ('reciprocal', 'realspace', or None)
            adaptive: Whether to use adaptive broadening
            adaptive_method: 'reference' or 'fast' for adaptive broadening
            broadening: Fixed broadening width (if not adaptive)
            shape: Broadening shape ('gauss' or 'lorentz')
            pdos: Whether to calculate PDOS by species
            weighting: 'dos', 'coherent', 'incoherent', or 'coherent-plus-incoherent'
        """
        if fc is None:
            return None

        # Generate MP grid
        qpts = mp_grid(grid)

        # Calculate phonon modes (with mode gradients if adaptive)
        if adaptive:
            phonons, mode_grads = fc.calculate_qpoint_phonon_modes(
                qpts,
                asr=asr,
                return_mode_gradients=True
            )
            mode_widths = mode_gradients_to_widths(mode_grads, fc.crystal.cell_vectors)
        else:
            phonons = fc.calculate_qpoint_phonon_modes(qpts, asr=asr)
            mode_widths = None

        # Get frequency range for energy bins.
        # Do not clamp e_min to 0: imaginary modes in dynamically unstable
        # materials appear as negative frequencies and must be included.
        freqs = phonons.frequencies.to(energy_unit).magnitude
        e_min = freqs.min() - 5
        e_max = freqs.max() * 1.1

        energy_bins = np.linspace(e_min, e_max, n_bins + 1) * ureg(energy_unit)

        # Prepare mode_widths argument for adaptive broadening
        mw_arg = mode_widths if (adaptive and mode_widths is not None) else None
        adaptive_method_arg = adaptive_method if adaptive else None

        # Calculate total DOS (always unweighted for the total)
        if mw_arg is not None:
            dos = phonons.calculate_dos(
                energy_bins,
                mode_widths=mw_arg,
                adaptive_method=adaptive_method_arg
            )
        else:
            dos = phonons.calculate_dos(energy_bins)
            if broadening > 0:
                dos = dos.broaden(
                    x_width=broadening * ureg(energy_unit),
                    shape=shape
                )

        result = {
            'total': dos, 
            'phonons': phonons, 
            'energy_unit': energy_unit, 
            'energy_bins': energy_bins,
            'weighting': weighting
        }

        # Compute per-atom PDOS whenever neutron weighting or per-species display
        # is requested.  A single calculate_pdos call serves both purposes.
        if weighting != 'dos' or pdos:
            # Build keyword arguments for calculate_pdos
            pdos_kwargs = {}
            if weighting != 'dos':
                pdos_kwargs['weighting'] = weighting
            if mw_arg is not None:
                pdos_kwargs['mode_widths'] = mw_arg
                pdos_kwargs['adaptive_method'] = adaptive_method_arg

            # calculate_pdos returns per-atom PDOS as Spectrum1DCollection
            pdos_collection = phonons.calculate_pdos(energy_bins, **pdos_kwargs)

            # Always compute neutron-weighted total DOS when weighting is active.
            # Previously this was gated on pdos=True, which caused the standalone
            # DOS plot to silently display *unweighted* data under a
            # "Neutron-weighted" heading whenever PDOS was not enabled.
            if weighting != 'dos':
                total_weighted = pdos_collection.sum()
                if mw_arg is None and broadening > 0:
                    total_weighted = total_weighted.broaden(
                        x_width=broadening * ureg(energy_unit),
                        shape=shape
                    )
                result['total_weighted'] = total_weighted

            # Per-species PDOS display
            if pdos:
                grouped_pdos = pdos_collection.group_by('species')

                # Apply broadening if not using adaptive
                if mw_arg is None and broadening > 0:
                    grouped_pdos = grouped_pdos.broaden(
                        x_width=broadening * ureg(energy_unit),
                        shape=shape
                    )

                result['pdos'] = grouped_pdos
                # Derive species names from grouped PDOS metadata to guarantee
                # the order matches the rows in grouped_pdos.y_data.
                # list(set(...)) gave non-deterministic ordering and could map
                # the wrong PDOS curve to each species label.
                # iter_metadata() yields per-spectrum dicts; .metadata alone is
                # the top-level collection dict, not a list.
                result['species'] = [m.get('species', '') for m in grouped_pdos.iter_metadata()]

        return result

    dos_data = calculate_dos(
        fc,
        [int(dos_grid_x.value), int(dos_grid_y.value), int(dos_grid_z.value)],
        energy_unit.value,
        int(dos_ebins.value),
        dos_asr.value,
        dos_adaptive.value,
        dos_adaptive_method.value,
        dos_energy_broadening.value,
        dos_shape.value,
        dos_pdos.value,
        dos_weighting.value
    )
    return (dos_data,)


@app.cell
def _(alt, dos_data, dos_pdos, dos_weighting, mo, pd):
    def plot_dos(data, show_pdos):
        """Create Altair DOS plot with optional neutron weighting."""
        if data is None:
            return None

        energy_unit = data['energy_unit']
        weighting = data.get('weighting', 'dos')

        # Use weighted total if available and weighting is enabled
        if weighting != 'dos' and 'total_weighted' in data:
            dos = data['total_weighted']
            total_label = f'Total ({weighting})'
            y_title = 'Neutron-weighted DOS'
        else:
            dos = data['total']
            total_label = 'Total'
            y_title = 'DOS (states/unit)'

        # Extract data from Spectrum1D
        energies = dos.x_data.to(energy_unit).magnitude
        # Use bin centres
        energy_centres = (energies[:-1] + energies[1:]) / 2
        dos_values = dos.y_data.magnitude

        rows = []
        for e, d in zip(energy_centres, dos_values):
            rows.append({'energy': e, 'dos': d, 'species': total_label})

        # Add PDOS if available (already neutron-weighted if weighting enabled)
        if show_pdos and 'pdos' in data:
            pdos_grouped = data['pdos']
            pdos_y = pdos_grouped.y_data.magnitude

            # Handle both single and multiple species cases
            if len(pdos_y.shape) == 1:
                pdos_y = [pdos_y]
            species_list = data['species']

            for sp_idx, sp in enumerate(species_list):
                sp_data = pdos_y[sp_idx] if len(pdos_y) > 1 else pdos_y[0]
                for e, d in zip(energy_centres, sp_data):
                    rows.append({'energy': e, 'dos': d, 'species': sp})

        df = pd.DataFrame(rows)

        # Determine title based on weighting
        if weighting != 'dos':
            title = f'Neutron-weighted Phonon DOS ({weighting})'
        else:
            title = 'Phonon Density of States'

        dos_chart = alt.Chart(df).mark_line(strokeWidth=2).encode(
            x=alt.X('energy:Q', title=f'Energy ({energy_unit})'),
            y=alt.Y('dos:Q', title=y_title),
            color=alt.Color('species:N', title='Species'),
            strokeDash=alt.condition(
                alt.datum.species == total_label,
                alt.value([]),
                alt.value([5, 5])
            )
        ).properties(
            width=400,
            height=400,
            title=title
        ).interactive()

        return dos_chart

    if dos_data:
        _dos_chart = plot_dos(dos_data, dos_pdos.value)
        _weighting_note = ""
        if dos_weighting.value != 'dos':
            _weighting_note = f"\n\n_Using **{dos_weighting.value}** neutron scattering cross-sections_"
        mo.output.append(mo.vstack([
            mo.md(f"## Density of States{_weighting_note}"),
            _dos_chart if _dos_chart else mo.md("_No DOS data_")
        ]))
    return


@app.cell
def _(alt, pd):
    def plot_combined(dispersion_data, dos_data, energy_unit):
        """Create combined dispersion + DOS plot with shared interactive y-axis zoom."""
        if dispersion_data is None or dos_data is None:
            return None

        # --- Shared y-axis selection for zoom/pan ---
        y_zoom = alt.selection_interval(bind='scales', encodings=['y'])

        # --- Dispersion chart data ---
        phonons = dispersion_data['phonons']
        qpts = dispersion_data['qpts']
        labels = dispersion_data['labels']
        label_positions = dispersion_data['label_positions']

        freqs = phonons.frequencies.to(energy_unit).magnitude
        n_qpts, n_branches = freqs.shape

        disp_rows = []
        for q_idx in range(n_qpts):
            for b_idx in range(n_branches):
                disp_rows.append({
                    'q_index': q_idx,
                    'frequency': freqs[q_idx, b_idx],
                    'branch': b_idx
                })
        disp_df = pd.DataFrame(disp_rows)

        # Build JavaScript object literal for label lookup
        label_lookup_js = "{" + ", ".join(f"{pos}: '{lbl}'" for pos, lbl in zip(label_positions, labels)) + "}"

        # Dispersion chart with custom x-axis tick labels
        dispersion_chart = alt.Chart(disp_df).mark_line(
            strokeWidth=1.5,
            opacity=0.8
        ).encode(
            x=alt.X('q_index:Q', 
                    title='',
                    scale=alt.Scale(domain=[min(label_positions), max(label_positions)]),
                    axis=alt.Axis(
                        values=label_positions,
                        labelExpr=f"{label_lookup_js}[datum.value] || ''",
                        labelFontSize=11,
                        labelFontWeight='bold',
                        labelOverlap=True,
                        labelFlush=False
                    )),
            y=alt.Y('frequency:Q', title=f'Energy ({energy_unit})'),
            color=alt.Color('branch:N', legend=None),
            detail='branch:N'
        ).properties(
            width=500,
            height=400,
            title='Phonon Dispersion'
        ).add_params(y_zoom)

        # Vertical lines at high-symmetry points
        label_df = pd.DataFrame({
            'q_index': label_positions,
            'label': labels
        })
        vlines = alt.Chart(label_df).mark_rule(
            opacity=0.3
        ).encode(x='q_index:Q')

        disp_layer = alt.layer(dispersion_chart, vlines).properties(width=500, height=400)

        # --- DOS chart data (rotated: energy on y-axis, DOS on x-axis) ---
        # Use the neutron-weighted total when available, matching the standalone
        # DOS panel.  Previously always used the unweighted 'total'.
        dos = dos_data.get('total_weighted', dos_data['total'])
        energies = dos.x_data.to(energy_unit).magnitude
        energy_centres = (energies[:-1] + energies[1:]) / 2
        dos_values = dos.y_data.magnitude

        # Build dataframe with energy and dos columns, sorted by energy
        dos_df = pd.DataFrame({
            'energy': energy_centres,
            'dos': dos_values
        }).sort_values('energy')

        # DOS chart: x=DOS values, y=energy (to align with dispersion y-axis)
        # Use 'order' encoding to connect points by energy (ascending)
        dos_chart = alt.Chart(dos_df).mark_line(
            strokeWidth=2, color='steelblue'
        ).encode(
            x=alt.X('dos:Q', title='DOS'),
            y=alt.Y('energy:Q', title=''),  # Shared with dispersion
            order='energy:Q'  # Connect points in order of energy
        ).properties(
            width=120,
            height=400,
            title='DOS'
        ).add_params(y_zoom)

        # Combine with shared y-axis
        combined = alt.hconcat(
            disp_layer, dos_chart
        ).resolve_scale(y='shared')

        return combined

    return (plot_combined,)


@app.cell
def _(dispersion_data, dos_data, energy_unit, mo, plot_combined):
    # Combined side-by-side view
    if dispersion_data and dos_data:
        mo.output.append(mo.md("---\n## Combined View"))
        mo.output.append(mo.md("_Scroll to zoom on y-axis, drag to pan_"))

        try:
            _combined = plot_combined(dispersion_data, dos_data, energy_unit.value)
            if _combined:
                mo.output.append(_combined)
        except Exception as e:
            mo.output.append(mo.callout(
                mo.md(f"**Error creating combined plot:** {e}"),
                kind="warn"
            ))
    return


@app.cell
def _(dispersion_data, dos_data):
    def get_fc_summary(fc) -> str:
        summary_text = "---\n## Data Summary\n"

        # Dispersion summary
        if dispersion_data:
            phonons = dispersion_data['phonons']
            freqs = phonons.frequencies.magnitude
            summary_text += f"""**Dispersion:**
            - Q-points: {len(dispersion_data['qpts'])}
            - Branches: {freqs.shape[1]}
            - Frequency range: {freqs.min():.2f} - {freqs.max():.2f} {phonons.frequencies.units}
            - High-symmetry path: {' → '.join(dispersion_data['labels'])}

            """

        # DOS summary
        if dos_data:
            dos = dos_data['total']
            summary_text += f"""**DOS:**
            - Energy bins: {len(dos.y_data.magnitude)}
            - Energy range: {dos.x_data.magnitude.min():.2f} - {dos.x_data.magnitude.max():.2f} {dos_data['energy_unit']}
            """

        return summary_text

    return (get_fc_summary,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
