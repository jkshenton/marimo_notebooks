# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "castep-outputs==0.2.0",
#     "pandas",
#     "numpy",
#     "ase",
#     "altair<6.0.0",
#     "pyarrow",
#     "weas-widget==0.1.26",
# ]
# ///
"""
CASTEP Geometry Optimization Convergence Dashboard

This uses the [castep-outputs](https://github.com/oerc0122/castep_outputs) package to parse CASTEP `.geom` or `.castep` files

Pipeline:
  1. File → list[Atoms]:  geom_to_trajectory() / castep_to_trajectory()
  2. list[Atoms] → DataFrame:  trajectory_to_dataframe()
  3. DataFrame → Summary:  dataframe_to_summary()

Usage:
  As marimo app: marimo run castep_geomopt_summary.py
  As CLI tool:   python castep_geomopt_summary.py input.geom [-o output.csv] [-q]
"""

from __future__ import annotations

import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")

# Setup cell: imports available to @app.function decorated functions
with app.setup:
    import castep_outputs as co
    from castep_outputs.parsers.md_geom_file_parser import parse_md_geom_frame
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from ase import Atoms, units


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # CASTEP Geometry Optimization Convergence Dashboard

    Upload a CASTEP `.geom` or `.castep` file to visualize the convergence of a geometry optimization calculation.

    **Note:** `.geom` files are preferred as they can be used while the calculation is still running. The `.castep` file can only be used for successfully completed calculations.
    """)
    return


@app.cell
def _():
    import altair as alt
    import tempfile
    from weas_widget.base_widget import BaseWidget
    from weas_widget.atoms_viewer import AtomsViewer
    from weas_widget.utils import ASEAdapter
    return ASEAdapter, AtomsViewer, BaseWidget, alt, tempfile


@app.function
def geom_to_trajectory(file_path):
    """Parse .geom file into ASE Atoms trajectory.

    Each Atoms object has:
      - arrays['forces']: Forces on each atom (eV/Å)
      - info['enthalpy']: Enthalpy in eV (includes PV work for variable cell)
      - info['energy']: Total energy in eV
      - info['stress']: Stress tensor as 6-element Voigt array (GPa), or None
      - info['iteration']: 1-based iteration number
    """
    file_path = Path(file_path)

    # Unit conversions
    Ha_to_eV = units.Ha
    Bohr_to_Ang = units.Bohr
    Ha_Bohr3_to_GPa = units.Ha / units.Bohr**3 / units.GPa

    # Parse file into frames
    with open(file_path, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    start_idx = 0
    for i, line in enumerate(lines):
        if 'END header' in line:
            start_idx = i + 2
            break

    # Split into frames by blank lines
    data_lines = lines[start_idx:]
    raw_frames = []
    current_frame = []
    for line in data_lines:
        if line.strip() == '':
            if current_frame:
                raw_frames.append(current_frame)
                current_frame = []
        else:
            current_frame.append(line)
    if current_frame:
        raw_frames.append(current_frame)

    # Parse each frame
    frames = [parse_md_geom_frame(iter(raw_frame)) for raw_frame in raw_frames]

    # Convert to Atoms objects
    trajectory = []
    for i, frame in enumerate(frames):
        lattice = frame.get('lattice_vectors', [])
        ions = frame.get('ions', {})
        if len(lattice) < 3 or not ions:
            continue

        # Cell and positions
        cell = np.array(lattice[-3:]) * Bohr_to_Ang
        symbols = [s for s, _ in ions.keys()]
        positions = np.array([ion['R'] for ion in ions.values()]) * Bohr_to_Ang

        atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)

        # Forces (eV/Å)
        forces = np.array([ion['F'] for ion in ions.values()]) * Ha_to_eV / Bohr_to_Ang
        atoms.set_array('forces', forces)

        # Energy and enthalpy from E field: [E_total, E_enthalpy]
        e_field = frame.get('E')
        if e_field:
            atoms.info['energy'] = e_field[0][0] * Ha_to_eV
            atoms.info['enthalpy'] = e_field[0][1] * Ha_to_eV

        # Stress tensor (convert to Voigt notation: xx, yy, zz, yz, xz, xy)
        stress_tensor = frame.get('S')
        if stress_tensor is not None:
            s = np.array(stress_tensor) * Ha_Bohr3_to_GPa
            atoms.info['stress'] = np.array([s[0,0], s[1,1], s[2,2], s[1,2], s[0,2], s[0,1]])

        atoms.info['iteration'] = i + 1
        trajectory.append(atoms)

    return trajectory


@app.function
def castep_to_trajectory(file_path):
    """Parse .castep file into ASE Atoms trajectory.

    Each Atoms object has:
      - arrays['forces']: Forces on each atom (eV/Å)
      - info['enthalpy']: Enthalpy in eV
      - info['stress_max']: Max stress (GPa) from minimisation data, or None
      - info['iteration']: 1-based iteration number

    Note: Only the first run is parsed. If the file contains multiple runs
    (e.g., continuation/restarted calculations), a warning is printed.
    """
    file_path = Path(file_path)

    parsed = co.parse_single(file_path)

    # Check for multiple runs (continuation/restarted calculations)
    if isinstance(parsed, list) and len(parsed) > 1:
        num_runs = len(parsed)
        total_iters = sum(
            len(entry.get('geom_opt', {}).get('iterations', []))
            for entry in parsed
        )
        import warnings
        warnings.warn(
            f"WARNING: {file_path.name} contains {num_runs} separate runs "
            f"(total ~{total_iters} iterations). Using the final run only."
        )

    # Use the last run (most recent/final for continuation calculations)
    data = parsed[-1] if isinstance(parsed, list) and parsed else parsed

    if 'geom_opt' not in data or 'iterations' not in data['geom_opt']:
        return []

    initial_cell = data.get('initial_cell', {}).get('real_lattice')
    trajectory = []
    prev_enthalpy = None
    iteration_num = 0

    for iteration in data['geom_opt']['iterations']:
        pos_data = iteration.get('atoms') or iteration.get('positions')
        if not pos_data:
            continue

        # Get enthalpy
        enthalpy_data = iteration.get('enthalpy')
        enthalpy = enthalpy_data[0] if isinstance(enthalpy_data, list) else enthalpy_data

        # Skip rejected LBFGS iterations (enthalpy unchanged)
        if enthalpy is not None and prev_enthalpy is not None and enthalpy == prev_enthalpy:
            continue
        prev_enthalpy = enthalpy
        iteration_num += 1

        # Cell
        cell = iteration.get('cell', {}).get('real_lattice', initial_cell)
        if cell is None:
            continue

        # Create Atoms - handle CIF-style labels like 'H [H3]' or 'Si[Si1]' -> element symbol
        import re
        def extract_element(label):
            """Extract element symbol from CIF-style label like 'H [H3]' or 'Si[Si1]'."""
            match = re.match(r'^([A-Z][a-z]?)', label)
            return match.group(1) if match else label

        atoms = Atoms(
            symbols=[extract_element(s) for s, _ in pos_data],
            scaled_positions=list(pos_data.values()),
            cell=cell, pbc=True
        )

        # Forces - check 'non_descript' first, then 'symmetrised'
        forces_dict = iteration.get('forces', {})
        forces_data = forces_dict.get('non_descript') or forces_dict.get('symmetrised', [])
        if forces_data:
            # Use last block (final forces after all corrections, matches f_max in minimisation)
            force_dict = forces_data[-1] if forces_data else {}
            forces = np.array([force_dict.get(k, [0,0,0]) for k in pos_data])
            atoms.set_array('forces', forces)

        # Enthalpy
        if enthalpy is not None:
            atoms.info['enthalpy'] = enthalpy

        # Convergence data from minimisation
        minimisation = iteration.get('minimisation', [])
        min_data = minimisation[-1] if minimisation else {}

        de_ion = min_data.get('de_ion', {})
        f_max = min_data.get('f_max', {})
        dr_max = min_data.get('dr_max', {})
        smax = min_data.get('smax', {})

        # Store max stress value - also check stresses dict if smax not in minimisation
        if isinstance(smax, dict) and smax.get('value') is not None:
            atoms.info['stress_max'] = smax['value']
        else:
            # Try to get from stresses block
            stresses_dict = iteration.get('stresses', {})
            stress_data = stresses_dict.get('symmetrised') or stresses_dict.get('non_descript')
            if stress_data:
                # Use last block (final stress after all corrections)
                last_stress = stress_data[-1] if stress_data else None
                if last_stress is not None:
                    atoms.info['stress_max'] = float(max(abs(x) for x in last_stress))

        atoms.info['iteration'] = iteration_num
        trajectory.append(atoms)

    return trajectory


@app.function
def file_to_trajectory(file_path):
    """Parse .geom or .castep file into ASE Atoms trajectory.

    Dispatches to geom_to_trajectory() or castep_to_trajectory() based on file extension.
    """
    file_path = Path(file_path)
    file_ext = file_path.suffix.lower()

    if file_ext == '.geom':
        return geom_to_trajectory(file_path)
    else:
        return castep_to_trajectory(file_path)


@app.function
def trajectory_to_dataframe(trajectory):
    """Convert ASE Atoms trajectory to convergence DataFrame.

    Calculates:
      - enthalpy_change_eV: Change from previous iteration
      - max_force_eV_ang: Maximum force magnitude
      - max_displacement_ang: Maximum atomic displacement from previous frame
      - max_stress_GPa: Maximum stress component (from info['stress'] or info['stress_max'])
      - volume_ang3: Cell volume
    """
    if not trajectory:
        return pd.DataFrame()

    results = []
    prev_enthalpy = None
    prev_positions = None

    for atoms in trajectory:
        iteration = atoms.info.get('iteration', len(results) + 1)
        enthalpy = atoms.info.get('enthalpy')

        # Enthalpy change
        enthalpy_change = None
        if enthalpy is not None and prev_enthalpy is not None:
            enthalpy_change = enthalpy - prev_enthalpy
        prev_enthalpy = enthalpy

        # Max force
        max_force = None
        if 'forces' in atoms.arrays:
            force_mags = np.linalg.norm(atoms.arrays['forces'], axis=1)
            max_force = float(force_mags.max())

        # Max displacement
        max_disp = None
        positions = atoms.get_positions()
        if prev_positions is not None and len(positions) == len(prev_positions):
            displacements = np.linalg.norm(positions - prev_positions, axis=1)
            max_disp = float(displacements.max())
        prev_positions = positions.copy()

        # Max stress
        max_stress = None
        if 'stress' in atoms.info:
            max_stress = float(np.max(np.abs(atoms.info['stress'])))
        elif 'stress_max' in atoms.info:
            max_stress = atoms.info['stress_max']

        # Volume
        volume = float(atoms.get_volume())

        results.append({
            'iteration': iteration,
            'enthalpy_eV': enthalpy,
            'enthalpy_change_eV': enthalpy_change,
            'max_force_eV_ang': max_force,
            'max_displacement_ang': max_disp,
            'max_stress_GPa': max_stress,
            'volume_ang3': volume,
        })

    return pd.DataFrame(results)


@app.function
def dataframe_to_summary(df, filename=""):
    """Generate text summary from convergence DataFrame."""
    if df is None or df.empty:
        return "No geometry optimization data found."

    lines = [
        f"\n{'='*60}",
        f"CASTEP Geometry Optimization Summary{': ' + filename if filename else ''}",
        f"{'='*60}",
        f"Total iterations:      {len(df)}",
        f"Final enthalpy:        {df['enthalpy_eV'].iloc[-1]:.6f} eV",
    ]

    if df['enthalpy_change_eV'].notna().any():
        lines.append(f"Final enthalpy change: {df['enthalpy_change_eV'].iloc[-1]:.2e} eV")
    if df['max_force_eV_ang'].notna().any():
        lines.append(f"Final max force:       {df['max_force_eV_ang'].iloc[-1]:.6f} eV/Å")
    if df['max_displacement_ang'].notna().any():
        lines.append(f"Final max displacement:{df['max_displacement_ang'].iloc[-1]:.6f} Å")
    if df['max_stress_GPa'].notna().any():
        lines.append(f"Final max stress:      {df['max_stress_GPa'].iloc[-1]:.6f} GPa")
    if df['volume_ang3'].notna().any():
        lines.append(f"Final volume:          {df['volume_ang3'].iloc[-1]:.3f} Å³")

    lines.append(f"{'='*60}\n")
    return '\n'.join(lines)


@app.cell
def _(mo):
    file_upload = mo.ui.file(
        label="Upload CASTEP file (.geom or .castep)",
        filetypes=[".geom", ".castep"],
        multiple=False
    )
    return (file_upload,)


@app.cell
def _(file_upload, mo):

    mo.vstack([
        mo.md("## Upload File"),
        file_upload,
        mo.md(f"✅ Loaded: **{file_upload.value[0].name}**") if file_upload.value else mo.md("_Supported formats: `.geom`, `.castep`_")
    ])
    return


@app.cell
def _(file_upload, tempfile):
    def parse_uploaded_file(uploaded_file):
        """Parse uploaded .geom or .castep file using the pipeline.

        Returns (trajectory, df) where trajectory has normalized force_magnitude for visualization.
        """
        if uploaded_file is None:
            return [], None

        suffix = f".{uploaded_file.name.split('.')[-1]}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded_file.contents)
            temp_path = Path(tmp.name)

        try:
            # Step 1: File → list[Atoms]
            trajectory = file_to_trajectory(temp_path)

            # Step 2: list[Atoms] → DataFrame  
            df = trajectory_to_dataframe(trajectory)

            # Add force_magnitude array for visualization (normalized log scale 0-1)
            if trajectory and 'forces' in trajectory[0].arrays:
                all_mags = np.concatenate([
                    np.linalg.norm(a.arrays['forces'], axis=1) 
                    for a in trajectory if 'forces' in a.arrays
                ])
                if len(all_mags):
                    log_min, log_max = np.log10(all_mags.min()), np.log10(all_mags.max())
                    for atoms in trajectory:
                        if 'forces' in atoms.arrays:
                            mags = np.linalg.norm(atoms.arrays['forces'], axis=1)
                            atoms.set_array('force_magnitude', (np.log10(mags) - log_min) / (log_max - log_min))

            return trajectory, df
        finally:
            temp_path.unlink()

    trajectory, df = parse_uploaded_file(file_upload.value[0]) if file_upload.value else ([], None)
    return df, trajectory


@app.cell
def _(df, mo):
    def format_value(series, fmt=".6f", suffix=""):
        """Format value or return 'N/A' if None/NaN."""
        val = series.iloc[-1] if len(series) > 0 else None
        if val is None or (isinstance(val, float) and val != val):  # NaN check
            return "N/A"
        return f"{val:{fmt}}{suffix}"

    if df is None or len(df) == 0:
        summary_table = "## Summary\n_No geometry optimization data found._"
    else:
        summary_table = f"""
            ## Summary
            | Metric | Value |
            |--------|-------|
            | Total iterations | {len(df)} |
            | Final enthalpy | {format_value(df['enthalpy_eV'])} eV |
            | Final max force | {format_value(df['max_force_eV_ang'])} eV/Å |
            | Final max stress | {format_value(df['max_stress_GPa'])} GPa |
            """
    mo.md(summary_table) if df is not None and len(df) > 0 else None
    return


@app.cell(hide_code=True)
def _(df, mo):
    mo.md("""
    ## Convergence Plots
    Drag on the **Enthalpy** chart to select an iteration range. The other plots and structure visualization will update.
    """) if df is not None and len(df) > 0 else None
    return


@app.cell
def _(mo):
    show_tolerances = mo.ui.checkbox(label="Show tolerance controls. Make sure to enter your own values of these.", value=False)
    enthalpy_tolerance = mo.ui.number(
        value=1e-5, start=1e-10, stop=1.0, step=1e-6,
        label="Enthalpy tolerance (eV)"
    )
    force_tolerance = mo.ui.number(
        value=0.01, start=1e-6, stop=10.0, step=0.001,
        label="Force tolerance (eV/Å)"
    )
    disp_tolerance = mo.ui.number(
        value=0.001, start=1e-6, stop=1.0, step=0.0001,
        label="Displacement tolerance (Å)"
    )
    stress_tolerance = mo.ui.number(
        value=0.01, start=1e-6, stop=10.0, step=0.001,
        label="Stress tolerance (GPa)"
    )
    return (
        disp_tolerance,
        enthalpy_tolerance,
        force_tolerance,
        show_tolerances,
        stress_tolerance,
    )


@app.cell
def _(alt, df, mo):
    # Master enthalpy chart - controls filtering for other charts and trajectory
    if df is not None and len(df) > 0:
        brush = alt.selection_interval(encodings=['x'])
        enthalpy_chart = mo.ui.altair_chart(
            alt.Chart(df).mark_line(point=True, color='steelblue').encode(
                x=alt.X('iteration:Q', title='Iteration'),
                y=alt.Y('enthalpy_eV:Q', title='Enthalpy (eV)', scale=alt.Scale(zero=False)),
                tooltip=['iteration', 'enthalpy_eV']
            ).add_params(brush).properties(title='Enthalpy (drag to select range)', width=700, height=250)
        )
    else:
        enthalpy_chart = None
    return (enthalpy_chart,)


@app.cell
def _(df, enthalpy_chart):
    # Get selected iteration range from enthalpy chart
    if df is not None and len(df) > 0 and enthalpy_chart is not None:
        chart_value = enthalpy_chart.value
        if chart_value is not None and len(chart_value) > 0:
            # Get iteration range from selection, then filter original df
            # This ensures we use the original data with all columns intact
            selected_iters = set(chart_value['iteration'])
            selected_df = df[df['iteration'].isin(selected_iters)].copy()
        else:
            selected_df = df.copy()
        # Ensure proper ordering by iteration
        selected_df = selected_df.sort_values('iteration').reset_index(drop=True)
        min_iter = int(selected_df['iteration'].min())
        max_iter = int(selected_df['iteration'].max())
        is_subset = len(selected_df) < len(df)
        selection_info = f"**Selected:** {min_iter}-{max_iter} ({len(selected_df)} pts)" if is_subset else ""
    else:
        selected_df, min_iter, max_iter, selection_info = df, 1, len(df) if df is not None else 1, ""
    return max_iter, min_iter, selected_df, selection_info


@app.cell
def _(alt, force_tolerance, selected_df, show_tolerances):
    # Force chart - only show if force data exists
    force_chart = None
    if selected_df is not None and len(selected_df) > 0:
        _df = selected_df[selected_df['max_force_eV_ang'].notna()].copy()
        if len(_df) > 0:
            _df['tolerance'] = force_tolerance.value
            force_chart = alt.layer(
                alt.Chart(_df).mark_line(point=True, color='green').encode(
                    x='iteration:Q', y=alt.Y('max_force_eV_ang:Q', title='Max Force (eV/Å)', scale=alt.Scale(type='log')),
                    tooltip=['iteration', 'max_force_eV_ang']),
                *([alt.Chart(_df).mark_rule(color='red', strokeDash=[5,5]).encode(y='tolerance:Q')] if show_tolerances.value else [])
            ).properties(title='Force', width=280, height=200)
    return (force_chart,)


@app.cell
def _(alt, enthalpy_tolerance, selected_df, show_tolerances):
    # Enthalpy change chart
    enthalpy_change_chart = None
    if selected_df is not None:
        _df = selected_df[selected_df['enthalpy_change_eV'].notna()].copy()
        if len(_df) > 0:
            # Use absolute value for log scale
            _df['abs_enthalpy_change'] = _df['enthalpy_change_eV'].abs()
            _df['tolerance'] = enthalpy_tolerance.value

            # Calculate domain including tolerance for proper scaling
            data_min = _df['abs_enthalpy_change'].min()
            data_max = _df['abs_enthalpy_change'].max()
            tol = enthalpy_tolerance.value
            min_val = max(min(data_min, tol) * 0.5, 1e-10)  # Include tolerance, avoid zero
            max_val = max(data_max, tol) * 2

            enthalpy_change_chart = alt.layer(
                alt.Chart(_df).mark_line(point=True, color='steelblue').encode(
                    x='iteration:Q', 
                    y=alt.Y('abs_enthalpy_change:Q', title='|ΔH| (eV)', 
                            scale=alt.Scale(type='log', domain=[min_val, max_val]),
                            axis=alt.Axis(format='.0e')),
                    tooltip=['iteration', 'enthalpy_change_eV']
                ),
                *([alt.Chart(_df).mark_rule(color='red', strokeDash=[5,5]).encode(y='tolerance:Q')] if show_tolerances.value else [])
            ).properties(title='Enthalpy Change', width=280, height=200)
    return (enthalpy_change_chart,)


@app.cell
def _(alt, disp_tolerance, selected_df, show_tolerances):
    # Displacement chart
    disp_chart = None
    if selected_df is not None:
        _df = selected_df[selected_df['max_displacement_ang'].notna()]
        if len(_df) > 0:
            _df = _df.assign(tolerance=disp_tolerance.value)
            disp_chart = alt.layer(
                alt.Chart(_df).mark_line(point=True, color='orange').encode(
                    x='iteration:Q', y=alt.Y('max_displacement_ang:Q', title='Max Disp (Å)', scale=alt.Scale(type='log')),
                    tooltip=['iteration', 'max_displacement_ang']),
                *([alt.Chart(_df).mark_rule(color='red', strokeDash=[5,5]).encode(y='tolerance:Q')] if show_tolerances.value else [])
            ).properties(title='Displacement', width=280, height=200)
    return (disp_chart,)


@app.cell
def _(alt, selected_df, show_tolerances, stress_tolerance):
    # Stress chart (only if stress data exists)
    stress_chart = None
    if selected_df is not None and selected_df['max_stress_GPa'].notna().any():
        _df = selected_df[selected_df['max_stress_GPa'].notna()]
        if len(_df) > 0:
            _df = _df.assign(tolerance=stress_tolerance.value)
            stress_chart = alt.layer(
                alt.Chart(_df).mark_line(point=True, color='crimson').encode(
                    x='iteration:Q', 
                    y=alt.Y('max_stress_GPa:Q', title='Max Stress (GPa)', scale=alt.Scale(type='log')),
                    tooltip=['iteration', 'max_stress_GPa']
                ),
                *([alt.Chart(_df).mark_rule(color='red', strokeDash=[5,5]).encode(y='tolerance:Q')] if show_tolerances.value else [])
            ).properties(title='Stress', width=280, height=200)
    return (stress_chart,)


@app.cell
def _(alt, selected_df):
    # Volume chart (only if volume varies)
    volume_chart = None
    if selected_df is not None and selected_df['volume_ang3'].notna().any():
        vol = selected_df['volume_ang3']
        if vol.max() - vol.min() > 0.01:
            volume_chart = alt.Chart(selected_df).mark_line(point=True, color='purple').encode(
                x='iteration:Q', y=alt.Y('volume_ang3:Q', title='Volume (Å³)', scale=alt.Scale(zero=False)),
                tooltip=['iteration', 'volume_ang3']
            ).properties(title='Volume', width=280, height=200)
    return (volume_chart,)


@app.cell
def _(
    disp_chart,
    disp_tolerance,
    enthalpy_change_chart,
    enthalpy_chart,
    enthalpy_tolerance,
    force_chart,
    force_tolerance,
    mo,
    selection_info,
    show_tolerances,
    stress_chart,
    stress_tolerance,
    volume_chart,
):
    # Display all charts in grid
    if enthalpy_chart:
        # Build rows for grid layout
        row1 = [c for c in [enthalpy_change_chart, force_chart] if c]
        row2 = [c for c in [disp_chart, stress_chart, volume_chart] if c]

        # Build tolerance controls (only show if corresponding chart exists)
        tolerances = [enthalpy_tolerance, force_tolerance, disp_tolerance]
        if stress_chart:
            tolerances.append(stress_tolerance)

        mo.output.append(mo.vstack([
            enthalpy_chart,
            mo.md(selection_info) if selection_info else None,
            show_tolerances,
            mo.hstack(tolerances, justify='start', gap=2) if show_tolerances.value else None,
            mo.hstack(row1, justify='start', gap=2) if row1 else None,
            mo.hstack(row2, justify='start', gap=2) if row2 else None
        ]))
    return


@app.cell(hide_code=True)
def _(mo, trajectory):
    mo.md("## Structure Visualization\nUse controls to step through the optimization trajectory.") if trajectory else None
    return


@app.cell
def _(ASEAdapter, AtomsViewer, BaseWidget, mo):
    def view_trajectory(atoms_list, model_style=1, show_bonded_atoms=False, color_by_force=False):
        """Display ASE Atoms trajectory using weas-widget."""
        if not atoms_list:
            return mo.md("_No trajectory data available._")

        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]

        v = AtomsViewer(BaseWidget(guiConfig={"controls": {"enabled": True}}))
        v.atoms = ASEAdapter.to_weas(atoms_list[0]) if len(atoms_list) == 1 else [ASEAdapter.to_weas(a) for a in atoms_list]
        v.model_style = model_style
        v.boundary = [[-0.1, 1.1], [-0.1, 1.1], [-0.1, 1.1]]
        v.show_bonded_atoms = show_bonded_atoms
        v.color_type = "VESTA"

        if color_by_force and hasattr(atoms_list[0], 'arrays') and 'force_magnitude' in atoms_list[0].arrays:
            v.color_by = "force_magnitude"
            v.color_ramp = ["#440154", "#31688e", "#35b779", "#fde724"]

        return v._widget
    return (view_trajectory,)


@app.cell
def _(mo, trajectory):
    # Viewer controls
    if trajectory:
        has_forces = hasattr(trajectory[0], 'arrays') and 'force_magnitude' in trajectory[0].arrays
        model_style_dropdown = mo.ui.dropdown(
            options={'Ball': 0, 'Ball and Stick': 1, 'Polyhedral': 2, 'Stick': 3},
            value='Ball and Stick', label='Model Style'
        )
        show_bonded = mo.ui.checkbox(label='Show bonded atoms', value=False)
        color_by_force_cb = mo.ui.checkbox(label='Color by force', value=False) if has_forces else None

        _controls = [model_style_dropdown, show_bonded]
        if color_by_force_cb:
            _controls.append(color_by_force_cb)
        mo.output.append(mo.hstack(_controls, justify='start', gap=2))
    else:
        has_forces = False
        model_style_dropdown = show_bonded = color_by_force_cb = None
    return color_by_force_cb, model_style_dropdown, show_bonded


@app.cell
def _(
    color_by_force_cb,
    max_iter,
    min_iter,
    mo,
    model_style_dropdown,
    show_bonded,
    trajectory,
    view_trajectory,
):
    if trajectory and model_style_dropdown:
        filtered_trajectory = trajectory[min_iter - 1 : max_iter]
        viewer = view_trajectory(
            filtered_trajectory,
            model_style=model_style_dropdown.value,
            show_bonded_atoms=show_bonded.value if show_bonded else False,
            color_by_force=color_by_force_cb.value if color_by_force_cb else False
        )
        mo.output.append(mo.vstack([
            mo.md(f"_Showing **{len(filtered_trajectory)}** structures (iterations {min_iter} to {max_iter})_"),
            viewer
        ]))
    else:
        mo.output.append(mo.md("_Upload a file to view the structure trajectory._"))
    return


@app.cell
def _(df, mo):
    # Show raw data table (collapsible)
    if df is not None and len(df) > 0:
        mo.output.append(mo.accordion({"📊 Raw Data Table": mo.ui.table(df, selection=None)}))
    return


def cli():
    """Command-line interface for extracting geometry optimization data."""
    import argparse
    import sys
    from pathlib import Path
    
    parser = argparse.ArgumentParser(
        description='Extract geometry optimization convergence data from CASTEP files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s calculation.geom                    # Print summary to stdout
  %(prog)s calculation.castep -o results.csv   # Save to CSV
  %(prog)s calculation.geom -o results.json    # Save to JSON
        '''
    )
    parser.add_argument('input', type=Path, help='Input .geom or .castep file')
    parser.add_argument('-o', '--output', type=Path, help='Output file (.csv, .json, .xlsx)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress summary output')
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    if args.input.suffix.lower() not in ['.geom', '.castep']:
        print(f"Error: Unsupported file type: {args.input.suffix}", file=sys.stderr)
        print("Supported types: .geom, .castep", file=sys.stderr)
        sys.exit(1)
    
    # Use the @app.function decorated functions directly
    # Step 1: File → list[Atoms]
    trajectory = file_to_trajectory(args.input)
    
    # Step 2: list[Atoms] → DataFrame
    df = trajectory_to_dataframe(trajectory)
    
    if df.empty:
        print("Error: No geometry optimization data found in file.", file=sys.stderr)
        sys.exit(1)
    
    # Step 3: DataFrame → Summary
    if not args.quiet:
        print(dataframe_to_summary(df, args.input.name))
    
    if args.output:
        suffix = args.output.suffix.lower()
        if suffix == '.csv':
            df.to_csv(args.output, index=False)
        elif suffix == '.json':
            df.to_json(args.output, orient='records', indent=2)
        elif suffix in ['.xlsx', '.xls']:
            df.to_excel(args.output, index=False)
        elif suffix == '.parquet':
            df.to_parquet(args.output, index=False)
        else:
            print(f"Warning: Unknown format '{suffix}', saving as CSV", file=sys.stderr)
            df.to_csv(args.output, index=False)
        
        if not args.quiet:
            print(f"Data saved to: {args.output}")


if __name__ == "__main__":
    import sys
    # Check if running via marimo or as standalone script
    if len(sys.argv) > 1 and sys.argv[1] not in ['run', 'edit']:
        cli()
    else:
        app.run()
