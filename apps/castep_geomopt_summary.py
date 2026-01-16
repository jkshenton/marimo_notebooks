# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair==6.0.0",
#     "ase==3.27.0",
#     "castep-outputs==0.2.0",
#     "marimo==0.19.4",
#     "numpy==2.4.1",
#     "pandas==2.3.3",
#     "weas-widget==0.2.4",
# ]
# ///

import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)

@app.cell
def _():
    import sys
    if sys.platform == "emscripten":  # Running in Pyodide/WASM
        import micropip
        await micropip.install("weas-widget==0.2.4", deps=False)
        await micropip.install(["castep-outputs==0.2.0", "ase==3.27.0", "anywidget"])


@app.cell
def _():
    import castep_outputs as co
    from castep_outputs.parsers.md_geom_file_parser import parse_md_geom_frame
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from ase import Atoms, units


    def format_file_scrollable(content, max_height="300px"):
        """Format full file content in a scrollable container for error display."""
        import html
        escaped = html.escape(content)
        return f'<pre style="max-height: {max_height}; overflow-y: auto; background: var(--gray-2); padding: 1em; border-radius: 6px; font-size: 0.85em;">{escaped}</pre>'
    return (
        Atoms,
        Path,
        co,
        format_file_scrollable,
        np,
        parse_md_geom_frame,
        pd,
        units,
    )





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


@app.cell
def _(Atoms, Path, format_file_scrollable, np, parse_md_geom_frame, units):
    def geom_to_trajectory(file_path):
        """Parse .geom file into ASE Atoms trajectory.

        Each Atoms object has:
          - arrays['forces']: Forces on each atom (eV/Å)
          - info['enthalpy']: Enthalpy in eV (includes PV work for variable cell)
          - info['energy']: Total energy in eV
          - info['stress']: Stress tensor as 6-element Voigt array (GPa), or None
          - info['iteration']: 1-based iteration number

        Returns:
            tuple: (trajectory, error_message) where error_message is None on success
        """
        file_path = Path(file_path)

        # Unit conversions
        Ha_to_eV = units.Ha
        Bohr_to_Ang = units.Bohr
        Ha_Bohr3_to_GPa = units.Ha / units.Bohr**3 / units.GPa

        # Parse file into frames
        try:
            with open(file_path, 'r') as f:
                content = f.read()
        except Exception as e:
            return [], f"Failed to read .geom file: {e}"

        if not content.strip():
            return [], "The .geom file is empty."

        file_preview = format_file_scrollable(content)

        lines = content.split('\n')
        start_idx = 0
        header_found = False
        for i, line in enumerate(lines):
            if 'END header' in line:
                start_idx = i + 2
                header_found = True
                break

        if not header_found:
            return [], f"Invalid .geom file: 'END header' marker not found. Is this a valid CASTEP .geom file?{file_preview}"

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

        if not raw_frames:
            return [], f"No geometry frames found in .geom file. The file may be incomplete or the calculation hasn't produced any output yet.{file_preview}"

        # Parse each frame
        frames = []
        parse_errors = []
        for i, raw_frame in enumerate(raw_frames):
            try:
                frame = parse_md_geom_frame(iter(raw_frame))
                frames.append(frame)
            except Exception as e:
                parse_errors.append(f"Frame {i+1}: {e}")

        if not frames:
            error_detail = "; ".join(parse_errors[:3])  # Show first 3 errors
            return [], f"Failed to parse any frames from .geom file. Errors: {error_detail}{file_preview}"

        # Convert to Atoms objects
        trajectory = []
        skipped_frames = 0
        for i, frame in enumerate(frames):
            lattice = frame.get('lattice_vectors', [])
            ions = frame.get('ions', {})
            if len(lattice) < 3 or not ions:
                skipped_frames += 1
                continue

            try:
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
            except Exception as e:
                skipped_frames += 1
                continue

        if not trajectory:
            return [], f"No valid geometry iterations found. Parsed {len(frames)} frames but all were missing required data (lattice vectors or ionic positions).{file_preview}"

        # Check for minimal required data
        warnings = []
        if not any('enthalpy' in a.info for a in trajectory):
            warnings.append("No enthalpy data found in any frame.")
        if not any('forces' in a.arrays for a in trajectory):
            warnings.append("No force data found in any frame.")

        warning_msg = " ".join(warnings) if warnings else None
        return trajectory, warning_msg
    return (geom_to_trajectory,)


@app.cell
def _(Atoms, Path, co, format_file_scrollable, np):
    def castep_to_trajectory(file_path):
        """Parse .castep file into ASE Atoms trajectory.

        Each Atoms object has:
          - arrays['forces']: Forces on each atom (eV/Å)
          - info['enthalpy']: Enthalpy in eV
          - info['stress_max']: Max stress (GPa) from minimisation data, or None
          - info['iteration']: 1-based iteration number

        Note: Only the final run is parsed. If the file contains multiple runs
        (e.g., continuation/restarted calculations), a warning is included.

        Returns:
            tuple: (trajectory, error_message) where error_message is None on success
        """
        file_path = Path(file_path)
        warning_msg = None

        # Read file content for error previews
        try:
            with open(file_path, 'r') as f:
                file_content = f.read()
            file_preview = format_file_scrollable(file_content)
        except Exception as e:
            file_preview = f"(Could not read file: {e})"

        # Parse the file
        try:
            parsed = co.parse_single(file_path)
        except Exception as e:
            return [], f"Failed to parse .castep file: {e}{file_preview}"

        if parsed is None:
            return [], f"Failed to parse .castep file: parser returned no data.{file_preview}"

        # Check for multiple runs (continuation/restarted calculations)
        if isinstance(parsed, list) and len(parsed) > 1:
            num_runs = len(parsed)
            total_iters = sum(
                len(entry.get('geom_opt', {}).get('iterations', []))
                for entry in parsed
            )
            warning_msg = f"File contains {num_runs} separate runs (total ~{total_iters} iterations). Showing final run only."

        # Use the last run (most recent/final for continuation calculations)
        data = parsed[-1] if isinstance(parsed, list) and parsed else parsed

        if not isinstance(data, dict):
            return [], f"Unexpected data format from parser: expected dict, got {type(data).__name__}{file_preview}"

        # Check for geometry optimization data
        if 'geom_opt' not in data:
            # Check if this might be a different calculation type
            calc_types = [k for k in data.keys() if k in ['singlepoint', 'md', 'phonon', 'ts_search', 'magres']]
            if calc_types:
                return [], f"This appears to be a {calc_types[0]} calculation, not a geometry optimization. Please upload a .castep file from a geometry optimization run.{file_preview}"
            return [], f"No geometry optimization data found in .castep file. Is this a geometry optimization calculation?{file_preview}"

        if 'iterations' not in data['geom_opt']:
            return [], f"Geometry optimization section found but contains no iterations. The calculation may have failed before completing any steps.{file_preview}"

        iterations = data['geom_opt']['iterations']
        if not iterations:
            return [], f"Geometry optimization contains an empty iterations list. The calculation may not have started properly.{file_preview}"

        initial_cell = data.get('initial_cell', {}).get('real_lattice')
        trajectory = []
        prev_enthalpy = None
        iteration_num = 0
        skipped_no_positions = 0
        skipped_no_cell = 0

        import re
        def extract_element(label):
            """Extract element symbol from CIF-style label like 'H [H3]' or 'Si[Si1]'."""
            match = re.match(r'^([A-Z][a-z]?)', label)
            return match.group(1) if match else label

        for iteration in iterations:
            pos_data = iteration.get('atoms') or iteration.get('positions')
            if not pos_data:
                skipped_no_positions += 1
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
                skipped_no_cell += 1
                continue

            try:
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
            except Exception as e:
                # Skip this iteration but continue processing
                continue

        if not trajectory:
            issues = []
            if skipped_no_positions:
                issues.append(f"{skipped_no_positions} iterations missing atomic positions")
            if skipped_no_cell:
                issues.append(f"{skipped_no_cell} iterations missing cell data")
            issue_str = "; ".join(issues) if issues else "unknown reason"
            return [], f"No valid geometry iterations could be extracted from {len(iterations)} iterations. Issues: {issue_str}{file_preview}"

        # Check for minimal required data and add warnings
        data_warnings = []
        if not any('enthalpy' in a.info for a in trajectory):
            data_warnings.append("No enthalpy data found.")
        if not any('forces' in a.arrays for a in trajectory):
            data_warnings.append("No force data found.")

        if data_warnings:
            extra_warning = " ".join(data_warnings)
            warning_msg = f"{warning_msg} {extra_warning}" if warning_msg else extra_warning

        return trajectory, warning_msg
    return (castep_to_trajectory,)


@app.cell
def _(Path, castep_to_trajectory, geom_to_trajectory):
    def file_to_trajectory(file_path):
        """Parse .geom or .castep file into ASE Atoms trajectory.

        Dispatches to geom_to_trajectory() or castep_to_trajectory() based on file extension.

        Returns:
            tuple: (trajectory, error_message) where error_message is None on success,
                   a warning string if there are non-fatal issues, or an error string on failure.
        """
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()

        if file_ext == '.geom':
            return geom_to_trajectory(file_path)
        elif file_ext == '.castep':
            return castep_to_trajectory(file_path)
        else:
            return [], f"Unsupported file extension '{file_ext}'. Please upload a .geom or .castep file."
    return (file_to_trajectory,)


@app.cell
def _(np, pd):
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
    return (trajectory_to_dataframe,)


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
def _(file_upload, mo, parse_error, trajectory):
    def _get_file_status():
        if not file_upload.value:
            return mo.md("_Supported formats: `.geom`, `.castep`_")

        filename = file_upload.value[0].name

        # Check for errors (no trajectory and error message)
        if not trajectory and parse_error:
            # Split error message and HTML file preview if present
            if "<pre style=" in parse_error:
                parts = parse_error.split("<pre style=", 1)
                error_text = parts[0].rstrip()
                file_html = "<pre style=" + parts[1]
                return mo.callout(
                    mo.vstack([
                        mo.md(f"**Error loading {filename}:**\\n\\n{error_text}"),
                        mo.accordion({"📄 Full file contents": mo.Html(file_html)})
                    ]),
                    kind="danger"
                )
            return mo.callout(
                mo.md(f"**Error loading {filename}:**\\n\\n{parse_error}"),
                kind="danger"
            )

        # Check for warnings (trajectory exists but with warning)
        if trajectory and parse_error:
            return mo.vstack([
                mo.md(f"✅ Loaded: **{filename}** ({len(trajectory)} iterations)"),
                mo.callout(mo.md(f"**Warning:** {parse_error}"), kind="warn")
            ])

        # Success with no issues
        if trajectory:
            return mo.md(f"✅ Loaded: **{filename}** ({len(trajectory)} iterations)")

        # Fallback
        return mo.md(f"⚠️ Loaded **{filename}** but no data extracted.")

    mo.vstack([
        mo.md("## Upload File"),
        file_upload,
        _get_file_status()
    ])
    return


@app.cell
def _(
    Path,
    file_to_trajectory,
    file_upload,
    np,
    tempfile,
    trajectory_to_dataframe,
):
    def parse_uploaded_file(uploaded_file):
        """Parse uploaded .geom or .castep file using the pipeline.

        Returns (trajectory, df, error_msg) where:
          - trajectory has normalized force_magnitude for visualization
          - error_msg is None on success, warning string for non-fatal issues, or error string on failure
        """
        if uploaded_file is None:
            return [], None, None

        suffix = f".{uploaded_file.name.split('.')[-1]}"
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(uploaded_file.contents)
                temp_path = Path(tmp.name)
        except Exception as e:
            return [], None, f"Failed to create temporary file: {e}"

        try:
            # Step 1: File → list[Atoms]
            trajectory, parse_msg = file_to_trajectory(temp_path)

            # If parsing completely failed, return early
            if not trajectory and parse_msg:
                return [], None, parse_msg

            # Step 2: list[Atoms] → DataFrame  
            df = trajectory_to_dataframe(trajectory)

            # Add force_magnitude array for visualization (normalized log scale 0-1)
            if trajectory and any('forces' in a.arrays for a in trajectory):
                all_mags = np.concatenate([
                    np.linalg.norm(a.arrays['forces'], axis=1) 
                    for a in trajectory if 'forces' in a.arrays
                ])
                if len(all_mags) and all_mags.min() > 0:
                    log_min, log_max = np.log10(all_mags.min()), np.log10(all_mags.max())
                    for atoms in trajectory:
                        if 'forces' in atoms.arrays:
                            mags = np.linalg.norm(atoms.arrays['forces'], axis=1)
                            # Avoid division by zero if all forces are identical
                            if log_max > log_min:
                                atoms.set_array('force_magnitude', (np.log10(mags) - log_min) / (log_max - log_min))
                            else:
                                atoms.set_array('force_magnitude', np.ones_like(mags) * 0.5)

            return trajectory, df, parse_msg
        except Exception as e:
            return [], None, f"Unexpected error while processing file: {e}"
        finally:
            try:
                temp_path.unlink()
            except:
                pass

    trajectory, df, parse_error = parse_uploaded_file(file_upload.value[0]) if file_upload.value else ([], None, None)
    return df, parse_error, trajectory


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
    if enthalpy_chart is not None:
        # Build rows for grid layout
        row1 = [c for c in [enthalpy_change_chart, force_chart] if c]
        row2 = [c for c in [disp_chart, stress_chart, volume_chart] if c]

        # Build tolerance controls (only show if corresponding chart exists)
        tolerances = [enthalpy_tolerance, force_tolerance, disp_tolerance]
        if stress_chart is not None:
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
    def view_trajectory(atoms_list,
                        model_style=1,
                        show_bonded_atoms=False,
                        color_by_force=False,
                        boundary=[[-0.1, 1.1], [-0.1, 1.1], [-0.1, 1.1]]):
        """Display ASE Atoms trajectory using weas-widget.

        Note: Changing parameters while animation is playing may cause a
        rendering error. This sometimes recovers on
        the next parameter change. Otherwise a page refresh may be required.
        """
        if not atoms_list:
            return mo.md("_No trajectory data available._")

        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]

        try:
            v = AtomsViewer(BaseWidget(guiConfig={"controls": {"enabled": False}}))
            v.atoms = ASEAdapter.to_weas(atoms_list[0]) if len(atoms_list) == 1 else [ASEAdapter.to_weas(a) for a in atoms_list]
            v.model_style = model_style
            v.boundary = boundary
            v.show_bonded_atoms = show_bonded_atoms
            v.color_type = "VESTA"

            if color_by_force and hasattr(atoms_list[0], 'arrays') and 'force_magnitude' in atoms_list[0].arrays:
                v.color_by = "force_magnitude"
                v.color_ramp = ["#440154", "#31688e", "#35b779", "#fde724"]

            return v._widget
        except Exception:
            # Race condition when widget is updated during animation
            return mo.callout(
                mo.md("⏳ **Viewer updating...** Stop the animation or wait a moment, then adjust a setting to refresh."),
                kind="warn"
            )
    return (view_trajectory,)


@app.cell
def _(mo, trajectory):
    # Viewer controls
    if len(trajectory) > 0:
        has_forces = hasattr(trajectory[0], 'arrays') and 'force_magnitude' in trajectory[0].arrays
        model_style_dropdown = mo.ui.dropdown(
            options={'Ball': 0, 'Ball and Stick': 1, 'Polyhedral': 2, 'Stick': 3},
            value='Ball and Stick', label='Model Style'
        )
        show_bonded = mo.ui.checkbox(label='Show bonded atoms outside boundary', value=False)
        color_by_force_cb = mo.ui.checkbox(label='Color by force', value=False) if has_forces else None

        # Boundary controls for unit cell visualization (debounced to prevent rapid updates)
        boundary_a_min = mo.ui.number(value=0.0, step=0.01, stop=0, label='**a**(min) = ', debounce=True)
        boundary_a_max = mo.ui.number(value=1.0, step=0.01, start=1, label='**a**(max) = ', debounce=True)
        boundary_b_min = mo.ui.number(value=0.0, step=0.01, stop=0, label='**b**(min) = ', debounce=True)
        boundary_b_max = mo.ui.number(value=1.0, step=0.01, start=1, label='**b**(max) = ', debounce=True)
        boundary_c_min = mo.ui.number(value=0.0, step=0.01, stop=0, label='**c**(min) = ', debounce=True)
        boundary_c_max = mo.ui.number(value=1.0, step=0.01, start=1, label='**c**(max) = ', debounce=True)

        _controls = [model_style_dropdown, show_bonded]
        if color_by_force_cb is not None:
            _controls.append(color_by_force_cb)

        _boundary_controls = mo.vstack([
            mo.hstack([boundary_a_min, boundary_a_max], justify='start'),
            mo.hstack([boundary_b_min, boundary_b_max], justify='start'),
            mo.hstack([boundary_c_min, boundary_c_max], justify='start'),
        ], justify='start')

        mo.output.append(mo.vstack([
            mo.hstack(_controls, justify='start'),
            mo.accordion({"⚙️ Boundary Settings": mo.vstack([
            mo.md('**Ranges of fractional coordinates:**'),
            _boundary_controls
            ])}),
        ], gap=1))
    else:
        has_forces = False
        model_style_dropdown = show_bonded = color_by_force_cb = None
        boundary_a_min = boundary_a_max = boundary_b_min = boundary_b_max = boundary_c_min = boundary_c_max = None
    return (
        boundary_a_max,
        boundary_a_min,
        boundary_b_max,
        boundary_b_min,
        boundary_c_max,
        boundary_c_min,
        color_by_force_cb,
        model_style_dropdown,
        show_bonded,
    )


@app.cell
def _(
    boundary_a_max,
    boundary_a_min,
    boundary_b_max,
    boundary_b_min,
    boundary_c_max,
    boundary_c_min,
    color_by_force_cb,
    max_iter,
    min_iter,
    mo,
    model_style_dropdown,
    show_bonded,
    trajectory,
    view_trajectory,
):
    # Clear previous output before rendering new widget
    mo.output.clear()

    if len(trajectory) > 0:
        filtered_trajectory = trajectory[min_iter - 1 : max_iter]
        boundary = [
            [boundary_a_min.value, boundary_a_max.value],
            [boundary_b_min.value, boundary_b_max.value],
            [boundary_c_min.value, boundary_c_max.value]
        ]
        viewer = view_trajectory(
            filtered_trajectory,
            model_style=model_style_dropdown.value,
            show_bonded_atoms=show_bonded.value if show_bonded.value else False,
            color_by_force=color_by_force_cb.value if color_by_force_cb.value else False,
            boundary=boundary
        )
        mo.output.append(mo.vstack([
            mo.md(f"_Showing **{len(filtered_trajectory)}** structures (iterations {min_iter} to {max_iter})_"),
            viewer
        ]))
    else:
        mo.output.append(mo.md("_Upload a file to view the structure trajectory._"))
    return


if __name__ == "__main__":
    app.run()
