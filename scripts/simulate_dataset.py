"""Generate pairs of 5e3- and 1e6-history proton dose simulations with TOPAS.

For every CT volume in ``--ct-dir`` (int16 Hounsfield units, as written by downsample_ct.py) one beam is
drawn: an incidence angle uniform in [0, 180) degrees (integer, rotation of the patient about the Y axis)
and an energy uniform in [--energy-min, --energy-max] MeV (rounded to 0.1 MeV), or the fixed --energy.
The same beam is then simulated with 5e3 and 1e6 primary histories.

Output in ``--output-dir`` (the layout read by the deep-learning code)::

    ct/CT_<id>.npy          CT (copy of the input volume)
    dose_5k/<id>_5k.npy     5e3-history dose, multiplied by 200 (1e6 / 5e3) to be on the 1e6 scale
    dose_1m/<id>_1M.npy     1e6-history dose
    simulations.csv         one row per TOPAS run: id, CT file, energy, angle, seed, histories, wall time

<id> is the last three characters of the CT file name (e.g. CT_001.npy -> 001).

Example (2 mm test split, energies of the test set):
    python scripts/simulate_dataset.py --ct-dir dataset_2mm/test/CT_scans --output-dir dataset_2mm/test \
        --energy-min 90 --energy-max 150
"""
import argparse
import csv
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[1]
HISTORIES = {"5k": 5_000, "1M": 1_000_000}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ct-dir", required=True, type=Path, help="downsampled CT volumes (.npy)")
    parser.add_argument("--output-dir", required=True, type=Path, help="dataset split directory to write")
    parser.add_argument("--voxel-size", type=float, default=2.0, help="isotropic voxel size in mm")
    energy = parser.add_mutually_exclusive_group(required=True)
    energy.add_argument("--energy", type=float, help="fixed beam energy in MeV")
    energy.add_argument("--energy-min", type=float, help="minimum beam energy in MeV (use with --energy-max)")
    parser.add_argument("--energy-max", type=float)
    parser.add_argument("--hu-table", type=Path, default=REPO_DIR / "topas" / "HUtoMaterialSchneider.txt",
                        help="TOPAS Schneider HU-to-material conversion file")
    parser.add_argument("--mcspec", type=Path, default=REPO_DIR / "topas" / "MCspec.txt",
                        help="TOPAS main parameter file (beam, physics, scoring)")
    parser.add_argument("--topas", default="topas", help="TOPAS executable (Geant4 data variables must be set)")
    parser.add_argument("--topas-seed", type=int, default=98905, help="i:Ts/Seed used for every run")
    parser.add_argument("--rng-seed", type=int, help="seed for drawing angles and energies")
    parser.add_argument("--first-id", type=int, default=0, help="skip CT files whose numeric id is below this")
    parser.add_argument("--max-volumes", type=int, help="stop after this many CT volumes")
    parser.add_argument("--previews", action="store_true", help="save a PNG of the central dose slice")
    args = parser.parse_args()
    if args.energy is None and args.energy_max is None:
        parser.error("--energy-min requires --energy-max")
    for path in (args.hu_table, args.mcspec):
        if not path.is_file():
            parser.error(f"{path} not found")
    if "/" in args.topas:  # TOPAS runs inside the work directory, so make a relative path absolute
        args.topas = str(Path(args.topas).resolve())
    return args


def write_ct_for_topas(ct, path):
    """TOPAS reads the CT rotated by -90 degrees in the X-Z plane, as raw int16."""
    np.rot90(ct.astype(np.int16), k=-1, axes=(0, 2)).copy().tofile(path)


def read_topas_dose(path, shape):
    """Read a binary TOPAS dose and rotate it back into the orientation of the CT."""
    dose = np.fromfile(path, dtype=np.float64).reshape(shape)
    return np.rot90(dose, k=1, axes=(0, 2)).copy()


def write_parameters(path, ct_bin, energy, angle, histories, dose_output, voxel_size, shape, topas_seed, hu_table):
    nz, ny, nx = shape
    lines = [
        f"i:Ts/Seed = {topas_seed}",
        f"d:So/Therapy/BeamEnergy = {energy} MeV",
        f's:Ge/Patient/InputFile = "{ct_bin.name}"',
        's:Ge/Patient/Type = "TsXiOPatient"',
        f's:Ge/Patient/InputDirectory = "{ct_bin.parent}/"',
        f"iv:Ge/Patient/NumberOfVoxelsZ = 1 {nz}",
        f"i:Ge/Patient/NumberOfVoxelsY = {ny}",
        f"i:Ge/Patient/NumberOfVoxelsX = {nx}",
        f"d:Ge/Patient/VoxelSizeX = {voxel_size:g} mm",
        f"d:Ge/Patient/VoxelSizeY = {voxel_size:g} mm",
        f"dv:Ge/Patient/VoxelSizeZ = 1 {voxel_size:g} mm",
        "d:So/Therapy/BeamPositionSpreadX = 0.30 cm",
        "d:So/Therapy/BeamPositionSpreadY = 0.30 cm",
        f"i:So/Therapy/NumberOfHistoriesInRun = {histories}",
        f"includeFile = {hu_table}",
        "d:Ge/PatientBox/TransX = 0.0 cm",
        "d:Ge/PatientBox/TransY = 0.0 cm",
        "d:Ge/PatientBox/TransZ = 0.0 cm",
        "d:Ge/Patient/TransX = 0.0 cm",
        "d:Ge/Patient/TransY = 0.0 cm",
        "d:Ge/Patient/TransZ = 0.0 cm",
        "d:Ge/Patient/RotX = 0.0 deg",
        f"d:Ge/Patient/RotY = {angle} deg",
        "d:Ge/Patient/RotZ = 0.0 deg",
        f's:Sc/DoseAtPatient/OutputFile = "{dose_output}"',
    ]
    path.write_text("\n".join(lines) + "\n")


def run_topas(topas, mcspec_template, parameters, work_dir):
    """Run TOPAS on a copy of the main file that includes ``parameters``; returns the wall time in seconds."""
    main_file = work_dir / "MCspec.txt"
    main_file.write_text(mcspec_template.replace("PARAMETERS_FILE", str(parameters.resolve())))
    start = time.perf_counter()
    subprocess.run([topas, str(main_file.resolve())], cwd=work_dir, check=True)
    return time.perf_counter() - start


def main():
    args = parse_args()
    if args.rng_seed is not None:
        np.random.seed(args.rng_seed)

    out = args.output_dir
    dirs = {name: out / name for name in ("ct", "ct_bin", "dose_5k", "dose_1m", "dose_5k_bin", "dose_1m_bin",
                                          "topas_work")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    mcspec_template = args.mcspec.read_text()
    log_path = out / "simulations.csv"
    log_fields = ["sample_id", "ct_file", "energy_mev", "angle_deg", "topas_seed", "histories", "seconds"]
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(log_fields)

    ct_files = sorted(args.ct_dir.glob("*.npy"))
    n_done = 0
    for ct_path in ct_files:
        sample_id = ct_path.stem[-3:]
        if int(sample_id) < args.first_id:
            continue
        if args.max_volumes is not None and n_done >= args.max_volumes:
            break

        angle = np.random.randint(0, 180)
        energy = args.energy if args.energy is not None else round(np.random.uniform(args.energy_min,
                                                                                     args.energy_max), 1)
        ct = np.load(ct_path)
        shutil.copy(ct_path, dirs["ct"] / f"CT_{sample_id}.npy")
        ct_bin = (dirs["ct_bin"] / f"{sample_id}.bin").resolve()
        write_ct_for_topas(ct, ct_bin)
        print(f"{ct_path.name}: id {sample_id}, angle {angle} deg, energy {energy} MeV", flush=True)

        for label, histories in HISTORIES.items():
            dose_output = dirs[f"dose_{label.lower()}_bin"] / f"dose_{label}_{sample_id}"
            parameters = dirs["topas_work"] / "parameters.txt"
            write_parameters(parameters, ct_bin, energy, angle, histories, dose_output.resolve(), args.voxel_size,
                             ct.shape, args.topas_seed, args.hu_table.resolve())
            seconds = run_topas(args.topas, mcspec_template, parameters, dirs["topas_work"])

            dose = read_topas_dose(f"{dose_output}.bin", ct.shape)
            if label == "5k":
                dose = dose * (HISTORIES["1M"] / HISTORIES["5k"])  # bring to the 1e6-history scale
            npy_path = dirs[f"dose_{label.lower()}"] / f"{sample_id}_{label}.npy"
            np.save(npy_path, dose)
            if args.previews:
                import matplotlib.pyplot as plt
                plt.imsave(npy_path.with_suffix(".png"), dose[:, dose.shape[1] // 2, :])

            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([sample_id, ct_path.name, energy, angle, args.topas_seed, histories,
                                        f"{seconds:.3f}"])
            print(f"  {label}: {seconds:.1f} s -> {npy_path}", flush=True)
        n_done += 1


if __name__ == "__main__":
    main()
