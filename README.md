# TOPAS dataset generation for proton dose denoising

Monte Carlo data generation for the paper **"Deep learning for fast Monte-Carlo proton dose denoising"**
(J. Jawaid, M. Spezialetti, G. Stilo, F. Mignosi, A. Langiu, M. Schwarz).

For every CT volume (2 mm isotropic voxels, 256×32×256) one proton beam is simulated twice with [TOPAS](https://www.topasmc.org/): with
5×10³ and with 10⁶ primary histories. The resulting pairs are the input and the target of the denoising
network in [MC_denoising_deep_learning](https://github.com/junaidjawaid1/MC_denoising_deep_learning).

## Repository structure

```
scripts/
  downsample_ct.py       1 mm CT slabs -> 2 mm simulation grid
  simulate_dataset.py    draws the beams, runs TOPAS (5e3 and 1e6 histories), writes the dataset
topas/
  MCspec.txt             TOPAS main file: beam, physics, geometry and scoring
  HUtoMaterialSchneider.txt   HU-to-material conversion (Schneider et al. 2000)
slurm/simulate.sbatch    example SLURM job
requirements.txt         Python dependencies
CITATION.cff, LICENSE
```

## Requirements

* **TOPAS 3.9** (Geant4 10.7) and its data sets (G4EMLOW 7.13, G4NDL 4.6, PhotonEvaporation 5.7,
  RadioactiveDecay 5.6, G4ENSDFSTATE 2.3, G4TENDL 1.3.2, G4PII 1.3, RealSurface 2.2, G4ABLA 3.1).
  `topas` must be on the `PATH` (or pass `--topas`) and the Geant4 data variables must be set, e.g.
  `TOPAS_G4_DATA_DIR`.
* Python 3.10 with `pip install -r requirements.txt`.

`topas/HUtoMaterialSchneider.txt` is the Schneider HU-to-material conversion used for the paper
(TOPAS's built-in Schneider converter).

## Pipeline

### 1. CT preparation

CTs come from [Head-Neck-PET-CT](https://doi.org/10.7937/K9/TCIA.2017.8oje5q00) (TCIA) and from the
[COVID-19 rural U.S. population chest imaging collection](https://doi.org/10.1038/s41597-020-00741-6).
Each CT was converted from DICOM to a 512×512×512 array at 1 mm³ (cubic interpolation) and placed so that
the couch is at the bottom, out of the beam path. Slabs of 512 × 64 × 512 mm were then extracted and saved
as `.npy` files in Hounsfield units, named with a three-digit ID (e.g. `CT_001.npy`). These slabs are
the input of this repository.

### 2. Downsampling

```bash
python scripts/downsample_ct.py --input-dir ct_1mm/train --output-dir dataset_2mm/train/CT_scans
```

Nearest-neighbour downsampling to 256×32×256 (2 mm), then a 180-degree rotation in the X–Z plane.
Repeat for `validation` and `test`.

### 3. Simulation

```bash
python scripts/simulate_dataset.py --ct-dir dataset_2mm/test/CT_scans --output-dir dataset_2mm/test \
    --energy-min 90 --energy-max 150
```

For each CT the script draws an incidence angle uniformly in [0, 180) degrees (integer, patient rotation
about the Y axis) and an energy uniformly in the given range (0.1 MeV steps), and runs TOPAS with
5×10³ and 10⁶ histories for that beam. Energies used in the paper:

| Split | Energy |
|---|---|
| train | 150 MeV (`--energy 150`, first experiments) plus 90–145 MeV (`--energy-min 90 --energy-max 145`) |
| validation, test | 90–150 MeV |

Output in `--output-dir`, ready for the deep-learning repository:

```
ct/CT_<id>.npy           CT on the simulation grid (Hounsfield units)
dose_5k/<id>_5k.npy      5e3-history dose multiplied by 200, i.e. on the 1e6-history scale
dose_1m/<id>_1M.npy      1e6-history dose
simulations.csv          id, CT file, energy, angle, TOPAS seed, histories and wall time of every run
ct_bin/, dose_*_bin/     raw TOPAS input and output (can be deleted)
```

`simulations.csv` records the beam of every sample and the simulation times reported in Table 4
of the paper. Use `--rng-seed` to make the drawn angles and energies repeatable, and `--first-id` /
`--max-volumes` to split a long run into several jobs.

## Simulation settings

| | |
|---|---|
| Particle | protons, energy spread 0.757504 % |
| Beam | Gaussian position, σ = 3 mm in X and Y (cut-off 10 cm); Gaussian angular spread 3.2 mrad |
| Geometry | `TsXiOPatient` from the CT, Schneider HU-to-material conversion, vacuum world |
| Physics | g4em-standard_opt4, g4h-phy_QGSP_BIC_HP, g4decay, g4ion-binarycascade, g4h-elastic_HP, g4stopping |
| Scoring | `DoseToMedium` on the patient grid, binary output |
| Histories | 5×10³ and 10⁶ per beam |
| Threads | 16 (`i:Ts/NumberOfThreads`) |
| Seed | `i:Ts/Seed = 98905` for every run |

## Citation

```bibtex
@article{jawaid2026proton,
  title   = {Deep learning for fast Monte-Carlo proton dose denoising},
  author  = {Jawaid, Junaid and Spezialetti, Matteo and Stilo, Giovanni and Mignosi, Filippo and
             Langiu, Alessio and Schwarz, Marco},
  year    = {2026},
  note    = {TODO: journal, volume, DOI}
}
```

Please also cite TOPAS (Faddegon et al., *Physica Medica* 72, 2020) and the CT collections above.

## License

MIT; see [LICENSE](LICENSE). TOPAS and Geant4 are distributed under their own licenses.
