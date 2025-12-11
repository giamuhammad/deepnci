import os
import numpy as np
import pandas as pd
import psi4
from scipy.ndimage import zoom
from pathlib import Path


# ============================================================
# KONFIGURASI
# ============================================================

INPUT_DIR = "./raw_data"
CUBE_OUT_DIR = "./density"
NPY_OUT_DIR = "./density_npy"

METHOD = "M06-2X"
PROP_METHOD = "B3LYP"
BASIS = "6-31G*"
TARGET_GRID = (137, 133, 124)

os.makedirs(CUBE_OUT_DIR, exist_ok=True)
os.makedirs(NPY_OUT_DIR, exist_ok=True)

psi4.core.set_num_threads(16)
psi4.set_memory("8GB")
#psi4.set_output_file(os.devnull, False)  # Redirect output ke null
psi4.core.be_quiet()  # Mode quiet untuk mengurangi output

psi4.set_options({
    "reference": "RKS",
    "cubeprop_tasks": ["density"],            # Only generate Dt.cube
    "dft_radial_points": 99,
    "dft_spherical_points": 590,
    "cubeprop_filepath": CUBE_OUT_DIR
})

# ============================================================
#  HELPER
# ============================================================

def load_cube_density(filename):
    """Parse cube file → kembalikan array density 3D."""
    with open(filename) as f:
        lines = f.readlines()

    idx = 2
    natoms = int(lines[idx].split()[0])
    idx += 1

    nx = int(lines[idx].split()[0]); idx += 1
    ny = int(lines[idx].split()[0]); idx += 1
    nz = int(lines[idx].split()[0]); idx += 1

    idx += natoms

    raw = " ".join(lines[idx:]).split()
    data = np.array(raw, dtype=float)

    return data.reshape((nx, ny, nz))


def resample_grid(vol, target):
    zoom_factors = (target[0]/vol.shape[0],
                    target[1]/vol.shape[1],
                    target[2]/vol.shape[2])
    return zoom(vol, zoom_factors, order=1).astype(np.float32)



# ============================================================
#  MAIN LOOP MOLEKUL
# ============================================================

all_props = []

xyz_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".xyz")])
print(f"Jumlah molekul: {len(xyz_files)}")
print("Mulai memproses...\n")

for fname in xyz_files:
    base = fname.replace(".xyz", "")
    fpath = os.path.join(INPUT_DIR, fname)

    print(f">>> Processing {fname}")


    with open(fpath, "r") as f:
        xyz_body = f.read()

    with open(fpath) as f:
        lines = f.read().strip().splitlines()
    

    mol = psi4.geometry(xyz_body)


    # -------------------------------
    # 2. DFT Energy + Wavefunction
    # -------------------------------
    E, wfn = psi4.energy(f"{METHOD}/{BASIS}", molecule=mol, return_wfn=True)

    # -------------------------------
    # 3. Frequency → Thermochemistry
    # -------------------------------
    psi4.frequency(f"{METHOD}/{BASIS}", molecule=mol)

    # -------------------------------
    # 4. cubeprop → menghasilkan Dt.cube
    # -------------------------------
    psi4.cubeprop(wfn)

    # PSI4 OUTPUT (selalu):
    # Da.cube, Db.cube, Ds.cube, Dt.cube, geom.xyz
    dt_file = "Dt.cube"

    dtf_path = os.path.join(CUBE_OUT_DIR, dt_file)
    if not os.path.exists(dtf_path):
        print(f"   ERROR: Dt.cube tidak ditemukan. SKIP.")
        continue

    # rename Dt.cube → nama_input.density.cube
    new_cube = os.path.join(CUBE_OUT_DIR, f"{base}.density.cube")
    os.rename(dtf_path, new_cube)

    # bersihkan file cube lain
    for extra in ["Da.cube", "Db.cube", "Ds.cube", "geom.xyz"]:
        if os.path.exists(extra):
            os.remove(extra)

    # -------------------------------
    # 5. Load density & resize grid
    # -------------------------------
    vol = load_cube_density(new_cube)
    vol_res = resample_grid(vol, TARGET_GRID)

    np.save(os.path.join(NPY_OUT_DIR, f"{base}.npy"), vol_res)


print("\n==== SELESAI ====")
print(f"Cube density disimpan di: {CUBE_OUT_DIR}")
print(f"NPY grid disimpan di:     {NPY_OUT_DIR}")
