"""
Download and validate TerraClimate per-year NetCDF files.

- Validates every existing file by opening it with netCDF4 and reading one value.
- Re-downloads any file that is missing, too small, or fails validation.
- Safe to re-run: already-valid files are skipped instantly.
- Uses wget for downloads (handles resume, retries, progress).

Usage:
    python download_terraclimate.py              # ppt + pet
    python download_terraclimate.py --vars ppt  # ppt only
"""

import argparse
import os
import subprocess
import sys

import netCDF4 as nc

# ── Configuration ────────────────────────────────────────────────────────────

BASE_URL   = 'https://climate.northwestknowledge.net/TERRACLIMATE-DATA'
INPUT_DIR  = os.path.join(os.path.dirname(__file__), 'input')
YEAR_START = 1958
YEAR_END   = 2025
MIN_BYTES  = 50 * 1024 * 1024   # 50 MB — any valid TerraClimate file is larger

# Variable name inside each NetCDF file (used for the read-one-value check)
VAR_NAMES = {
    'ppt':  'ppt',
    'pet':  'pet',
    'tmax': 'tmax',
    'tmin': 'tmin',
    'tmean': 'tmean',
    'ws':   'ws',
    'vpd':  'vpd',
    'aet':  'aet',
    'def':  'def',
    'swe':  'swe',
    'soil': 'soil',
    'srad': 'srad',
    'pdsi': 'PDSI',
    'PDSI': 'PDSI',
    'q':    'q',
    'vap':  'vap',
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def validate(path, var):
    """
    Return (ok, reason) for a file.
    Opens the file with netCDF4 and reads one value from the variable array
    to catch partial writes, truncation, and HDF5-level corruption.
    """
    if not os.path.exists(path):
        return False, 'missing'
    size = os.path.getsize(path)
    if size < MIN_BYTES:
        return False, f'too small ({size/1e6:.1f} MB)'
    try:
        with nc.Dataset(path, 'r') as ds:
            if var not in ds.variables:
                # Try to find the right variable name
                candidates = [v for v in ds.variables if not v.startswith('_')]
                return False, f'variable {var!r} not found; has {candidates}'
            _ = ds.variables[var][0, 0, 0]   # read one scalar — triggers full header + chunk decode
        return True, 'ok'
    except Exception as e:
        return False, str(e)


def download(url, dest):
    """
    Download url → dest using wget.
    Flags:
      -c          continue/resume a partial download
      --tries=5   retry up to 5 times on transient errors
      -q          quiet (no progress bar spam in logs)
      --show-progress  single-line progress bar
    """
    tmp = dest + '.part'
    # If a partial file exists from a previous interrupted run, wget -c resumes it
    cmd = [
        'wget', '-c', '--tries=5', '--timeout=60',
        '--show-progress', '-q',
        '-O', tmp, url,
    ]
    print(f'  Downloading {os.path.basename(dest)} ...', flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f'  ERROR: wget failed (exit {result.returncode})', file=sys.stderr)
        return False
    os.replace(tmp, dest)   # atomic rename — dest is never left in a partial state
    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vars', nargs='+', default=['ppt', 'pet'],
                        help='Variables to download (default: ppt pet)')
    parser.add_argument('--years', nargs=2, type=int,
                        default=[YEAR_START, YEAR_END],
                        metavar=('START', 'END'),
                        help=f'Year range (default: {YEAR_START} {YEAR_END})')
    args = parser.parse_args()

    os.makedirs(INPUT_DIR, exist_ok=True)
    years = range(args.years[0], args.years[1] + 1)

    total_ok = total_fixed = total_failed = 0

    for var in args.vars:
        nc_var = VAR_NAMES.get(var, var)
        bad = []

        print(f'\n[{var}] Validating {len(years)} files ...')
        for year in years:
            fname = f'TerraClimate_{var}_{year}.nc'
            path  = os.path.join(INPUT_DIR, fname)
            ok, reason = validate(path, nc_var)
            if ok:
                total_ok += 1
            else:
                print(f'  BAD  {fname}  ({reason})')
                bad.append((year, fname, path))

        if not bad:
            print(f'  All {len(years)} files valid.')
            continue

        print(f'\n[{var}] Re-downloading {len(bad)} files ...')
        for year, fname, path in bad:
            url = f'{BASE_URL}/{fname}'
            # Remove corrupt file so wget starts fresh (not -c resume of garbage)
            if os.path.exists(path):
                os.remove(path)
            part = path + '.part'
            if os.path.exists(part):
                os.remove(part)

            ok = download(url, path)
            if not ok:
                total_failed += 1
                continue

            # Validate the freshly downloaded file
            ok2, reason2 = validate(path, nc_var)
            if ok2:
                total_fixed += 1
            else:
                print(f'  STILL BAD after download: {fname} ({reason2})', file=sys.stderr)
                total_failed += 1

    print(f'\n{"="*50}')
    print(f'Valid (no action): {total_ok}')
    print(f'Fixed (re-downloaded): {total_fixed}')
    print(f'Failed: {total_failed}')
    if total_failed:
        print('Re-run this script to retry failed downloads.', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
