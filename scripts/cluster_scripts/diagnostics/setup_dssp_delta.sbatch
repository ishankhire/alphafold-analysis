#!/bin/bash
#SBATCH --job-name=SetupDSSP
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4g
#SBATCH --time=00:30:00
#SBATCH --constraint="scratch"
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhire3@gatech.edu

set -euo pipefail

module reset

MICROMAMBA_EXE="$HOME/.local/bin/micromamba"
MAMBA_ROOT_PREFIX="$HOME/.micromamba"
DSSP_PREFIX="$HOME/dssp_env"
export MAMBA_ROOT_PREFIX

mkdir -p "$HOME/.local/bin"

if [ ! -x "$MICROMAMBA_EXE" ]; then
    SETUP_TMP=$(mktemp -d "${TMPDIR:-/tmp}/micromamba-setup.XXXXXX")

    curl -LsS \
        https://micro.mamba.pm/api/micromamba/linux-64/latest \
        -o "$SETUP_TMP/micromamba.tar.bz2"

    tar -xjf "$SETUP_TMP/micromamba.tar.bz2" \
        -C "$SETUP_TMP" bin/micromamba

    install -m 755 "$SETUP_TMP/bin/micromamba" "$MICROMAMBA_EXE"
    rm -r "$SETUP_TMP"
fi

"$MICROMAMBA_EXE" --version

if [ -d "$DSSP_PREFIX/conda-meta" ]; then
    "$MICROMAMBA_EXE" install \
        --yes \
        --prefix "$DSSP_PREFIX" \
        --channel conda-forge \
        dssp=4.6.1
else
    "$MICROMAMBA_EXE" create \
        --yes \
        --prefix "$DSSP_PREFIX" \
        --channel conda-forge \
        dssp=4.6.1
fi

"$DSSP_PREFIX/bin/mkdssp" --version
