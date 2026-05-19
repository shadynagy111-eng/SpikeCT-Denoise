"""
scripts/eval/per_patient_direct_psnr.py
----------------------------------------
Per-patient PSNR evaluation for SNN-Direct-IF and SNN-Direct-LIF.

Runs inference on all 200 slices per test patient, reports mean PSNR,
and verifies that the overall mean matches the expected values:
  SNN-Direct-IF:  29.4094 dB
  SNN-Direct-LIF: 29.1467 dB

Usage:
  cd /mnt/c/Projects/SpikeCT-Denoise
  python scripts/eval/per_patient_direct_psnr.py
"""

import sys
import numpy as np
import torch
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.seed import set_seed
from src.models.snn_direct import SNNDirectIF, SNNDirectLIF, direct_encode
from src.data.dataset import load_split
from spikingjelly.activation_based import functional
import h5py

set_seed(42)

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
H5         = Path.home() / 'SpikeCT_Data/mayo_50patients_fast.h5'
SPLIT_JSON = Path.home() / 'SpikeCT_Data/patient_split_50.json'
CKPT_DIR   = PROJECT_ROOT / 'checkpoints'

TEST_PATIENTS = ['C004', 'C050', 'C111', 'C121', 'C249']

# Known reference values for sanity check
CNN_PER_PATIENT = {
    'C004': 30.62, 'C050': 31.36, 'C111': 29.44,
    'C121': 28.84, 'C249': 30.39
}
EXPECTED_OVERALL = {
    'SNN-Direct-IF':  29.4094,
    'SNN-Direct-LIF': 29.1467,
}


def infer_direct(model, arr, T):
    """Run direct coding inference on a single (512,512) float32 slice."""
    img = torch.from_numpy(arr).unsqueeze(0)             # (1, H, W)
    di  = direct_encode(img, T=T).unsqueeze(1).to(DEVICE) # (T, 1, 1, H, W)
    functional.reset_net(model)
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            pred = model(di)
    return pred.squeeze().cpu().float().numpy().clip(0, 1)


def evaluate_model(model, model_name, T):
    """Evaluate model on all test patients. Returns per-patient means."""
    model.eval()
    per_patient = {}
    all_psnrs   = []

    print(f'\n  {model_name}:')
    with h5py.File(H5, 'r') as hf:
        for pid in TEST_PATIENTS:
            low_vol  = hf[pid]['low_dose'][:]   # (200, 512, 512) float32
            full_vol = hf[pid]['full_dose'][:]

            psnrs = []
            for i in range(low_vol.shape[0]):
                low  = low_vol[i].astype(np.float32)
                full = full_vol[i].astype(np.float32)
                pred = infer_direct(model, low, T=T)
                psnrs.append(psnr_fn(full, pred, data_range=1.0))

            mean_psnr = float(np.mean(psnrs))
            per_patient[pid] = mean_psnr
            all_psnrs.extend(psnrs)
            gap = mean_psnr - CNN_PER_PATIENT[pid]
            print(f'    {pid}: {mean_psnr:.4f} dB  (CNN={CNN_PER_PATIENT[pid]:.2f}, gap={gap:+.2f} dB)')

    overall = float(np.mean(all_psnrs))
    expected = EXPECTED_OVERALL[model_name]
    diff = abs(overall - expected)
    status = '✓ MATCH' if diff < 0.01 else f'✗ MISMATCH (expected {expected:.4f})'
    print(f'    Overall mean: {overall:.4f} dB  {status}')
    return per_patient, overall


def main():
    print('='*60)
    print('Per-Patient PSNR — SNN-Direct-IF and SNN-Direct-LIF')
    print('='*60)
    print(f'Device: {DEVICE}')

    split = load_split(SPLIT_JSON)
    assert set(split['test']) == set(TEST_PATIENTS), \
        f'Test split mismatch: {split["test"]} vs {TEST_PATIENTS}'

    # ── Load SNN-Direct-IF ────────────────────────────────────────────────────
    print('\nLoading SNN-Direct-IF...')
    dif_ckpt = torch.load(CKPT_DIR / 'snn_direct_if_best.pth',
                          map_location=DEVICE, weights_only=False)
    dif = SNNDirectIF(T=8).to(DEVICE)
    dif.load_state_dict(dif_ckpt['model_state_dict'])
    print(f'  Checkpoint: epoch {dif_ckpt["epoch"]}, val_psnr={dif_ckpt["val_psnr"]:.4f} dB')

    # ── Load SNN-Direct-LIF ───────────────────────────────────────────────────
    print('Loading SNN-Direct-LIF...')
    dlf_ckpt = torch.load(CKPT_DIR / 'snn_direct_lif_best.pth',
                          map_location=DEVICE, weights_only=False)
    dlf = SNNDirectLIF(T=8).to(DEVICE)
    dlf.load_state_dict(dlf_ckpt['model_state_dict'])
    print(f'  Checkpoint: epoch {dlf_ckpt["epoch"]}, val_psnr={dlf_ckpt["val_psnr"]:.4f} dB')

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print('\nRunning inference (200 slices × 5 patients × 2 models)...')
    dif_results, dif_overall = evaluate_model(dif, 'SNN-Direct-IF',  T=8)
    dlf_results, dlf_overall = evaluate_model(dlf, 'SNN-Direct-LIF', T=8)

    # ── Print final table ─────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('RESULTS TABLE')
    print('='*60)
    print(f'{"Patient":<10} {"CNN (dB)":>10} {"SNN-Direct-IF":>15} {"SNN-Direct-LIF":>16} {"Gap IF":>8} {"Gap LIF":>9}')
    print('-'*68)

    for pid in TEST_PATIENTS:
        cnn_v  = CNN_PER_PATIENT[pid]
        dif_v  = dif_results[pid]
        dlf_v  = dlf_results[pid]
        gap_dif = dif_v - cnn_v
        gap_dlf = dlf_v - cnn_v
        print(f'{pid:<10} {cnn_v:>10.4f} {dif_v:>15.4f} {dlf_v:>16.4f} '
              f'{gap_dif:>+8.2f} {gap_dlf:>+9.2f}')

    print('-'*68)
    # Overall means weighted equally across patients
    dif_pat_mean = float(np.mean([dif_results[p] for p in TEST_PATIENTS]))
    dlf_pat_mean = float(np.mean([dlf_results[p] for p in TEST_PATIENTS]))
    cnn_pat_mean = float(np.mean([CNN_PER_PATIENT[p] for p in TEST_PATIENTS]))
    print(f'{"Mean":<10} {cnn_pat_mean:>10.4f} {dif_pat_mean:>15.4f} {dlf_pat_mean:>16.4f}')

    print()
    print('Sanity check (per-patient mean vs known overall):')
    for name, val, expected in [
        ('SNN-Direct-IF',  dif_overall, EXPECTED_OVERALL['SNN-Direct-IF']),
        ('SNN-Direct-LIF', dlf_overall, EXPECTED_OVERALL['SNN-Direct-LIF']),
    ]:
        diff   = abs(val - expected)
        status = '✓' if diff < 0.01 else '✗ CHECK CHECKPOINT'
        print(f'  {name}: {val:.4f} dB  (expected {expected:.4f})  {status}')

    print()
    print('Note: per-patient mean above is mean of 5 patient means (equal weight).')
    print('Overall test mean weights all 1,000 slices equally — may differ slightly.')


if __name__ == '__main__':
    main()