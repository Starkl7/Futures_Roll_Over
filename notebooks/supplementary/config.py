"""Shared constants for all Supplementary_notebooks."""
from pathlib import Path

RESULTS_DIR = Path('../../results')
SEAGATE_DIR = Path('/Volumes/SEAGATE/Databento_Futures')
FIGS_DIR    = Path('figures')
TICK        = 0.25
MULT        = 50

WINDOWS_META = {
    'W1': dict(
        front='ESU4', back='ESZ4', roll_start='2024-09-12',
        result_key='ESU4_ESZ4_20240912',
        days=['2024-09-12','2024-09-13','2024-09-15','2024-09-16',
              '2024-09-17','2024-09-18','2024-09-19'],
        day_labels=['D1\nThu','D2\nFri','D3\nSun*','D4\nMon',
                    'D5\nTue','D6\nFOMC','D7\nExp'],
        rth_start_min=12*60+30, rth_end_min=19*60+15,
        fomc_date='2024-09-18',
    ),
    'W2': dict(
        front='ESZ4', back='ESH5', roll_start='2024-12-12',
        result_key='ESZ4_ESH5_20241212',
        days=['2024-12-12','2024-12-13','2024-12-15','2024-12-16',
              '2024-12-17','2024-12-18','2024-12-19'],
        day_labels=['D1\nThu','D2\nFri','D3\nSun*','D4\nMon',
                    'D5\nTue','D6\nFOMC','D7\nExp'],
        rth_start_min=13*60+30, rth_end_min=20*60+15,
        fomc_date='2024-12-18',
    ),
    'W3': dict(
        front='ESH5', back='ESM5', roll_start='2025-03-13',
        result_key='ESH5_ESM5_20250313',
        days=['2025-03-13','2025-03-14','2025-03-16','2025-03-17',
              '2025-03-18','2025-03-19','2025-03-20'],
        day_labels=['D1\nThu','D2\nFri','D3\nSun*','D4\nMon',
                    'D5\nTue','D6\nWed','D7\nThu'],
        rth_start_min=12*60+30, rth_end_min=19*60+15,
        fomc_date='2025-03-19',
    ),
    'W4': dict(
        front='ESM5', back='ESU5', roll_start='2025-06-12',
        result_key='ESM5_ESU5_20250612',
        days=['2025-06-12','2025-06-13','2025-06-15','2025-06-16',
              '2025-06-17','2025-06-18','2025-06-19'],
        day_labels=['D1\nThu','D2\nFri','D3\nSun*','D4\nMon',
                    'D5\nTue','D6\nFOMC','D7\nExp'],
        rth_start_min=12*60+30, rth_end_min=19*60+15,
        fomc_date='2025-06-18',
    ),
}

BASELINE_STATS = {
    'W1': dict(n=24, wr=0.708, gross=5.62,   pf=0.94),
    'W2': dict(n=45, wr=0.556, gross=-45.62,  pf=0.90),
    'W3': dict(n=48, wr=0.125, gross=-675.0,  pf=0.16),
    'W4': dict(n=20, wr=0.300, gross=-50.0,   pf=0.73),
}

UPDATED_STATS = {
    'W1': dict(n=19, wr=0.737, gross=57.50),
    'W2': dict(n=23, wr=0.696, gross=94.38),
    'W3': dict(n=31, wr=0.258, gross=-162.50),
    'W4': dict(n=13, wr=0.462, gross=0.00),
}

WIN_COLORS = {
    'W1': '#2ecc71',
    'W2': '#3498db',
    'W3': '#e74c3c',
    'W4': '#f39c12',
}


def save_fig(fig, name):
    import matplotlib.pyplot as plt
    FIGS_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved --> figures/{name}')
