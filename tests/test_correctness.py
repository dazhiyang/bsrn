"""Offline correctness checks using synthetic data and published SPA values."""
import gzip
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from pydantic import ValidationError

from bsrn.archive import LR0001, LR0100, LR_SPECS
from bsrn.io.reader import read_bsrn_archive
from bsrn.physics.geometry import get_solar_position
from bsrn.qc.ppl import ghi_ppl_test, bni_ppl_test, dhi_ppl_test, lwd_ppl_test
from bsrn.modeling.separation import erbs_separation


class CorrectnessTests(unittest.TestCase):
    """Check observable scientific and archive behavior without remote services."""

    def test_ghi_limits_and_missing(self):
        """Check inclusive noon limits, exceedances and missing measurements."""
        ghi = np.array([-4.01, -4, 1600, 1600.01, np.nan])
        np.testing.assert_array_equal(
            ghi_ppl_test(ghi, 0, 1000), [False, True, True, False, False])

    def test_dhi_limits(self):
        """Check diffuse irradiance limits at a zenith of zero degrees."""
        np.testing.assert_array_equal(
            dhi_ppl_test(np.array([-4.01, -4, 1000, 1000.01]), 0, 1000),
            [False, True, True, False])

    def test_bni_limits_and_missing(self):
        """Check beam irradiance limits and missing measurements."""
        np.testing.assert_array_equal(
            bni_ppl_test(np.array([-5, -4, 1000, 1001, np.nan]), 1000),
            [False, True, True, False, False])

    def test_longwave_limits(self):
        """Check both longwave boundaries and adjacent invalid values."""
        np.testing.assert_array_equal(
            lwd_ppl_test(np.array([39.99, 40, 700, 700.01, np.nan])),
            [False, True, True, False, False])

    def test_nighttime_limits(self):
        """Check that below-horizon solar geometry has finite QC limits."""
        self.assertTrue(ghi_ppl_test(100, 120, 1000))
        self.assertFalse(ghi_ppl_test(101, 120, 1000))
        self.assertTrue(dhi_ppl_test(50, 120, 1000))
        self.assertFalse(dhi_ppl_test(51, 120, 1000))

    def test_spa_published_example(self):
        """Compare to Reda and Andreas SPA example (NREL/TP-560-34302).

        The package fixes delta_t at 69.1 s rather than the example's 67 s;
        the 0.001 degree tolerance accommodates that documented difference.
        Reference: https://www.nrel.gov/docs/fy08osti/34302.pdf
        """
        times = pd.DatetimeIndex(['2003-10-17 19:30:30'], tz='UTC')
        out = get_solar_position(times, 39.742476, -105.1786, 1830.14,
                                 pressure=820, temp=11)
        self.assertAlmostEqual(out.apparent_zenith.iloc[0], 50.111622, delta=0.001)
        self.assertAlmostEqual(out.azimuth.iloc[0], 194.340241, delta=0.001)

    def test_timestamp_resolution_and_timezone(self):
        """Representing the same instants must preserve solar position."""
        times = pd.date_range('2025-06-21 10:00', periods=3, tz='UTC')
        expected = get_solar_position(times, 40, 0)
        for alternative in [times.as_unit('us'), times.tz_convert('Asia/Shanghai')]:
            np.testing.assert_allclose(
                get_solar_position(alternative, 40, 0), expected, atol=1e-10)

    def test_archive_roundtrip(self):
        """Round-trip a synthetic full February, including a missing value."""
        ghi = np.full(28 * 1440, 123.0)
        ghi[123] = np.nan
        columns = {name: np.full(28 * 1440, np.nan)
                   for name in LR_SPECS['LR0100'] if name != 'yearMonth'}
        columns['ghi_avg'] = ghi
        record = LR0100(yearMonth='2025-02', **columns)
        header = LR0001(stationNumber=94, month=2, year=2025, version=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'qiq0225.dat.gz'
            with gzip.open(path, 'wt', encoding='ascii') as stream:
                stream.write(header.get_bsrn_format() + '\n')
                stream.write(record.get_bsrn_format())
            out = read_bsrn_archive(path, strict=True)
        self.assertEqual((out['station_code'], out['year'], out['month']),
                         ('qiq', 2025, 2))
        # Raw archive records retain the BSRN missing-value sentinel.
        expected = ghi.copy()
        expected[123] = -999
        np.testing.assert_allclose(out['lr0100'].ghi_avg, expected)
        self.assertEqual(out['metadata_lrs']['lr0001'].stationNumber, 94)

    def test_wrong_month_length_rejected(self):
        """Reject a minute vector that cannot represent the specified month."""
        with self.assertRaises(ValidationError):
            LR0100(yearMonth='2025-02', ghi_avg=np.ones(1440))

    def test_separation_energy_closure(self):
        """Separated irradiance must reconstruct the daytime measurement."""
        times = pd.date_range('2025-06-21 10:00', periods=4, freq='h', tz='UTC')
        ghi = np.array([500., 600., 700., 600.])
        out = erbs_separation(times, ghi, 40, 0)
        zenith = get_solar_position(times, 40, 0).zenith.to_numpy()
        np.testing.assert_allclose(
            out.dhi + out.bni * np.cos(np.radians(zenith)), ghi, atol=1e-8)
        self.assertTrue(out.k.between(0, 1).all())


if __name__ == '__main__':
    unittest.main()
