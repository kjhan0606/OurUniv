import io
import hashlib
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import cf4_r2_sdss_mock_stream_summary as summary
from scripts.cf4_r2_sdss_mock_stream_summary import summarize_catalog


HEADER = "ID,RA,Dec,z_true,z_obs,z_obs_cen,cen_flag,subhalomass,parenthalomass,deVMag_r,gmr,kcor,extinction_r,rtrue,r,er,strue,s,es,itrue,i,ei,x,y,z,vxcen,vycen,vzcen,vx,vy,vz,nbar,logdist_true,logdist,logdist_err,logdist_alpha\n"


def row(gid, central, logdist, truth, sigma, mass, zcen, vx, vy, vz, submass=None):
    submass = mass if submass is None else submass
    values = [gid, 1, 2, 0.02, 0.021, zcen, central, submass, mass,
              15, 0.8, 0.1, 0.02, 1, 1, 0.01, 2.5, 2.5, 0.02,
              2, 2, 0.02, 10, 20, 30, vx, vy, vz,
              vx if central else vx + 10, vy if central else vy + 10,
              vz if central else vz + 10, 0.001, truth, logdist, sigma, 0]
    return ",".join(map(str, values)) + "\n"


class SDSSMockStreamSummaryTests(unittest.TestCase):
    def test_composite_host_key_and_group_mean_uncertainty(self):
        content = HEADER
        content += row(100, 1, 0.10, 0.00, 0.10, 14.0, 0.02, 100, 200, 300)
        content += row(100, 0, -0.10, 0.00, 0.10, 14.0, 0.02, 100, 200, 300)
        content += row(200, 0, 0.20, 0.00, 0.20, 13.0, 0.04, 400, 500, 600, 12.0)
        got = summarize_catalog("mocks/MOCK_HAMHOD_SDSS_v5_R19051.0_err_corr", io.BytesIO(content.encode()))
        self.assertEqual(got["galaxies"], 3)
        self.assertEqual(got["host_groups"], 2)
        self.assertEqual(got["groups_without_selected_central"], 1)
        self.assertEqual(got["bins"]["2-4"]["groups"], 1)
        self.assertAlmostEqual(got["bins"]["2-4"]["standardized_residual_sd"], 1.0)

    def test_repeated_id_cannot_cross_host_keys(self):
        content = HEADER
        content += row(100, 1, 0.10, 0.00, 0.10, 14.0, 0.02, 100, 200, 300)
        content += row(100, 0, 0.10, 0.00, 0.10, 13.0, 0.04, 400, 500, 600, 12.0)
        with self.assertRaisesRegex(ValueError, "repeated ID maps to multiple host keys"):
            summarize_catalog("mocks/MOCK_HAMHOD_SDSS_v5_R19051.0_err_corr", io.BytesIO(content.encode()))

    def test_host_key_cannot_merge_distinct_ids_or_two_centrals(self):
        content = HEADER
        content += row(100, 1, 0.10, 0.00, 0.10, 14.0, 0.02, 100, 200, 300)
        content += row(101, 1, 0.10, 0.00, 0.10, 14.0, 0.02, 100, 200, 300)
        with self.assertRaisesRegex(ValueError, "multiple selected centrals|host key collision"):
            summarize_catalog("mocks/MOCK_HAMHOD_SDSS_v5_R19051.0_err_corr", io.BytesIO(content.encode()))

    def test_streaming_tar_checksum_and_ensemble_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "mocks.tar.gz"
            payload = (HEADER + row(100, 1, 0.0, 0.0, 0.1, 14.0, 0.02, 100, 200, 300)).encode()
            with tarfile.open(archive, mode="w:gz") as tar:
                for simulation in range(19000, 19256):
                    for observer in range(8):
                        name = f"mocks/MOCK_HAMHOD_SDSS_v5_R{simulation}.{observer}_err_corr"
                        info = tarfile.TarInfo(name)
                        info.size = len(payload)
                        tar.addfile(info, io.BytesIO(payload))
            size = archive.stat().st_size
            digest = hashlib.md5(archive.read_bytes()).hexdigest()
            with mock.patch.object(summary, "EXPECTED_BYTES", size), mock.patch.object(summary, "EXPECTED_MD5", digest):
                got = summary.summarize_archive(archive)
        self.assertEqual(got["mock_catalogues"], 2048)
        self.assertEqual(got["independent_simulation_boxes"], 256)
        self.assertEqual(got["archive_md5"], digest)
        self.assertEqual(got["richness_bins"]["1"]["galaxies"], 2048)


if __name__ == "__main__":
    unittest.main()
