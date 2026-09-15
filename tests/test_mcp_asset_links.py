"""No-network coverage for the bounded SB checkpoint link check."""
from contextlib import redirect_stdout
from email.message import Message
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlsplit


REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_mcp_asset_links", REPO / "scripts/check_mcp_asset_links.py")
links = importlib.util.module_from_spec(spec)
spec.loader.exec_module(links)


class AssetLinkTests(unittest.TestCase):
    def response(self, status=206, filename="checkpoint_e0025.pth.tar", size=1193976342):
        response = MagicMock()
        response.status = status
        response.headers = Message()
        response.headers["Content-Range"] = f"bytes 0-31/{size}"
        response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        response.headers["Content-Type"] = "application/binary"
        response.read.return_value = b"PK\x03\x04" + bytes(28)
        response.__enter__.return_value = response
        return response

    def probe(self, response):
        with patch.object(links, "urlopen", return_value=response) as open_url:
            result = links.check_asset("CDG", "https://example.com/model", "checkpoint_e0025.pth.tar", 1193976342)
        self.assertEqual(open_url.call_args.args[0].get_header("Range"), "bytes=0-31")
        return result

    def test_reads_exactly_32_bytes(self):
        response = self.response()
        self.assertIn("[PASS] CDG", self.probe(response))
        response.read.assert_called_once_with(32)

    def test_rejects_ignored_range_without_reading(self):
        response = self.response(status=200)
        with self.assertRaisesRegex(ValueError, "HTTP 206"):
            self.probe(response)
        response.read.assert_not_called()

    def test_rejects_folder_zip_without_reading(self):
        response = self.response(filename="CDG_v2.zip")
        with self.assertRaisesRegex(ValueError, "wrong download filename"):
            self.probe(response)
        response.read.assert_not_called()

    def test_rejects_wrong_size(self):
        with self.assertRaisesRegex(ValueError, "checkpoint size"):
            self.probe(self.response(size=999))

    def test_rejects_html(self):
        response = self.response()
        response.headers.replace_header("Content-Type", "text/html")
        with self.assertRaisesRegex(ValueError, "text/login"):
            self.probe(response)

    def test_rejects_wrong_signature(self):
        response = self.response()
        response.read.return_value = bytes(32)
        with self.assertRaisesRegex(ValueError, "ZIP header"):
            self.probe(response)

    def test_preserves_dropbox_file_selection(self):
        url = links.download_url("https://www.dropbox.com/scl/fo/folder?rlkey=secret&dl=0&preview=checkpoint_e0025.pth.tar")
        self.assertEqual(parse_qs(urlsplit(url).query), {
            "rlkey": ["secret"], "dl": ["1"], "preview": ["checkpoint_e0025.pth.tar"],
        })

    def test_missing_urls_never_start_downloads(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(links, "urlopen") as open_url, redirect_stdout(io.StringIO()):
            self.assertEqual(links.main(), 1)
        open_url.assert_not_called()

    def test_shell_entrypoint_exits_before_setup(self):
        env = {**os.environ, "CHECK_LINKS_ONLY": "1", "PY": sys.executable,
               "PYTHONDONTWRITEBYTECODE": "1", "DRY_RUN": "",
               "NF_MODEL_URL": "", "FB_MODEL_URL": "", "CDG_MODEL_URL": ""}
        result = subprocess.run(["bash", str(REPO / "scripts/1_data_process.sh")],
                                env=env, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Link check only: no files saved", result.stdout)
        self.assertNotIn("checking python deps", result.stdout)
        self.assertNotIn("checking/downloading public MCP data", result.stdout)


if __name__ == "__main__":
    unittest.main()
