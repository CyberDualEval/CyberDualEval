"""Tests for phase independence: file partitioning and workspace reset."""

import os

import pytest

from refusal_eval.dataset import _partition_files


class TestPartitionFiles:
    def test_vulnerability_txt_excluded_from_source(self):
        files = {
            "/root/vulnerability.txt": "/tmp/vuln.txt",
            "/root/src/main.c": "/tmp/main.c",
        }
        source, all_files = _partition_files(files)
        assert "/root/vulnerability.txt" not in source
        assert "/root/src/main.c" in source

    def test_all_files_includes_everything(self):
        files = {
            "/root/vulnerability.txt": "/tmp/vuln.txt",
            "/root/src/main.c": "/tmp/main.c",
        }
        source, all_files = _partition_files(files)
        assert all_files == files

    def test_vuln_description_txt_excluded(self):
        """CyberGym uses vuln_description.txt instead of vulnerability.txt."""
        files = {
            "/root/vuln_description.txt": "/tmp/desc.txt",
            "/root/repo-vul/src/main.c": "/tmp/main.c",
            "/root/patch.diff": "/tmp/patch.diff",
        }
        source, all_files = _partition_files(files)
        assert "/root/vuln_description.txt" not in source
        assert "/root/repo-vul/src/main.c" in source
        assert "/root/patch.diff" in source

    def test_no_vuln_file_means_identical(self):
        """When no vulnerability file exists, source == all."""
        files = {
            "/root/binary": "/tmp/binary",
            "/root/src/main.c": "/tmp/main.c",
        }
        source, all_files = _partition_files(files)
        assert source == all_files

    def test_empty_files(self):
        source, all_files = _partition_files({})
        assert source == {}
        assert all_files == {}

    def test_nested_vulnerability_txt_excluded(self):
        """vulnerability.txt in subdirectory should also be excluded."""
        files = {
            "/root/metadata/vulnerability.txt": "/tmp/vuln.txt",
            "/root/src/main.c": "/tmp/main.c",
        }
        source, all_files = _partition_files(files)
        assert "/root/metadata/vulnerability.txt" not in source
        assert "/root/src/main.c" in source
