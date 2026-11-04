"""Crash-aware lossless storage for retained canonical posterior states.

The held-out evaluator needs each Markov-chain state, including repeats after
rejection. Online field moments are not enough to reconstruct nonlinear joint
posterior-predictive scores. This store writes a fixed-capacity float64 NumPy
memmap and a small append-only JSONL index; it does not read held-out data.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


class CanonicalStateStore:
    """Append and verify retained states for one independently run chain.

    The caller owns the output directory and supplies a frozen contract ID.
    Each `append` first flushes the exact state row, then records its checksum
    and transition metadata. An incomplete final JSONL line is discarded on
    resume; an indexed state with a bad checksum is an error, never skipped.
    """

    VERSION = 1
    STATE_FILE = "canonical_states.npy"
    META_FILE = "canonical_states.meta.json"
    INDEX_FILE = "canonical_states.jsonl"

    def __init__(self, directory, *, chain_id, capacity, dimension, contract_id):
        self.directory = Path(directory)
        self.chain_id = str(chain_id)
        self.capacity = int(capacity)
        self.dimension = int(dimension)
        self.contract_id = str(contract_id)
        if (not self.chain_id or not self.contract_id or self.capacity < 1
                or self.dimension < 1):
            raise ValueError("nonempty chain/contract IDs and positive shape required")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.state_path = self.directory / self.STATE_FILE
        self.meta_path = self.directory / self.META_FILE
        self.index_path = self.directory / self.INDEX_FILE
        expected = dict(version=self.VERSION, chain_id=self.chain_id,
            capacity=self.capacity, dimension=self.dimension,
            dtype="<f8", contract_id=self.contract_id)
        present = [p.exists() for p in (self.state_path, self.meta_path, self.index_path)]
        if any(present) and not all(present):
            raise ValueError("partial canonical-state store; preserve and inspect before resume")
        if all(present):
            metadata = json.loads(self.meta_path.read_text())
            if metadata != expected:
                raise ValueError("canonical-state store metadata/contract mismatch")
            self.states = np.load(self.state_path, mmap_mode="r+", allow_pickle=False)
            if (self.states.shape != (self.capacity, self.dimension)
                    or self.states.dtype != np.dtype("<f8")):
                raise ValueError("canonical-state array shape or dtype mismatch")
            self.records = self._read_index(repair_partial_tail=True)
            if self.records:
                self.read_state(len(self.records) - 1)
        else:
            self.states = np.lib.format.open_memmap(self.state_path, mode="w+",
                dtype=np.dtype("<f8"), shape=(self.capacity, self.dimension))
            self.states.flush()
            self.meta_path.write_text(json.dumps(expected, sort_keys=True) + "\n")
            self.index_path.write_bytes(b"")
            self.records = []

    @staticmethod
    def _checksum(state):
        value = np.ascontiguousarray(np.asarray(state, dtype=np.dtype("<f8")))
        return hashlib.sha256(value.tobytes(order="C")).hexdigest()

    def _read_index(self, *, repair_partial_tail):
        with self.index_path.open("r+b" if repair_partial_tail else "rb") as stream:
            content = stream.read()
            last_newline = content.rfind(b"\n")
            if last_newline != len(content) - 1:
                if not repair_partial_tail:
                    raise ValueError("canonical-state index has an incomplete final record")
                stream.truncate(last_newline + 1)
                stream.flush()
                os.fsync(stream.fileno())
                content = content[:last_newline + 1]
        records = []
        for line_number, raw in enumerate(content.splitlines(), start=1):
            try:
                record = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid canonical-state index line {line_number}") from error
            expected_index = len(records)
            checksum = record.get("state_sha256")
            previous_checksum = record.get("previous_state_sha256")
            transition = record.get("transition")
            if (record.get("index") != expected_index
                    or record.get("chain_id") != self.chain_id
                    or record.get("contract_id") != self.contract_id
                    or not isinstance(checksum, str) or len(checksum) != 64
                    or any(ch not in "0123456789abcdef" for ch in checksum)
                    or not isinstance(previous_checksum, str) or len(previous_checksum) != 64
                    or any(ch not in "0123456789abcdef" for ch in previous_checksum)
                    or not isinstance(record.get("accepted"), bool)
                    or not isinstance(transition, int) or isinstance(transition, bool)
                    or transition < 1
                    or not math.isfinite(float(record.get("target_value", math.nan)))
                    or not math.isfinite(float(record.get("previous_target_value", math.nan)))):
                raise ValueError(f"invalid canonical-state metadata at index {expected_index}")
            if expected_index and transition != records[-1]["transition"] + 1:
                raise ValueError("retained chain transitions must be consecutive, including repeats")
            if expected_index and (previous_checksum != records[-1]["state_sha256"]
                    or float(record["previous_target_value"]) != float(records[-1]["target_value"])):
                raise ValueError("canonical-state transition linkage is broken")
            if not record["accepted"] and (checksum != previous_checksum
                    or float(record["target_value"]) != float(record["previous_target_value"])):
                raise ValueError("rejected transition metadata does not retain its prior state")
            records.append(record)
        if len(records) > self.capacity:
            raise ValueError("canonical-state index exceeds fixed array capacity")
        return records

    def __len__(self):
        return len(self.records)

    def append(self, state, *, previous_state, previous_target, transition,
               accepted, target_value):
        """Persist one post-Metropolis state, even when the proposal rejected."""
        if len(self.records) >= self.capacity:
            raise ValueError("canonical-state store capacity exhausted")
        if not isinstance(transition, int) or isinstance(transition, bool) or transition < 1:
            raise ValueError("transition must be a positive integer")
        if self.records and transition != self.records[-1]["transition"] + 1:
            raise ValueError("transition index must continue the retained chain")
        if not isinstance(accepted, (bool, np.bool_)):
            raise ValueError("accepted must be boolean")
        target_value = float(target_value)
        previous_target = float(previous_target)
        if not math.isfinite(target_value) or not math.isfinite(previous_target):
            raise ValueError("retained target value must be finite")
        raw_q = np.asarray(state)
        raw_previous = np.asarray(previous_state)
        if (raw_q.dtype.kind != "f" or raw_q.dtype.itemsize != 8
                or raw_previous.dtype.kind != "f" or raw_previous.dtype.itemsize != 8):
            raise ValueError("lossless canonical storage requires float64 input")
        q = np.asarray(raw_q, dtype=np.dtype("<f8"))
        previous = np.asarray(raw_previous, dtype=np.dtype("<f8"))
        if (q.shape != (self.dimension,) or previous.shape != (self.dimension,)
                or not np.isfinite(q).all() or not np.isfinite(previous).all()):
            raise ValueError("finite canonical state has the wrong dimension")

        index = len(self.records)
        if self.records:
            stored_previous = self.read_state(index - 1, verify=False)
            stored_target = float(self.records[-1]["target_value"])
            if (not np.array_equal(previous, stored_previous)
                    or previous_target != stored_target):
                raise ValueError("transition start differs from the preceding retained state")
            previous = stored_previous
            previous_target = stored_target
        if not accepted and (not np.array_equal(q, previous) or target_value != previous_target):
            raise ValueError("rejected transition must retain the exact prior state and target")
        checksum = self._checksum(q)
        previous_checksum = self._checksum(previous)
        self.states[index] = q
        self.states.flush()
        with self.state_path.open("r+b") as state_file:
            os.fsync(state_file.fileno())
        record = dict(index=index, chain_id=self.chain_id,
            contract_id=self.contract_id, transition=transition,
            accepted=bool(accepted), target_value=target_value,
            state_sha256=checksum, previous_target_value=previous_target,
            previous_state_sha256=previous_checksum)
        encoded = (json.dumps(record, sort_keys=True, allow_nan=False,
            separators=(",", ":")) + "\n").encode("utf-8")
        with self.index_path.open("ab", buffering=0) as stream:
            stream.write(encoded)
            os.fsync(stream.fileno())
        self.records.append(record)
        return dict(record)

    def read_state(self, index, *, verify=True):
        index = int(index)
        if index < 0 or index >= len(self.records):
            raise IndexError("canonical-state index is not committed")
        value = np.array(self.states[index], copy=True)
        if verify and self._checksum(value) != self.records[index]["state_sha256"]:
            raise ValueError(f"canonical-state checksum mismatch at index {index}")
        return value

    def iter_states(self, *, verify=True):
        for index in range(len(self.records)):
            yield self.read_state(index, verify=verify), self.records[index]

    def verify(self):
        """Read/check every committed state; useful before held-out scoring."""
        for index in range(len(self.records)):
            self.read_state(index, verify=True)
        return dict(chain_id=self.chain_id, states=len(self.records),
            dimension=self.dimension, contract_id=self.contract_id,
            all_checksums_valid=True)
