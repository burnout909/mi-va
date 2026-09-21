"""load_dicom_record reads a 12-lead ECG Waveform Storage file into the canonical layout."""

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")

from mival.signal import LEADS_12  # noqa: E402
from mival.stages.preprocess import load_dicom_record  # noqa: E402

# MDC codes in the channel order MIMIC files use: aVF comes before aVL.
CHANNELS = [("2:1", "Lead I"), ("2:2", "Lead II"), ("2:61", "Lead III"), ("2:62", "aVR, augmented voltage, right"),
            ("2:64", "aVF, augmented voltage, foot"), ("2:63", "aVL, augmented voltage, left"),
            ("2:3", "Lead V1"), ("2:4", "Lead V2"), ("2:5", "Lead V3"), ("2:6", "Lead V4"), ("2:7", "Lead V5"), ("2:8", "Lead V6")]


def write_dicom(path, data_mv, fs=500.0, sensitivity=0.005):
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.9.1.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "ECG"

    item = Dataset()
    item.MultiplexGroupTimeOffset = 0
    item.WaveformOriginality = "ORIGINAL"
    item.NumberOfWaveformChannels = len(CHANNELS)
    item.NumberOfWaveformSamples = data_mv.shape[1]
    item.SamplingFrequency = fs
    item.WaveformBitsAllocated = 16
    item.WaveformSampleInterpretation = "SS"
    channels = []
    for code, meaning in CHANNELS:
        source = Dataset(); source.CodeValue = code; source.CodingSchemeDesignator = "MDC"; source.CodeMeaning = meaning
        unit = Dataset(); unit.CodeValue = "mV"; unit.CodingSchemeDesignator = "UCUM"; unit.CodeMeaning = "millivolt"
        channel = Dataset()
        channel.ChannelSourceSequence = [source]
        channel.ChannelSensitivity = sensitivity
        channel.ChannelSensitivityUnitsSequence = [unit]
        channel.ChannelBaseline = 0.0
        channel.ChannelSampleSkew = 0
        channel.WaveformBitsStored = 16
        channels.append(channel)
    item.ChannelDefinitionSequence = channels
    item.WaveformData = np.round(data_mv.T / sensitivity).astype("<i2").tobytes()
    ds.WaveformSequence = [item]
    # pydicom 3.0 renamed write_like_original to enforce_file_format
    # (inverted sense); support both the laptop's 2.4 and the server's 3.0.
    try:
        ds.save_as(str(path), enforce_file_format=True)
    except TypeError:
        ds.save_as(str(path), write_like_original=False)
    return path


def test_loader_returns_mv_with_normalised_lead_names(tmp_path):
    data = np.zeros((12, 100), dtype=np.float32)
    data[4, 10] = 1.0   # the file's fifth channel is aVF
    path = write_dicom(tmp_path / "x.dcm", data)
    array, source = load_dicom_record(path)
    assert array.shape == (12, 100) and array.dtype == np.float32
    assert source.leads == ("I", "II", "III", "aVR", "aVF", "aVL", "V1", "V2", "V3", "V4", "V5", "V6")
    assert set(source.leads) == set(LEADS_12)
    assert source.sampling_rate_hz == 500.0 and source.n_samples == 100 and source.unit == "mV"
    assert np.isclose(array[4, 10], 1.0, atol=0.005)
