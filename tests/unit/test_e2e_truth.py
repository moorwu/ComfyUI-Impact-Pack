import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "person_e2e"))

import run_e2e

FX = {
    "people": [
        {"key": "man_back", "gender": "male"},
        {"key": "woman_yellow", "gender": "female"},
        {"key": "boy_far", "gender": "male"},
    ]
}


def test_expected_positions_v1_format():
    assert run_e2e.expected_positions(FX, {"expected": [3, 1]}) == [1, 3]


def test_expected_positions_by_pos():
    assert run_e2e.expected_positions(FX, {"expect_pos": [2]}) == [2]


def test_expected_positions_by_keys():
    assert run_e2e.expected_positions(FX, {"expect": ["boy_far", "man_back"]}) == [1, 3]


def test_case_ok_ids_only():
    ok, exp, why = run_e2e.case_ok([2], 1, FX, {"expect": ["woman_yellow"]})
    assert ok and exp == [2] and why == ""


def test_case_ok_wrong_ids():
    ok, exp, why = run_e2e.case_ok([1], 1, FX, {"expect": ["woman_yellow"]})
    assert not ok and exp == [2] and "ids" in why


def test_case_ok_expect_face_true_and_false():
    assert run_e2e.case_ok([2], 1, FX, {"expect": ["woman_yellow"], "expect_face": True})[0]
    ok, _, why = run_e2e.case_ok([1], 1, FX, {"expect": ["man_back"], "expect_face": False})
    assert not ok and "face" in why
    assert run_e2e.case_ok([1], 0, FX, {"expect": ["man_back"], "expect_face": False})[0]


def test_unknown_key_raises():
    with pytest.raises(ValueError, match="unknown person key"):
        run_e2e.expected_positions(FX, {"expect": ["nobody"]})


def test_selection_workflow_sam_wiring():
    case = {"index": "", "gender": "any", "description": ""}
    wf = run_e2e.selection_workflow("x.png", case, False, "p")
    assert "20" not in wf and "sam_model" not in wf["5"]["inputs"]
    wf = run_e2e.selection_workflow("x.png", case, False, "p", sam="sam_vit_b_01ec64.pth")
    assert wf["20"] == {"class_type": "SAMLoader", "inputs": {"model_name": "sam_vit_b_01ec64.pth", "device_mode": "AUTO"}}
    assert wf["5"]["inputs"]["sam_model"] == ["20", 0]
    dwf = run_e2e.detailer_workflow("x.png", case, "p", "person", "prompt", sam="sam_vit_b_01ec64.pth")
    assert dwf["5"]["inputs"]["sam_model"] == ["20", 0] and dwf["14"]["inputs"]["segs"] == ["6", 1]


def test_selection_workflow_uses_segm_output_and_calibrated_defaults():
    wf = run_e2e.selection_workflow("x.png", {"index": "", "gender": "any", "description": ""}, False, "p")
    inputs = wf["5"]["inputs"]
    assert inputs["person_segm_detector"] == ["2", 1]
    assert inputs["min_person_ratio"] == run_e2e.MIN_PERSON_RATIO == 0.015
    assert inputs["min_relative_size"] == run_e2e.MIN_RELATIVE_SIZE == 0.31
    assert inputs["min_sharpness"] == run_e2e.MIN_SHARPNESS
    assert inputs["extra_person_detector"] == ["21", 0] and inputs["head_detector"] == ["22", 0]
    assert wf["21"]["inputs"]["model_name"] == "bbox/person_detect_v1.1_m.pt"
    assert wf["22"]["inputs"]["model_name"] == "bbox/head_detect_v2.0_s_yv11.pt"
