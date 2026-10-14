import numpy as np
import pytest

from wheatpheno.features import (BANDS, block_mean, extract_features, quantize,
                              spectral_features, structural_features, texture_features)


def bands(shape=(3, 3)):
    return {key: np.full(shape, value) for key, value in
            {"B": 0.1, "G": 0.2, "R": 0.3, "RE": 0.4, "NIR": 0.8}.items()}


def test_ndre_is_normalized_difference_and_nir2_is_separate():
    result = spectral_features(bands(), include_nir2=True)
    assert len(result) == 16
    np.testing.assert_allclose(result["NDRE"], 1 / 3)
    np.testing.assert_allclose(result["NIR2"], 1.6)
    assert "NIR2" not in spectral_features(bands())


def test_tcari_profiles_are_explicit_and_do_not_change_ndre():
    standard = spectral_features(bands())
    historical = spectral_features(bands(), tcari_profile="historical")
    np.testing.assert_allclose(standard["TCARI"], 3 * (0.1 - 0.2 * 0.2 * (0.4 / 0.3)))
    np.testing.assert_allclose(historical["TCARI"], 3 * 0.1 - 0.2 * 0.2 * (0.8 / 0.3))
    np.testing.assert_equal(standard["NDRE"], historical["NDRE"])


def test_spectral_zero_denominators_and_missingness():
    data = {name: np.zeros((2, 2)) for name in BANDS}
    result = spectral_features(data)
    assert np.isnan(result["NDRE"]).all()
    assert np.isnan(result["RVI"]).all()
    assert (result["OSAVI"] == 0).all()
    data = bands()
    data["RE"][1, 1] = np.nan
    result = spectral_features(data)
    assert np.isnan(result["NDRE"][1, 1])
    assert np.isfinite(result["NDVI"][1, 1])


@pytest.mark.parametrize("bad", [-0.1, 1.2, 255])
def test_spectral_rejects_wrong_reflectance_scale(bad):
    data = bands()
    data["B"][0, 0] = bad
    with pytest.raises(ValueError, match="reflectance fractions"):
        spectral_features(data)


def test_spectral_rejects_misaligned_or_missing_bands():
    data = bands()
    data["B"] = np.zeros((2, 2))
    with pytest.raises(ValueError, match="same shape"):
        spectral_features(data)
    del data["B"]
    with pytest.raises(ValueError, match="exactly"):
        spectral_features(data)


def test_fixed_quantization_keeps_true_zero_and_missing_distinct():
    q, valid = quantize([[0, 0.5, 1, np.nan]], levels=8)
    np.testing.assert_array_equal(q, [[0, 4, 7, -1]])
    np.testing.assert_array_equal(valid, [[True, True, True, False]])
    with pytest.raises(ValueError, match="outside"):
        quantize([[1.1]])


def test_minmax_quantization_ignores_nodata():
    q, _ = quantize([[0.3, 0.4, 0.5, np.nan]], levels=8, policy="image_minmax")
    np.testing.assert_array_equal(q, [[0, 4, 7, -1]])


def test_glcm_matches_hand_calculated_two_pair_distribution():
    image = np.tile([0.0, 1.0, 0.0], (3, 1))
    result = texture_features(image, levels=2, window=3)
    expected = {"asm": 0.5, "contrast": 1, "dissimilarity": 1,
                "energy": np.sqrt(0.5), "entropy": np.log(2),
                "homogeneity": 0.5, "mean": 0.5, "std": 0.5, "correlation": -1}
    for name, value in expected.items():
        assert result[name][1, 1] == pytest.approx(value)


def test_standard_glcm_excludes_neighbor_outside_the_local_patch():
    image = np.zeros((7, 7))
    image[:, 5] = 1
    # Center (3,3) has columns 2..4 in its 3x3 patch; column5 cannot enter.
    result = texture_features(image, levels=2, window=3)
    assert result["contrast"][3, 3] == 0
    assert result["asm"][3, 3] == 1
    assert result["mean"][3, 3] == 0


@pytest.mark.parametrize("offset", [(0, 1), (0, -1), (1, 0), (-1, 1), (2, -2)])
def test_standard_glcm_matches_direct_patch_reference(offset):
    image = np.random.default_rng(38).uniform(0, 1, (7, 8))
    image[2, 2] = np.nan
    actual = texture_features(image, levels=4, window=5, offset=offset)
    q, valid = quantize(image, levels=4)
    for r, c in [(0, 0), (3, 4), (6, 7)]:
        counts = np.zeros((4, 4), dtype=float)
        for rr in range(max(0, r-2), min(7, r+3)):
            for cc in range(max(0, c-2), min(8, c+3)):
                nr, nc = rr+offset[0], cc+offset[1]
                if (max(0, r-2) <= nr < min(7, r+3) and
                        max(0, c-2) <= nc < min(8, c+3) and valid[rr,cc] and valid[nr,nc]):
                    counts[q[rr,cc], q[nr,nc]] += 1
        p = counts/counts.sum()
        i, j = np.indices(p.shape)
        mu_x, mu_y = (i*p).sum(), (j*p).sum()
        sd_x = np.sqrt(((i-mu_x)**2*p).sum())
        sd_y = np.sqrt(((j-mu_y)**2*p).sum())
        assert actual["contrast"][r,c] == pytest.approx(((i-j)**2*p).sum())
        assert actual["asm"][r,c] == pytest.approx((p*p).sum())
        assert actual["mean"][r,c] == pytest.approx(mu_x)
        assert actual["std"][r,c] == pytest.approx(sd_x)
        corr = ((i-mu_x)*(j-mu_y)*p).sum()/(sd_x*sd_y) if sd_x*sd_y else 1.
        assert actual["correlation"][r,c] == pytest.approx(corr)


def test_standard_glcm_constant_descriptor_semantics():
    result = texture_features(np.full((5, 5), 0.6))
    assert result["mean"][2, 2] == pytest.approx(4)
    assert result["std"][2, 2] == pytest.approx(0)
    assert result["correlation"][2, 2] == pytest.approx(1)
    assert result["asm"][2, 2] == pytest.approx(1)


def test_historical_glcm_preserves_original_nonstandard_equations():
    result = texture_features(np.full((3, 3), 255, dtype=np.uint8), profile="historical", levels=2, window=3)
    assert result["mean"][1, 1] == pytest.approx(0.25)
    assert result["std"][1, 1] == pytest.approx(np.sqrt(0.75))
    assert result["correlation"][1, 1] == pytest.approx(0.75 ** 2 / (0.75 + 1e-10))
    with pytest.raises(ValueError, match="horizontal"):
        texture_features(np.ones((3, 3)), profile="historical", offset=(1, 0))


def test_historical_glcm_matches_original_nonconstant_window_equations():
    image = np.arange(20, dtype=float).reshape(4, 5) / 20
    actual = texture_features(image, profile="historical", levels=4, window=3)
    q, _ = quantize(image, levels=4, policy="image_minmax", historical=True)
    neighbor = np.column_stack([q[:,1:], q[:,-1]])
    origins, targets = np.pad(q, 1, mode="reflect"), np.pad(neighbor, 1, mode="reflect")
    i, j = np.indices((4,4))
    for r,c in [(0,0), (1,2), (3,4)]:
        p = np.zeros((4,4), dtype=float)
        np.add.at(p, (origins[r:r+3,c:c+3].ravel(), targets[r:r+3,c:c+3].ravel()), 1)
        p /= 9
        mean = (p*i).sum()/16
        std = np.sqrt(((p*i-mean)**2).sum())
        correlation = (p*(i-mean)*(j-mean)).sum()/(std**2+1e-10)
        assert actual["mean"][r,c] == pytest.approx(mean)
        assert actual["std"][r,c] == pytest.approx(std)
        assert actual["correlation"][r,c] == pytest.approx(correlation)
        assert actual["contrast"][r,c] == pytest.approx((p*(i-j)**2).sum())
        assert actual["asm"][r,c] == pytest.approx((p**2).sum())


@pytest.mark.parametrize("profile", ["standard", "historical"])
def test_texture_nodata_is_not_interpreted_as_dark_reflectance(profile):
    image = np.full((5, 5), 0.5)
    image[2, 2] = np.nan
    result = texture_features(image, profile=profile)
    assert all(np.isnan(value[2, 2]) for value in result.values())
    empty = texture_features(np.full((3, 3), np.nan), profile=profile)
    assert all(np.isnan(value).all() for value in empty.values())


def test_texture_valid_pair_threshold_rejects_sparse_context():
    image = np.full((5, 5), np.nan)
    image[2, 2:4] = [0.2, 0.8]
    result = texture_features(image, window=5, minimum_valid_fraction=0.5)
    assert np.isnan(result["mean"][2, 2])


def test_structural_native_percentile_and_canopy_fraction():
    result = structural_features([[0.1, 0.2], [0.3, 0.4]], [[0, 1], [1, 1]], block_size=2)
    assert len(result) == 12
    assert result["CC"][0, 0] == pytest.approx(0.75)
    assert result["CHM_PCT95"][0, 0] == pytest.approx(0.385)
    assert result["CHM_MEAN"][0, 0] == pytest.approx(0.25)
    assert result["CHM_STD"][0, 0] == pytest.approx(np.std([0.1, 0.2, 0.3, 0.4]))


def test_structural_height_mask_does_not_change_cc_or_impose_osavi():
    height = np.array([[0.1, 0.2], [0.3, 0.4]])
    mask = np.array([[False, True], [True, True]])
    result = structural_features(height, mask, block_size=2, height_mask=mask)
    assert result["CC"][0, 0] == 0.75
    assert result["CHM_PCT95"][0, 0] == pytest.approx(0.39)
    empty = structural_features(np.zeros((2, 2)), np.zeros((2, 2)), block_size=2)
    assert empty["CHM_MEAN"][0, 0] == 0
    assert np.isnan(empty["CHM_CV"][0, 0])


def test_partial_blocks_and_unknown_masks_do_not_add_zero_samples():
    height = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0]])
    mask = np.array([[1.0, 0.0, np.nan], [1.0, 1.0, 1.0]])
    result = structural_features(height, mask, block_size=2)
    np.testing.assert_allclose(result["CC"], [[2 / 3, 1]])
    np.testing.assert_allclose(result["CHM_MEAN"], [[7 / 3, 4.5]])
    np.testing.assert_allclose(block_mean(height, 2), [[7 / 3, 4.5]])


def test_all_feature_extraction_has_documented_grid_and_correct_ndre():
    data = bands((7, 5))
    result = extract_features(data, np.ones((7, 5)), np.ones((7, 5)), block_size=3)
    assert len(result) == 72
    assert {x.shape for x in result.values()} == {(3, 2)}
    np.testing.assert_allclose(result["NDRE"], 1 / 3)
    compatibility = extract_features(data, np.ones((7, 5)), np.ones((7, 5)), block_size=3, include_nir2=True)
    assert len(compatibility) == 73
    np.testing.assert_allclose(compatibility["NIR2"], 1.6)


@pytest.mark.parametrize("kwargs", [{"window": 2}, {"levels": 1}, {"offset": (0, 0)}, {"minimum_valid_fraction": 0}])
def test_invalid_texture_settings_fail(kwargs):
    with pytest.raises(ValueError):
        texture_features(np.ones((3, 3)), **kwargs)
