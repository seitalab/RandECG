from functions.augment_search_funcs import *

augment_choices = ["scale", "sine", "square", "sine_p", "square_p",
                    "cutout", "shift", "drop", "flip", "sine_p_w", "square_p_w",
                    "wn_p_w", "sine_a", "square_a", "wn_a", "fir_l_f"]

def get_Max_and_Min(augment_type):
    # augment_type: search-<augmentation>-max<val>-min<val>
    _, augmentation, max_val, min_val = augment_type.split("-")
    assert(augmentation in augment_choices)
    assert(max_val.startswith("max"))
    assert(min_val.startswith("min"))
    max_val = float(max_val[3:])
    min_val = float(min_val[3:])
    return augmentation, max_val, min_val

def fix_aug(X_batch, augmentation_setting):
    """
    Apply fixed augmentaion function.
    """
    augmentation, max_val, min_val = get_Max_and_Min(augmentation_setting)

    if augmentation == "scale":
        X_batch = search_random_scale(X_batch, max_val, min_val)

    if augmentation == "flip":
        X_batch = search_flip_range(X_batch, max_val, min_val)

    if augmentation == "drop":
        X_batch = search_drop_range(X_batch, max_val, min_val)

    if augmentation == "cutout":
        X_batch = search_cutout_range(X_batch, max_val, min_val)

    if augmentation == "shift":
        X_batch = search_shift_range(X_batch, max_val, min_val)

    if augmentation == "sine":
        X_batch = search_sine_noise(X_batch, max_val, min_val)

    if augmentation == "square":
        X_batch = search_square_noise(X_batch, max_val, min_val)

    if augmentation == "sine_p":
        X_batch = search_sine_noise_partial(X_batch, max_val, min_val)

    if augmentation == "square_p":
        X_batch = search_square_noise_partial(X_batch, max_val, min_val)

    if augmentation == "sine_p_w":
        X_batch = search_sine_partial_width(X_batch, max_val, min_val)

    if augmentation == "square_p_w":
        X_batch = search_square_partial_width(X_batch, max_val, min_val)

    if augmentation == "wn_p_w":
        X_batch = search_wn_partial_width(X_batch, max_val, min_val)

    if augmentation == "sine_a":
        X_batch = search_sine_amp(X_batch, max_val, min_val)

    if augmentation == "square_a":
        X_batch = search_square_amp(X_batch, max_val, min_val)

    if augmentation == "fir_l_f":
        X_batch = search_fir_l(X_batch, max_val, min_val)

    if augmentation == "wn_a":
        X_batch = search_wn_amp(X_batch, max_val, min_val)
        
    X_batch = X_batch.astype(np.float32)
    return X_batch
