def calibrate_confidence(raw_confidence):
    """
    Simulates probability calibration.
    In real systems this uses Platt scaling / isotonic regression.
    """
    calibrated = raw_confidence * 0.95
    return round(calibrated, 2)
