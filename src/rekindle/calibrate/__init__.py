"""Deriving rekindle's thresholds from YOUR photographs, once.

    rekindle calibrate

Not a settings page: a guided sequence with a visible position, one plain
question at each step, and an end. It shows real photographs from the library
it is pointed at, chosen near the current cut, and turns the answers into
numbers - the same method that produced the shipped values, automated.

The layers, smallest first:

    judge      answers -> a number, plus how much to trust it
    sampling   which photographs to ask about, and in what order
    plan       the sequence: what is asked, when, and in which mode
    state      calibration.json: finished, when, against how many, how far
    preview    what a proposed number does to this library, before writing
    impact     the exit question - which memories would actually change
    session    the state machine both front ends drive
    cli        the terminal front end

Nothing here imports numpy, torch, onnxruntime or Pillow at module level. The
first-run experience has to work on a plain `uv sync`, which is four packages.
"""
