## Analysis  
* Decompose guitar note
* Extract parameters (frequency, module, phase)
* FFT signal

## envelope
* Filter the straightened signal by a convolution with a low-pass FIR filter.
* Demonstrate how N was obtained

## Synthesis
* Add sin (sin extracted from the requirements)
* Multiply the result with the envelope
* Produce the 5th harmony from Beethoven

## Filtering
* Filter a 1000Hz sine out of `note_basson_plus_sine_1000_hz` with a band-stop FIR filter 
    * rejected band: 960 Hz to 1040hz
    * Gain: 0dB
    * N: 6000
* plot the impulse response of the filter
