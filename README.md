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
* Filter 1000Hz with a band-stop FIR filter 
    * rejected band: 960 Hz to 1040hz
    * Gain: 0dB
    * N: 6000
* Impulse response for the filtered 1000Hz sin
