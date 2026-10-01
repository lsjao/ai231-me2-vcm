# Breadboard + GPIO tutorial (for learning, not just doing)

This explains *why* the wiring in `GPIO_WIRING.md` works, not just what to plug where. Written
to paste into a chat and ask follow-up questions about.

## What a breadboard actually is

A breadboard is a grid of spring clips under the plastic, not just holes. Components pushed
into holes that are electrically connected get wired together with no soldering.

- **The two side rails** (marked + and -, usually red and blue stripes) run the *full length*
  of the board. Everything plugged into the red rail is electrically the same point; same for
  the blue rail. These are for power and ground.
- **The main grid** is split into short columns of 5 holes each, grouped in two halves
  separated by a center gap. Each column of 5 is one electrical point -- plug two things into
  the same column and they're connected; plug them into different columns and they're not,
  even though the columns look identical side by side.

So "wire A to B" on a breadboard really means: put A and B's legs in the *same column* (or run
a jumper wire between two columns).

## Why an LED needs a resistor at all

An LED is not like a light bulb that resists current on its own -- it's a diode, which means
once it turns on, it tries to let through as much current as it can, limited only by
whatever's in series with it. A Raspberry Pi GPIO pin can only safely supply about 16mA. Without
a resistor, an LED would try to pull far more than that the instant it turns on -- burning out
the LED, or worse, damaging the GPIO pin.

The resistor's job is purely to *limit current to a safe level*, using Ohm's law
(`V = I * R`, or rearranged, `I = V / R`). A GPIO pin outputs 3.3V. A typical LED "uses up"
about 2V of that just by conducting (its forward voltage). That leaves about 1.3V for the
resistor to absorb. With a 220 ohm resistor: `I = 1.3V / 220 ohm ≈ 6mA` -- safely under the
16mA limit, comfortably bright. This is why the kit's 220-330 ohm resistors are the ones to use
for LEDs specifically (the kit likely has other values too, for other circuits) -- lower values
let through more current (brighter, but riskier), higher values are dimmer but safer. The
project's choice of 220-330 ohm is a normal, safe middle ground, not a precise requirement.

## LED polarity: why orientation matters

An LED only conducts current in one direction (that's what makes it a diode). The two legs are
different lengths on purpose:

- **Long leg = anode** = the side current flows *into*. Connects toward the positive/GPIO side.
- **Short leg = cathode** = the side current flows *out of*. Connects toward ground.

Get it backwards and the LED simply won't light (it won't break anything at these voltages --
LEDs wired backwards just block the current entirely).

## How a GPIO pin turns an LED on and off

A GPIO pin configured as an output is really just a switch between two states: "3.3V" (high)
or "0V / ground" (low). To light an LED:

`GPIO pin (3.3V when on) -> resistor -> LED long leg -> LED short leg -> GND`

When the code sets the pin high, 3.3V pushes current through the resistor and LED down to
ground, and it lights. When the pin goes low (0V), there's no voltage difference to push
current anywhere, so it's dark. This is the entire mechanism behind `LED.on()` / `LED.off()`
in the `gpiozero` library used in this project's `dispatch.py`.

## RGB LEDs and color mixing

An RGB LED is really three LEDs (red, green, blue) in one package, sharing one common leg.
Every color you see is a *mix* of how bright each of the three is -- yellow is red+green with
no blue, white is all three at full brightness, etc. That's why the RGB LED needs 3 separate
GPIO pins (one per color) plus one shared ground/common leg, versus the single-color LED's one
pin.

**Common-cathode vs common-anode** is about which way that shared leg points:
- *Common-cathode* (this project assumes this, most common in beginner kits): the shared leg
  goes to GND, and each color pin needs to go *high* (3.3V) to light that color -- matches how
  the single-color LED works, so the same mental model applies.
- *Common-anode*: the shared leg goes to 3.3V instead, and confusingly, each color pin needs to
  go *low* to light that color (current flows from the always-on 3.3V rail, through the LED,
  out through the GPIO pin sinking to ground). If your code (which assumes common-cathode)
  gets this backwards -- e.g. wired for common-cathode but the LED is actually common-anode --
  every color reads as inverted: "off" commands light it up, "on" commands don't.

## Dimming: why brightness needs PWM, not just on/off

A plain digital pin only has two states, high or low -- there's no "half-on." To fake a dimmer
brightness, the Pi switches the pin on and off *very fast* (thousands of times per second) and
varies the fraction of time it's on vs off. This is PWM (pulse-width modulation). At 20% "duty
cycle," the pin is high 20% of the time and low 80% -- too fast for your eye to see the
flickering, so it looks like a steady 20%-bright light. `gpiozero.PWMLED` handles this
automatically; setting `.value = 0.2` in code just sets that duty-cycle fraction.

## How this maps to the actual wiring (see GPIO_WIRING.md for the exact pin numbers)

- **Single LED (on/off)**: one GPIO pin -> resistor -> LED long leg; LED short leg -> GND.
  Digital on/off only, no PWM needed -- matches `light_on_off`'s on/off semantics exactly.
- **RGB LED (brightness/color)**: three GPIO pins (R, G, B), each through its own resistor to
  that color's leg; the shared leg -> GND (assuming common-cathode). Each pin is driven with
  PWM so brightness can be any level, and mixing the three gives any color -- matches
  `light_dim_color`'s brightness-and-color semantics.

## Questions worth asking in chat if this doesn't fully click
- "Why does a resistor value being *too low* matter, not just too high?"
- "What actually happens electrically when I wire an LED backwards?"
- "How does gpiozero decide the PWM frequency, and does it matter here?"
- "If I wanted a fourth color the RGB LED can't make directly, how would I get it?"
