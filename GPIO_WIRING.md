# Breadboard LED wiring guide

Matches the pins already coded in `src/vcm/dispatch.py`'s `GPIOLightController` -- wire it
exactly like this and the code will just work, no pin renumbering needed.

Physical pin numbers below refer to position on the Pi 4's 40-pin GPIO header (counting along
the header, not the BCM/GPIO number) -- easier to count by hand than BCM numbers.

## What you need from the kit
- 1 single-color LED (for on/off)
- 1 RGB LED (for brightness/color) -- almost certainly **common-cathode** (4 legs: the
  longest leg is the shared one). If your kit's datasheet says common-*anode* instead, see the
  troubleshooting note at the bottom before wiring the common leg.
- 4 resistors, 220-330 ohm (any value in that range is fine for this)
- Jumper wires + breadboard

## 1. Single-color LED (on/off)

| LED leg | Connects to |
|---|---|
| Long leg (anode) -- through a resistor | Physical pin 11 (GPIO17) |
| Short leg (cathode) | Physical pin 9 (GND) |

Order on the breadboard: `Pi pin 11 -> jumper -> resistor -> LED long leg`, and
`LED short leg -> jumper -> Pi pin 9`.

## 2. RGB LED (brightness/color)

| RGB LED leg | Connects to |
|---|---|
| Red leg -- through a resistor | Physical pin 15 (GPIO22) |
| Green leg -- through a resistor | Physical pin 16 (GPIO23) |
| Blue leg -- through a resistor | Physical pin 18 (GPIO24) |
| Common leg (longest of the 4) -- **no resistor** | Physical pin 14 (GND) |

Physical pins 15 and 16 are right next to each other (same row of the header), and 18 is the
next row over, so this stays a short, simple run of wires.

## 3. Testing it

Once wired, power the Pi on and run (needs `gpiozero` installed -- it's already in
`requirements-pi.txt`):

```
cd ~/vcm/src && source ../.venv/bin/activate
python -m vcm.gpio_smoke_test
```

It cycles: off -> full white -> red -> green -> blue -> yellow -> off, printing each step so
you can match what you see against what it says. If the single LED doesn't light for
"on/off", double check pin 11/9. If the RGB LED lights up the *wrong* colors, check the wiring
against the table above -- LED leg order varies by package, so double-check which leg is
which against your specific LED before assuming the code is wrong.

## Troubleshooting: colors are inverted (e.g. everything's on when it should be off, or off
when it should be on)

That means the RGB LED is common-**anode**, not common-cathode -- this needs both a rewire
*and* a one-line code change, not just a rewire:
1. Move the common leg from GND (pin 14) to a 3.3V pin (e.g. physical pin 1 or 17) instead.
2. Tell me (or edit `GPIOLightController.__init__` in `src/vcm/dispatch.py` yourself) to pass
   `active_high=False` to the three `PWMLED(...)` calls -- `gpiozero` does *not* auto-detect
   this, it has to be told which way the LED is wired.

Wiring the common leg to GND first and testing is the safe way to find out which kind you
have -- a common-anode LED just won't light correctly when wired for common-cathode, it won't
damage anything either way.
