
import os
import pty
import time
import random
import tty

# Create a virtual serial connection
master, slave = pty.openpty()
tty.setraw(slave)

port = os.ttyname(slave)

print("Virtual Arduino is running!")
print("Serial port:", port)
print("Keep this terminal open.")

temperature = 29.0
humidity = 69.3
light = 450

try:
    while True:
        temperature += random.uniform(-0.3, 0.3)
        humidity += random.uniform(-0.8, 0.8)
        light += random.randint(-10, 10)

        temperature = max(20, min(38, temperature))
        humidity = max(30, min(90, humidity))
        light = max(0, min(1023, light))

        message = f"{temperature:.1f},{humidity:.1f},{light}\n"

        os.write(master, message.encode())

        print("Sent:", message.strip())

        time.sleep(1)

except KeyboardInterrupt:
    print("\nVirtual Arduino stopped.")

finally:
    os.close(master)
    os.close(slave)
