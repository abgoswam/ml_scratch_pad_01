import time
import random

print("Starting program...")

# Run in a loop
for i in range(100):
    print(f"Sleeping at {i}...")
    time.sleep(16)
    # Generate and print a random number
    random_number = random.random()
    print(f"Random number: {random_number}")
    print()  # Empty line for readability

print("Program completed!")