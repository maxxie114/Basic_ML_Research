# fib gen code created by deepseek
def generate_fibonacci(n):
    fib_sequence = [0, 1]  # Initialize with the first two Fibonacci numbers
    for i in range(2, n):  # Generate up to n Fibonacci numbers
        next_num = fib_sequence[i-1] + fib_sequence[i-2]
        fib_sequence.append(next_num)
    return fib_sequence

# Example usage:
n = 1000  # Change this number to generate more or fewer Fibonacci numbers
fib_numbers = generate_fibonacci(n)
print(fib_numbers)
