import psutil
import platform

def print_machine_resources():
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print(f"CPU Count: {psutil.cpu_count(logical=True)}")
    print(f"CPU Frequency: {psutil.cpu_freq().current:.2f}Mhz")
    print(f"Total Memory: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB")
    print(f"Available Memory: {psutil.virtual_memory().available / (1024 ** 3):.2f} GB")
    print(f"Used Memory: {psutil.virtual_memory().used / (1024 ** 3):.2f} GB")
    print(f"Memory Usage: {psutil.virtual_memory().percent}%")
    print(f"Disk Usage: {psutil.disk_usage('/').percent}%")
    print(f"Disk Total: {psutil.disk_usage('/').total / (1024 ** 3):.2f} GB")
    print(f"Disk Used: {psutil.disk_usage('/').used / (1024 ** 3):.2f} GB")
    print(f"Disk Free: {psutil.disk_usage('/').free / (1024 ** 3):.2f} GB")

if __name__ == "__main__":
    print_machine_resources()
